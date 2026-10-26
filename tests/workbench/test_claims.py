"""Exercise durable action claims through the public store and HTTP entry points."""

import hashlib
import json
import multiprocessing
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from phaseeval.kernel.errors import Busy, Conflict
from phaseeval.kernel.lock import ProcessOwner
from phaseeval.workbench.store import ActionRequest, IdempotencyScope, idempotency_slot, request_digest
from tests.support.store import ADD, OWNER, REFINE, World


def _slot(world, key):
    return idempotency_slot(IdempotencyScope(principal_id=OWNER.principal_id, root=world.root, run=None), key)


def _seed_claim(world, parent, owner, generation=0):
    slot = _slot(world, 'refit')
    path = world.store.root / 'claims' / slot / f'claim-{generation}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(slot=slot, generation=generation, request_sha256=request_digest(parent, REFINE),
                                   principal_id=OWNER.principal_id, owner=owner.model_dump())))
    return path


def _worker(root, parent, ready, release, results, barrier=None):
    world = World(root)
    execute = world.store._execute

    def held(ctx, action):
        ready.set()
        assert release.wait(15)
        return execute(ctx, action)

    world.store._execute = held
    if barrier is not None:
        barrier.wait(15)
    try:
        result = world.act(OWNER, parent, 'refit', REFINE)
        results.put(('ok', result.digest, len(world.refine.calls)))
    except Busy:
        results.put(('busy',))


def test_concurrent_requests_with_one_key_execute_once(world):
    from phaseeval.workbench.executors import RefitAdmission

    world.store._admission = RefitAdmission(slots=8, per_principal=8)
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    start, entered, release = threading.Barrier(8), threading.Event(), threading.Event()
    execute = world.store._execute
    completed = queue.Queue()

    def held(ctx, action):
        entered.set()
        assert release.wait(10)
        return execute(ctx, action)

    world.store._execute = held

    def act(_):
        start.wait(10)
        try:
            return world.act(OWNER, parent, 'refit', REFINE).digest
        except Busy:
            completed.put('busy')
            return 'busy'

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(act, i) for i in range(8)]
        try:
            assert entered.wait(10)
            assert [completed.get(timeout=5) for _ in range(7)] == ['busy'] * 7
        finally:
            release.set()
        results = [f.result() for f in futures]
    assert results.count('busy') == 7
    assert len(world.refine.calls) == 1
    assert world.act(OWNER, parent, 'refit', REFINE).digest == next(r for r in results if r != 'busy')


def test_two_processes_with_one_key_execute_once(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    ctx = multiprocessing.get_context('spawn')
    ready, release, results, barrier = ctx.Event(), ctx.Event(), ctx.Queue(), ctx.Barrier(2)
    processes = [ctx.Process(target=_worker, args=(world.store.root, parent, ready, release, results, barrier)) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert ready.wait(15)
        assert results.get(timeout=15) == ('busy',)
        release.set()
        status, digest, calls = results.get(timeout=15)
        assert (status, calls) == ('ok', 1)
        assert world.act(OWNER, parent, 'refit', REFINE).digest == digest
    finally:
        release.set()
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.kill()
                process.join()
        assert [p.exitcode for p in processes] == [0, 0]


def test_a_dead_owners_claim_is_taken_over_and_the_retry_converges(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    ctx = multiprocessing.get_context('spawn')
    ready, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(target=_worker, args=(world.store.root, parent, ready, release, results))
    process.start()
    try:
        assert ready.wait(15)
        process.kill()
        process.join(15)
        first = world.store.root / 'claims' / _slot(world, 'refit') / 'claim-0'
        assert first.exists()
        result = world.act(OWNER, parent, 'refit', REFINE)
        assert world.act(OWNER, parent, 'refit', REFINE).digest == result.digest
        assert first.with_name('claim-1').exists()
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def test_a_live_owners_claim_is_busy(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    _seed_claim(world, parent, ProcessOwner.current())
    with pytest.raises(Busy):
        world.act(OWNER, parent, 'refit', REFINE)
    assert not world.refine.calls


def test_a_different_request_on_a_bound_slot_is_a_conflict(bench):
    parent = bench.post(OWNER, f'/{bench.root}/actions', {'steps': [{'idempotency_key': 'add', 'action': ADD}]}).json()['attempts'][0]['digest']
    execute = bench.store._execute

    def failed(*_):
        raise RuntimeError('executor failed')

    bench.store._execute = failed
    with pytest.raises(RuntimeError, match='executor failed'):
        bench.store.act(OWNER, parent, ActionRequest(idempotency_key='refit', action=REFINE))
    bench.store._execute = execute
    changed = REFINE | {'recipe': 'pawley_only'}
    response = bench.post(OWNER, f'/{parent}/actions', {'steps': [{'idempotency_key': 'refit', 'action': changed}]})
    assert response.status_code == 409
    response = bench.post(OWNER, f'/{parent}/actions', {'steps': [{'idempotency_key': 'refit', 'action': REFINE}]})
    assert response.status_code == 201


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_an_exception_abandons_the_generation_and_a_same_process_retry_succeeds(world, failure):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    execute = world.store._execute

    def failed(*_):
        raise failure('interrupted')

    world.store._execute = failed
    with pytest.raises(failure):
        world.act(OWNER, parent, 'refit', REFINE)
    claim = world.store.root / 'claims' / _slot(world, 'refit') / 'claim-0'
    assert claim.with_name('abandoned-0').exists()
    world.store._execute = execute
    result = world.act(OWNER, parent, 'refit', REFINE)
    assert claim.with_name('claim-1').exists()
    assert world.act(OWNER, parent, 'refit', REFINE).digest == result.digest


def test_a_taken_over_owner_never_commits(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    execute = world.store._execute

    def replaced(ctx, action):
        _seed_claim(world, parent, ProcessOwner.current(), generation=1)
        return execute(ctx, action)

    world.store._execute = replaced
    with pytest.raises(Conflict):
        world.act(OWNER, parent, 'refit', REFINE)
    slot = world.store.root / 'claims' / _slot(world, 'refit')
    assert not (slot / 'commit').exists()
    assert not (slot / 'abandoned-1').exists()
    assert world.tree() == [world.root, parent]


def test_a_retry_repairs_markers_after_commit_publication(world, monkeypatch):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    with monkeypatch.context() as patch:
        def failed(*_):
            raise OSError('marker publication failed')
        patch.setattr(world.store, '_mark', failed)
        with pytest.raises(OSError, match='marker publication'):
            world.act(OWNER, parent, 'refit', REFINE)
    result = world.act(OWNER, parent, 'refit', REFINE)
    assert len(world.refine.calls) == 1
    assert world.tree() == [world.root, parent, result.digest]


def test_legacy_commits_without_claim_zero_still_bind_the_request(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    result = world.act(OWNER, parent, 'refit', REFINE)
    slot = _slot(world, 'refit')
    legacy = world.store.root / 'idem' / slot
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_bytes(json.dumps({'child': result.digest, 'request_sha256': request_digest(parent, REFINE)}).encode())
    shutil.rmtree(world.store.root / 'claims' / slot, ignore_errors=True)
    with pytest.raises((Conflict, ValueError)):
        world.act(OWNER, parent, 'refit', REFINE | {'recipe': 'pawley_only'})
    assert world.act(OWNER, parent, 'refit', REFINE).digest == result.digest
    assert len(world.refine.calls) == 1


def test_slot_and_request_digest_encodings_are_unchanged():
    scope = IdempotencyScope(principal_id='rater', root='a' * 64, run=None)
    action = {'verb': 'refine', 'parameters': {'z': 1, 'a': ['é', None]}}
    expected_slot = hashlib.sha256(json.dumps([scope.model_dump(mode='json'), 'key'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    expected_request = hashlib.sha256(json.dumps(['b' * 64, action], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert idempotency_slot(scope, 'key') == expected_slot
    assert request_digest('b' * 64, action) == expected_request


def test_a_live_claim_returns_429_with_a_retry_hint(bench):
    parent = bench.store.act(OWNER, bench.root, ActionRequest(idempotency_key='add', action=ADD)).digest
    _seed_claim(bench, parent, ProcessOwner.current())
    response = bench.post(OWNER, f'/{parent}/actions', {'steps': [{'idempotency_key': 'refit', 'action': REFINE}]})
    assert response.status_code == 429
    assert response.headers['Retry-After'] == '30'


@pytest.mark.parametrize('takeover', [False, True])
def test_commit_rechecks_validate_a_legacy_request_even_after_fence_loss(world, takeover):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    execute = world.store._execute

    def replaced(ctx, action):
        if takeover:
            _seed_claim(world, parent, ProcessOwner.current(), generation=1)
        legacy = world.store.root / 'idem' / _slot(world, 'refit')
        legacy.parent.mkdir(exist_ok=True)
        legacy.write_text(json.dumps({'child': parent, 'request_sha256': 'f' * 64}))
        return execute(ctx, action)

    world.store._execute = replaced
    with pytest.raises(Conflict):
        world.act(OWNER, parent, 'refit', REFINE)
    assert world.tree() == [world.root, parent]


def test_child_registration_requires_the_lock_and_the_current_generation(world):
    from phaseeval.workbench.store import ClaimRecord, register_child_locked

    claim = ClaimRecord(slot='a' * 64, generation=0, request_sha256='b' * 64,
                        principal_id=OWNER.principal_id, owner=ProcessOwner.current())
    with pytest.raises(AssertionError, match='requires the store lock'):
        register_child_locked(world.store.ledger, None, claim, claim, 'stage', ProcessOwner.current())
    with world.store.ledger.transaction(), pytest.raises(Conflict):
        register_child_locked(world.store.ledger, None, claim.model_copy(update={'generation': 1}),
                              claim, 'stage', ProcessOwner.current())
