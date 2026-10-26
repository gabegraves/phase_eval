import json
import multiprocessing
import threading
import time

import pytest

from phaseeval.kernel.errors import Busy
from phaseeval.kernel.lock import ProcessOwner
from tests.support.store import ADD, OWNER, REFINE, World


def test_a_refit_beyond_the_slots_is_busy_without_waiting():
    from phaseeval.workbench.executors import RefitAdmission

    admission = RefitAdmission(slots=2, per_principal=1)
    with admission.try_acquire('one'), admission.try_acquire('two'):
        start = time.monotonic()
        with pytest.raises(Busy):
            with admission.try_acquire('three'):
                pytest.fail('admitted beyond the slots')
        assert time.monotonic() - start < 0.5
    with admission.try_acquire('one'):
        with pytest.raises(Busy):
            with admission.try_acquire('one'):
                pytest.fail('admitted beyond the principal limit')
        with admission.try_acquire('two'):
            pass


def test_admission_refusal_creates_no_claim_and_same_process_retry_succeeds(world):
    from phaseeval.workbench.executors import RefitAdmission

    world.store._admission = RefitAdmission(slots=1, per_principal=1)
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    before = sorted((world.store.root / 'claims').rglob('*'))
    with world.store._admission.try_acquire('another'):
        with pytest.raises(Busy):
            world.act(OWNER, parent, 'refit', REFINE)
    assert sorted((world.store.root / 'claims').rglob('*')) == before
    assert world.act(OWNER, parent, 'refit', REFINE).attempt.outcome.status == 'completed'


@pytest.mark.parametrize('fails', [False, True])
def test_the_workdir_is_removed_on_success_and_failure(world, fails):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    paths = []
    execute = world.store._execute

    def run(ctx, action):
        paths.append(ctx.workdir)
        if fails:
            raise RuntimeError('failed')
        return execute(ctx, action)

    world.store._execute = run
    if fails:
        with pytest.raises(RuntimeError):
            world.act(OWNER, parent, 'refit', REFINE)
    else:
        world.act(OWNER, parent, 'refit', REFINE)
    assert paths and all(not path.exists() for path in paths)
    assert not list((world.store.root / 'scratch').iterdir())


def _scratch_owner(root, ready, child=None):
    world = World(root)
    parent = world.act(OWNER, world.root, 'add', ADD).digest

    def run(ctx, action):
        if child is not None:
            ctx.register_child('stage1', child.model_copy(update={'start_token': 'dead'}))
            ctx.register_child('stage2', child)
        ready.set()
        threading.Event().wait(30)
        raise RuntimeError('expired')

    world.store._execute = run
    world.act(OWNER, parent, 'refit', REFINE)


def test_a_second_open_sweeps_only_dead_owners_directories(tmp_path):
    root = tmp_path / 'store'
    ctx = multiprocessing.get_context('spawn')
    ready = ctx.Event()
    process = ctx.Process(target=_scratch_owner, args=(root, ready))
    process.start()
    try:
        assert ready.wait(15)
        paths = list((root / 'scratch').iterdir())
        assert len(paths) == 1
        owner = json.loads((paths[0] / 'owner.json').read_bytes())
        assert owner['generation'] == 0 and owner['slot']
        World(root)
        assert paths[0].exists()
        process.kill()
        process.join(15)
        World(root)
        assert not paths[0].exists()
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def test_live_stage_children_block_takeover_and_scratch_sweep(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    execute = world.store._execute
    workdirs = []

    def run(ctx, action):
        workdirs.append(ctx.workdir)
        ctx.register_child('stage1', ProcessOwner.current().model_copy(update={'start_token': 'dead'}))
        ctx.register_child('stage2', ProcessOwner.current())
        raise RuntimeError('owner failed after stage two')

    world.store._execute = run
    with pytest.raises(RuntimeError):
        world.act(OWNER, parent, 'refit', REFINE)
    world.store._execute = execute
    World(world.store.root)
    assert workdirs[0].exists()
    with pytest.raises(Busy):
        world.act(OWNER, parent, 'refit', REFINE)
    assert not world.refine.calls


def test_child_registration_requires_the_current_generation(world):
    parent = world.act(OWNER, world.root, 'add', ADD).digest
    callbacks = []
    execute = world.store._execute

    def run(ctx, action):
        callbacks.append(ctx.register_child)
        return execute(ctx, action)

    world.store._execute = run
    world.act(OWNER, parent, 'refit', REFINE)
    from phaseeval.kernel.errors import Conflict
    with pytest.raises(Conflict):
        callbacks[0]('late', ProcessOwner.current())


def test_a_live_stage_child_blocks_recovery_after_its_owner_is_killed(tmp_path):
    root = tmp_path / 'store'
    ctx = multiprocessing.get_context('spawn')
    ready = ctx.Event()
    process = ctx.Process(target=_scratch_owner, args=(root, ready, ProcessOwner.current()))
    process.start()
    try:
        assert ready.wait(15)
        scratch, = (root / 'scratch').iterdir()
        process.kill()
        process.join(15)
        recovered = World(root)
        assert scratch.exists()
        with pytest.raises(Busy):
            recovered.act(OWNER, recovered.tree()[-1], 'refit', REFINE)
        assert not recovered.refine.calls
    finally:
        if process.is_alive():
            process.kill()
        process.join()


def test_open_preserves_scratch_without_an_owner_record(world):
    legacy = world.store.root / 'scratch' / 'legacy-refit'
    legacy.mkdir(parents=True)
    (legacy / 'result').write_bytes(b'unowned legacy output')
    reopened = World(world.store.root)
    assert (legacy / 'result').read_bytes() == b'unowned legacy output'
    assert reopened.store.view(OWNER, reopened.root).digest == world.root
