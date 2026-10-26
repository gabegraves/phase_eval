import multiprocessing
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st

from phaseeval.api.auth import CaseOperation
from phaseeval.workbench import cli
from phaseeval.workbench.labels import (
    Assignment, AssignmentV1, FamilyFacts, LabelRequest, LabelStore, Rater, RevealOpened, SignOff, SignOffRequest,
    cohorts_of, exposure_of, stage_of,
)
from phaseeval.workbench.model import canonical_bytes
from tests.support.store import BLOBS, CASE, OWNER, RATER_A, RATER_B, World, _import, _snapshot


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob('*')}


def test_reads_write_nothing(assigned):
    assigned.sign_all(RATER_A)
    assigned.sign_all(RATER_B)
    before = tree_bytes(assigned.store.root)
    snap = assigned.snaps[0]
    for path in (
        f'/roots/{assigned.root}/reveal-state', f'/snapshots/{snap}',
        f'/snapshots/{snap}/labels', f'/snapshots/{snap}/review',
        f'/snapshots/{snap}/revealed', f'/snapshots/{snap}/files.zip',
        f'/step-feedback?attempt={assigned.runs[0]}', '/queue',
    ):
        response = assigned.get(RATER_A, path)
        assert response.status_code == 200, (path, response.text)
        assert tree_bytes(assigned.store.root) == before, path
    for operation, args in ((assigned.store.view, (assigned.runs[0],)),
                            (assigned.store.path, (assigned.runs[0],)),
                            (assigned.store.tree, (assigned.root,)),
                            (assigned.store.compare, assigned.runs)):
        operation(OWNER, *args)
        assert tree_bytes(assigned.store.root) == before


def test_exposure_carries_across_roots_of_one_pattern(assigned):
    assigned.sign_all(RATER_A)
    other_case = CASE.model_copy(update={'case_id': 'DEV901'})
    root = assigned.store.load_case(OWNER, other_case, BLOBS)
    assigned.store.grant(OWNER, root, RATER_A.principal_id, (CaseOperation.INSPECT,))
    origin = _import(assigned.store, root, 'task-dev901__later', ('CIF000001',))
    snap = assigned.labels.put_snapshot(_snapshot(root, origin, ('CIF000001',)).model_copy(update={'case_id': 'DEV901'}))
    assigned.labels.assign(OWNER, root, (RATER_A.principal_id,), (snap,))
    label = assigned.labels.label(RATER_A, LabelRequest(snapshot=snap, item='amounts_defensible', answer='yes'))
    assert label.exposure == 'revealed'
    assert assigned.get(RATER_A, f'/snapshots/{snap}/revealed').status_code == 403


def test_queueing_before_any_signoff_joins_the_open_cohort(assigned):
    late = _import(assigned.store, assigned.root, 'task-dev900__open', ('CIF000003',))
    with patch.object(cli, 'OWNER', OWNER):
        digest = cli.queue_for_review(assigned.store, assigned.root, late)
    assignment = assigned.labels.latest_assignment(assigned.root)
    assert assignment.cohort == 1 and assignment.version == 2
    assert set(assignment.snapshots) == {*assigned.snaps, digest}


def test_queueing_after_signoff_opens_next_cohort(assigned):
    assigned.sign_all(RATER_A)
    late = _import(assigned.store, assigned.root, 'task-dev900__closed', ('CIF000003',))
    with patch.object(cli, 'OWNER', OWNER):
        digest = cli.queue_for_review(assigned.store, assigned.root, late)
    assignment = assigned.labels.latest_assignment(assigned.root)
    assert (assignment.cohort, assignment.version, assignment.snapshots) == (2, 1, (digest,))
    assert assigned.get(RATER_A, f'/snapshots/{assigned.snaps[0]}/revealed').status_code == 200
    assert assigned.get(RATER_A, f'/snapshots/{digest}/revealed').status_code == 403
    assert assigned.label(RATER_A, digest, item='amounts_defensible', phase_id=None).json()['exposure'] == 'revealed'


def test_an_old_reveal_never_includes_a_later_unsigned_answers_identity(assigned):
    assigned.sign_all(RATER_A)
    assigned.sign_all(RATER_B)
    late = _import(assigned.store, assigned.root, 'task-dev900__unsigned-sibling', ('CIF000003',))
    with patch.object(cli, 'OWNER', OWNER):
        digest = cli.queue_for_review(assigned.store, assigned.root, late)
    response = assigned.get(RATER_A, f'/snapshots/{assigned.snaps[0]}/revealed')
    assert response.status_code == 200, response.text
    assert {s['snapshot'] for s in response.json()['siblings']} == set(assigned.snaps)
    assert digest not in response.text and 'unsigned-sibling' not in response.text


def test_triage_waits_for_the_snapshots_own_cohort(assigned):
    from phaseeval.workbench.triage import PROVISIONAL_2026_10_21, Triage

    assigned.sign_all(RATER_A)
    queued = []
    for name in ('one', 'two'):
        late = _import(assigned.store, assigned.root, f'task-dev900__next-{name}', ('CIF000003',))
        with patch.object(cli, 'OWNER', OWNER):
            queued.append(cli.queue_for_review(assigned.store, assigned.root, late))
    triage = Triage(assigned.store, assigned.labels, PROVISIONAL_2026_10_21)
    triage.sweep(OWNER)
    assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=queued[0], verdict='would_sign'))
    assert triage.revealed(RATER_A, queued[0]) is None
    assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=queued[1], verdict='would_sign'))
    assert triage.revealed(RATER_A, queued[0]).snapshot == queued[0]


def test_an_explicit_assign_into_a_closed_cohort_is_a_conflict(assigned):
    from phaseeval.kernel.errors import Conflict

    assigned.sign_all(RATER_A)
    with pytest.raises(Conflict):
        assigned.labels.assign(OWNER, assigned.root, ('rater-a',), assigned.snaps, cohort=1)


def test_every_human_record_uses_the_family_stamp(assigned):
    assigned.sign_all(RATER_A)
    snap = assigned.snaps[0]
    calls = (
        (f'/snapshots/{snap}/peak-calls', {'peak_id': 'p23.95', 'cause': 'missing_phase'}),
        (f'/snapshots/{snap}/signoff', {'verdict': 'would_sign'}),
        ('/step-feedback', {'snapshot': snap, 'step': {'attempt': assigned.runs[0], 'step_kind': 'attempt'}, 'mark': 'good_move'}),
    )
    for path, body in calls:
        response = assigned.post(RATER_A, path, body)
        assert response.status_code == 201, response.text
        assert response.json()['exposure'] == 'revealed'
    assert response.json()['schema_version'] == 'phaseeval.step-feedback.v2'
    assert response.json()['snapshot'] == snap and 'step' not in response.json()


def test_legacy_reveals_remain_readable_and_are_validated(assigned):
    from phaseeval.workbench.store import CustodyError

    marker = assigned.store.root / 'reveals' / assigned.root / 'rater-a.json'
    marker.parent.mkdir(parents=True)
    record = RevealOpened(root=assigned.root, principal_id='rater-a', opened_at_utc=datetime(2026, 1, 1, tzinfo=UTC))
    marker.write_bytes(canonical_bytes(record))
    before = tree_bytes(assigned.store.root)
    assert assigned.labels.reveal_state(RATER_A, assigned.root).revealed
    assert assigned.labels._projection.exposure('rater-a', assigned.root) == 'exposed'
    assert tree_bytes(assigned.store.root) == before
    marker.write_bytes(canonical_bytes(record.model_copy(update={'principal_id': 'rater-b'})))
    with pytest.raises(CustodyError, match='path'):
        assigned.labels.reveal_state(RATER_A, assigned.root)


def test_extending_an_open_legacy_cohort_cannot_reveal_its_old_subset(bench):
    legacy = AssignmentV1(root=bench.root, raters=('rater-a', 'rater-b'), snapshots=(bench.snaps[0],),
                          assigned_by=OWNER.principal_id, created_at_utc=bench.store._clock())
    with bench.store.ledger.transaction():
        bench.labels._collections['assignments'].append((bench.root,), legacy)
    bench.allow(RATER_A)
    bench.allow(RATER_B)
    current = bench.labels.assign(OWNER, bench.root, legacy.raters, (bench.snaps[1],))
    assert set(current.snapshots) == set(bench.snaps)
    for rater in (RATER_A, RATER_B):
        bench.labels.sign_off(rater, SignOffRequest(snapshot=bench.snaps[0], verdict='would_sign'))
    assert not bench.labels.reveal_state(RATER_A, bench.root).revealed
    assert bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}/revealed').status_code == 403
    assert bench.label(RATER_A, bench.snaps[1]).json()['exposure'] == 'blind'
    bench.labels.sign_off(RATER_A, SignOffRequest(snapshot=bench.snaps[1], verdict='would_sign'))
    assert bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}/revealed').status_code == 200
    assert not bench.labels.reveal_state(RATER_A, bench.root).revealed
    bench.labels.sign_off(RATER_B, SignOffRequest(snapshot=bench.snaps[1], verdict='would_sign'))
    assert bench.labels.reveal_state(RATER_A, bench.root).revealed


def test_the_legacy_bridge_retains_the_original_full_reference_gate(bench):
    for rater in (RATER_A, RATER_B):
        bench.allow(rater)

    def legacy(snapshots):
        record = AssignmentV1(root=bench.root, raters=('rater-a', 'rater-b'), snapshots=snapshots,
                              assigned_by=OWNER.principal_id, created_at_utc=bench.store._clock())
        with bench.store.ledger.transaction():
            bench.labels._collections['assignments'].append((bench.root,), record)

    legacy((bench.snaps[0],))
    bench.labels.sign_off(RATER_A, SignOffRequest(snapshot=bench.snaps[0], verdict='would_sign'))
    legacy(bench.snaps)
    origin = _import(bench.store, bench.root, 'task-dev900__legacy-bridge', ('CIF000003',))
    added = bench.labels.put_snapshot(_snapshot(bench.root, origin, ('CIF000003',)))
    bench.labels.assign(OWNER, bench.root, ('rater-a', 'rater-b'), (added,))
    for rater in (RATER_A, RATER_B):
        for snapshot in (bench.snaps[1], added):
            bench.labels.sign_off(rater, SignOffRequest(snapshot=snapshot, verdict='would_sign'))
    assert not bench.labels.reveal_state(RATER_A, bench.root).revealed
    bench.labels.sign_off(RATER_B, SignOffRequest(snapshot=bench.snaps[0], verdict='would_sign'))
    assert bench.labels.reveal_state(RATER_A, bench.root).revealed


def _review_worker(root, snapshot, barrier, begin, ready, release, started, read, result, action):
    labels = LabelStore(World(root).store)
    barrier.wait(15)
    assert begin.wait(15)
    started.set()
    if action == 'sign':
        append = labels._append_locked

        def held(directory, record, at):
            ready.set()
            assert release.wait(15)
            append(directory, record, at)

        labels._append_locked = held
        labels.sign_off(RATER_A, SignOffRequest(snapshot=snapshot, verdict='would_sign'))
    else:
        facts = labels._projection.facts

        def observe(root):
            read.set()
            return facts(root)

        labels._projection.facts = observe
        result.put(labels.label(RATER_A, LabelRequest(snapshot=snapshot, item='amounts_defensible', answer='yes')).exposure)


def test_two_processes_cannot_stamp_a_blind_label_after_the_signoff(assigned):
    assigned.labels.sign_off(RATER_A, SignOffRequest(snapshot=assigned.snaps[0], verdict='would_sign'))
    ctx = multiprocessing.get_context('spawn')
    ready, release, signing, labeling = (ctx.Event() for _ in range(4))
    sign_begin, label_begin, read = (ctx.Event() for _ in range(3))
    barrier = ctx.Barrier(3)
    result = ctx.Queue()
    sign = ctx.Process(target=_review_worker, args=(assigned.store.root, assigned.snaps[1], barrier, sign_begin, ready, release, signing, read, result, 'sign'))
    label = ctx.Process(target=_review_worker, args=(assigned.store.root, assigned.snaps[0], barrier, label_begin, ready, release, labeling, read, result, 'label'))
    sign.start()
    label.start()
    try:
        barrier.wait(15)
        sign_begin.set()
        assert ready.wait(15)
        label_begin.set()
        assert labeling.wait(15)
        assert not read.wait(0.3)
        release.set()
        assert result.get(timeout=15) == 'revealed'
    finally:
        release.set()
        sign_begin.set()
        label_begin.set()
        for process in (sign, label):
            if process.pid is not None:
                process.join(15)
                if process.is_alive():
                    process.kill()
                    process.join()
        assert [p.exitcode for p in (sign, label)] == [0, 0]


def _assign_worker(root, family_root, snapshots, barrier, result):
    labels = LabelStore(World(root).store)
    barrier.wait(15)
    assignment = labels.assign(OWNER, family_root, ('rater-a',), snapshots)
    result.put((assignment.cohort, assignment.version))


def test_two_assigners_never_publish_one_version_number(assigned):
    ctx = multiprocessing.get_context('spawn')
    barrier, result = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_assign_worker, args=(assigned.store.root, assigned.root, assigned.snaps, barrier, result)) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert {result.get(timeout=15), result.get(timeout=15)} == {(1, 2), (1, 3)}
    finally:
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.kill()
                process.join()
        assert [p.exitcode for p in processes] == [0, 0]


@settings(max_examples=80, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 2), st.integers(0, 1), st.integers(0, 2)), min_size=1, max_size=40))
def test_no_interleaving_of_appends_lowers_a_stage(events):
    root, snapshots, raters = 'f' * 64, tuple(c * 64 for c in 'abc'), ('one', 'two')
    records, signs = [], []
    previous = {r: [0, 0, 0] for r in raters}
    exposed = {r: False for r in raters}
    for index, (operation, who, which) in enumerate(events):
        at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)
        cohorts = cohorts_of(records, signs)
        if operation == 0:
            current = cohorts[-1] if cohorts and not cohorts[-1].closed else None
            number = current.number if current else len(cohorts) + 1
            versions = [r.version for r in records if r.cohort == number]
            records.append(Assignment(root=root, cohort=number, version=max(versions, default=0) + 1,
                                      raters=tuple(sorted({*(current.raters if current else ()), raters[who]})),
                                      snapshots=tuple(sorted({*(current.snapshots if current else ()), snapshots[which]})),
                                      assigned_by='owner', created_at_utc=at))
        else:
            signs.append(SignOff(root=root, snapshot=snapshots[which], verdict='would_sign',
                                 rater=Rater(principal_id=raters[who], kind='expert'), exposure='blind', created_at_utc=at))
        facts = FamilyFacts(family=root, cohorts=cohorts_of(records, signs), signoffs=tuple(signs), declarations=(), legacy_reveals=())
        for rater in raters:
            stages = [stage_of(rater, sid, facts) for sid in snapshots]
            assert all(now >= old for now, old in zip(stages, previous[rater]))
            assert not exposed[rater] or exposure_of(rater, facts) == 'revealed'
            previous[rater], exposed[rater] = stages, exposure_of(rater, facts) == 'revealed'
