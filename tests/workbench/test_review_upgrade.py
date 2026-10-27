import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from tests.support.store import app
from phaseeval.workbench import cli, labels
from tests.support.store import OWNER, RATER_A, RATER_B, World, _import


@pytest.mark.parametrize('sign_before_reduction', [False, True])
def test_legacy_snapshot_reduction_preserves_signed_access_on_reopen(bench, monkeypatch, sign_before_reduction):
    legacy_type = getattr(labels, 'AssignmentV1', labels.Assignment)
    for rater in (RATER_A, RATER_B):
        bench.allow(rater)

    def assign(snapshots):
        record = legacy_type(root=bench.root, raters=('rater-a', 'rater-b'), snapshots=snapshots,
                             assigned_by=OWNER.principal_id, created_at_utc=bench.store._clock())
        bench.labels._append(bench.store.root / 'assignments' / bench.root, record, record.created_at_utc)

    def sign(rater):
        response = bench.post(rater, f'/snapshots/{bench.snaps[0]}/signoff', {'verdict': 'would_sign'})
        assert response.status_code == 201, response.text

    assign(bench.snaps)
    if sign_before_reduction:
        sign(RATER_A)
    assign((bench.snaps[0],))
    for rater in (RATER_A, RATER_B):
        sign(rater)

    bench.store = World(bench.store.root).store
    bench.labels = labels.LabelStore(bench.store)
    monkeypatch.setattr(app.state, 'workbench_store', bench.store)
    for rater in (RATER_A, RATER_B):
        state = bench.get(rater, f'/roots/{bench.root}/reveal-state')
        assert state.status_code == 200, state.text
        assert state.json()['signed_off'], state.json()
        assert state.json()['revealed'], state.json()
        for resource in ('revealed', 'files.zip'):
            response = bench.get(rater, f'/snapshots/{bench.snaps[0]}/{resource}')
            assert response.status_code == 200, response.text


def test_concurrent_queueing_does_not_assign_one_answer_twice(assigned, monkeypatch):
    assigned.sign_all(RATER_A)
    child = _import(assigned.store, assigned.root, 'task-dev900__concurrent-queue', ('CIF000003',))
    directory = assigned.store.root / 'assignments' / assigned.root
    before = set(directory.glob('*.json'))
    paused, finished = threading.Event(), threading.Event()
    latest_assignment = labels.LabelStore.latest_assignment
    delayed_thread = None

    def pause_after_lookup(self, root):
        latest = latest_assignment(self, root)
        if threading.current_thread() is delayed_thread:
            paused.set()
            finished.wait(1)
        return latest

    def delayed():
        nonlocal delayed_thread
        delayed_thread = threading.current_thread()
        return cli.queue_for_review(assigned.store, assigned.root, child)

    def competing():
        assert paused.wait(10)
        try:
            digest = cli.queue_for_review(assigned.store, assigned.root, child)
            response = assigned.post(RATER_A, f'/snapshots/{digest}/signoff', {'verdict': 'would_sign'})
            assert response.status_code == 201, response.text
            return digest
        finally:
            finished.set()

    monkeypatch.setattr(labels.LabelStore, 'latest_assignment', pause_after_lookup)
    with patch.object(cli, 'OWNER', OWNER), ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(delayed)
        second = pool.submit(competing)
        assert first.result(timeout=15) == second.result(timeout=15)
    assert len(set(directory.glob('*.json')) - before) == 1
