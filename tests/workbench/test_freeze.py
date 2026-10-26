from __future__ import annotations

import multiprocessing
import os
from datetime import UTC, datetime, timedelta

import pytest

from phaseeval.workbench import cli, snapshot
from phaseeval.workbench.labels import LabelRequest, LabelStore
from phaseeval.workbench.model import digest_of
from phaseeval.workbench.triage import PROVISIONAL_2026_10_21, Triage
from tests.support.store import OWNER, RATER_A


def test_retry_after_index_publication_fails_keeps_the_first_snapshot(bench, monkeypatch):
    from tests.support.store import _import

    origin = _import(bench.store, bench.root, 'task-dev900__index-retry', ('CIF000001',))
    first = snapshot.build_snapshot(bench.store, OWNER, origin, 'agent_answer')
    sid = digest_of(first)
    freezer = getattr(bench.store, 'snapshots', None)
    target = freezer._index if freezer else bench.store._snapshot_collection
    original = target.put

    def fail_publication(*args):
        if freezer is None:
            original(*args)
        raise OSError('index publication failed')

    with monkeypatch.context() as patch:
        patch.setattr(target, 'put', fail_publication)
        with pytest.raises(OSError, match='index publication failed'):
            bench.labels.put_snapshot(first)
    assert bench.labels._snapshot(sid) == first
    changed = first.model_copy(update={'claims': ('changed on retry',)})
    try:
        retried = bench.labels.put_snapshot(changed)
    except ValueError:
        retried = None
    assert retried == sid
    assert [d for d, s in bench.labels._snapshots() if s.origin == origin] == [sid]
    reopened = cli.open_store(bench.store.root, bench.store.import_root)
    assert LabelStore(reopened).put_snapshot(changed) == sid


def test_every_caller_gets_one_snapshot_per_origin(assigned, monkeypatch):
    monkeypatch.setattr(cli, "OWNER", OWNER)
    bench = assigned
    first = bench.snaps[0]
    assert bench.labels.freeze(RATER_A, bench.runs[0], 'agent_answer') == first
    changed = bench.labels._snapshot(first).model_copy(update={'claims': ('changed derivation',)})
    assert bench.labels.put_snapshot(changed) == first
    assert cli.queue_for_review(bench.store, bench.root, bench.runs[0]) == first
    routed = Triage(bench.store, bench.labels, PROVISIONAL_2026_10_21).sweep(OWNER)
    assert first in {r.snapshot for r in routed.routed}
    snapshot.attempt_payload(bench.store, OWNER, bench.runs[0])
    assert bench.store.snapshots.freeze(OWNER, bench.runs[0], 'agent_answer') == first
    assert len([s for _, s in bench.labels._snapshots() if s.origin == bench.runs[0] and s.kind == 'agent_answer']) == 1


@pytest.mark.parametrize('history', ['timestamps', 'index_order', 'tie', 'unassigned'])
def test_two_legacy_snapshots_of_one_origin_both_stay_resolvable(bench, monkeypatch, history):
    from phaseeval.workbench.labels import AssignmentV1

    monkeypatch.setattr(cli, 'OWNER', OWNER)
    bench.allow(RATER_A)
    first = bench.snaps[0]
    original = bench.labels._snapshot(first)
    for number in range(100):
        duplicate = original.model_copy(update={'claims': (f'legacy duplicate {number}',)})
        other = digest_of(duplicate)
        if other < first:
            break
    assert other < first
    bench.store._snapshot_collection.put((other,), duplicate)
    at = datetime(2026, 10, 21, tzinfo=UTC)
    histories = {
        'timestamps': [(at + timedelta(seconds=1), (first, other)), (at, (first,))],
        'index_order': [(at, (first, other))],
        'tie': [(at, (first,)), (at, (other,))],
        'unassigned': [],
    }
    for when, items in histories[history]:
        record = AssignmentV1(root=bench.root, raters=(RATER_A.principal_id,), snapshots=items,
                              assigned_by=OWNER.principal_id, created_at_utc=when)
        bench.labels._collections['assignments'].append((bench.root,), record, at=when)
    index = bench.store.root / 'snapshot-index' / original.origin / original.kind
    index.unlink(missing_ok=True)
    before = {sid: (bench.store.root / 'snapshots' / f'{sid}.json').read_bytes()
              for sid in (first, other)}
    reopened = cli.open_store(bench.store.root, bench.store.import_root)
    expected = first if history in ('timestamps', 'index_order') else other
    assert cli.queue_for_review(reopened, bench.root, original.origin) == expected
    labels = LabelStore(reopened)
    labels.assign(OWNER, bench.root, (RATER_A.principal_id,), (other, first))
    for sid in (other, first):
        assert labels.snapshot(RATER_A, sid).origin == original.origin
        label = labels.label(RATER_A, LabelRequest(snapshot=sid, item='fit_supports_conclusion', answer='yes'))
        assert labels.snapshot(RATER_A, label.snapshot).origin == original.origin
        assert (reopened.root / 'snapshots' / f'{sid}.json').read_bytes() == before[sid]
    assert len(Triage(reopened, labels, PROVISIONAL_2026_10_21).sweep(OWNER).routed) == 2


def _freeze_worker(root, import_root, origin, start, output):
    store = cli.open_store(root, import_root)
    try:
        built = snapshot.build_snapshot(store, cli.OWNER, origin, 'agent_answer')
        built = built.model_copy(update={'claims': (f'freeze worker {os.getpid()}',)})
        start.wait(15)
        LabelStore(store).put_snapshot(built)
        output.put(store.snapshots.freeze(cli.OWNER, origin, 'agent_answer'))
    except Exception as exc:
        output.put(repr(exc))


def test_two_processes_freezing_one_origin_publish_one_index_entry(demo_store, tmp_path):
    store, (origin,) = demo_store.import_into(tmp_path / 'race', [demo_store.origin()])
    ctx = multiprocessing.get_context('spawn')
    start, output = ctx.Barrier(2), ctx.Queue()
    workers = [ctx.Process(target=_freeze_worker, args=(store.root, store.import_root, origin, start, output)) for _ in range(2)]
    for worker in workers:
        worker.start()
    results = [output.get(timeout=60) for _ in workers]
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    assert results[0] == results[1] and len(results[0]) == 64, results
    assert len(list((store.root / 'snapshot-index' / origin).iterdir())) == 1
    assert len(LabelStore(store)._snapshots()) == 1


def test_an_unfrozen_attempt_preview_writes_nothing(bench):
    from tests.support.store import _import

    origin = _import(bench.store, bench.root, 'task-dev900__preview', ('CIF000001',))
    before = {p.relative_to(bench.store.root): p.read_bytes()
              for p in bench.store.root.rglob('*') if p.is_file()}
    snapshot.attempt_payload(bench.store, OWNER, origin)
    assert {p.relative_to(bench.store.root): p.read_bytes()
            for p in bench.store.root.rglob('*') if p.is_file()} == before


def test_export_keeps_legacy_duplicates_and_their_label_targets(assigned, tmp_path, monkeypatch):
    import json
    from phaseeval.workbench import exports

    monkeypatch.setattr(cli, 'OWNER', OWNER)
    bench = assigned
    original = bench.labels._snapshot(bench.snaps[0])
    duplicate = original.model_copy(update={'claims': ('legacy duplicate for export',)})
    other = digest_of(duplicate)
    bench.store._snapshot_collection.put((other,), duplicate)
    bench.labels.assign(OWNER, bench.root, (RATER_A.principal_id,), (other,))
    for sid in (bench.snaps[0], other):
        bench.labels.label(RATER_A, LabelRequest(snapshot=sid, item='fit_supports_conclusion', answer='yes'))
    out = tmp_path / 'export'
    assert cli.main(['export', '--store', str(bench.store.root), '--out', str(out)]) == 0
    snapshots = {digest_of(snapshot.AnalysisSnapshot.model_validate_json(line))
                 for line in (out / 'snapshots.jsonl').read_text().splitlines()}
    labels = [json.loads(line) for line in (out / 'labels.expert.jsonl').read_text().splitlines()]
    assert {bench.snaps[0], other} <= snapshots
    assert {row['snapshot'] for row in labels} == {bench.snaps[0], other}
    assert exports.validate_dataset(out, exports.dataset_json_schemas()) == []
