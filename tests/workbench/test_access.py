from __future__ import annotations

import pytest

from phaseeval.api import auth
from phaseeval.workbench import snapshot
from tests.support.store import BLOBS, CASE, JUDGE, OWNER, RATER_A, _import


def test_a_judge_reads_the_blind_view_of_a_granted_root_only(bench):
    bench.store.grant(OWNER, bench.root, JUDGE.principal_id, (auth.CaseOperation.INSPECT,))
    response = bench.get(JUDGE, f'/snapshots/{bench.snaps[0]}')
    assert response.status_code == 200, response.text
    assert 'origin' not in response.json() and 'root' not in response.json()
    assert bench.get(JUDGE, f'/snapshots/{bench.snaps[0]}/review').status_code == 200
    other_root = bench.store.load_case(OWNER, CASE.model_copy(update={'case_id': 'DEV901'}), BLOBS)
    origin = _import(bench.store, other_root, 'task-dev901__qwen', ('CIF000001',))
    other = bench.labels.put_snapshot(snapshot.build_snapshot(bench.store, OWNER, origin, 'agent_answer'))
    assert bench.get(JUDGE, f'/snapshots/{other}').status_code == 403
    assert bench.get(JUDGE, f'/snapshots/{other}/review').status_code == 403
    assert bench.get(JUDGE, f'/{bench.runs[0]}').status_code == 403
    assert bench.get(JUDGE, f'/{bench.runs[0]}/review').status_code == 403


def test_a_judge_without_a_grant_is_denied(bench):
    assert bench.get(JUDGE, f'/snapshots/{bench.snaps[0]}').status_code == 403


def test_the_rules_table_drives_every_route(assigned, monkeypatch):
    from phaseeval.kernel.errors import Denied

    bench = assigned
    assert set(auth.RULES) == set(auth.Op)
    calls = []
    original = auth.Access.require

    def require(self, op, **kwargs):
        calls.append(op)
        return original(self, op, **kwargs)

    monkeypatch.setattr(auth.Access, 'require', require)
    assert bench.get(OWNER, f'/{bench.runs[0]}').status_code == 200
    assert auth.Op.TREE_READ in calls
    calls.clear()
    assert bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}').status_code == 200
    assert auth.Op.REVIEW_READ in calls
    monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_READ, auth.OpRule(frozenset(), None))
    assert bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}').status_code == 403
    with pytest.raises(Denied):
        bench.labels.snapshot(RATER_A, bench.snaps[0])


@pytest.mark.parametrize('operation', ['freeze', 'step_feedback'])
def test_review_writes_use_the_write_rule(assigned, monkeypatch, operation):
    monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_WRITE, auth.OpRule(frozenset(), None))
    if operation == 'freeze':
        response = assigned.post(RATER_A, '/snapshots', {
            'origin': assigned.runs[0], 'kind': 'agent_answer'})
    else:
        response = assigned.post(RATER_A, '/step-feedback', {
            'snapshot': assigned.snaps[0], 'step': {'attempt': assigned.runs[0], 'call_index': 0,
                                                  'step_kind': 'refine'}, 'mark': 'good_move'})
    assert response.status_code == 403, response.text


def test_export_uses_the_export_rule_without_requiring_scorecard(bench, monkeypatch):
    monkeypatch.setitem(auth.RULES, auth.Op.SCORECARD, auth.OpRule(frozenset(), None))
    from phaseeval.workbench import exports
    from tests.support.runs import INPUTS
    dataset = exports.build_dataset(bench.store, bench.labels, OWNER, (), INPUTS, 'internal_evaluation',
                                    exports.RaterPseudonyms(bench.store.root / 'private'),
                                    review_payload=exports.blind_view)
    assert dataset.episodes == ()


def test_a_service_can_triage_without_raw_tree_access(bench):
    from phaseeval.workbench.triage import PROVISIONAL_2026_10_21, Triage

    service = auth.Principal('triage-service', auth.PrincipalType.SERVICE, frozenset({'adjudicator'}))
    routed = Triage(bench.store, bench.labels, PROVISIONAL_2026_10_21).sweep(service)
    assert {r.snapshot for r in routed.routed} == set(bench.snaps)
    from phaseeval.kernel.errors import Denied
    with pytest.raises(Denied):
        bench.store.view(service, bench.runs[0])
