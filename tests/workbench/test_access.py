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


@pytest.mark.parametrize('method', ['get', 'post'])
def test_composition_routes_use_their_review_rule(assigned, monkeypatch, method):
    assigned.store.grant(OWNER, assigned.root, RATER_A.principal_id, (auth.CaseOperation.ACTION,))
    path = f'/roots/{assigned.root}/composition'
    body = {'basis': 'weight', 'method': 'XRF', 'fractions': {'Ni': 0.3}, 'source': 'report 1'}
    if hasattr(auth, 'RULES'):
        monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_REFIT, auth.OpRule(frozenset(), None))

    def request():
        return (assigned.get(RATER_A, path) if method == 'get' else
                assigned.post(RATER_A, path, body))

    assert request().status_code == (200 if method == 'get' else 201)
    if hasattr(auth, 'RULES'):
        operation = auth.Op.REVIEW_READ if method == 'get' else auth.Op.REVIEW_WRITE
        monkeypatch.setitem(auth.RULES, operation, auth.OpRule(frozenset(), None))
    assert request().status_code == 403


def test_composition_read_ignores_write_rule_and_writes_need_action(assigned, monkeypatch):
    path = f'/roots/{assigned.root}/composition'
    body = {'basis': 'weight', 'method': 'XRF', 'fractions': {'Ni': 0.3}, 'source': 'report 1'}
    assert assigned.post(RATER_A, path, body).status_code == 403
    if hasattr(auth, 'RULES'):
        monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_WRITE, auth.OpRule(frozenset(), None))
    assert assigned.get(RATER_A, path).status_code == 200
    assigned.store.grant(OWNER, assigned.root, RATER_A.principal_id, (auth.CaseOperation.ACTION,))
    assert assigned.post(RATER_A, path, body).status_code == 403
    assert assigned.get(RATER_A, path).json() == []


def test_the_rules_table_drives_every_route(assigned, monkeypatch):
    from phaseeval.kernel.errors import Denied

    bench = assigned
    calls = []
    if hasattr(auth, 'Access'):
        original = auth.Access.require

        def require(self, op, **kwargs):
            calls.append(op.value)
            return original(self, op, **kwargs)

        monkeypatch.setattr(auth.Access, 'require', require)
    assert bench.get(OWNER, f'/{bench.runs[0]}').status_code == 200
    assert 'tree_read' in calls
    calls.clear()
    assert bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}').status_code == 200
    assert 'review_read' in calls
    assert set(auth.RULES) == set(auth.Op)
    monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_READ, auth.OpRule(frozenset(), None))
    assert bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}').status_code == 403
    with pytest.raises(Denied):
        bench.labels.snapshot(RATER_A, bench.snaps[0])


@pytest.mark.parametrize('gate', ['mode', 'grant'])
def test_freeze_requires_the_complete_write_rule(assigned, monkeypatch, gate):
    rule = (auth.OpRule(frozenset(), None) if gate == 'mode' else
            auth.OpRule(frozenset({auth.Mode.RATER}), auth.CaseOperation.EXPORT))
    monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_WRITE, rule)
    response = assigned.post(RATER_A, '/snapshots', {
        'origin': assigned.runs[0], 'kind': 'expert_correction'})
    assert response.status_code == 403, response.text
    assert not any(s.kind == 'expert_correction' for _, s in assigned.labels._snapshots())
    assert not (assigned.store.root / 'snapshot-index' / assigned.runs[0] / 'expert_correction').exists()


@pytest.mark.parametrize('gate', ['mode', 'grant'])
def test_review_reads_do_not_require_write_authorization(assigned, monkeypatch, gate):
    write_rule = (auth.OpRule(frozenset(), None) if gate == 'mode' else
                  auth.OpRule(frozenset({auth.Mode.RATER}), auth.CaseOperation.EXPORT))
    monkeypatch.setitem(auth.RULES, auth.Op.REVIEW_WRITE, write_rule)
    for path in (f'/snapshots/{assigned.snaps[0]}', f'/snapshots/{assigned.snaps[0]}/review',
                 f'/roots/{assigned.root}/reveal-state'):
        response = assigned.get(RATER_A, path)
        assert response.status_code == 200, response.text


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
