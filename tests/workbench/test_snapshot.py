import hashlib
import json

import numpy as np
import pytest

from phaseeval.workbench import snapshot as snapshot_module
from phaseeval.workbench.cli import OWNER
from phaseeval.workbench.snapshot import Dia, build_snapshot, k2
from phaseeval.workbench.store import ActionRefused


@pytest.fixture
def imported(demo_store, tmp_path):
    store, (origin,) = demo_store.import_into(tmp_path / 'imported', [demo_store.origin()])
    return store, origin


def test_a_demo_import_freezes_with_computed_k1_k2_k5(imported):
    store, origin = imported
    snap = build_snapshot(store, OWNER, origin, 'agent_answer')
    checks = {c.code: c for c in snap.checks}
    assert {code: checks[code].status for code in ('K1', 'K2', 'K5')} == dict.fromkeys(('K1', 'K2', 'K5'), 'pass')
    assert [p.phase_id for p in snap.phases] == ['CIF012075', 'COD2310703', 'CIF001783']
    (peak,) = snap.residual_peaks
    assert peak.peak_id == 'p23.95' and peak.height_percent == 3.18
    assert snap.rwp_percent == pytest.approx(3.54, abs=0.01)
    assert snap.chemistry is not None
    assert snap == build_snapshot(store, OWNER, origin, 'agent_answer')


def test_the_page_and_the_judge_see_the_answer_as_submitted(imported):
    store, origin = imported
    snap = build_snapshot(store, OWNER, origin, 'agent_answer')
    result = store.lineage(OWNER, origin)[-1][1].outcome.result
    final = json.loads(store.read_blob(result.end.answer_sha256))
    assert [p.reported_formula for p in snap.phases] == ['NiO', 'Li2CO3', 'Bi2O3']
    assert snap.reported_answer['weight_fractions'] == final['weight_fractions']
    assert snap.reported_quantity_basis == final['quantity_basis']
    body = snapshot_module.attempt_payload(store, OWNER, origin)
    assert body['answer']['submitted'] == snap.reported_answer
    assert body['fit']['native_points'] == 8282
    assert body['fit']['points'] <= 1200
    assert len(body['fit']['phase_curves']) == 3
    assert body['residual_windows'][0]['peak_id'] == 'p23.95'
    assert {w['phase_id'] for w in body['phase_windows']} == {p.phase_id for p in snap.phases}


def test_drafts_with_the_answers_phases_and_amounts_are_never_shown_as_the_answer(demo_store, tmp_path):
    source = demo_store.store
    (_, root), (_, attempt) = source.lineage(OWNER, demo_store.origin())
    result = attempt.outcome.result
    final = json.loads(source.read_blob(result.end.answer_sha256))
    blobs = {ref.sha256: source.read_blob(ref.sha256) for ref in attempt.outcome.origin.receipts}
    nonce = 0
    while True:
        draft = final | {'quantity_basis': f'Draft basis, superseded by the submitted answer ({nonce}).'}
        draft_bytes = json.dumps(draft).encode()
        draft_digest = hashlib.sha256(draft_bytes).hexdigest()
        if draft_digest < result.end.answer_sha256:
            break
        nonce += 1
    assert draft_digest < result.end.answer_sha256
    blobs[draft_digest] = draft_bytes
    from phaseeval.workbench import cli
    store = cli.open_store(tmp_path / 'drafts', tmp_path)
    case_blobs = {p.name: p.read_bytes() for p in (source.root / 'blobs').iterdir()}
    parent = store.load_case(OWNER, root.case, case_blobs)
    origin = store.import_run(cli.IMPORTER, parent, attempt.action, result, blobs)
    snap = build_snapshot(store, OWNER, origin, 'agent_answer')
    assert snap.reported_quantity_basis == final['quantity_basis']
    assert 'Draft basis' not in snap.model_dump_json()


@pytest.mark.parametrize('binding', [None, 'receipt'])
def test_an_import_bound_to_no_answer_file_shows_no_answer_words(demo_store, tmp_path, binding):
    from phaseeval.workbench import cli
    source = demo_store.store
    (_, root), (_, attempt) = source.lineage(OWNER, demo_store.origin())
    result = attempt.outcome.result
    blobs = {ref.sha256: source.read_blob(ref.sha256) for ref in attempt.outcome.origin.receipts}
    non_answer = b'{"not": "an answer"}'
    digest = hashlib.sha256(non_answer).hexdigest()
    blobs[digest] = non_answer
    end = result.end.model_copy(update={'answer_sha256': None if binding is None else digest})
    store = cli.open_store(tmp_path / 'unbound', tmp_path)
    parent = store.load_case(OWNER, root.case, {p.name: p.read_bytes() for p in (source.root / 'blobs').iterdir()})
    origin = store.import_run(cli.IMPORTER, parent, attempt.action, result.model_copy(update={'end': end}), blobs)
    snap = build_snapshot(store, OWNER, origin, 'agent_answer')
    assert snap.answered and snap.phases
    assert (snap.claims, snap.reported_quantity_basis, snap.reported_answer) == ((), None, None)
    assert all(p.reported_formula is None for p in snap.phases)


def test_an_import_naming_an_answer_file_outside_its_receipts_is_refused(demo_store, tmp_path):
    from phaseeval.workbench import cli
    source = demo_store.store
    (_, root), (_, attempt) = source.lineage(OWNER, demo_store.origin())
    store = cli.open_store(tmp_path / 'invalid', tmp_path)
    parent = store.load_case(OWNER, root.case, {p.name: p.read_bytes() for p in (source.root / 'blobs').iterdir()})
    result = attempt.outcome.result
    with pytest.raises(ActionRefused, match='not one of its receipts'):
        store.import_run(cli.IMPORTER, parent, attempt.action,
                         result.model_copy(update={'end': result.end.model_copy(update={'answer_sha256': '0' * 64})}),
                         {ref.sha256: source.read_blob(ref.sha256) for ref in attempt.outcome.origin.receipts})


@pytest.mark.parametrize('case_id', ['DEV006', 'DEV102'])
def test_the_brief_keeps_recorded_masses_and_withholds_policy_amounts(demo_store, case_id):
    body = snapshot_module.attempt_payload(demo_store.store, OWNER, demo_store.origin(case=case_id))
    brief = body['brief']
    if case_id == 'DEV006':
        assert brief['precursor_amounts_note'] == 'withheld by task policy'
        assert 'reference weight fractions' not in json.dumps(brief)
        assert 'brief.precursors[].mass_g' in body['missing']
    else:
        assert [p['mass_g'] for p in brief['precursors']] == [1.38156, 0.62231]
        assert brief['precursor_amounts_note'] is None


def test_a_residual_peak_at_exactly_the_flag_height_flags() -> None:
    """K2 boundary: a peak whose height is the policy's flag height flags (>=, not >)."""
    tt = np.arange(10.0, 60.0, 0.01)
    background = np.full_like(tt, 1000.0)
    calculated = background + 1.0e5 * np.exp(-0.5 * ((tt - 30.0) / 0.05) ** 2)
    bump = np.exp(-0.5 * ((tt - 45.0) / 0.05) ** 2)

    def dia(amplitude: float) -> Dia:
        observed = calculated + amplitude * bump
        return Dia(None, "rietveld", (), np.column_stack([tt, observed, calculated, background]))

    first = k2(dia(1.0e4))["peaks"][0]["pct_of_strongest_peak"]  # the height scales linearly
    check = k2(dia(1.0e4 * 5.0 / first))
    (peak,) = check["peaks"]
    assert peak["pct_of_strongest_peak"] == 5.0 and peak["status"] == "flag"
    assert check["status"] == "flag"
    assert k2(dia(1.0e4 * 4.99 / first))["status"] == "pass"

def test_the_scrub_replaces_paths_and_run_names_and_keeps_formulas_ratios_and_numbers() -> None:
    names = ("u6-dev006-astra-007", "task-dev006__nrBCdaa", "gpt-6-astra", "astra", "codex")
    value = {
        "evidence": "P2_1/c, Rwp/Rexp = 1.8; see /workspace/output/run1/result.json, https://doi.org/10.1107/x",
        "result_file": "/workspace/output/h1_search/result.json",
        "note": "Astra's run task-dev006__nrBCdaa in codex; Astragalus and gpt-6-astra-like stay apart",
        "values": [891.7, None, True, 0, "counts"],
    }
    assert snapshot_module._scrubbed(value, names) == {
        "evidence": "P2_1/c, Rwp/Rexp = 1.8; see [path], https://doi.org/10.1107/x",
        "result_file": "[path]",
        "note": "[run identity]'s run [run identity] in [run identity]; Astragalus and [run identity]-like stay apart",
        "values": [891.7, None, True, 0, "counts"],
    }


def test_files_zip_requires_the_raters_own_cohort_signoff(assigned):
    from tests.support.store import RATER_A, RATER_B

    url = f'/snapshots/{assigned.snaps[0]}/files.zip'
    assert assigned.get(RATER_A, url).status_code == 403
    assigned.post(RATER_A, f'/snapshots/{assigned.snaps[0]}/signoff', {'verdict': 'would_sign'})
    assert assigned.get(RATER_A, url).status_code == 403
    assigned.sign_all(RATER_A)
    assert assigned.get(RATER_A, url).status_code == 200
    assert assigned.get(RATER_B, url).status_code == 403
