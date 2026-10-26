import hashlib
import io
import json
import re
import shutil
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from phaseeval.api.app import app
from phaseeval.api.auth import CaseOperation
from phaseeval.workbench import cli, snapshot
from phaseeval.workbench.labels import LabelStore, PermissionRequest, SignOffRequest
from phaseeval.workbench.store import ActionRequest
from tests.support.demo import BUNDLE, REPO
from tests.support.store import (
    ADD, AGENT, ALLOW, CASE, OWNER, RATER_A, RUN_START, _headers, _import,
)


def test_an_oversized_action_is_refused_and_the_case_stays_readable(bench):
    bench.store.grant(OWNER, bench.root, AGENT.principal_id, (CaseOperation.INSPECT, CaseOperation.ACTION))
    run = bench.store.act(AGENT, bench.root, ActionRequest(idempotency_key='run', action=RUN_START)).digest
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(f'/api/attempts/{run}/actions', headers=_headers(AGENT), json={
        'steps': [{'idempotency_key': 'large', 'action': {'verb': 'unknown', 'value': 'x' * (33 * 1024 * 1024)}}],
    })
    readable = bench.get(OWNER, f'/roots/{bench.root}/tree')
    assert (response.status_code, readable.status_code) == (413, 200)


def test_queueing_after_a_signoff_keeps_the_first_cohort_signed(assigned):
    assigned.sign_all(RATER_A)
    assert assigned.labels.reveal_state(RATER_A, assigned.root).signed_off
    late = _import(assigned.store, assigned.root, 'task-dev900__queued', ('CIF000003',))
    with patch.object(cli, 'OWNER', OWNER):
        cli.queue_for_review(assigned.store, assigned.root, late)
    assert assigned.labels.reveal_state(RATER_A, assigned.root).signed_off


def test_a_nan_body_is_422(bench):
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(f'/api/attempts/{bench.root}/actions', headers=_headers(OWNER) | {'Content-Type': 'application/json'},
                           content='{"steps":[{"idempotency_key":NaN,"action":{"verb":"refine"}}]}')
    assert response.status_code == 422


@pytest.mark.xfail(strict=True, reason="ledger S15; fixed by P15d", raises=AssertionError)
def test_a_validation_error_never_echoes_input(bench):
    secret = 'untrusted-input-sentinel'
    response = bench.post(OWNER, '/roots/' + bench.root + '/assignment', {'raters': [secret], 'snapshots': [secret]})
    assert response.status_code in (400, 422)
    assert secret not in response.text
    response = bench.post(OWNER, f'/{bench.root}/actions', {
        'steps': [{'idempotency_key': 'bad', 'action': {'verb': 'refine', 'recipe': secret}}],
    })
    assert response.status_code in (400, 422)
    assert secret not in response.text
    assert 'input_value' not in response.text
    response = bench.post(OWNER, f'/{bench.root}/actions', {
        'steps': [{'idempotency_key': 'unknown', 'action': {'verb': secret}}],
    })
    assert response.status_code == 422
    assert secret not in response.text


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_unparsed_records_refuse_nested_nonfinite_json(value):
    from pydantic import ValidationError
    from phaseeval.workbench.model import Unparsed

    with pytest.raises(ValidationError):
        Unparsed.model_validate_json('{"verb":"unparsed","raw":{"values":[{"value":' + value + '}]}}')


def test_a_missing_referenced_blob_is_a_controlled_refusal(bench):
    ref = CASE.pattern.sha256
    path = f'/{bench.root}/artifacts/{ref}'
    assert bench.get(OWNER, path).status_code == 200
    (bench.store.root / 'blobs' / ref).unlink()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get('/api/attempts' + path, headers=_headers(OWNER))
    assert response.status_code == 400, response.text


@pytest.mark.xfail(strict=True, reason="ledger S15; fixed by P15d", raises=AssertionError)
def test_unknown_and_ungranted_ids_get_the_same_403_body(bench):
    missing = bench.get(RATER_A, '/snapshots/' + 'f' * 64)
    existing = bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}')
    assert missing.status_code == existing.status_code == 403
    assert missing.json() == existing.json()


@pytest.mark.xfail(strict=True, reason="ledger U17; fixed by P15b", raises=AssertionError)
def test_scorecard_data_without_a_token_is_401_even_with_the_local_owner_variable(bench, monkeypatch):
    monkeypatch.setenv('AXA_WORKBENCH_LOCAL_OWNER', '1')
    monkeypatch.setenv('AXA_WORKBENCH_DATASET', str(REPO / 'data/dataset-v0-public'))
    client = TestClient(app, base_url='http://127.0.0.1:18800', client=('127.0.0.1', 12345))
    response = client.get('/api/attempts/scorecard')
    assert response.status_code == 401


@pytest.mark.xfail(strict=True, reason="ledger X13; fixed by P22")
def test_a_file_name_alone_does_not_assert_rietveld(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    for name in ('DEV006_rietveld.sav', 'DEV006_rietveld.lst', 'DEV006_rietveld.dia', 'DEV006.xy'):
        shutil.copy(BUNDLE / name, bundle / name)
    with pytest.raises(cli.CliError, match='stage'):
        cli.main(['import-bgmn', '--case-dir', str(REPO / 'harness/examples/dev006/input'),
                  '--run-dir', str(bundle), '--store', str(tmp_path / 'store')])


@pytest.mark.xfail(strict=True, reason="ledger U18; fixed by P22", raises=AssertionError)
def test_derived_gewicht_reaches_the_snapshot_as_derived(tmp_path):
    bundle = tmp_path / 'bundle'
    shutil.copytree(BUNDLE, bundle)
    control = bundle / 'DEV006_rietveld.sav'
    control.write_text(re.sub(r'^(?:sum|Q\w+|GOAL\[\d+\])=.*\n', '', control.read_text(), flags=re.M))
    root = tmp_path / 'store'
    assert cli.main(['import-bgmn', '--case-dir', str(REPO / 'harness/examples/dev006/input'),
                     '--run-dir', str(bundle), '--store', str(root)]) == 0
    out = tmp_path / 'export'
    assert cli.main(['export', '--store', str(root), '--out', str(out)]) == 0
    records = [json.loads(line) for line in (out / 'snapshots.jsonl').read_text().splitlines()]
    assert 'derived_gewicht' in json.dumps(records)


@pytest.mark.xfail(strict=True, reason="ledger E12; fixed by P23", raises=AssertionError)
def test_files_zip_after_signoff_is_scrubbed_and_complete(demo_store, bench, tmp_path):
    store, (origin,) = demo_store.import_into(tmp_path / 'files-store', [demo_store.origin()])
    (_, root_attempt), (_, attempt) = store.lineage(cli.OWNER, origin)
    label = snapshot.evidence(store, cli.OWNER, origin, 'agent_answer').fit.label
    job = attempt.action.harbor.job_name
    receipts = {}
    for ref in attempt.outcome.origin.receipts:
        data = store.read_blob(ref.sha256)
        try:
            value = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            value = {}
        if data.startswith(b'TITEL=') or (isinstance(value, dict) and (
            'working_dir' in value or isinstance(value.get('config'), dict) and 'working_dir' in value['config']
        )):
            data = data.replace(label.encode(), job.encode())
        receipts[hashlib.sha256(data).hexdigest()] = data
    store = cli.open_store(tmp_path / 'named-files-store', tmp_path)
    root = store.load_case(cli.OWNER, root_attempt.case, {
        p.name: p.read_bytes() for p in (demo_store.store.root / 'blobs').iterdir()
    })
    origin = store.import_run(cli.IMPORTER, root, attempt.action, attempt.outcome.result, receipts)
    labels = LabelStore(store)
    snap = labels.put_snapshot(snapshot.build_snapshot(store, cli.OWNER, origin, 'agent_answer'))
    store.grant(cli.OWNER, root, RATER_A.principal_id, (CaseOperation.INSPECT,))
    labels.assign(cli.OWNER, root, (RATER_A.principal_id,), (snap,))
    labels.record_permission(RATER_A, PermissionRequest(**ALLOW))
    labels.sign_off(RATER_A, SignOffRequest(snapshot=snap, verdict='would_sign'))
    app.state.workbench_store = store
    response = bench.get(RATER_A, f'/snapshots/{snap}/files.zip')
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert 'pattern/DEV006.xy' in names
        assert len([n for n in names if n.startswith('cif/')]) == 3
        assert any(n.endswith('request.json') for n in names)
        assert any(n.endswith('result.json') for n in names)
        assert any(n.endswith('rietveld.dia') for n in names)
        assert all(job not in n for n in names)
        assert all(b'u6-dev006' not in archive.read(n) for n in names)


def test_an_action_over_the_action_cap_is_refused_before_claiming(bench):
    before = sorted((bench.store.root / 'claims').rglob('*'))

    response = bench.post(OWNER, f'/{bench.root}/actions', {
        'steps': [{'idempotency_key': 'large', 'action': {'verb': 'unknown', 'value': 'x' * (256 * 1024)}}],
    })
    assert response.status_code == 413
    assert bench.get(OWNER, f'/roots/{bench.root}/tree').status_code == 200
    assert sorted((bench.store.root / 'claims').rglob('*')) == before


@pytest.mark.parametrize('patch', [
    {'declared_target': 'x' * 4001},
    {'precursors': [{'formula': 'x' * 4001}]},
    {'processing': [{'operation': 'heat', 'atmosphere': 'x' * 4001}]},
])
def test_patch_strings_are_bounded_before_any_claim(bench, patch):
    before = sorted((bench.store.root / 'claims').rglob('*'))
    response = bench.post(OWNER, f'/{bench.root}/actions', {
        'steps': [{'idempotency_key': 'patch', 'action': {'verb': 'correct_material', 'patch': patch}}],
    })
    assert response.status_code == 422
    assert sorted((bench.store.root / 'claims').rglob('*')) == before
    assert bench.get(OWNER, f'/roots/{bench.root}/tree').status_code == 200
