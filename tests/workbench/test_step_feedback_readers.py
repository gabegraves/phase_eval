import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('version', (1, 2))
def test_the_failure_brief_reads_both_feedback_versions(tmp_path, version):
    export = tmp_path / 'export'
    shutil.copytree(ROOT / 'docs/examples/failure-brief/export-dev006-qwen', export)
    snapshot = json.loads((export / 'snapshots.jsonl').read_text().splitlines()[0])
    record = json.loads((ROOT / 'schemas/examples/step-feedback.v1.example.json').read_text())
    record['step']['attempt'] = snapshot['origin']
    record['note'] = 'Check the missing phase'
    if version == 2:
        step = record.pop('step')
        record.update(schema_version='phaseeval.step-feedback.v2', snapshot='a' * 64,
                      attempt=step['attempt'], call_index=step['call_index'])
    (export / 'step_feedback.jsonl').write_text(json.dumps(record) + '\n')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/failure_brief.py'), str(export), '--json'],
                            capture_output=True, text=True, check=True, timeout=30)
    [brief] = json.loads(result.stdout)
    assert any(c['text'] == 'Check the missing phase' for c in brief['critiques'])


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
@pytest.mark.parametrize('version', (1, 2))
def test_the_page_restores_both_feedback_versions(tmp_path, version):
    record = {'mark': 'good_move', 'note': 'kept', 'exposure': 'blind'}
    step = {'attempt': 'run', 'call_index': 0}
    record.update({'step': step} if version == 1 else step)
    script = tmp_path / 'restore.js'
    script.write_text('''
const fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const context = vm.createContext({ console });
vm.runInContext(source.slice(0, source.indexOf('\\n$("auth").addEventListener')), context);
context.records = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(vm.runInContext('(() => { const V = {marks: {}}; restoreMarks(V, records); return V.marks; })()', context)));
''')
    result = subprocess.run(['node', str(script), str(ROOT / 'src/phaseeval/api/assets/workbench.js'), json.dumps([record])],
                            capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout) == {'run:0': [{'mark': 'good_move', 'note': 'kept', 'exposure': 'blind'}]}


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_the_page_binds_child_feedback_to_the_viewed_answer(assigned, tmp_path):
    from phaseeval.api.auth import CaseOperation
    from phaseeval.workbench.store import ActionRequest
    from tests.support.store import OWNER, RATER_A

    assigned.sign_all(RATER_A)
    assigned.store.grant(OWNER, assigned.root, RATER_A.principal_id, (CaseOperation.INSPECT, CaseOperation.ACTION))
    child = assigned.store.act(RATER_A, assigned.runs[1], ActionRequest(
        idempotency_key='feedback-child', action={'verb': 'remove_phases', 'candidates': ['CIF000003']},
    )).digest
    script = tmp_path / 'save-mark.js'
    script.write_text('''
const fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const context = vm.createContext({ console });
vm.runInContext(source.slice(0, source.indexOf('\\n$("auth").addEventListener')), context);
context.snapshot = process.argv[3];
context.attempt = process.argv[4];
vm.runInContext(`
S.ep = {digest: snapshot};
render = () => {};
call = async (name, params, options) => { requestBody = options.body; return {}; };
saveMark({attempt, step_kind: 'attempt'}, 'good_move', 'child of answer two')
`, context).then(() => process.stdout.write(JSON.stringify(context.requestBody)));
''')
    result = subprocess.run(['node', str(script), str(ROOT / 'src/phaseeval/api/assets/workbench.js'), assigned.snaps[1], child],
                            capture_output=True, text=True, check=True, timeout=30)
    body = json.loads(result.stdout)
    assert body.get('snapshot') == assigned.snaps[1], body
    response = assigned.post(RATER_A, '/step-feedback', body)
    assert response.status_code == 201, response.text
    assert response.json()['snapshot'] == assigned.snaps[1]
    assert assigned.get(RATER_A, f'/step-feedback?attempt={child}').json()[-1]['snapshot'] == assigned.snaps[1]


def test_ambiguous_legacy_child_feedback_is_refused_without_storing_it(assigned):
    from phaseeval.api.auth import CaseOperation
    from phaseeval.workbench.store import ActionRequest
    from tests.support.store import OWNER, RATER_A

    assigned.sign_all(RATER_A)
    assigned.store.grant(OWNER, assigned.root, RATER_A.principal_id, (CaseOperation.INSPECT, CaseOperation.ACTION))
    child = assigned.store.act(RATER_A, assigned.runs[1], ActionRequest(
        idempotency_key='legacy-feedback-child', action={'verb': 'remove_phases', 'candidates': ['CIF000003']},
    )).digest
    response = assigned.post(RATER_A, '/step-feedback', {
        'step': {'attempt': child, 'step_kind': 'attempt'}, 'mark': 'good_move',
    })
    assert response.status_code == 422, response.text
    assert assigned.get(RATER_A, f'/step-feedback?attempt={child}').json() == []


def test_unambiguous_legacy_feedback_still_resolves_its_answer(assigned):
    from tests.support.store import RATER_A

    response = assigned.post(RATER_A, '/step-feedback', {
        'step': {'attempt': assigned.runs[1], 'step_kind': 'attempt'}, 'mark': 'good_move',
    })
    assert response.status_code == 201, response.text
    assert response.json()['snapshot'] == assigned.snaps[1]
