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
