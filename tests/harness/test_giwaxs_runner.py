"""Scripted actors and synthetic physics test the runner, not scientific quality."""
import argparse
import json
from pathlib import Path
import time

import pytest

from harness.agent import run_agent as mod
from harness.giwaxs.environment import EXPECTED_RUNTIME
from test_giwaxs import evidence


@pytest.fixture
def args(evidence, tmp_path, monkeypatch):
    execute = evidence.execute
    def process(argv, **kwargs):
        stdout = json.dumps(EXPECTED_RUNTIME) if argv[-1] == '--probe' else execute(argv)
        return stdout, '', 0
    monkeypatch.setattr(mod, 'bounded_process', process)
    return argparse.Namespace(input_dir=evidence.inputs, case='synthetic', out=tmp_path / 'run',
        task_kind='giwaxs', giwaxs_python=Path('/synthetic/python'),
        base_url='https://invalid.example/v1', model='scripted', endpoint_label='test',
        api_key_env='NO_TEST_KEY', max_turns=4, time_limit=30)


def reply(*calls):
    return {'choices': [{'message': {'tool_calls': [
        {'id': 'duplicate', 'function': {'name': name, 'arguments': json.dumps(arguments)}}
        for name, arguments in calls]}}]}


PARTIAL = {'status': 'partial', 'claims': [], 'limitations': ['Calibration unverified'],
           'evidence_needed': ['Independent calibration']}
TRANSFORM = {'label': 'map', 'mask_policy': 'supplied', 'incident_angle': 0.1,
             'sample_orientation': 3, 'tilt_angle': 0.0}


def test_rejection_repair_terminal_batch_and_unique_events(args, monkeypatch):
    runner = mod.Runner(args)
    replies = iter([reply(('submit', PARTIAL | {'status': 'supported'})),
                    reply(('transform', TRANSFORM), ('submit', PARTIAL), ('transform', TRANSFORM | {'label': 'forbidden'}))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    assert runner.run() == 0
    assert runner.receipt['ending']['kind'] == 'submitted'
    assert runner.receipt['workbench_import'] == 'reduction_bound'
    assert [a['outcome'] for a in runner.receipt['actions']] == ['rejected', 'success', 'success', 'rejected']
    ids = [a['event_id'] for a in runner.receipt['actions']]
    assert len(set(ids)) == 4
    trace = json.loads((runner.out / 'agent/trajectory.json').read_text())
    results = [r for s in trace['steps'] for r in s.get('observation', {}).get('results', [])]
    assert [r['event_id'] for r in results] == ids
    assert 'error:' in runner.call('transform', TRANSFORM | {'label': 'later'})
    with pytest.raises(ValueError, match='submission'):
        runner.evidence.call('transform', TRANSFORM | {'label': 'direct'})
    assert runner.evidence.call('read_evidence', {})
    assert not (runner.output / 'forbidden').exists()
    for name, sha in runner.receipt['evidence_bindings'].items():
        assert sha == mod.digest((runner.out / name).read_bytes())


def test_post_submit_calls_are_recorded_without_execution_even_after_deadline(args, monkeypatch):
    runner = mod.Runner(args)
    response = reply(('submit', PARTIAL), ('transform', TRANSFORM), ('read_evidence', {}), ('submit', PARTIAL))
    response['choices'][0]['message']['tool_calls'][-1]['function']['arguments'] = '{invalid'
    monkeypatch.setattr(runner, 'chat', lambda _: response)
    call = runner.call
    executed = []
    def submit(name, arguments):
        assert runner.evidence.submission is None, 'post-submit action executed'
        executed.append(name)
        result = call(name, arguments)
        runner.deadline = time.monotonic() - 1
        return result
    monkeypatch.setattr(runner, 'call', submit)
    assert runner.run() == 0
    assert executed == ['submit']
    assert runner.receipt['ending']['kind'] == 'submitted'
    actions = runner.receipt['actions']
    assert [action['status'] for action in actions] == ['completed'] * 4
    assert [action['outcome'] for action in actions] == ['success', 'rejected', 'rejected', 'rejected']
    trace = json.loads((runner.out / 'agent/trajectory.json').read_text())
    calls = trace['steps'][-1]['tool_calls']
    results = trace['steps'][-1]['observation']['results']
    assert [c['function_name'] for c in calls] == ['submit', 'transform', 'read_evidence', 'submit']
    assert calls[-1]['arguments'] == {'_unparsed': '{invalid'}
    assert len({action['event_id'] for action in actions}) == 4
    assert [c['event_id'] for c in calls] == [a['event_id'] for a in actions] == [r['event_id'] for r in results]
    assert [r['outcome'] for r in results] == [c['outcome'] for c in calls] == [a['outcome'] for a in actions]
    assert [r['content'] for r in results[1:]] == ['error: episode already submitted'] * 3
    assert not (runner.output / TRANSFORM['label']).exists()


def test_partial_without_reduction_and_kind_mismatch(args, monkeypatch):
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda _: reply(('submit', PARTIAL)))
    assert runner.run() == 0
    assert runner.receipt['workbench_import'] == 'unsupported_no_reduction'
    args.out = args.out.parent / 'wrong'
    args.task_kind = 'powder'
    wrong = mod.Runner(args)
    assert wrong.run() == 1
    assert 'task kind' in wrong.receipt['setup']['detail']


def test_invalid_runtime_is_durable_without_provider(args, monkeypatch):
    monkeypatch.setattr(mod, 'bounded_process', lambda *a, **kw: (json.dumps(EXPECTED_RUNTIME | {'python': '3.12.0'}), '', 0))
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda _: pytest.fail('provider must not run'))
    assert runner.run() == 1
    assert runner.receipt['ending']['kind'] == 'setup_failure'
    assert 'runtime' in runner.receipt['setup']['detail']
    assert runner.receipt['runtime']['giwaxs_observed_worker']['python'] == '3.12.0'


def test_comparison_binds_worker_runtime_and_processing_numpy(args, monkeypatch):
    runner = mod.Runner(args)
    identity = runner.receipt['comparison']['identity']
    assert 'harness/giwaxs/worker.py' in identity['sources']
    assert identity['giwaxs_runtime']['worker'] == EXPECTED_RUNTIME
    assert identity['giwaxs_runtime']['processing_numpy']
    monkeypatch.setattr(mod, 'bounded_process', lambda *a, **kw: (json.dumps(EXPECTED_RUNTIME | {'numpy': '2.0.0'}), '', 0))
    with pytest.raises(mod.InputChanged, match='runtime'):
        runner.check_inputs()


def test_worker_failure_keeps_stderr_and_is_recoverable(args, monkeypatch):
    runner = mod.Runner(args)
    process = mod.bounded_process
    monkeypatch.setattr(mod, 'bounded_process', lambda argv, **kw: process(argv, **kw) if argv[-1] == '--probe' else ('', 'integration failed', 2))
    replies = iter([reply(('transform', TRANSFORM)), reply(('submit', PARTIAL))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    assert runner.run() == 0
    assert runner.receipt['actions'][0]['outcome'] == 'failure'
    assert 'integration failed' in (runner.out / 'agent/trajectory.json').read_text()


@pytest.mark.parametrize('manifest', [None, [], 'invalid'])
def test_invalid_manifest_retains_setup_failure(args, manifest):
    (args.input_dir / 'manifest.json').write_text(json.dumps(manifest))
    runner = mod.Runner(args)
    assert runner.run() == 1
    assert runner.receipt['ending']['kind'] == 'setup_failure'
    assert 'must be an object' in runner.receipt['setup']['detail']


@pytest.mark.parametrize('change', ['prompt', 'tools', 'input', 'numpy'])
def test_frozen_contract_rechecked_before_actions(args, monkeypatch, change):
    runner = mod.Runner(args)
    if change == 'prompt':
        contract = mod.task_contract
        monkeypatch.setattr(mod, 'task_contract', lambda args: (contract(args)[0] + ' changed', contract(args)[1]))
    elif change == 'tools':
        from phaseeval import giwaxs
        monkeypatch.setattr(giwaxs, 'TOOLS', [*giwaxs.TOOLS, {'type': 'changed'}])
    elif change == 'input':
        (args.input_dir / 'raw.tif').write_bytes(b'changed')
    else:
        import numpy as np
        monkeypatch.setattr(np, '__version__', 'changed')
    assert runner.call('submit', PARTIAL).startswith('error:')
    assert runner.input_failure
    assert not (runner.output / 'analysis.json').exists()


def test_fingerprint_changes_with_worker_source(args, monkeypatch):
    before = mod.comparison_fingerprint(args)
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: original(p) + (b'\n# changed' if p.name == 'worker.py' else b''))
    assert mod.comparison_fingerprint(args) != before


@pytest.mark.skipif(not hasattr(mod.os, 'killpg'), reason='POSIX process groups required')
def test_shared_process_timeout_kills_descendants():
    # The child inherits stdout: communicate cannot finish promptly unless it also dies.
    script = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"]); time.sleep(10)'
    began = time.monotonic()
    with pytest.raises(TimeoutError, match='process group terminated'):
        mod.bounded_process([mod.sys.executable, '-c', script], cwd=mod.REPO,
                            env=mod.os.environ.copy(), timeout=0.1, deadline=began + 30)
    assert time.monotonic() - began < 1


def test_worker_timeout_is_an_explicit_failure(args, monkeypatch):
    runner = mod.Runner(args)
    process = mod.bounded_process
    def timeout(argv, **kwargs):
        if argv[-1] == '--probe':
            return process(argv, **kwargs)
        raise TimeoutError('physics tool exceeded time limit; process group terminated')
    monkeypatch.setattr(mod, 'bounded_process', timeout)
    replies = iter([reply(('transform', TRANSFORM)), reply(('submit', PARTIAL))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    assert runner.run() == 0
    assert runner.receipt['actions'][0]['outcome'] == 'failure'
    assert 'process group terminated' in (runner.out / 'agent/trajectory.json').read_text()


def test_configured_provider_key_removed_from_worker_environment(args, monkeypatch):
    monkeypatch.setenv(args.api_key_env, 'secret-value')
    process = mod.bounded_process
    def checked(argv, **kwargs):
        assert args.api_key_env not in kwargs['env']
        assert argv[0] == str(args.giwaxs_python)
        return process(argv, **kwargs)
    monkeypatch.setattr(mod, 'bounded_process', checked)
    assert mod.Runner(args).receipt['setup']['status'] == 'valid'
