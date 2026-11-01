"""Receipts retain failed scheduled attempts, input identity and completed calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import urllib.error

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('integrity_runner', REPO / 'harness/agent/run_agent.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture
def args(tmp_path):
    inputs = tmp_path / 'input'
    shutil.copytree(REPO / 'harness/examples/dev006/input', inputs)
    return argparse.Namespace(input_dir=inputs, case='DEV006', out=tmp_path / 'trial',
        base_url='https://invalid.example/v1', model='scripted', endpoint_label='test',
        api_key_env='NO_TEST_KEY', max_turns=5, time_limit=30)


def receipt(runner):
    return json.loads((runner.out / 'runner_receipt.json').read_text())


def response(*calls, usage=None):
    return {'choices': [{'message': {'content': 'observed', 'tool_calls': [
        {'id': str(i), 'function': {'name': name, 'arguments': json.dumps(arguments)}}
        for i, (name, arguments) in enumerate(calls)]}}], 'usage': usage}


def test_clean_entry_and_stable_comparison(args):
    first = mod.Runner(args)
    second = mod.Runner(argparse.Namespace(**(vars(args) | {'out': args.out.parent / 'second'})))
    assert receipt(first)['comparison']['fingerprint'] == receipt(second)['comparison']['fingerprint']
    assert receipt(first)['session_id'] != receipt(second)['session_id']
    with pytest.raises(ValueError, match='empty|claimed'):
        mod.Runner(args)
    assert mod.comparison_fingerprint(args) == receipt(first)['comparison']['fingerprint']
    changed = argparse.Namespace(**(vars(args) | {'max_turns': 6}))
    assert mod.comparison_fingerprint(changed) != mod.comparison_fingerprint(args)


@pytest.mark.parametrize('defect', ['hash', 'duplicate', 'nan', 'escape', 'case_id', 'empty_scan', 'duplicate_json'])
def test_invalid_setup_is_retained_before_any_tool(args, defect, monkeypatch):
    manifest = args.input_dir / 'manifest.json'
    data = json.loads(manifest.read_text())
    if defect == 'hash':
        (args.input_dir / 'DEV006.xy').write_text('10 1\n11 2\n')
    elif defect == 'duplicate':
        library = args.input_dir / 'candidates/DEV006/manifest.json'
        rows = json.loads(library.read_text()); rows['candidates'].append(rows['candidates'][0])
        library.write_text(json.dumps(rows))
    elif defect in ('nan', 'empty_scan'):
        raw = b'10 NaN\n11 2\n' if defect == 'nan' else b''
        (args.input_dir / 'DEV006.xy').write_bytes(raw)
        data['cases'][0]['pattern_sha256'] = hashlib.sha256(raw).hexdigest()
        manifest.write_text(json.dumps(data))
    elif defect == 'escape':
        data['cases'][0]['pattern_file'] = '../outside.xy'; manifest.write_text(json.dumps(data))
    elif defect == 'case_id':
        args.case = '../DEV006'
    else:
        manifest.write_text('{"cases":[],"cases":[]}')
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda *_: pytest.fail('provider called on invalid setup'))
    assert runner.run() == 1
    saved = receipt(runner)
    assert saved['setup']['status'] == 'invalid'
    assert saved['ending']['kind'] == 'setup_failure'
    assert saved['comparison'] is None
    assert saved['workbench_import'] == 'unsupported_setup_invalid'


def test_changed_inputs_block_all_tools(args, monkeypatch):
    runner = mod.Runner(args)
    (args.input_dir / 'DEV006.xy').write_text('changed')
    monkeypatch.setattr(runner, 'tool', lambda *_a, **_kw: pytest.fail('physics ran'))
    assert runner.call('candidates', {}).startswith('error:')


@pytest.mark.parametrize('arguments', [
    {'candidates': ['cod_1010093', 'cod_1010093'], 'run_label': 'a'},
    {'candidates': ['cod_1010093'], 'run_label': 'a.b'},
    {'candidates': ['cod_1010093'], 'run_label': 'a', 'pawley_only': 1},
])
def test_invalid_action_never_invokes_physics(args, arguments, monkeypatch):
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'tool', lambda *_a, **_kw: pytest.fail('invalid action ran'))
    assert runner.call('refine', arguments).startswith('error:')


@pytest.mark.parametrize('name,arguments,field,value', [
    ('refine', {'candidates': ['cod_1010093'], 'run_label': '-fit'}, 'run_label', '-fit'),
    ('screen', {'residual': '-fit'}, 'residual', '-fit'),
    ('case_brief', {}, 'case', '-DEV006'),
])
def test_leading_hyphen_scalar_reaches_physics_parser(args, monkeypatch, name, arguments, field, value):
    if field == 'case':
        for path in (args.input_dir / 'manifest.json', args.input_dir / 'candidates/DEV006/manifest.json'):
            path.write_text(path.read_text().replace('"DEV006"', '"-DEV006"'))
        (args.input_dir / 'candidates/DEV006').rename(args.input_dir / 'candidates/-DEV006')
        args.case = value
    runner = mod.Runner(args)
    if name == 'screen':
        runner.fit_phases['-fit'] = {'cod_1010093'}
    actual_popen = mod.subprocess.Popen
    # Execute the real CLI parser, exiting before the scientific tool runs.
    script = '''import argparse, json, runpy, sys
from pathlib import Path
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
original = argparse.ArgumentParser.parse_args
def parse(self, *args, **kwargs):
    print(json.dumps(vars(original(self, *args, **kwargs)), default=str))
    raise SystemExit(0)
argparse.ArgumentParser.parse_args = parse
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    monkeypatch.setattr(mod.subprocess, 'Popen',
        lambda cmd, **kw: actual_popen([cmd[0], '-c', script, *cmd[1:]], **kw))
    observed = runner.call(name, arguments)
    assert not observed.startswith('error:'), observed
    assert json.loads(observed.split('\n[exit ')[0])[field] == value


@pytest.mark.parametrize('patch', [
    {'weight_fractions': {'cod_1010093': True}},
    {'weight_fractions': {'cod_1010093': float('nan')}},
    {'weight_fractions': {'cod_9008283': 0.5}},
    {'phases': [{'candidate_id': 'cod_1010093'}] * 2},
    {'fit_artifacts': ['/workspace/output/invented/result.json']},
    {'fit_artifacts': ['/etc/passwd']},
    {'fit_artifacts': ['/workspace/output/../foreign/result.json']},
])
def test_invalid_answers_rejected(args, patch):
    runner = mod.Runner(args)
    answer = {'case_id': args.case, 'phases': [{'candidate_id': 'cod_1010093'}], **patch}
    assert runner.write_analysis(answer).startswith('error:')
    assert not (runner.output / 'analysis.json').exists()


def test_no_fit_unresolved_submission_and_missing_usage(args, monkeypatch):
    runner = mod.Runner(args)
    replies = iter([response(('write_analysis', {'analysis': {'case_id': args.case,
        'phases': [], 'reason': 'unresolved'}}), usage={'prompt_tokens': 0, 'completion_tokens': 2}), response()])
    monkeypatch.setattr(runner, 'chat', lambda *_: next(replies))
    assert runner.run() == 0
    saved = receipt(runner)
    assert saved['ending']['kind'] == 'submitted'
    assert saved['workbench_import'] == 'unsupported_no_refinement'
    assert saved['usage']['prompt_tokens'] == {'known_subtotal': 0, 'complete': False, 'total': None}
    assert saved['usage']['completion_tokens']['known_subtotal'] == 2


def test_completed_call_saved_before_next_call_throws(args, monkeypatch):
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda *_: response(('case_brief', {}), ('candidates', {})))
    def call(name, _args):
        if name == 'candidates':
            persisted = json.loads((runner.out / 'agent/trajectory.json').read_text())
            assert persisted['steps'][-1]['observation']['results'][0]['content'] == 'retained evidence'
            raise OSError('tool unavailable')
        return 'retained evidence'
    monkeypatch.setattr(runner, 'call', call)
    assert runner.run() == 1
    assert receipt(runner)['actions'][0]['status'] == 'completed'
    assert receipt(runner)['actions'][0]['duration_seconds'] >= 0
    assert receipt(runner)['ending']['kind'] == 'tool_failure'


def test_budget_stops_second_batch_call_and_preserves_answer(args, monkeypatch):
    args.time_limit = 0.05
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda *_: response(('write_analysis', {}), ('candidates', {})))
    def call(name, _args):
        assert name == 'write_analysis'
        runner.write_analysis({'case_id': args.case, 'phases': []})
        time.sleep(0.06)
        return 'saved'
    monkeypatch.setattr(runner, 'call', call)
    assert runner.run() == 1
    assert receipt(runner)['ending']['kind'] == 'budget_exhausted'
    assert (runner.output / 'analysis.json').exists()
    assert len(receipt(runner)['actions']) == 1


def test_group_slot_claim_is_before_setup_and_never_refilled(args, tmp_path):
    from phaseeval.reward_profile import profile_group

    manifest = tmp_path / 'group.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'group_id': 'g1', 'created_at': mod.now(),
        'comparison_fingerprint': mod.comparison_fingerprint(args), 'episode_ids': ['one', 'two']}))
    args.group_manifest, args.episode_id = manifest, 'one'
    (args.input_dir / 'DEV006.xy').unlink()
    first = mod.Runner(args)
    assert first.run() == 1
    assert receipt(first)['group']['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    retry = mod.Runner(argparse.Namespace(**(vars(args) | {'out': tmp_path / 'retry'})))
    assert retry.run() == 1
    assert 'already attempted' in receipt(retry)['setup']['detail']
    assert receipt(first)['group']['episode_id'] == 'one'
    paths = [r.out / 'runner_receipt.json' for r in (first, retry)]
    report = profile_group(manifest, paths)
    assert report['counts']['attempted'] == 1
    assert report['counts']['missing'] == 1
    assert report['counts']['unscheduled'] == 1
    assert report['episodes'][0]['scientific_status'] == 'execution_invalid'
    assert report['unscheduled'][0]['episode_id'] is None
    assert report['unscheduled'][0]['requested_episode_id'] == 'one'
    assert report['unscheduled'][0]['scientific_status'] == 'unclaimed_attempt'
    assert receipt(first)['group']['claimed'] is True
    assert receipt(retry)['group']['claimed'] is False
    retry_only = profile_group(manifest, paths[1:])
    assert retry_only['counts']['attempted'] == 0
    assert retry_only['counts']['missing'] == 2
    assert all(row['scientific_status'] == 'missing_attempt' for row in retry_only['episodes'])


def test_fingerprint_covers_prompt_tools_manifests_and_decoding(args, monkeypatch):
    baseline = mod.comparison_fingerprint(args)
    original = mod.system_prompt
    monkeypatch.setattr(mod, 'system_prompt', lambda case: original(case) + '\nchanged')
    assert mod.comparison_fingerprint(args) != baseline
    monkeypatch.setattr(mod, 'system_prompt', original)
    monkeypatch.setattr(mod, 'TOOLS', [*mod.TOOLS, {'type': 'test'}])
    assert mod.comparison_fingerprint(args) != baseline
    monkeypatch.undo()
    args.decoding = {'temperature': 0}
    assert mod.comparison_fingerprint(args) != baseline
    args.decoding = {'model': 'replacement'}
    with pytest.raises(ValueError, match='decoding'):
        mod.comparison_fingerprint(args)
    del args.decoding
    manifest = args.input_dir / 'manifest.json'
    manifest.write_text(manifest.read_text() + '\n')
    assert mod.comparison_fingerprint(args) != baseline


def test_stale_group_identity_and_duplicate_slot_are_recorded(args, tmp_path, monkeypatch):
    group = tmp_path / 'group.json'
    group.write_text(json.dumps({'schema_version': 1, 'group_id': 'g', 'created_at': mod.now(),
        'comparison_fingerprint': '0' * 64, 'episode_ids': ['a']}))
    args.group_manifest, args.episode_id = group, 'a'
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda *_: pytest.fail('stale comparison invoked provider'))
    assert runner.run() == 1
    assert 'fingerprint differs' in receipt(runner)['setup']['detail']


def saved_refine(runner, label='fit', candidates=None):
    candidates = candidates or ['cod_1010093']
    folder = runner.output / label
    folder.mkdir()
    selected = [row for row in runner.evidence.candidates if row['candidate_id'] in candidates]
    (folder / 'request.json').write_text(json.dumps({'case_id': runner.args.case,
        'pattern_sha256': runner.evidence.case['pattern_sha256'], 'candidates': selected}))
    (folder / 'result.json').write_text('{}')
    return 'saved refinement request'


@pytest.mark.parametrize('ending,cause', [('provider', 'provider_error'), ('tool', 'executor_unavailable'), ('setup', 'executor_unavailable')])
def test_persisted_refinement_bound_infra_ending_imports(args, tmp_path, monkeypatch, ending, cause):
    from phaseeval.workbench import cli, importer
    package = cli.harness_case_package(args.input_dir, args.case, tmp_path / 'normalized')
    labels, custody = cli.user_case_records(package, args.case, tmp_path)
    case, _ = importer.read_case_package(package, labels, custody=custody)
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'tool', lambda *_a, **_kw: saved_refine(runner))
    replies = 0
    def chat(_messages):
        nonlocal replies
        replies += 1
        if replies == 1:
            return response(('refine', {'candidates': ['cod_1010093'], 'run_label': 'fit'}))
        if ending == 'provider':
            raise urllib.error.URLError('provider unavailable')
        if ending == 'setup':
            (args.input_dir / 'DEV006.xy').write_text('changed after refinement')
        return response(('case_brief', {}))
    monkeypatch.setattr(runner, 'chat', chat)
    original = runner.call
    def call(name, arguments):
        if ending == 'tool' and name == 'case_brief':
            raise OSError('executor could not launch')
        return original(name, arguments)
    monkeypatch.setattr(runner, 'call', call)
    assert runner.run() == 1
    assert receipt(runner)['workbench_import'] == 'refinement_bound'
    imported_action, imported, _ = importer.read_harbor_run(runner.out, case)
    assert imported.end.end == 'invalid'
    assert imported.end.cause == cause
    assert imported_action.harbor.trial_id == receipt(runner)['trial_id']
    trace = json.loads((runner.out / 'agent/trajectory.json').read_text())
    assert trace['steps'][2]['observation']['results'][0]['content'] == 'saved refinement request'


def test_owned_fit_phase_identity_and_symlink_scrub(args, tmp_path, monkeypatch):
    runner = mod.Runner(args)
    monkeypatch.setattr(runner, 'tool', lambda *_a, **_kw: saved_refine(runner))
    assert not runner.call('refine', {'candidates': ['cod_1010093'], 'run_label': 'fit'}).startswith('error:')
    answer = {'case_id': args.case, 'phases': [{'candidate_id': 'cod_1010093'}],
              'weight_fractions': {'cod_1010093': 1}, 'fit_artifacts': ['/workspace/output/fit/result.json']}
    assert runner.write_analysis(answer).startswith('wrote')
    answer['phases'] = [{'candidate_id': 'cod_9008283'}]
    answer['weight_fractions'] = {}
    assert 'phases differ' in runner.write_analysis(answer)
    target = tmp_path / 'outside.json'
    target.write_text('{"host":"outside"}')
    def bad_tool(*_a, **_kw):
        folder = runner.output / 'bad'
        folder.mkdir()
        (folder / 'result.json').symlink_to(target)
        return 'bad'
    monkeypatch.setattr(runner, 'tool', bad_tool)
    with pytest.raises(ValueError, match='symlink'):
        runner.call('refine', {'candidates': ['cod_1010093'], 'run_label': 'bad'})
    assert target.read_text() == '{"host":"outside"}'


def test_http_read_uses_total_remaining_wall_budget(args, monkeypatch):
    import http.server
    import threading
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            raw = json.dumps(response()).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            try:
                for byte in raw:
                    self.wfile.write(bytes([byte])); self.wfile.flush()
                    time.sleep(0.015)
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *_):
            pass
    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    args.base_url = f'http://127.0.0.1:{server.server_port}'
    args.time_limit = 0.2
    runner = mod.Runner(args)
    began = time.monotonic()
    try:
        assert runner.run() == 1
        assert time.monotonic() - began < 0.6
        assert receipt(runner)['ending']['kind'] == 'budget_exhausted'
        assert receipt(runner)['usage']['prompt_tokens']['total'] is None
    finally:
        thread.join(timeout=1)
        server.server_close()


def test_successful_http_response_with_explicit_zero_usage(args):
    import http.server
    import threading
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            raw = json.dumps(response(usage={'prompt_tokens': 0, 'completion_tokens': 0})).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(raw))); self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *_):
            pass
    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True); thread.start()
    args.base_url = f'http://127.0.0.1:{server.server_port}'
    runner = mod.Runner(args)
    try:
        assert runner.run() == 0
        assert receipt(runner)['usage']['prompt_tokens'] == {'known_subtotal': 0, 'complete': True, 'total': 0}
    finally:
        thread.join(timeout=1); server.server_close()


@pytest.mark.skipif(not hasattr(mod.os, 'killpg'), reason='POSIX process groups required')
def test_physics_timeout_kills_descendants_and_remains_recoverable(args, monkeypatch):
    runner = mod.Runner(args)
    actual_popen = mod.subprocess.Popen
    script = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"]); time.sleep(10)'
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda cmd, **kw: actual_popen([mod.sys.executable, '-c', script], **kw))
    monkeypatch.setattr(runner, 'remaining', lambda maximum=None: 0.1)
    began = time.monotonic()
    assert 'process group terminated' in runner.tool('case_brief.py')
    # The child inherits stdout: communicate cannot return promptly unless it was killed too.
    assert time.monotonic() - began < 1


def test_nonzero_physics_exit_is_observed_then_policy_can_repair(args, monkeypatch):
    runner = mod.Runner(args)
    actual_popen = mod.subprocess.Popen
    script = 'import sys; print("numerical refinement did not converge"); sys.exit(2)'
    monkeypatch.setattr(mod.subprocess, 'Popen', lambda cmd, **kw: actual_popen([mod.sys.executable, '-c', script], **kw))
    replies = iter([response(('refine', {'candidates': ['cod_1010093'], 'run_label': 'fit'})),
                    response(('write_analysis', {'analysis': {'case_id': args.case, 'phases': [], 'reason': 'unresolved after failed fit'}})), response()])
    def chat(messages):
        if messages[-1]['role'] == 'tool' and 'numerical' in messages[-1]['content']:
            assert '[exit 2]' in messages[-1]['content']
        return next(replies)
    monkeypatch.setattr(runner, 'chat', chat)
    assert runner.run() == 0
    assert receipt(runner)['ending']['kind'] == 'submitted'
    trace = json.loads((runner.out / 'agent/trajectory.json').read_text())
    assert 'numerical refinement did not converge' in trace['steps'][2]['observation']['results'][0]['content']
