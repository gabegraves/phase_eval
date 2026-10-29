"""Run an AgentXRD-style agent with any OpenAI-compatible model on one case folder.

Writes a Harbor-shaped trial folder that phaseeval.workbench.importer.read_harbor_run reads:
config.json, result.json, agent/trajectory.json, artifacts/workspace/output/.
The API key is read from the environment and never written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import signal
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHYSICS = HERE.parent / "physics"
PYTHON = REPO / "launcher" / ".venv" / "bin" / "python"
HARNESS = "agentxrd-openai-runner"
SHOWN_CHARS = 20000  # bounded actor observation, including any recovery notice
sys.path[:0] = [str(REPO), str(REPO / "src")]
from harness.agent.evidence import checked_bytes
from harness.agent.episode import SAFE_ID as LABEL, atomic_json, bind_group, digest, preflight as episode_preflight, strict_json

TOOLS = [
    {"type": "function", "function": {"name": "read_evidence", "description": "List registered evidence (omit handle), or read native scan/fit rows or exact text pages of context, CIFs and clipped tool output. Row indexes are local to the artifact. Angle bounds are inclusive; repeat them with next_start.", "parameters": {"type": "object", "additionalProperties": False, "properties": {"handle": {"type": "string"}, "start": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 2000}, "two_theta_min": {"type": "number"}, "two_theta_max": {"type": "number"}}}}},
    {"type": "function", "function": {"name": "case_brief", "description": "Chemistry, measurement, precursors, processing and declared target of the case.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "candidates", "description": "One line per candidate CIF: id, formula, space group, crystal system, cell.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "screen", "description": "Rank candidates against observed peaks (no tree search). With residual=<run_label>, list the largest residual peaks of that saved fit.", "parameters": {"type": "object", "properties": {"residual": {"type": ["string", "null"]}}}}},
    {"type": "function", "function": {"name": "refine", "description": "Staged BGMN refinement (Pawley then Rietveld) of the named candidates. Writes /workspace/output/<run_label>/.", "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5, "uniqueItems": True}, "run_label": {"type": "string"}, "pawley_only": {"type": "boolean"}}, "required": ["candidates", "run_label"]}}},
    {"type": "function", "function": {"name": "write_analysis", "description": "Validate and write the answer to /workspace/output/analysis.json.", "parameters": {"type": "object", "properties": {"analysis": {"type": "object"}}, "required": ["analysis"]}}},
]


# A fixed stdlib worker bounds DNS, connection, headers and body together. Request
# credentials travel over stdin; they are never command arguments or saved artifacts.
HTTP_REQUEST = """
import json, sys, urllib.error, urllib.request
request = json.load(sys.stdin)
try:
    req = urllib.request.Request(request['url'], request['body'].encode(), request['headers'])
    with urllib.request.urlopen(req, timeout=request['timeout']) as response:
        result = {'body': response.read().decode('utf-8')}
except urllib.error.HTTPError as exc:
    result = {'error': 'HTTP ' + str(exc.code)}
except Exception as exc:
    result = {'error': type(exc).__name__ + ': ' + str(exc)}
json.dump(result, sys.stdout, allow_nan=False)
"""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class BudgetExpired(TimeoutError):
    pass


class ToolFailure(RuntimeError):
    pass


class InputChanged(ValueError):
    pass


def system_prompt(case_id: str) -> str:
    return (PHYSICS / 'README.md').read_text() + '\n\n' + (HERE / 'task.md').read_text().replace('{case_id}', case_id)


def task_contract(args):
    if getattr(args, 'task_kind', 'powder') == 'giwaxs':
        from phaseeval.giwaxs import TOOLS as giwaxs_tools
        return (f"Analyze GIWAXS case {args.case} using only the registered evidence tools. "
                "Inspect the catalog and context before choosing reduction geometry and mask policy. "
                "The supplied geometry is acquisition context; alternatives are exploratory assumptions. "
                "Maps use q=2pi/d in nm^-1. Extraction bounds are q in nm^-1 and use nearest-bin, "
                "upper-exclusive strips. Declare notebook or coverage_aware aggregation; inspect coverage. "
                "Use artifact handles to cite claims, and compare only compatible coordinate conventions. "
                "Use a new label for every transform, extraction or comparison, including retries after failure. "
                "Submit supported claims or a justified partial result with limitations and evidence needed. "
                "Submission ends the episode immediately. Scientific validity is unassessed and reward is unavailable.",
                giwaxs_tools)
    return system_prompt(args.case), TOOLS


def bounded_process(cmd, *, cwd, env, timeout, deadline):
    """One process group owns each physics call, including any worker descendants."""
    if os.name not in ('posix', 'nt'):
        raise ToolFailure('bounded physics process-tree termination is unavailable on this platform')
    group = {'start_new_session': True} if os.name == 'posix' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, **group) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                try:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   capture_output=True, timeout=5, check=True)
                except (OSError, subprocess.SubprocessError) as exc:
                    process.kill()
                    raise ToolFailure('native process-tree cancellation failed') from exc
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.communicate(timeout=5)
            if deadline <= time.monotonic():
                raise BudgetExpired('wall time expired during physics call') from None
            raise TimeoutError('physics tool exceeded time limit; process group terminated') from None
    return stdout, stderr, process.returncode


def physics_environment(args):
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'src')}
    env.pop(args.api_key_env, None)
    return env


def giwaxs_execute(args, argv, *, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetExpired('wall time budget exhausted')
    stdout, stderr, code = bounded_process(argv, cwd=REPO / 'harness/giwaxs',
        env=physics_environment(args), timeout=min(30 if argv[-1] == '--probe' else 1800, remaining), deadline=deadline)
    if code:
        raise ToolFailure(f'GIWAXS worker failed [exit {code}]\n{stdout}\n[stderr]\n{stderr}')
    return stdout


def preflight(args, *, execute=None):
    """Validate/fingerprint without a trial or provider call; GIWAXS probes its runtime."""
    system, tools = task_contract(args)
    if execute is None:
        deadline = time.monotonic() + args.time_limit
        execute = lambda argv: giwaxs_execute(args, argv, deadline=deadline)
    return episode_preflight(args, repo=REPO, system=system, tools=tools, execute=execute)


def comparison_fingerprint(args) -> str:
    return preflight(args)[1]['fingerprint']


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task_kind = getattr(args, 'task_kind', 'powder')
        self.call_outcome = None
        original = args.out.absolute()
        if any(p.is_symlink() for p in (original, *original.parents)):
            raise ValueError('symlink trial directory is not allowed')
        original.mkdir(parents=True, exist_ok=True)
        if any(original.iterdir()):
            raise ValueError('trial directory must be empty')
        self.out = original.resolve()
        self.sid = str(uuid.uuid4())
        self.started, self.t0 = now(), time.monotonic()
        self.deadline = None
        self.output = self.out / 'artifacts/workspace/output'
        self.evidence, self.library, self.labels, self.fit_phases = None, set(), set(), {}
        self.input_failure = None
        self.has_run = False
        self.receipt = {
            'schema_version': 1, 'session_id': self.sid, 'trial_id': trial_name(self.out, self.sid),
            'started_at': self.started, 'finished_at': None, 'group': None, 'comparison': None,
            'setup': {'status': 'pending', 'detail': None}, 'ending': None,
            'runtime': {'python': sys.version, 'platform': sys.platform, 'served_models': [],
                        'system_fingerprints': [], 'bgmn_binary_sha256': None, 'packages': {}},
            'usage': {key: {'known_subtotal': 0, 'complete': True, 'total': 0}
                      for key in ('prompt_tokens', 'completion_tokens')},
            'provider_responses': [], 'actions': [], 'wall_seconds': None, 'cost_usd': None,
            'workbench_import': 'unsupported_no_reduction' if self.task_kind == 'giwaxs' else 'unsupported_no_refinement',
        }
        # An exclusive claim protects reusable construction, including competing callers.
        with (self.out / 'runner_receipt.json').open('x') as stream:
            json.dump(self.receipt, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            bind_group(args, self.receipt, self.save_receipt)
            self.output.mkdir(parents=True)
            (self.out / 'agent').mkdir()
            for name in ('numpy', 'scipy', 'pymatgen', 'pydantic'):
                try:
                    self.receipt['runtime']['packages'][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    self.receipt['runtime']['packages'][name] = None
            if type(args.time_limit) in (int, float) and math.isfinite(args.time_limit) and args.time_limit > 0:
                self.deadline = self.t0 + args.time_limit
            self.evidence, self.receipt['comparison'] = preflight(args, execute=self.execute_giwaxs_raw)
            if self.task_kind == 'giwaxs':
                self.receipt['runtime']['giwaxs'] = self.receipt['comparison']['identity']['giwaxs_runtime']
                self.evidence.execute = self.execute_giwaxs
            self.library = {row['candidate_id'] for row in self.evidence.candidates}
            self.deadline = self.t0 + args.time_limit
            group = self.receipt['group']
            if group and group['comparison_fingerprint'] != self.receipt['comparison']['fingerprint']:
                raise ValueError('comparison fingerprint differs from the scheduled group')
            self.receipt['setup'] = {'status': 'valid', 'detail': None}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            self.receipt['setup'] = {'status': 'invalid', 'detail': self.scrub(str(exc))}
            self.receipt['workbench_import'] = 'unsupported_setup_invalid'
        self.save_receipt()

    def save_receipt(self):
        atomic_json(self.out / 'runner_receipt.json', self.receipt)

    def remaining(self, maximum=None):
        if self.deadline is None:
            raise InputChanged(self.receipt['setup']['detail'] or 'setup is invalid')
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExpired('wall time budget exhausted')
        return min(remaining, maximum) if maximum else remaining

    def check_inputs(self):
        if self.receipt['setup']['status'] != 'valid':
            raise InputChanged(self.receipt['setup']['detail'] or 'setup is invalid')
        try:
            for name, expected in self.receipt['comparison']['identity']['inputs'].items():
                checked_bytes(self.args.input_dir.absolute(), name, expected)
            identity = self.receipt['comparison']['identity']
            for name, expected in identity['sources'].items():
                checked_bytes(REPO, name, expected)
            system, tools = task_contract(self.args)
            if digest(system.encode()) != identity['prompt_sha256']:
                raise ValueError('prompt changed after setup')
            if digest(json.dumps(tools, sort_keys=True, allow_nan=False).encode()) != identity['tools_sha256']:
                raise ValueError('tools changed after setup')
            if self.task_kind == 'giwaxs':
                from harness.giwaxs.environment import probe_runtime
                import numpy as np
                actual = {'worker': probe_runtime(self.args.giwaxs_python, self.execute_giwaxs_raw),
                          'processing_numpy': np.__version__}
                if actual != identity['giwaxs_runtime']:
                    raise ValueError('GIWAXS runtime changed after setup')
        except BudgetExpired:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            self.input_failure = self.scrub(str(exc))
            raise InputChanged(self.input_failure) from exc

    def tool(self, script: str, *extra: str, output: bool = False) -> str:
        self.check_inputs()
        timeout = self.remaining(1800)
        cmd = [str(PYTHON if PYTHON.exists() else sys.executable), str(PHYSICS / script), '--input-dir', str(self.args.input_dir.resolve()), f'--case={self.args.case}']
        if output:
            cmd += ['--output-dir', str(self.output)]
        cmd += list(extra)
        try:
            stdout, stderr, code = bounded_process(cmd, cwd=PHYSICS, env=physics_environment(self.args),
                                                   timeout=timeout, deadline=self.deadline)
        except BudgetExpired:
            raise
        except TimeoutError:
            return 'error: physics tool exceeded 1800 seconds; process group terminated'
        text = self.scrub(stdout + (f'\n[stderr]\n{stderr}' if stderr.strip() else '') + f'\n[exit {code}]')
        return ('error: physics tool failed\n' if code else '') + text

    def execute_giwaxs_raw(self, argv):
        self.remaining()
        stdout = giwaxs_execute(self.args, argv, deadline=self.deadline)
        if argv[-1] == '--probe':
            self.receipt['runtime']['giwaxs_observed_worker'] = strict_json(stdout)
        return stdout

    def execute_giwaxs(self, argv):
        self.check_inputs()
        return self.execute_giwaxs_raw(argv)

    def scrub(self, text: str) -> str:
        """Container paths only: no host paths, base URL or key reach actor records."""
        for real, shown in ((str(self.output), '/workspace/output'), (str(self.args.input_dir.resolve()), '/workspace/input'), (str(REPO), '.'), (self.args.base_url, '<endpoint>'), (str(Path.home()), '~')):
            text = text.replace(real, shown)
        key = os.environ.get(self.args.api_key_env)
        return text.replace(key, '<key>') if key else text

    def call(self, name: str, a: dict) -> str:
        self.call_outcome = None
        try:
            self.check_inputs()
        except InputChanged as exc:
            self.call_outcome = 'failure'
            return 'error: ' + self.scrub(str(exc))
        if not isinstance(a, dict):
            return 'error: arguments must be a JSON object'
        try:
            json.dumps(a, allow_nan=False)
        except (ValueError, TypeError):
            return 'error: arguments must contain finite JSON values'
        if self.task_kind == 'giwaxs':
            try:
                result = self.evidence.call(name, a)
                self.call_outcome = 'success'
                if self.evidence.reduction_bound:
                    self.receipt['workbench_import'] = 'reduction_bound'
                return self.scrub(json.dumps(result, ensure_ascii=False, allow_nan=False))
            except BudgetExpired:
                raise
            except (ValueError, TypeError, KeyError) as exc:
                self.call_outcome = 'rejected'
                return 'error: ' + self.scrub(str(exc))
            except (RuntimeError, OSError) as exc:
                self.call_outcome = 'failure'
                self.remaining()
                return 'error: ' + self.scrub(str(exc))
        if name == 'read_evidence':
            try:
                return json.dumps(self.evidence.read(**a), ensure_ascii=False, allow_nan=False)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                return 'error: ' + self.scrub(str(exc))
        if name == 'case_brief':
            return self.tool('case_brief.py')
        if name == 'candidates':
            return self.tool('candidates.py')
        if name == 'screen':
            r = a.get('residual')
            if r is not None:
                if not isinstance(r, str) or not LABEL.fullmatch(r) or r not in self.fit_phases:
                    return 'error: residual must name a refinement from this episode'
                return self.tool('screen.py', f'--residual={r}', output=True)
            return self.tool('screen.py', '--no-search', output=True)
        if name == 'refine':
            c, label = a.get('candidates'), a.get('run_label')
            if not isinstance(c, list) or not 1 <= len(c) <= 5 or not all(isinstance(x, str) for x in c) or len(set(c)) != len(c):
                return 'error: candidates must be one to five distinct candidate IDs'
            if bad := sorted(set(c) - self.library):
                return f'error: not in the candidate library: {bad}'
            if not isinstance(label, str) or not LABEL.fullmatch(label):
                return 'error: run_label must match [A-Za-z0-9_-]{1,64}'
            if 'pawley_only' in a and type(a['pawley_only']) is not bool:
                return 'error: pawley_only must be boolean'
            if label in self.labels or (self.output / label).exists():
                return f'error: run label {label} already used'
            self.labels.add(label)
            self.fit_phases[label] = set(c)
            extra = ['--candidates', *c, f'--run-label={label}'] + (['--pawley-only'] if a.get('pawley_only') else [])
            try:
                return self.tool('refine.py', *extra, output=True)
            finally:
                # Never follow a tool-created symlink while scrubbing saved host paths.
                for path in (self.output / label).rglob('*.json'):
                    raw = checked_bytes(self.output, path.relative_to(self.output).as_posix())
                    atomic_json(path, strict_json(self.scrub(raw.decode())))
                self.evidence.register_fit(label)
                request = self.output / label / 'request.json'
                if request.is_file():
                    record = strict_json(checked_bytes(self.output, f'{label}/request.json'))
                    expected = {row['candidate_id']: row['sha256'] for row in self.evidence.candidates}
                    if (record['case_id'] != self.args.case or record['pattern_sha256'] != self.evidence.case['pattern_sha256']
                            or {row['candidate_id'] for row in record['candidates']} != set(c)
                            or any(expected.get(row['candidate_id']) != row['sha256'] for row in record['candidates'])):
                        raise ToolFailure('refinement request does not bind to the selected inputs')
                    self.receipt['workbench_import'] = 'refinement_bound'
        if name == 'write_analysis':
            return self.write_analysis(a.get('analysis'))
        return f'error: unknown tool {name}'

    def write_analysis(self, x: object) -> str:
        try:
            self.check_inputs()
            x = strict_json(x if isinstance(x, str) else json.dumps(x, allow_nan=False))
            if not isinstance(x, dict):
                raise ValueError('analysis must be an object')
            if x.get('case_id') != self.args.case:
                raise ValueError(f'case_id must be {self.args.case}')
            phases = x.get('phases')
            if not isinstance(phases, list) or not all(isinstance(p, dict) and isinstance(p.get('candidate_id'), str) for p in phases):
                raise ValueError('phases must be objects with candidate_id')
            ids = [p['candidate_id'] for p in phases]
            if len(set(ids)) != len(ids):
                raise ValueError('phase IDs must be distinct')
            wf = x.get('weight_fractions', {})
            if not isinstance(wf, dict) or set(wf) - set(ids):
                raise ValueError('weight_fractions must be an object keyed only by reported phases')
            if set(ids) - self.library:
                raise ValueError('phase is not in the candidate library')
            for key, value in wf.items():
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f'weight fraction for {key} must be a finite number within 0..1')
            artifacts = x.get('fit_artifacts', [])
            if not isinstance(artifacts, list) or not all(isinstance(p, str) for p in artifacts):
                raise ValueError('fit_artifacts must be paths')
            for name in artifacts:
                relative = Path(name).relative_to('/workspace/output')
                if len(relative.parts) < 2 or relative.parts[0] not in self.fit_phases:
                    raise ValueError('fit artifact is not owned by this episode')
                checked_bytes(self.output, relative.as_posix())
                if self.fit_phases[relative.parts[0]] != set(ids):
                    raise ValueError('cited fit phases differ from the answer')
            if any(value is not None for value in wf.values()) and not artifacts:
                raise ValueError('quantified phases require an owned matching fit artifact')
            atomic_json(self.output / 'analysis.json', x)
            return 'wrote /workspace/output/analysis.json'
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return 'error: ' + self.scrub(str(exc))

    def chat(self, messages: list) -> dict:
        self.check_inputs()
        body = json.dumps({'model': self.args.model, 'messages': messages, 'tools': task_contract(self.args)[1],
                           'tool_choice': 'auto', **getattr(self.args, 'decoding', {})}, allow_nan=False).encode()
        headers = {'Content-Type': 'application/json'}
        key = os.environ.get(self.args.api_key_env)
        if key:
            headers['Authorization'] = f'Bearer {key}'
        timeout = self.remaining(900)
        payload = json.dumps({'url': self.args.base_url.rstrip('/') + '/chat/completions',
                              'body': body.decode(), 'headers': headers, 'timeout': timeout})
        with subprocess.Popen([sys.executable, '-c', HTTP_REQUEST], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            try:
                stdout, stderr = process.communicate(payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()  # This fixed HTTP worker cannot create descendants.
                process.communicate()
                self.remaining()
                raise TimeoutError('provider request exceeded 900 seconds') from None
        self.remaining()
        if process.returncode:
            raise RuntimeError('provider worker failed: ' + self.scrub(stderr[-1000:]))
        result = strict_json(stdout)
        if 'error' in result:
            raise RuntimeError(self.scrub(result['error']))
        return strict_json(result['body'])

    def record_response(self, response):
        usage = response.get('usage') or {}
        observed = {}
        for key, total in self.receipt['usage'].items():
            value = usage.get(key)
            value = value if type(value) is int and value >= 0 else None
            observed[key] = value
            total['known_subtotal'] += value if value is not None else 0
            total['complete'] = total['complete'] and value is not None
            total['total'] = total['known_subtotal'] if total['complete'] else None
        self.receipt['provider_responses'].append({'usage': observed,
            'model': response.get('model'), 'system_fingerprint': response.get('system_fingerprint')})
        for key, field in (('model', 'served_models'), ('system_fingerprint', 'system_fingerprints')):
            value = response.get(key)
            if value is not None and value not in self.receipt['runtime'][field]:
                self.receipt['runtime'][field].append(value)
        self.save_receipt()

    def run(self) -> int:
        if self.has_run:
            raise ValueError('this trial has already run')
        self.has_run = True
        a, steps, turns, exception, stage = self.args, [], 0, None, 'setup'
        task_sha, provider_pending = None, False
        try:
            self.check_inputs()
            system = task_contract(a)[0]
            task_sha = digest(system.encode())
            if task_sha != self.receipt['comparison']['identity']['prompt_sha256']:
                raise InputChanged('prompt changed after setup')
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': f'Solve case {a.case}.'}]
            steps = [{'step_id': i + 1, 'source': msg['role'], 'message': msg['content']} for i, msg in enumerate(messages)]
            self.save_trajectory(steps)
            while True:
                self.remaining()
                if turns >= a.max_turns:
                    raise BudgetExpired(f'model turn budget exhausted after {turns} turns')
                turns += 1
                stage = 'provider'
                self.check_inputs()
                provider_pending = True
                resp = strict_json(json.dumps(self.chat(messages), allow_nan=False))
                self.record_response(resp)
                provider_pending = False
                self.remaining()
                msg = resp['choices'][0]['message']
                calls = msg.get('tool_calls') or []
                messages.append({k: v for k, v in msg.items() if k in ('role', 'content', 'tool_calls') and v is not None} | {'role': 'assistant'})
                step = {'step_id': len(steps) + 1, 'source': 'agent', 'timestamp': now(), 'message': msg.get('content') or '',
                        'reasoning_content': msg.get('reasoning_content'), 'tool_calls': [], 'observation': {'results': []}}
                steps.append(step)
                self.save_trajectory(steps)
                for c in calls:
                    self.remaining()
                    stage = 'tool'
                    fn = c['function']
                    try:
                        arguments = strict_json(fn.get('arguments') or '{}')
                    except (ValueError, TypeError):
                        arguments = {'_unparsed': fn.get('arguments')}
                    event_id = str(uuid.uuid4())
                    action = {'event_id': event_id, 'outcome': None, 'tool_call_id': c['id'], 'function_name': fn['name'], 'started_at': now(),
                              'duration_seconds': None, 'status': 'started'}
                    self.receipt['actions'].append(action)
                    step['tool_calls'].append({'event_id': event_id, 'tool_call_id': c['id'], 'function_name': fn['name'], 'arguments': arguments})
                    self.save_trajectory(steps)
                    self.save_receipt()
                    began = time.monotonic()
                    self.call_outcome = None
                    try:
                        full = self.call(fn['name'], arguments) if isinstance(arguments, dict) and '_unparsed' not in arguments else 'error: arguments are not a finite JSON object'
                        action['status'] = 'completed'
                        action['outcome'] = self.call_outcome or ('rejected' if full.startswith('error:') else 'success')
                    except Exception:
                        action['status'] = 'failed'
                        action['outcome'] = 'failure'
                        raise
                    finally:
                        action['duration_seconds'] = time.monotonic() - began
                        step['tool_calls'][-1]['outcome'] = action['outcome']
                        self.save_receipt()
                    result = observation(self.out, c['id'], full)
                    result.update(event_id=event_id, outcome=action['outcome'])
                    if 'extra' in result:
                        self.evidence.register_output(result['extra'])
                    step['observation']['results'].append(result)
                    messages.append({'role': 'tool', 'tool_call_id': c['id'], 'content': result['content']})
                    self.save_trajectory(steps)
                    if self.input_failure:
                        raise InputChanged(self.input_failure)
                    if self.task_kind == 'giwaxs' and self.evidence.submission is not None:
                        break
                if not calls or self.task_kind == 'giwaxs' and self.evidence.submission is not None:
                    break
            ending = 'submitted' if (self.output / 'analysis.json').is_file() else 'no_submission'
        except Exception as exc:
            if provider_pending:
                for usage in self.receipt['usage'].values():
                    usage['complete'], usage['total'] = False, None
            if isinstance(exc, BudgetExpired) or (isinstance(exc, TimeoutError) and self.deadline is not None and time.monotonic() >= self.deadline):
                ending, kind = 'budget_exhausted', 'AgentTimeoutError'
            elif isinstance(exc, InputChanged) or stage == 'setup':
                ending, kind = 'setup_failure', 'SetupError'
            elif stage == 'provider':
                ending, kind = 'provider_failure', 'ProviderError'
            else:
                ending, kind = 'tool_failure', 'ToolExecutionError'
            detail = f'HTTP {exc.code}' if isinstance(exc, urllib.error.HTTPError) else str(exc)
            exception = {'exception_type': kind, 'exception_message': self.scrub(detail)}
        (self.out / 'agent').mkdir(exist_ok=True)
        self.save_trajectory(steps)
        if self.task_kind == 'giwaxs':
            self.receipt['evidence_bindings'] = {}
            for name in ('giwaxs-evidence.json', 'artifacts/workspace/output/analysis.json', 'agent/trajectory.json'):
                self.receipt['evidence_bindings'][name] = None
                try:
                    if (self.out / name).exists() or (self.out / name).is_symlink():
                        self.receipt['evidence_bindings'][name] = digest(checked_bytes(self.out, name))
                except (OSError, ValueError) as exc:
                    ending = 'setup_failure'
                    exception = {'exception_type': 'SetupError', 'exception_message': self.scrub(str(exc))}
        finished = now()
        self.receipt.update(finished_at=finished, ending={'kind': ending, 'detail': exception['exception_message'] if exception else None},
                            wall_seconds=time.monotonic() - self.t0)
        self.save_receipt()
        label = a.endpoint_label or 'openai-compatible'
        atomic_json(self.out / 'config.json', {
            'trials_dir': self.out.parent.name, 'task': {'case_id': a.case, 'input_dir': '/workspace/input', 'task_kind': self.task_kind},
            'agent': {'name': HARNESS, 'model_name': a.model, 'endpoint_label': label, 'override_timeout_sec': a.time_limit if type(a.time_limit) in (int, float) and math.isfinite(a.time_limit) else None, 'max_turns': a.max_turns},
            'tool_versions': {'comparison_fingerprint': self.receipt['comparison']['fingerprint'] if self.receipt['comparison'] else None},
        })
        atomic_json(self.out / 'result.json', {
            'trial_name': self.receipt['trial_id'], 'task_name': a.case, 'task_checksum': task_sha,
            'agent_info': {'name': HARNESS, 'version': 'receipt-v1', 'model_info': {'name': a.model, 'provider': label}},
            'agent_result': {'n_input_tokens': self.receipt['usage']['prompt_tokens']['total'],
                             'n_output_tokens': self.receipt['usage']['completion_tokens']['total'], 'cost_usd': None},
            'agent_execution': {'started_at': self.started, 'finished_at': finished},
            'exception_info': exception, 'turns': turns, 'wall_seconds': self.receipt['wall_seconds'],
            'status': 'error' if exception else ending,
        })
        print(json.dumps({'out': a.out.name, 'turns': turns, 'exception': exception, 'answer': (self.output / 'analysis.json').is_file()}))
        return 0 if exception is None else 1

    def save_trajectory(self, steps: list) -> None:
        atomic_json(self.out / 'agent/trajectory.json', {'schema_version': 'ATIF-v1.2', 'session_id': self.sid,
            'agent': {'name': HARNESS, 'model_name': self.args.model}, 'steps': steps})

def trial_name(out: Path, session_id: str) -> str:
    """The output folder's name and the run's session id: unique per run, so runs written to
    folders of one name do not share the trial name an import is keyed on."""
    return f"{out.resolve().name}-{session_id[:8]}"


def observation(out: Path, call_id: str, full: str) -> dict:
    """The trajectory's record of one call's output: the tail the model is shown, and, when that
    is a truncation, the executor's full output kept at agent/tool-output/<its sha256>.txt (by
    content, so a server that reuses call ids cannot overwrite an earlier call's output)."""
    result = {"source_call_id": call_id, "content": full[-SHOWN_CHARS:]}
    if len(full) > SHOWN_CHARS:
        sha = digest(full.encode())
        name = f"agent/tool-output/{sha}.txt"
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(full.encode())
        handle = f"output:{sha}"
        notice = f"[output clipped; recover full text with read_evidence(handle=\"{handle}\", start=0)]\n"
        result["content"] = notice + full[-(SHOWN_CHARS - len(notice)):]
        result["extra"] = {
            "evidence_handle": handle,
            "executor_output": name,
            "executor_sha256": sha,
            "executor_chars": len(full),
            "shown_chars": SHOWN_CHARS - len(notice),
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--task-kind", choices=("powder", "giwaxs"), default="powder")
    p.add_argument("--giwaxs-python", type=Path, help="absolute path to the pinned GIWAXS worker interpreter")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint-label", default=None, help="optional name for the endpoint, recorded instead of the URL")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--time-limit", type=int, default=3440)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decoding-json", dest="decoding", type=strict_json, default={}, help="explicit decoding settings; omitted provider defaults remain unknown")
    p.add_argument("--model-revision", help="declared model revision, if known")
    p.add_argument("--group-manifest", type=Path, help="predeclared fixed attempt group")
    p.add_argument("--episode-id", help="scheduled slot, attempted at most once")
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error(f"{a.out} is not empty")
    return Runner(a).run()


if __name__ == "__main__":
    sys.exit(main())
