"""Runner-local input/comparison records; these are not workbench export schemas."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from harness.agent.evidence import Evidence, checked_bytes

SAFE_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')
SHA256 = re.compile(r'[a-f0-9]{64}')


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    # JSON exponents such as 1e999 bypass parse_constant.
    json.dumps(value, allow_nan=False)
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preflight(args, *, repo: Path, system: str, tools: list) -> tuple[Evidence, dict]:
    """Read and validate only: no model, physics execution, output creation or slot claim."""
    if not isinstance(args.case, str) or not SAFE_ID.fullmatch(args.case):
        raise ValueError('case ID must be 1..64 letters, digits, underscores or hyphens')
    if type(args.max_turns) is not int or args.max_turns <= 0:
        raise ValueError('max_turns must be a positive integer')
    if type(args.time_limit) not in (int, float) or not math.isfinite(args.time_limit) or args.time_limit <= 0:
        raise ValueError('time_limit must be positive and finite')
    decoding = getattr(args, 'decoding', {})
    allowed_decoding = {'temperature', 'top_p', 'seed', 'max_tokens', 'max_completion_tokens',
                        'presence_penalty', 'frequency_penalty'}
    if not isinstance(decoding, dict) or set(decoding) - allowed_decoding:
        raise ValueError('unsupported decoding field')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in decoding.values()):
        raise ValueError('decoding values must be finite numbers')
    for name in ('model', 'base_url'):
        if not isinstance(getattr(args, name), str) or not getattr(args, name):
            raise ValueError(f'{name} must be a nonempty string')
    inputs = args.input_dir.absolute()
    manifest = checked_bytes(inputs, 'manifest.json')
    cases = strict_json(manifest)['cases']
    ids = [row['case_id'] for row in cases]
    if any(not isinstance(x, str) or not SAFE_ID.fullmatch(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('case IDs must be safe and distinct')
    library_name = f'candidates/{args.case}/manifest.json'
    library_bytes = checked_bytes(inputs, library_name)
    library = strict_json(library_bytes)
    if library['case_id'] != args.case:
        raise ValueError('candidate manifest case ID differs')
    ids = [row['candidate_id'] for row in library['candidates']]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('candidate IDs must be nonempty and distinct')
    evidence = Evidence(inputs, args.case, args.out / 'artifacts/workspace/output', args.out)
    bound = {'manifest.json': digest(manifest), library_name: digest(library_bytes)}
    for root, name, _kind, expected in evidence.files.values():
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError('input digest must be SHA256')
        raw = checked_bytes(root, name, expected)
        if not raw.strip():
            raise ValueError('empty input evidence')
        bound[(root / name).relative_to(evidence.inputs).as_posix()] = expected
    scan = evidence.read(handle='scan', limit=1)
    if scan['total_rows'] < 2:
        raise ValueError('scan must have at least two usable rows')
    sources = {repo / 'harness/agent/run_agent.py', repo / 'harness/agent/evidence.py', Path(__file__),
               repo / 'src/phaseeval/workbench/evidence.py', repo / 'src/phaseeval/workbench/importer.py',
               repo / 'src/phaseeval/workbench/importers/external.py'}
    sources.update((repo / 'harness/physics').glob('*.py'))
    sources.update(p for p in (repo / 'src/phaseeval/refinement').rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    sources.update(repo / name for name in ('pyproject.toml', 'launcher/uv.lock', 'launcher/pyproject.toml'))
    identity = {
        'case_id': args.case, 'inputs': bound,
        'prompt_sha256': digest(system.encode()), 'tools_sha256': digest(json.dumps(tools, sort_keys=True, allow_nan=False).encode()),
        'sources': {p.relative_to(repo).as_posix(): digest(p.read_bytes()) for p in sorted(sources)},
        'policy': {'model_requested': args.model, 'endpoint_label': args.endpoint_label,
                   'endpoint_sha256': digest(args.base_url.rstrip('/').encode()),
                   'model_revision': getattr(args, 'model_revision', None),
                   'decoding': getattr(args, 'decoding', {}), 'unspecified_decoding': 'provider_defaults_unknown'},
        'budget': {'max_turns': args.max_turns, 'wall_seconds': args.time_limit, 'tool_seconds': 1800},
        'unknown_pins': ['served_model_revision', 'provider_decoding_defaults', 'bgmn_binary', 'installed_environment_matches_lock'],
    }
    comparison = {'fingerprint': digest(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()),
                  'identity': identity}
    return evidence, comparison


def bind_group(args, receipt: dict, save) -> None:
    """Bind exact manifest bytes, then atomically consume its slot before input setup."""
    path, episode_id = getattr(args, 'group_manifest', None), getattr(args, 'episode_id', None)
    if path is None and episode_id is None:
        return
    if path is None or not isinstance(episode_id, str) or not SAFE_ID.fullmatch(episode_id):
        raise ValueError('group_manifest and a safe episode_id are required together')
    path = Path(path).absolute()
    receipt['group'] = {'group_id': None, 'manifest_sha256': None,
                        'episode_id': episode_id, 'comparison_fingerprint': None}
    save()
    raw = checked_bytes(path.parent, path.name)
    receipt['group'] = {'group_id': None, 'manifest_sha256': digest(raw),
                        'episode_id': episode_id, 'comparison_fingerprint': None}
    save()
    manifest = strict_json(raw)
    if not isinstance(manifest, dict):
        raise ValueError('group manifest must be an object')
    receipt['group'] = {'group_id': manifest.get('group_id'), 'manifest_sha256': digest(raw),
                        'episode_id': episode_id, 'comparison_fingerprint': manifest.get('comparison_fingerprint')}
    save()
    if type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('unsupported group manifest version')
    if not isinstance(manifest.get('group_id'), str) or not SAFE_ID.fullmatch(manifest['group_id']):
        raise ValueError('invalid group ID')
    created = dt.datetime.fromisoformat(manifest['created_at'])
    if created.tzinfo is None or created > dt.datetime.now(dt.timezone.utc):
        raise ValueError('group creation must be timezone-aware and precede execution')
    ids = manifest.get('episode_ids')
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not SAFE_ID.fullmatch(i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('scheduled episode IDs must be nonempty and distinct')
    if episode_id not in ids:
        raise ValueError('episode ID was not scheduled')
    fingerprint = manifest.get('comparison_fingerprint')
    if not isinstance(fingerprint, str) or not SHA256.fullmatch(fingerprint):
        raise ValueError('invalid group comparison fingerprint')
    claims = path.parent / ('.' + path.name + '.attempts')
    if claims.is_symlink():
        raise ValueError('symlink slot directory is not allowed')
    claims.mkdir(exist_ok=True)
    try:
        with (claims / (episode_id + '.json')).open('x') as stream:
            json.dump(receipt['group'] | {'session_id': receipt['session_id'], 'trial_id': receipt['trial_id']}, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise ValueError('scheduled episode already attempted; create a new group for retries') from None
