import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_tool(name, *args, exit_code=0):
    result = subprocess.run(
        [sys.executable, str(REPO / 'tools/demo' / name), *map(str, args)],
        env={**os.environ, 'PYTHONPATH': str(REPO / 'src')},
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    return result.stdout


def files(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}


def write_archive(path, contents):
    with tarfile.open(path, 'w:xz', preset=0) as archive:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def replace_episodes(contents, rows):
    contents['dataset/episodes.jsonl'] = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    manifest = json.loads(contents['dataset/manifest.json'])
    for entry in manifest['files']:
        if entry['path'] == 'episodes.jsonl':
            entry['sha256'] = hashlib.sha256(contents['dataset/episodes.jsonl']).hexdigest()
    contents['dataset/manifest.json'] = json.dumps(manifest).encode()


def test_synthetic_dirty_receipts_have_14_address_hits_and_20_path_placeholders(tmp_path):
    from phaseeval.publish.leaks import scan_bytes

    old, new = 'http://100.64.12.34:40000/v1', 'http://redacted.invalid/v1'
    contents = {}
    for index in range(20):
        receipt = {'path': f'/private-runs/jobs/synthetic/receipt-{index}'}
        if index < 14:
            receipt['base_url'] = old
        data = (json.dumps(receipt, sort_keys=True) + '\n').encode()
        contents[f'store/blobs/{hashlib.sha256(data).hexdigest()}'] = data
    original, rebuilt = tmp_path / 'dirty.tar.xz', tmp_path / 'clean.tar.xz'
    mapping = tmp_path / 'map.json'
    write_archive(original, contents)
    before = files(original)
    assert sum(b'/private-runs/' in data for data in before.values()) == 20
    for rule in ('ipv4', 'url_with_port'):
        assert sum(any(f.rule == rule for f in scan_bytes(data)) for data in before.values()) == 14
    assert all(f.rule != 'posix_private' for data in before.values() for f in scan_bytes(data))
    output = run_tool('rewrite_store.py', original, rebuilt, old, new, mapping)
    assert 'fixed point' in output
    assert len(json.loads(mapping.read_text())) == 14
    after = files(rebuilt)
    assert len(after) == 20
    assert sum(b'/private-runs/' in data for data in after.values()) == 20
    assert all(not scan_bytes(data) for data in after.values())
    assert all(Path(name).name == hashlib.sha256(data).hexdigest() for name, data in after.items())
    again, no_mapping = tmp_path / 'again.tar.xz', tmp_path / 'no-map.json'
    run_tool('rewrite_store.py', rebuilt, again, old, new, no_mapping)
    assert json.loads(no_mapping.read_text()) == {}
    assert rebuilt.read_bytes() == again.read_bytes()


@pytest.mark.parametrize('change', ['wall_seconds', 'trajectory', 'coordinates', 'record_id', 'size_bytes'])
@pytest.mark.parametrize('scope', ['dataset', 'export'])
def test_comparison_detects_bundled_and_fresh_record_changes(tmp_path, monkeypatch, change, scope):
    spec = importlib.util.spec_from_file_location('compare_exports', REPO / 'tools/demo/compare_exports.py')
    compare = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(REPO / 'tools/demo'))
    spec.loader.exec_module(compare)
    contents = files(REPO / 'launcher/demo/phaseeval-demo.tar.xz')
    from phaseeval.workbench.exports import _record_id

    rows = [json.loads(line) for line in contents['dataset/episodes.jsonl'].splitlines()]
    for row in rows:
        row['record_id'] = _record_id({k: v for k, v in row.items() if k != 'record_id'})
    replace_episodes(contents, rows)
    bundled = {name: data for name, data in contents.items() if name.startswith('dataset/')}
    original, changed = tmp_path / 'original.tar.xz', tmp_path / 'changed.tar.xz'
    write_archive(original, contents)

    calls = 0

    def fresh_export(argv):
        nonlocal calls
        calls += 1
        out = Path(argv[argv.index('--out') + 1])
        out.mkdir()
        for name, data in (altered if scope == 'export' and calls == 2 else bundled).items():
            (out / Path(name).name).write_bytes(data)
        return 0

    monkeypatch.setattr(compare.cli, 'main', fresh_export)
    altered = dict(bundled)
    rows = [json.loads(line) for line in altered['dataset/episodes.jsonl'].splitlines()]
    if change == 'wall_seconds':
        rows[0]['effort']['agent_wall_seconds'] = 123456.0
    elif change == 'trajectory':
        trajectory = rows[0]['steps'][0]['attempt']['outcome']['result']['trajectory']
        assert len(trajectory) > 1
        trajectory.reverse()
    elif change == 'coordinates':
        coordinates = rows[0]['root']['case']['measurement']['two_theta_deg']
        assert coordinates[0] != coordinates[1]
        coordinates.reverse()
    elif change == 'record_id':
        rows[0]['record_id'] = '0' * 64
    else:
        rows[0]['root']['case']['pattern']['size_bytes'] += 1
    if change != 'record_id':
        for row in rows:
            row['record_id'] = _record_id({k: v for k, v in row.items() if k != 'record_id'})
    replace_episodes(altered, rows)
    write_archive(changed, {**contents, **altered} if scope == 'dataset' else contents)
    mapping = tmp_path / 'map.json'
    mapping.write_text('{}')
    if change in ('record_id', 'size_bytes'):
        with pytest.raises(ValueError, match=change):
            compare.compare_exports(original, changed, mapping, 'http://old.invalid/v1', 'http://new.invalid/v1')
    else:
        differences = compare.compare_exports(original, changed, mapping, 'http://old.invalid/v1', 'http://new.invalid/v1')
        assert [Path(name).name for name in differences] == ['episodes.jsonl']


@pytest.mark.parametrize('change', ['count', 'policy', 'timestamp'])
@pytest.mark.parametrize('scope', ['dataset', 'export'])
def test_comparison_checks_manifest_facts(tmp_path, monkeypatch, capsys, change, scope):
    spec = importlib.util.spec_from_file_location('compare_exports', REPO / 'tools/demo/compare_exports.py')
    compare = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(REPO / 'tools/demo'))
    spec.loader.exec_module(compare)
    manifest = json.loads((REPO / 'data/dataset-v0-public/manifest.json').read_text())
    name = f'{scope}/manifest.json'
    before = {name: json.dumps(manifest)}
    if change == 'count':
        next(row for row in manifest['counts'] if row['name'] == 'snapshots_with_chemistry')['count'] -= 1
    elif change == 'policy':
        manifest['episode_export']['policy_version'] = 'changed-policy'
    else:
        manifest['generated_at_utc'] = '2026-10-26T00:00:00Z'
    after = {name: json.dumps(manifest)}
    mapping = tmp_path / 'map.json'
    mapping.write_text('{}')
    monkeypatch.setattr(compare, 'export', lambda tarball, work: (before if tarball.name == 'original' else after, {}))
    result = compare.main(['original', 'rebuilt', str(mapping), 'old.invalid', 'new.invalid'])
    output = capsys.readouterr().out
    assert result == (0 if change == 'timestamp' else 1), output
    assert (f'{scope}/manifest.json' in output) == (change != 'timestamp')


def test_rewrite_reaches_a_fixed_point_and_the_exports_agree(tmp_path):
    original = REPO / 'launcher/demo/phaseeval-demo.tar.xz'
    rebuilt, mapping = tmp_path / 'rebuilt.tar.xz', tmp_path / 'map.json'
    old, new = 'http://redacted.invalid/v1', 'https://redacted.invalid/v1'
    output = run_tool('rewrite_store.py', original, rebuilt, old, new, mapping)
    assert 'fixed point' in output
    assert json.loads(mapping.read_text())
    before, after = files(original), files(rebuilt)
    from phaseeval.workbench.exports import _record_id

    for line in after['dataset/episodes.jsonl'].splitlines():
        row = json.loads(line)
        assert row['record_id'] == _record_id({k: v for k, v in row.items() if k != 'record_id'})
    assert len(before) == len(after) == 1761
    with tarfile.open(rebuilt) as archive:
        members = archive.getmembers()
        assert [m.name for m in members] == sorted(m.name for m in members)
        assert all(m.mtime == 0 for m in members)
    assert any(old.encode() in data for data in before.values())
    assert all(old.encode() not in data for data in after.values())
    for name, data in after.items():
        if name.startswith('store/blobs/'):
            assert Path(name).name == hashlib.sha256(data).hexdigest()
    output = run_tool('compare_exports.py', original, rebuilt, mapping, old, new)
    assert 'differing beyond the scrub: none' in output
    wrong_mapping = tmp_path / 'wrong-map.json'
    wrong_mapping.write_text('{}')
    output = run_tool('compare_exports.py', original, rebuilt, wrong_mapping, old, new, exit_code=1)
    assert 'episodes.jsonl' in output
    again, no_mapping = tmp_path / 'again.tar.xz', tmp_path / 'no-map.json'
    run_tool('rewrite_store.py', rebuilt, again, old, new, no_mapping)
    assert json.loads(no_mapping.read_text()) == {}
    assert rebuilt.read_bytes() == again.read_bytes()


@pytest.mark.parametrize('kind', ['strings', 'artifacts'])
def test_rewrite_preserves_ordered_arguments_when_hash_order_changes(tmp_path, kind):
    old, new = 'old', 'new'
    for index in range(100):
        blobs = [b'old first', f'old second {index}'.encode()]
        blobs.sort(key=lambda data: hashlib.sha256(data).hexdigest())
        rewritten = [hashlib.sha256(data.replace(b'old', b'new')).hexdigest() for data in blobs]
        if rewritten != sorted(rewritten):
            break
    assert rewritten != sorted(rewritten)
    hashes = [hashlib.sha256(data).hexdigest() for data in blobs]
    arguments = hashes if kind == 'strings' else [dict(sha256=h, size_bytes=len(b)) for h, b in zip(hashes, blobs)]
    contents = {f'store/blobs/{h}': b for h, b in zip(hashes, blobs)}
    contents['store/command.json'] = (json.dumps({'arguments': arguments, 'inputs': hashes}) + '\n').encode()
    original, rebuilt, mapping = (tmp_path / name for name in ('original.tar.xz', 'rebuilt.tar.xz', 'map.json'))
    write_archive(original, contents)
    run_tool('rewrite_store.py', original, rebuilt, old, new, mapping)
    result = json.loads(files(rebuilt)['store/command.json'])
    actual = result['arguments'] if kind == 'strings' else [v['sha256'] for v in result['arguments']]
    assert actual == rewritten, 'rewrite reordered command arguments'
    assert result['inputs'] == sorted(rewritten)


def test_rewrite_refuses_to_overwrite_its_input(tmp_path):
    source = tmp_path / 'input.tar.xz'
    source.write_bytes(b'keep this input')
    result = subprocess.run(
        [sys.executable, str(REPO / 'tools/demo/rewrite_store.py'), str(source), str(source),
         'old', 'new', str(tmp_path / 'map.json')], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'paths must differ' in result.stderr
    assert source.read_bytes() == b'keep this input'
    assert not (tmp_path / 'map.json').exists()
