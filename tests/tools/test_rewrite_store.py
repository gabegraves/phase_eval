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
    with tarfile.open(path, 'w:xz') as archive:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


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


@pytest.mark.parametrize('change', ['wall_seconds', 'trajectory', 'coordinates'])
@pytest.mark.parametrize('scope', ['dataset', 'export'])
def test_comparison_detects_bundled_and_fresh_record_changes(tmp_path, monkeypatch, change, scope):
    spec = importlib.util.spec_from_file_location('compare_exports', REPO / 'tools/demo/compare_exports.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    bundled = {name: data for name, data in files(REPO / 'launcher/demo/phaseeval-demo.tar.xz').items()
               if name.startswith('dataset/')}
    original, changed = tmp_path / 'original.tar.xz', tmp_path / 'changed.tar.xz'
    write_archive(original, bundled)

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
    else:
        coordinates = rows[0]['root']['case']['measurement']['two_theta_deg']
        assert coordinates[0] != coordinates[1]
        coordinates.reverse()
    altered['dataset/episodes.jsonl'] = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    manifest = json.loads(altered['dataset/manifest.json'])
    for entry in manifest['files']:
        if entry['path'] == 'episodes.jsonl':
            entry['sha256'] = hashlib.sha256(altered['dataset/episodes.jsonl']).hexdigest()
    altered['dataset/manifest.json'] = json.dumps(manifest).encode()
    write_archive(changed, altered if scope == 'dataset' else bundled)
    mapping = tmp_path / 'map.json'
    mapping.write_text('{}')
    differences = compare.compare_exports(original, changed, mapping, 'http://old.invalid/v1', 'http://new.invalid/v1')
    assert [Path(name).name for name in differences] == ['episodes.jsonl']


def test_rewrite_reaches_a_fixed_point_and_the_exports_agree(tmp_path):
    original = REPO / 'launcher/demo/phaseeval-demo.tar.xz'
    rebuilt, mapping = tmp_path / 'rebuilt.tar.xz', tmp_path / 'map.json'
    old, new = 'http://redacted.invalid/v1', 'https://redacted.invalid/v1'
    output = run_tool('rewrite_store.py', original, rebuilt, old, new, mapping)
    assert 'fixed point' in output
    assert json.loads(mapping.read_text())
    before, after = files(original), files(rebuilt)
    assert len(before) == len(after) == 1761
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
