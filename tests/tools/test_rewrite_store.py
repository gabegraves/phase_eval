import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

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
