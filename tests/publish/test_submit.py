import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('p02_submit', REPO / 'scripts/submit_run.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


def test_a_labels_file_is_refused(tmp_path):
    path = tmp_path / 'labels.expert.jsonl'
    path.write_text(json.dumps(dict(schema_version='label.v1', snapshot='1' * 64, item='fit_supports_conclusion',
        phase_id=None, answer='yes', rater={'principal_id':'r-0123456789ab', 'kind':'expert'},
        exposure='blind', created_at_utc='2026-10-22T00:00:00Z')) + '\n')
    with pytest.raises(submit.Refused, match='episodes'):
        submit.check(path, False)


def test_a_bundled_demo_run_is_refused(tmp_path):
    path = tmp_path / 'episodes.jsonl'
    path.write_text((REPO / 'data/dataset-v0-public/episodes.jsonl').read_text().splitlines()[0] + '\n')
    with pytest.raises(submit.Refused, match='demo_run'):
        submit.check(path, False)


def test_a_valid_non_demo_bundle_passes(published):
    import hashlib
    import re

    path = published / 'episodes.jsonl'
    data = re.sub(rb'u6-(dev006|dev102)-qwen-(006|007|008)', rb'contributor-\1-run-\2', path.read_bytes())
    path.write_bytes(data)
    manifest_path = published / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    next(row for row in manifest['files'] if row['path'] == 'episodes.jsonl')['sha256'] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    assert len(submit.check(path, False)) == 6
    manifest_path.unlink()
    with pytest.raises(submit.Refused, match='manifest.json: missing'):
        submit.check(path, False)
