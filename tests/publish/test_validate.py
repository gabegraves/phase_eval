import json

import pytest

from phaseeval.workbench import exports


@pytest.mark.parametrize("line", [b"{bad json\n", b"\xff\n"])
def test_a_malformed_line_is_reported_not_raised(published, line):
    (published / "episodes.jsonl").write_bytes(line)
    problems = exports.validate_dataset(published, exports.dataset_json_schemas())
    assert any('episodes.jsonl:1: not JSON' in p for p in problems)


def test_a_schema_valid_label_failing_its_model_is_refused(published):
    snapshot = json.loads((published / 'manifest.json').read_text())['snapshot_index'][0]['digest']
    label = dict(schema_version='label.v1', snapshot=snapshot, item='fit_supports_conclusion', phase_id='nacl',
        answer='yes', rater={'principal_id': 'r-0123456789ab', 'kind': 'expert', 'judge': None},
        exposure='blind', created_at_utc='2026-10-22T00:00:00Z')
    from jsonschema import Draft202012Validator
    schemas = exports.dataset_json_schemas()
    assert Draft202012Validator(schemas['label.v1.json']).is_valid(label)
    (published / 'labels.expert.jsonl').write_text(json.dumps(label) + '\n')
    assert any('labels.expert.jsonl:1: model' in p for p in exports.validate_dataset(published, schemas))


def test_counts_that_disagree_with_the_rows_fail(published):
    path = published / 'manifest.json'
    manifest = json.loads(path.read_text())
    next(r for r in manifest['counts'] if r['name'] == 'snapshots')['count'] = 5
    path.write_text(json.dumps(manifest))
    assert any('count_mismatch' in p for p in exports.validate_dataset(published, exports.dataset_json_schemas()))


@pytest.mark.parametrize('name', ['extra.json', 'extra.txt', 'episodes.jsonl'])
def test_every_file_is_scanned(published, name):
    with (published / name).open('a') as handle:
        handle.write(json.dumps(['alice@example.org']) + '\n')
    assert any(name in p and 'email' in p for p in exports.validate_dataset(published, exports.dataset_json_schemas()))


def test_the_published_v0_dataset_validates(published):
    assert exports.validate_dataset(published, exports.dataset_json_schemas()) == []


@pytest.mark.parametrize('name, content', [('manifest.json', '{bad'), ('judge_manifest.json', '{bad')])
def test_malformed_manifest_is_reported(empty_export, tmp_path, name, content):
    out = tmp_path / 'dataset'
    exports.write_dataset(empty_export[2](), out)
    (out / name).write_text(content)
    assert exports.validate_dataset(out, exports.dataset_json_schemas())


def test_v2_export_names_walked_exported_and_count_files(empty_export, tmp_path):
    out = tmp_path / 'dataset'
    exports.write_dataset(empty_export[2](), out)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['schema_version'] == 'phaseeval.dataset-manifest.v2'
    assert manifest['episode_export']['episodes_walked'] == manifest['episode_export']['episodes_exported'] == 0
    assert 'episodes' not in manifest['episode_export']
    assert all(row['file'] for row in manifest['counts'])
    manifest['episode_export']['episodes_walked'] = 0
    manifest['episode_export']['episodes_exported'] = 1
    (out / 'manifest.json').write_text(json.dumps(manifest))
    assert any('below' in p for p in exports.validate_dataset(out, exports.dataset_json_schemas()))


@pytest.mark.parametrize('name', ['episodes_incomplete', 'agent_snapshots', 'snapshots_with_chemistry',
                                'snapshots_judged', 'snapshots_with_expert_labels', 'snapshots_with_model_judge_labels'])
def test_subset_counts_match_the_records(published, name):
    path = published / 'manifest.json'
    manifest = json.loads(path.read_text())
    row = next(row for row in manifest['counts'] if row['name'] == name)
    row['count'] = row['count'] - 1 if row['count'] else 1
    path.write_text(json.dumps(manifest))
    assert any('count_mismatch' in p for p in exports.validate_dataset(published, exports.dataset_json_schemas()))
