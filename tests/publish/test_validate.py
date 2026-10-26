import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phaseeval.workbench import exports


def replace_file(directory, name, text):
    data = text.encode()
    (directory / name).write_bytes(data)
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    entry = next(entry for entry in manifest['files'] if entry['path'] == name)
    entry.update(lines=len(data.splitlines()), sha256=hashlib.sha256(data).hexdigest())
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize('identity', [
    'u6-dev006-qwen-006', 'harbor-local-mcp', 'harbor-live', 'harbor-at', 'series-1',
    'task-dev006__foo', 'qwen', 'astra', 'gpt-6', '/jobs/run', 'data/custody', 'codex',
])
def test_run_identity_text_is_refused_in_judge_briefs(published, identity):
    rows = [json.loads(line) for line in (published / 'judge_inputs.jsonl').read_text().splitlines()]
    rows[0]['brief']['specimen'] = identity
    replace_file(published, 'judge_inputs.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
    problems = exports.validate_dataset(published, exports.dataset_json_schemas())
    assert any('judge_inputs.jsonl:1: run identity text' in p for p in problems), problems


@pytest.mark.parametrize('number', ['1e400', 'NaN', 'Infinity', '-Infinity'])
def test_nonfinite_numbers_are_reported_by_the_validator_cli(published, number):
    rows = [json.loads(line) for line in (published / 'episodes.jsonl').read_text().splitlines()]
    rows[0]['effort']['cost_usd'] = number
    text = ''.join(json.dumps(row) + '\n' for row in rows)
    replace_file(published, 'episodes.jsonl', text.replace(f'"cost_usd": "{number}"', f'"cost_usd": {number}'))
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / 'scripts/workbench/validate_eval_dataset.py'), str(published)],
        cwd=repo, env={**os.environ, 'PYTHONPATH': str(repo / 'src')}, capture_output=True, text=True,
    )
    assert 'episodes.jsonl:1: nonfinite number' in result.stdout, (result.stdout, result.stderr)
    assert result.returncode == 1
    assert 'Traceback' not in result.stderr


@pytest.mark.parametrize('field, value, detail', [
    ('episodes', 0, 'publicly exported episodes exceed accepted episodes'),
    ('excluded', 1, 'episode_export.excluded differs from excluded.jsonl'),
])
def test_v1_totals_are_checked_without_optional_count_rows(published, field, value, detail):
    path = published / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['counts'] = []
    path.write_text(json.dumps(manifest))
    assert exports.validate_dataset(published, exports.dataset_json_schemas()) == []
    manifest['episode_export'][field] = value
    path.write_text(json.dumps(manifest))
    problems = exports.validate_dataset(published, exports.dataset_json_schemas())
    assert any('count_mismatch' in p and detail in p for p in problems), problems


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


@pytest.mark.parametrize('legacy', [True, False])
@pytest.mark.parametrize('claim', ['walked', 'gradable_expectations', 'qualification_reports',
                                  'runs_imported', 'unknown_count'])
def test_manifest_counts_reconcile_with_their_evidence(published, tmp_path, legacy, claim):
    out = published
    if not legacy:
        out = tmp_path / 'migrated'
        exports.migrate_dataset(published, out)
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text())
    if claim == 'walked':
        field = 'episodes' if legacy else 'episodes_walked'
        manifest['episode_export'][field] = 777
        for row in manifest['counts']:
            if row['name'] in ('episodes_exported', 'episodes_excluded'):
                row['denominator'] = 777
    elif claim in ('gradable_expectations', 'qualification_reports'):
        row = next(row for row in manifest['counts'] if row['name'] == claim)
        assert not (out / exports.COUNT_FILES[claim]).exists()
        row.update(count=123, denominator=123)
    else:
        row = dict(name=claim, count=777, denominator=999, denominator_name='runs in the load manifest')
        if not legacy:
            row['file'] = 'episodes.jsonl'
        manifest['counts'].append(row)
    path.write_text(json.dumps(manifest))

    problems = exports.validate_dataset(out, exports.dataset_json_schemas())

    detail = 'leaves' if claim == 'walked' else claim
    assert any('count_mismatch' in p and detail in p for p in problems), problems


def test_export_omits_counts_without_exported_source_records(empty_export, tmp_path):
    from phaseeval.api.auth import Principal, PrincipalType
    from phaseeval.workbench import policy

    store, labels, _ = empty_export
    dataset = exports.build_dataset(
        store, labels, Principal('owner', PrincipalType.USER, frozenset({'owner'})), (),
        exports.ExportInputs(dataset_version='test', eligibility=policy.ELIGIBILITY_LEDGER_V0,
                             rights=policy.rights_registry_v0(()), now=store._clock()),
        'internal_evaluation', exports.RaterPseudonyms(store.root / 'private'),
        build_snapshot=lambda *a: None, review_payload=lambda *a: {},
        store_manifest={'rows': [{}]},
    )
    out = tmp_path / 'dataset'
    manifest = exports.write_dataset(dataset, out)

    assert 'runs_imported' not in {row.name for row in manifest.counts}
    assert exports.validate_dataset(out, exports.dataset_json_schemas()) == []


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


@pytest.mark.parametrize('legacy', [True, False])
def test_agent_snapshot_denominator_matches_eligible_episodes(published, tmp_path, legacy):
    import tarfile
    from pathlib import Path

    from phaseeval.workbench import cli

    out = published if legacy else tmp_path / 'export'
    if not legacy:
        repo = Path(__file__).resolve().parents[2]
        with tarfile.open(repo / 'launcher/demo/phaseeval-demo.tar.xz') as archive:
            archive.extractall(tmp_path, filter='data')
        assert cli.main(['export', '--store', str(tmp_path / 'store'), '--out', str(out)]) == 0
    assert exports.validate_dataset(out, exports.dataset_json_schemas()) == []
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text())
    count = next(row for row in manifest['counts'] if row['name'] == 'agent_snapshots')
    assert count['count'] == count['denominator'] == 6
    count['denominator'] = 999
    path.write_text(json.dumps(manifest))
    assert any('count_mismatch: agent_snapshots: denominator' in p
               for p in exports.validate_dataset(out, exports.dataset_json_schemas()))
