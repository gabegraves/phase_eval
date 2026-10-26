import hashlib
import json

import pytest

from phaseeval.workbench import exports


def test_the_published_v1_dataset_still_validates(published):
    assert exports.validate_dataset(published, exports.dataset_json_schemas()) == []


@pytest.mark.parametrize('withheld_evidence', [False, True])
def test_migrate_drops_expert_only_fields_and_validates(published, tmp_path, withheld_evidence):
    if withheld_evidence:
        rows = [json.loads(line) for line in (published / 'episodes.jsonl').read_bytes().splitlines()]
        private = {'value': 'expert specimen sentinel', 'source': {
            'kind': 'document', 'channel': 'custody_record', 'adapter': {'adapter_id': 'fixture', 'version': '1'},
            'document': None,
        }}
        artifact = {'sha256': '9' * 64, 'size_bytes': 1, 'media_type': 'text/csv', 'role': 'custody_withheld'}
        for row in rows:
            row['root']['case']['material']['specimen'] = private
            for step in row['steps']:
                step['state_after']['material']['specimen'] = private
                outcome = step['attempt']['outcome']
                if outcome['status'] == 'completed':
                    outcome['artifacts'].append(artifact)
                    result = outcome['result']
                    fit = result.get('basis_fit') or result.get('final')
                    if fit:
                        fit['curve'] = artifact
            row['rights'].append(row['rights'][0] | {'artifact': artifact['sha256'], 'role': artifact['role']})
            body = {key: value for key, value in row.items() if key != 'record_id'}
            row['record_id'] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        data = b''.join(json.dumps(row).encode() + b'\n' for row in rows)
        (published / 'episodes.jsonl').write_bytes(data)
        manifest = json.loads((published / 'manifest.json').read_bytes())
        next(entry for entry in manifest['files'] if entry['path'] == 'episodes.jsonl')['sha256'] = hashlib.sha256(data).hexdigest()
        (published / 'manifest.json').write_text(json.dumps(manifest))
        assert exports.validate_dataset(published, exports.dataset_json_schemas()) == []
    before = {p.name: p.read_bytes() for p in published.iterdir() if p.is_file()}
    target = tmp_path / 'migrated'

    assert exports.migrate_dataset(published, target) == []

    assert exports.validate_dataset(target, exports.dataset_json_schemas()) == []
    rows = [json.loads(line) for line in (target / 'episodes.jsonl').read_bytes().splitlines()]
    assert len(rows) == len(before['episodes.jsonl'].splitlines())
    for row in rows:
        assert row['schema_version'] == 'phaseeval.episode.v2'
        assert 'ingest' not in row['root']['case']
        assert not {'original_name', 'rationale'} & set(exports._keys(row))
        encoded = json.dumps(row)
        assert '"channel": "custody_record"' not in encoded
        assert '"channel": "operator_note"' not in encoded
        assert 'custody_withheld' not in encoded
        assert 'expert specimen sentinel' not in encoded
        episode = exports.EpisodeRecord.model_validate(row)
        assert exports.playback_states(episode) == tuple(step.state_after for step in episode.steps)
        body = {key: value for key, value in row.items() if key != 'record_id'}
        assert row['record_id'] == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for name, data in before.items():
        assert (published / name).read_bytes() == data
        if name not in ('episodes.jsonl', 'manifest.json'):
            assert (target / name).read_bytes() == data
    second = tmp_path / 'migrated-again'
    assert exports.migrate_dataset(target, second) == []
    assert (second / 'episodes.jsonl').read_bytes() == (target / 'episodes.jsonl').read_bytes()


def test_migration_refuses_invalid_input_without_publishing(published, tmp_path):
    (published / 'episodes.jsonl').write_text('{broken\n')
    target = tmp_path / 'migrated'
    with pytest.raises(ValueError, match='not JSON'):
        exports.migrate_dataset(published, target)
    assert not target.exists()


def test_migration_preserves_an_existing_target(published, tmp_path):
    target = tmp_path / 'migrated'
    target.mkdir()
    (target / 'keep').write_text('original')
    with pytest.raises(FileExistsError):
        exports.migrate_dataset(published, target)
    assert (target / 'keep').read_text() == 'original'


def test_migration_refuses_a_target_inside_the_source(empty_export, tmp_path):
    source = tmp_path / 'source'
    exports.write_dataset(empty_export[2](), source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}

    with pytest.raises(ValueError, match='outside the source'):
        exports.migrate_dataset(source, source / 'nested')

    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
