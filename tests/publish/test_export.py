import hashlib
import json

import pytest

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench import cli, exports
from phaseeval.workbench.labels import PermissionRequest, Rater, StepFeedback, StepRef
from phaseeval.workbench.model import canonical_bytes


@pytest.mark.parametrize('leak', [r'C:\Users\alice\run', '/root/private/run', '/private/var/run', 'host.local', '2001:db8::1'])
def test_a_refused_export_leaves_no_output_directory(empty_export, tmp_path, monkeypatch, leak):
    store, labels, build = empty_export
    dataset = build()
    dataset = dataset.model_copy(update={'manifest': dataset.manifest.model_copy(update={'exclusions': (
        exports.DatasetExclusion(kind='run', ref='run', reasons=(leak,)),
    )})})
    monkeypatch.setattr(cli, 'open_store', lambda *a: store)
    monkeypatch.setattr(exports, 'build_dataset', lambda *a, **k: dataset)
    out = tmp_path / 'export'
    with pytest.raises(SystemExit, match='leak'):
        cli.main(['export', '--store', str(store.root), '--out', str(out)])
    assert not out.exists()
    assert not list(tmp_path.glob('.export*'))


@pytest.mark.parametrize('release, note, expected', [(True, 'Contact alice@example.org', '[redacted]'), (False, 'private interpretation', '[withheld]')])
def test_free_text_policy(empty_export, tmp_path, release, note, expected):
    store, labels, build = empty_export
    rater = Principal('rater', PrincipalType.USER, frozenset({'rater'}))
    labels.record_permission(rater, PermissionRequest(participation_and_storage=True,
        internal_evaluation_and_calibration=True, public_release=release, training=False, public_identity_disclosure=False))
    record = StepFeedback(step=StepRef(attempt='1' * 64, step_kind='attempt'), mark='note', note=note,
        rater=Rater(principal_id='rater', kind='expert'), exposure='blind', created_at_utc=store._clock())
    directory = store.root / 'step-feedback' / ('1' * 64)
    directory.mkdir(parents=True)
    data = canonical_bytes(record)
    (directory / f'record-{hashlib.sha256(data).hexdigest()}.json').write_bytes(data)
    out = tmp_path / 'export'
    exports.write_dataset(build(), out)
    assert json.loads((out / 'step_feedback.jsonl').read_text())['note'] == expected


def test_existing_output_is_preserved(empty_export, tmp_path):
    out = tmp_path / 'export'
    out.mkdir()
    (out / 'keep').write_text('original')
    with pytest.raises(FileExistsError):
        exports.write_dataset(empty_export[2](), out)
    assert [p.name for p in out.iterdir()] == ['keep']


def test_prose_changes_recompute_an_episodes_record_id(published):
    record = exports.EpisodeRecord.model_validate_json((published / 'episodes.jsonl').read_text().splitlines()[0])
    assert not record.human_contributions
    step = record.steps[0]
    changed = record.model_copy(update={'steps': (step.model_copy(update={
        'attempt': step.attempt.model_copy(update={'rationale': '[withheld]'})
    }), *record.steps[1:])})
    result = exports.pseudonymous_episode(changed, lambda value: value)
    body = result.model_dump(mode='json', exclude={'record_id'})
    assert result.record_id == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert result.record_id != record.record_id
