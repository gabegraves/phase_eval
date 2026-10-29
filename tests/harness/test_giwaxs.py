"""GIWAXS mechanics only: synthetic arrays are not scientific reference labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from harness.giwaxs.environment import GiwaxsEvidence, EXPECTED_RUNTIME, probe_runtime


def test_runtime_probe_checks_actual_pins():
    assert probe_runtime('/python', lambda argv: json.dumps(EXPECTED_RUNTIME)) == EXPECTED_RUNTIME
    with pytest.raises(ValueError, match='runtime'):
        probe_runtime('/python', lambda argv: json.dumps(EXPECTED_RUNTIME | {'python': '3.12.0'}))


@pytest.fixture
def evidence(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    files = {}
    for role, filename in [('raw', 'raw.tif'), ('geometry', 'geometry.poni'), ('mask', 'mask.edf')]:
        (inputs / filename).write_text('synthetic ' + role)
        files[role] = {'file': filename, 'sha256': hashlib.sha256((inputs / filename).read_bytes()).hexdigest()}
    case = {'case_id': 'synthetic', **files, 'source': {'doi': '10.example/test', 'license': 'test',
            'attribution': 'Synthetic mechanics test', 'archive_sha256': 'a'*64},
            'context': {'sample': 'synthetic'}, 'unknowns': ['scientific validity'],
            'acquisition': {'incident_angle': 0.1, 'sample_orientation': 3, 'tilt_angle': 0.0}, 'development_only': True}
    (inputs / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'task_kind': 'giwaxs', 'cases': [case]}))
    def execute(argv):
        request = json.loads(Path(argv[-1]).read_text())
        target = Path(request['output'])
        if request['operation'] == 'decode':
            np.savez_compressed(target, intensity=np.arange(120).reshape(10, 12))
        else:
            np.savez_compressed(target, intensity=np.array([[0., -4., 9., 0.], [2., -2., 0., 0.], [3., 3., 3., 3.]]),
                                count=np.array([[1., 1., 1., 0.], [1., 1., 0., 0.], [1., 1., 1., 1.]]),
                                qr=np.array([0., 1., 2., 3.]), qz=np.array([0., 1., 2.]))
        return json.dumps({'runtime': EXPECTED_RUNTIME, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                           'warnings': [], 'executed_geometry': {'distance_m': 1.}})
    return GiwaxsEvidence(inputs, 'synthetic', tmp_path / 'trial/output', tmp_path / 'trial',
                         python='/synthetic/python', execute=execute, runtime=EXPECTED_RUNTIME)


def transform(evidence, label='map1'):
    return evidence.call('transform', {'label': label, 'mask_policy': 'supplied', 'incident_angle': 0.1,
                                      'sample_orientation': 3, 'tilt_angle': 0.0})['handle']


def extract(evidence, parent, label='cut', mode='coverage_aware', **kw):
    return evidence.call('extract', {'label': label, 'map_handle': parent, 'lower': 0., 'upper': 2.,
                                    'mode': mode, **kw})['handle']


def test_native_pages_exact_indexes_and_text(evidence):
    first = evidence.read(handle='raw', limit=100)
    second = evidence.read(handle='raw', start=first['next_start'])
    assert first['shape'] == [10, 12]
    assert first['row_indexes'] + second['row_indexes'] == list(range(120))
    assert [row[2] for row in first['rows'] + second['rows']] == list(range(120))
    assert evidence.read(handle='geometry')['text'] == 'synthetic geometry'
    assert not evidence.reduction_bound
    with pytest.raises(ValueError):
        evidence.read(handle='raw', limit=101)


def test_measured_nonpositive_values_coverage_and_notebook_fill(evidence):
    parent = transform(evidence)
    cut = extract(evidence, parent)
    rows = evidence.read(handle=cut)['rows']
    assert rows == [[0., 1., 2, True], [1., -3., 2, True], [2., 9., 1, True], [3., 0., 0, False]]
    notebook = extract(evidence, parent, label='notebook', mode='notebook')
    rows = evidence.read(handle=notebook)['rows']
    assert rows[2] == [2., 4.5, 1, False]
    assert evidence.artifacts[notebook]['warnings']
    comparison = evidence.call('compare', {'label': 'compare', 'left': cut, 'right': notebook})
    assert json.loads(evidence.read(handle=comparison['handle'])['text'])['common_bins'] == 2
    assert evidence.reduction_bound


@pytest.mark.parametrize('kwargs', [{'lower': -1.}, {'upper': 5.}, {'lower': 0.1, 'upper': 0.2}, {'lower': 2., 'upper': 1.}])
def test_invalid_rois_rejected(evidence, kwargs):
    parent = transform(evidence)
    with pytest.raises(ValueError):
        extract(evidence, parent, **kwargs)


def test_coordinate_mismatch_and_same_profile_comparison(evidence):
    parent = transform(evidence)
    q = extract(evidence, parent, 'q')
    s = extract(evidence, parent, 's', coordinate_definition='s=1/d')
    with pytest.raises(ValueError, match='coordinate'):
        evidence.call('compare', {'label': 'bad', 'left': q, 'right': s})
    same = evidence.call('compare', {'label': 'same', 'left': q, 'right': q})
    assert json.loads(evidence.read(handle=same['handle'])['text'])['relative_l2_change'] == 0


def test_comparison_never_interpolates_missing_gaps():
    from harness.giwaxs.environment import compare_profiles
    left = {'x': np.arange(9)/2, 'intensity': np.ones(9), 'usable': np.ones(9, bool)}
    right = {'x': np.arange(5), 'intensity': np.ones(5), 'usable': np.array([1, 1, 0, 1, 1], bool)}
    assert compare_profiles(left, right)['common_left_indexes'] == [0, 1, 2, 6, 7, 8]


@pytest.mark.parametrize('label', ['../escape', '/absolute', '.hidden', '', 'x'*65])
def test_unsafe_labels_rejected(evidence, label):
    with pytest.raises(ValueError):
        transform(evidence, label)


def test_strict_arguments_and_partial_submission(evidence):
    for args in ({'start': True}, {'limit': 1.5}, {'path': 'raw.tif'}, {'start': float('inf')}):
        with pytest.raises(ValueError):
            evidence.call('read_evidence', args)
    partial = evidence.call('submit', {'status': 'partial', 'claims': [], 'limitations': ['No valid reduction'],
                                       'evidence_needed': ['Acquisition calibration uncertainty']})
    assert partial['scientific_reward'] is None and partial['scientific_qualified'] is False
    assert evidence.submission == json.loads((evidence.output / 'analysis.json').read_text())
    assert not evidence.reduction_bound


def test_failed_worker_never_registers_artifact_and_label_consumed(evidence):
    def failed(argv):
        raise RuntimeError('worker timed out')
    evidence.execute = failed
    with pytest.raises(RuntimeError, match='timed out'):
        transform(evidence)
    assert not evidence.artifacts and not evidence.reduction_bound
    with pytest.raises(ValueError, match='already used'):
        transform(evidence)


def test_parent_tampering_blocks_reads_extraction_and_citation(evidence):
    parent = transform(evidence)
    cut = extract(evidence, parent)
    root, name, _, _ = evidence.files[parent]
    (root / name).write_bytes(b'tampered')
    for operation in (lambda: evidence.read(handle=cut), lambda: extract(evidence, parent, 'again'),
                      lambda: evidence.call('submit', {'status': 'supported', 'claims': [{'text': 'claim', 'evidence': [cut]}],
                                                       'limitations': ['synthetic'], 'evidence_needed': ['truth']})):
        with pytest.raises(ValueError, match='digest'):
            operation()


@pytest.mark.parametrize('replacement', ['changed', 'symlink'])
def test_input_hash_or_symlink_substitution(evidence, replacement, tmp_path):
    raw = evidence.inputs / evidence.case['raw']['file']
    if replacement == 'changed':
        raw.write_text('changed')
    else:
        sibling = tmp_path / 'other.tif'
        sibling.write_bytes(raw.read_bytes())
        raw.unlink()
        raw.symlink_to(sibling)
    with pytest.raises(ValueError):
        evidence.read(handle='context')


def test_owned_citations_only_and_recorded_hashes(evidence):
    parent = transform(evidence)
    cut = extract(evidence, parent)
    request = {'status': 'supported', 'claims': [{'text': 'Bounded mechanics claim', 'evidence': ['profile:other_episode']}],
               'limitations': ['No scientific assessment'], 'evidence_needed': ['Expert review']}
    with pytest.raises(ValueError, match='unknown'):
        evidence.call('submit', request)
    request['claims'][0]['evidence'] = [cut]
    submitted = evidence.call('submit', request)
    assert submitted['citations'][cut] == evidence.files[cut][3]
    assert (evidence.trial / 'giwaxs-evidence.json').is_file()


def test_schema_type_parity(evidence):
    from phaseeval.giwaxs import GiwaxsManifest, ArtifactReceipt, Submission
    import jsonschema
    repo = Path(__file__).resolve().parents[2]
    parent = transform(evidence)
    for name, model, sample in [('giwaxs-case.v1', GiwaxsManifest, json.loads((evidence.inputs / 'manifest.json').read_text())),
                              ('giwaxs-artifact.v1', ArtifactReceipt, evidence.artifacts[parent])]:
        schema = json.loads((repo / 'schemas' / (name + '.json')).read_text())
        assert schema == model.model_json_schema()
        jsonschema.validate(sample, schema)


def test_failed_domain_action_label_cannot_be_reused(evidence):
    parent = transform(evidence)
    with pytest.raises(ValueError, match='outside'):
        extract(evidence, parent, lower=-1.)
    with pytest.raises(ValueError, match='already used'):
        extract(evidence, parent)


def test_receipt_reads_also_validate_artifact_and_ancestry(evidence):
    parent = transform(evidence)
    cut = extract(evidence, parent)
    root, name, _, _ = evidence.files[parent]
    (root / name).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='digest'):
        evidence.read(handle=cut + ':receipt')


def test_symlink_output_never_creates_external_directory(evidence, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    evidence.output.parent.mkdir()
    evidence.output.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        transform(evidence)
    assert not list(outside.iterdir())


def test_published_contracts_and_submission_schema(evidence):
    from phaseeval.giwaxs import TOOLS, SubmissionReceipt, EvidenceRegistry, GiwaxsManifest
    import jsonschema
    repo = Path(__file__).resolve().parents[2]
    result = evidence.call('submit', {'status': 'partial', 'claims': [], 'limitations': ['No reduction'], 'evidence_needed': ['Calibration']})
    registry = json.loads((evidence.trial / 'giwaxs-evidence.json').read_text())
    for name, model, sample in [('giwaxs-submission.v1', SubmissionReceipt, result), ('giwaxs-evidence.v1', EvidenceRegistry, registry)]:
        schema = json.loads((repo / 'schemas' / (name + '.json')).read_text())
        assert schema == model.model_json_schema()
        jsonschema.validate(sample, schema)
    tools_schema = json.loads((repo / 'schemas/giwaxs-tools.v1.json').read_text())
    assert tools_schema == {'type': 'array', 'const': TOOLS}
    jsonschema.Draft202012Validator.check_schema(tools_schema)
    jsonschema.validate(TOOLS, tools_schema)
    manifest = json.loads((evidence.inputs / 'manifest.json').read_text())
    with pytest.raises(ValueError, match='integer'):
        GiwaxsManifest.model_validate(manifest | {'schema_version': True})
    with pytest.raises(ValueError, match='literal'):
        SubmissionReceipt.model_validate(result | {'scientific_qualified': 0})


def test_clipped_output_registered_for_native_text_recovery(evidence):
    evidence.trial.mkdir()
    raw = ('full ' * 600).encode()
    (evidence.trial / 'tool-output.txt').write_bytes(raw)
    evidence.register_output({'evidence_handle': 'tool:one', 'executor_output': 'tool-output.txt',
                              'executor_sha256': hashlib.sha256(raw).hexdigest()})
    assert evidence.read(handle='tool:one')['text'] == raw.decode()[:2000]


def test_failed_registry_write_does_not_publish_artifact(evidence, monkeypatch):
    def failed():
        raise OSError('disk full')
    monkeypatch.setattr(evidence, '_save_registry', failed)
    with pytest.raises(OSError, match='disk full'):
        transform(evidence)
    assert not evidence.artifacts and not evidence.reduction_bound
    assert set(evidence.files) == {'raw', 'geometry', 'mask'}
