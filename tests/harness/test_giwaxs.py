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


@pytest.mark.parametrize('handle', ['raw', 'mask'])
@pytest.mark.parametrize('failure', ['timeout', 'invalid_array'])
def test_native_decode_retries_without_actor_label_collision(evidence, handle, failure):
    transform(evidence, 'native_' + handle)
    execute = evidence.execute
    attempts = []

    def fail_once(argv):
        request = json.loads(Path(argv[-1]).read_text())
        target = Path(request['output'])
        attempts.append(target.parent)
        if len(attempts) == 1:
            if failure == 'timeout':
                raise RuntimeError('worker timed out')
            np.savez_compressed(target, intensity=np.arange(3))
            return json.dumps({'runtime': EXPECTED_RUNTIME,
                               'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        return execute(argv)

    evidence.execute = fail_once
    with pytest.raises(RuntimeError, match='timed out|invalid detector array'):
        evidence.read(handle=handle)
    assert handle not in evidence.native
    first = evidence.read(handle=handle, limit=1)
    assert first['rows'] == [[0, 0, 0]]
    assert evidence.read(handle=handle, limit=1) == first
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    assert all(directory.is_dir() for directory in attempts)


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


@pytest.mark.parametrize('mode,expected_intensity,expected_usable', [
    ('coverage_aware', [2.5, -2., 3.], [True, True, True]),
    ('notebook', [2.5, -1., 3.], [True, False, True]),
])
def test_qz_profile_axes_coverage_and_s_coordinates(evidence, mode, expected_intensity, expected_usable):
    parent = transform(evidence)
    q = extract(evidence, parent, 'qz', mode=mode, axis='qz', lower=1., upper=3.)
    s = extract(evidence, parent, 's', mode=mode, axis='qz', lower=1., upper=3., coordinate_definition='s=1/d')
    q_rows, s_rows = (np.array(evidence.read(handle=handle)['rows']) for handle in (q, s))
    np.testing.assert_array_equal(q_rows[:, 0], [0., 1., 2.])
    np.testing.assert_array_equal(q_rows[:, 1], expected_intensity)
    np.testing.assert_array_equal(q_rows[:, 2], [2, 1, 2])
    np.testing.assert_array_equal(q_rows[:, 3], expected_usable)
    np.testing.assert_allclose(s_rows[:, 0], q_rows[:, 0] / (2*np.pi))
    np.testing.assert_array_equal(s_rows[:, 1:], q_rows[:, 1:])
    for handle in (q, s):
        parameters = evidence.artifacts[handle]['parameters']
        assert parameters['requested_roi'] == [1., 3.]
        assert parameters['slice'] == [1, 3]
        assert parameters['realized_roi'] == [1., 2.]


@pytest.mark.parametrize('kwargs', [{'lower': -1.}, {'upper': 5.}, {'lower': 0.1, 'upper': 0.2}, {'lower': 2., 'upper': 1.}])
def test_invalid_rois_rejected(evidence, kwargs):
    parent = transform(evidence)
    with pytest.raises(ValueError):
        extract(evidence, parent, **kwargs)


def test_coordinate_mismatch_and_same_profile_comparison(evidence):
    parent = transform(evidence)
    q = extract(evidence, parent, 'q')
    s = extract(evidence, parent, 's', coordinate_definition='s=1/d')
    q_rows, s_rows = (np.array(evidence.read(handle=handle)['rows']) for handle in (q, s))
    np.testing.assert_allclose(s_rows[:, 0], q_rows[:, 0] / (2*np.pi))
    np.testing.assert_array_equal(s_rows[:, 1:], q_rows[:, 1:])
    with pytest.raises(ValueError, match='coordinate'):
        evidence.call('compare', {'label': 'bad', 'left': q, 'right': s})
    same = evidence.call('compare', {'label': 'same', 'left': q, 'right': q})
    assert json.loads(evidence.read(handle=same['handle'])['text'])['relative_l2_change'] == 0


def test_comparison_never_interpolates_missing_gaps():
    from harness.giwaxs.environment import compare_profiles
    left = {'x': np.arange(9)/2, 'intensity': np.ones(9), 'usable': np.ones(9, bool)}
    right = {'x': np.arange(5), 'intensity': np.ones(5), 'usable': np.array([1, 1, 0, 1, 1], bool)}
    assert compare_profiles(left, right)['common_left_indexes'] == [0, 1, 2, 6, 7, 8]


def test_comparison_zero_norm_and_no_common_support():
    from harness.giwaxs.environment import compare_profiles
    left = {'x': np.arange(3), 'intensity': np.zeros(3), 'usable': np.ones(3, bool)}
    right = left | {'intensity': np.ones(3)}
    result = compare_profiles(left, right)
    assert result['relative_l2_change'] is None
    assert result['max_absolute_change'] == 1.
    assert result['scientific_reward'] is None
    with pytest.raises(ValueError, match='no common covered support'):
        compare_profiles(left, right | {'usable': np.zeros(3, bool)})


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
    with pytest.raises(ValueError, match='unknown evidence handle') as rejected:
        evidence.call('submit', request)
    assert str(rejected.value) == "unknown evidence handle: 'profile:other_episode'"
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


def test_context_is_registered_citable_and_manifest_bound(evidence):
    page = evidence.read(handle='context')
    submitted = evidence.call('submit', {'status': 'partial',
        'claims': [{'text': 'Acquisition context is supplied, not validated', 'evidence': ['context']}],
        'limitations': ['Synthetic mechanics'], 'evidence_needed': ['Calibration']})
    assert submitted['citations']['context'] == page['sha256'] == evidence.files['context'][3]
    registry = json.loads((evidence.trial / 'giwaxs-evidence.json').read_text())
    assert registry['files']['context']['sha256'] == page['sha256']
    (evidence.inputs / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        evidence._bytes('context')


def test_context_file_tampering_rejected(evidence):
    evidence.read(handle='context')
    root, name, _, _ = evidence.files['context']
    (root / name).write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        evidence.read(handle='context')


def test_profile_extent_and_support_are_separate_from_orthogonal_roi(evidence):
    parent = transform(evidence)
    profile = evidence.call('extract', {'label': 's', 'map_handle': parent, 'axis': 'qr',
        'lower': 0., 'upper': 2., 'mode': 'coverage_aware', 'coordinate_definition': 's=1/d'})
    parameters = profile['parameters']
    assert parameters['roi_axis'] == 'qz'
    assert parameters['requested_roi'] == [0., 2.]
    assert parameters['roi_coordinate_definition'] == 'q=2pi/d'
    np.testing.assert_allclose(parameters['output_axis_extent'], [0., 3/(2*np.pi)])
    summary = profile['summary']
    assert summary == evidence.read(handle=profile['handle'])['summary']
    assert summary['usable_bins'] == 3 and summary['unusable_bins'] == 1
    assert summary['usable_support']['contiguous_runs'] == 1
    np.testing.assert_allclose(summary['usable_support']['bounding_extent'], [0., 2/(2*np.pi)])


def test_numeric_selection_paginated_in_native_indexes(evidence):
    parent = transform(evidence)
    selection = {'kind': 'map', 'qr_min': 1., 'qr_max': 3., 'qz_min': 0., 'qz_max': 1.}
    first = evidence.read(handle=parent, selection=selection, limit=2)
    second = evidence.read(handle=parent, selection=selection, start=first['next_start'], limit=2)
    third = evidence.read(handle=parent, selection=selection, start=second['next_start'], limit=2)
    assert first['total_rows'] == 6 and first['native_total_rows'] == 12
    assert first['row_indexes'] + second['row_indexes'] + third['row_indexes'] == [1, 2, 3, 5, 6, 7]
    assert third['next_start'] is None
    assert first['rows'][0][2:] == [-4., 1., True]
    assert second['rows'][0][2:] == [0., 0., False]
    assert first['summary']['covered_zero_bins'] == 1
    cut = extract(evidence, parent, coordinate_definition='s=1/d')
    page = evidence.read(handle=cut, selection={'kind': 'profile', 'lower': 1/(2*np.pi), 'upper': 2/(2*np.pi)})
    assert page['row_indexes'] == [1, 2]
    page = evidence.read(handle='raw', selection={'kind': 'detector', 'row_min': 1, 'row_max': 2,
                                                'column_min': 10, 'column_max': 11}, limit=3)
    assert page['row_indexes'] == [22, 23, 34] and page['next_start'] == 3
    assert page['summary']['shape'] == [10, 12]
    mask = evidence.read(handle='mask', limit=1)['summary']
    assert mask['masked_pixels'] == 119 and mask['unmasked_pixels'] == 1


@pytest.mark.parametrize('handle,selection', [
    ('context', {'kind': 'profile', 'lower': 0., 'upper': 1.}),
    ('raw', {'kind': 'map', 'qr_min': 0., 'qr_max': 1., 'qz_min': 0., 'qz_max': 1.}),
    ('raw', {'kind': 'detector', 'row_min': True, 'row_max': 1, 'column_min': 0, 'column_max': 1}),
    ('raw', {'kind': 'detector', 'row_min': 0., 'row_max': 1, 'column_min': 0, 'column_max': 1}),
    ('raw', {'kind': 'detector', 'row_min': -1, 'row_max': 1, 'column_min': 0, 'column_max': 1}),
    ('raw', {'kind': 'detector', 'row_min': 1, 'row_max': 0, 'column_min': 0, 'column_max': 1}),
    ('raw', {'kind': 'detector', 'row_min': 0, 'row_max': 10, 'column_min': 0, 'column_max': 1}),
    ('raw', {'kind': 'detector', 'row_min': 0, 'row_max': 1, 'column_min': 0, 'column_max': 1, 'extra': 1}),
])
def test_targeted_selection_rejects_wrong_kinds_and_bounds(evidence, handle, selection):
    with pytest.raises(ValueError):
        evidence.read(handle=handle, selection=selection)


def test_comparison_summary_precedes_large_index_payload_and_counts_gaps(evidence):
    from harness.giwaxs.environment import compare_profiles
    parent = transform(evidence)
    cut = extract(evidence, parent)
    comparison = evidence.call('compare', {'label': 'same', 'left': cut, 'right': cut})
    assert comparison['summary']['relative_l2_change'] == 0
    assert 'common_left_indexes' not in comparison['summary']
    assert evidence.read(handle=comparison['handle'], limit=1)['summary'] == comparison['summary']
    left = {'x': np.arange(3000.), 'intensity': np.ones(3000), 'usable': np.ones(3000, bool)}
    left['usable'][1000] = False
    value = compare_profiles(left, left)
    directory = evidence._allocate('long')
    long = evidence._register(directory, 'comparison', {}, evidence.artifacts[cut]['coordinate'],
                              {cut: evidence.files[cut][3]}, [], value)
    page = evidence.read(handle=long['handle'], limit=1)
    assert page['summary']['common_bins'] == 2999
    assert page['summary']['contiguous_runs'] == 2
    assert page['summary']['common_interval_is_contiguous'] is False
    assert page['next_start'] == 1
    assert len(evidence._bytes(long['handle'])) > 10000


@pytest.mark.parametrize('registered', [False, True])
def test_context_reconstruction_is_read_only_and_missing_files_are_not_recreated(evidence, registered):
    if registered:
        evidence.read(handle='context')
    before = {p: p.read_bytes() for p in evidence.trial.rglob('*') if p.is_file()}
    rebuilt = GiwaxsEvidence(evidence.inputs, 'synthetic', evidence.output, evidence.trial)
    assert {p: p.read_bytes() for p in evidence.trial.rglob('*') if p.is_file()} == before
    if registered:
        rebuilt.files.update(evidence.files)
        assert rebuilt._bytes('context') == evidence._bytes('context')
        root, name, _, _ = evidence.files['context']
        (root / name).unlink()
        with pytest.raises(ValueError, match='missing'):
            rebuilt.read(handle='context')
        assert not (root / name).exists()
    else:
        with pytest.raises(ValueError, match='unknown'):
            rebuilt._bytes('context')
        assert not evidence.trial.exists()


@pytest.mark.parametrize('selection', [
    {'kind': 'profile', 'lower': -1., 'upper': 1.},
    {'kind': 'profile', 'lower': 0., 'upper': 4.},
    {'kind': 'profile', 'lower': 2., 'upper': 1.},
    {'kind': 'profile', 'lower': 0.1, 'upper': 0.2},
    {'kind': 'profile', 'lower': True, 'upper': 2.},
    {'kind': 'profile', 'lower': '0', 'upper': 2.},
    {'kind': 'profile', 'lower': float('nan'), 'upper': 2.},
    {'kind': 'profile', 'lower': 0., 'upper': float('inf')},
    {'kind': 'profile', 'lower': 0.},
])
def test_profile_selection_rejects_outside_empty_and_nonfinite(evidence, selection):
    cut = extract(evidence, transform(evidence))
    with pytest.raises(ValueError):
        evidence.read(handle=cut, selection=selection)


def test_selection_single_coordinate_and_exhausted_page(evidence):
    cut = extract(evidence, transform(evidence))
    selection = {'kind': 'profile', 'lower': 3., 'upper': 3.}
    assert evidence.read(handle=cut, selection=selection)['rows'] == [[3., 0., 0, False]]
    page = evidence.read(handle=cut, selection=selection, start=1)
    assert page['rows'] == [] and page['next_start'] is None and page['total_rows'] == 1
    with pytest.raises(ValueError, match='start'):
        evidence.read(handle=cut, selection=selection, start=2)
    with pytest.raises(ValueError, match='limit'):
        evidence.read(handle=cut, selection=selection, limit=101)


def test_profile_summary_does_not_bridge_gaps_or_treat_uncovered_as_zero():
    from harness.giwaxs.environment import numeric_summary
    profile = {'x': np.arange(5.), 'intensity': np.array([0., -2., 0., 9., 0.]),
               'covered_count': np.array([1, 1, 0, 1, 0]), 'usable': np.array([True, True, False, True, False])}
    summary = numeric_summary(profile, 'profile')
    assert summary['usable_support']['bounding_extent'] == [0., 3.]
    assert summary['usable_support']['contiguous_runs'] == 2
    assert summary['usable_bins'] == 3 and summary['unusable_bins'] == 2
    assert summary['usable_intensity_extent'] == [-2., 9.]
    profile['usable'][:] = False
    summary = numeric_summary(profile, 'profile')
    assert summary['usable_support']['bounding_extent'] is None
    assert summary['usable_support']['contiguous_runs'] == 0
    assert summary['usable_intensity_extent'] is None


def test_context_registered_digest_cannot_substitute_for_manifest_case(evidence):
    evidence.read(handle='context')
    root, name, kind, _ = evidence.files['context']
    (root / name).write_bytes(b'{}')
    evidence.files['context'] = (root, name, kind, hashlib.sha256(b'{}').hexdigest())
    with pytest.raises(ValueError, match='manifest case'):
        evidence._bytes('context')


@pytest.mark.parametrize('change', [
    {'qr_min': -1.}, {'qr_max': 4.}, {'qz_min': -1.}, {'qz_max': 3.},
    {'qr_min': 2., 'qr_max': 1.}, {'qz_min': 2., 'qz_max': 1.},
    {'qr_min': 0.1, 'qr_max': 0.2}, {'qz_min': 0.1, 'qz_max': 0.2},
])
def test_map_selection_checks_both_axes(evidence, change):
    parent = transform(evidence)
    selection = {'kind': 'map', 'qr_min': 0., 'qr_max': 3., 'qz_min': 0., 'qz_max': 2.} | change
    with pytest.raises(ValueError):
        evidence.read(handle=parent, selection=selection)
