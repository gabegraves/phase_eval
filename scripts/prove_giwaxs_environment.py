"""Real GIWAXS physics through the runner with scripted replies; no scientific reward claim."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
import numpy as np
from harness.agent.episode import atomic_json, digest, strict_json
from harness.agent.evidence import checked_bytes
from harness.agent.run_agent import Runner, comparison_fingerprint, git_commit, now
from harness.giwaxs.environment import arrays
from harness.giwaxs.review import freeze_packet, validate_annotation
from phaseeval.giwaxs import GiwaxsManifest
from phaseeval.reward_profile import profile_group

ARCHIVE_SHA256 = 'fcc15f7f3a7d2b2824435975dbcbdabc1f16a29ff1755264857c32a9df42079f'
CASE = 'synchrotron_s16'
GEOMETRY = {'incident_angle': 0.07249400217733637, 'sample_orientation': 3, 'tilt_angle': 0.0}
LIMITS = ['One exposed development specimen; no independent scientific reference labels.',
          'Supplied geometry has no established scan-specific calibration uncertainty.',
          'No scientific evaluator qualification, live-model usability or RL learning was tested.']
NEEDED = ['Scan-linked calibration and correction history.',
          'Qualified scientific method/outcome judgments and independent specimens.']


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def response(*calls):
    return {'choices': [{'message': {'role': 'assistant', 'content': 'SCRIPTED development proof; no model inference.',
        'tool_calls': [{'id': f'call-{i}', 'function': {'name': name, 'arguments': json.dumps(arguments)}}
                       for i, (name, arguments) in enumerate(calls)]}}]}


def build_case(archive, inputs):
    inputs.mkdir()
    refs = {}
    with zipfile.ZipFile(archive) as bundle:
        for kind, name in (('raw', 's16_00401-00029.tif'), ('geometry', 'lab6_402.poni'), ('mask', 'pe_mask_1.edf')):
            raw = bundle.read('GIWAXS/synchrotron/' + name)
            (inputs / name).write_bytes(raw)
            refs[kind] = {'file': name, 'sha256': digest(raw)}
    manifest = GiwaxsManifest.model_validate({'schema_version': 1, 'task_kind': 'giwaxs', 'cases': [{
        'case_id': CASE, **refs, 'source': {'doi': '10.5281/zenodo.18484796', 'license': 'CC-BY-4.0',
            'attribution': 'Wu, Kraus and colleagues, Electron Microscopy and GIWAXS Raw Data and Code',
            'archive_sha256': ARCHIVE_SHA256},
        'context': {'source_scope': 'Deposited synchrotron notebook and publication-level context; not joined LIMS history.',
            'paper_doi': '10.1038/s41467-026-70690-y', 'sample': 'DRCN5T:PC71BM film; reported blend mass ratio 1:0.8.',
            'instrument': 'Reported P08/DESY synchrotron acquisition, 25 keV, 10 s exposure.',
            'notebook': 'The deposited 2D transform omits the mask; its later sector integration applies it.',
            'coordinates': 'Map q=2pi/d in nm^-1; published plots also use s=q/(2pi)=1/d.'},
        'unknowns': ['Complete sample/measurement linkage and experimental history.',
            'Scan-specific calibration uncertainty, dark/flat corrections and full detector history.',
            'Independent phase/indexing labels and qualified scientific interpretation.'],
        'acquisition': GEOMETRY, 'development_only': True}]}).model_dump()
    atomic_json(inputs / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--giwaxs-python', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='prior notebook NPZ with intensity, qr, qz and ip')
    parser.add_argument('--baseline-sha256', required=True, help='digest recorded by the prior notebook run')
    parser.add_argument('--out', type=Path, required=True, help='new external proof directory')
    args = parser.parse_args()
    script_sha256 = digest(Path(__file__).read_bytes())
    archive = args.archive.absolute()
    baseline = args.baseline.absolute()
    checked_bytes(archive.parent, archive.name, ARCHIVE_SHA256)
    baseline_bytes = checked_bytes(baseline.parent, baseline.name, args.baseline_sha256)
    reference = arrays(baseline_bytes)
    require({'intensity', 'qr', 'qz', 'ip'} <= reference.keys(), 'baseline lacks required notebook arrays')
    out = args.out.absolute()
    require(not any(p.is_symlink() for p in (out, *out.parents)), 'symlink proof output is not allowed')
    out = out.resolve()
    require(not out.is_relative_to(REPO), 'keep generated proof data outside the repository')
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    build_case(archive, inputs)
    common = dict(input_dir=inputs, case=CASE, task_kind='giwaxs', giwaxs_python=args.giwaxs_python.absolute(),
        base_url='https://scripted.invalid/v1', model='SCRIPTED-no-model', endpoint_label='SCRIPTED-real-GIWAXS',
        api_key_env='PHASEEVAL_UNUSED_PROOF_KEY', max_turns=30, time_limit=900,
        decoding={}, model_revision='script-sha256:' + script_sha256, out=out / 'trials' / 'main')
    fingerprint = comparison_fingerprint(argparse.Namespace(**common))
    group = out / 'group.json'
    atomic_json(group, {'schema_version': 1, 'group_id': 'giwaxs-development-proof', 'created_at': now(),
        'comparison_fingerprint': fingerprint, 'episode_ids': ['main', 'partial', 'failed', 'missing']})

    def runner(slot, input_dir=inputs):
        return Runner(argparse.Namespace(**(common | {'input_dir': input_dir, 'out': out / 'trials' / slot,
                                                      'group_manifest': group, 'episode_id': slot})))

    transform = GEOMETRY | {'distance_scale': 1.0, 'mask_policy': 'omit'}
    extract = {'map_handle': 'map:map_notebook', 'axis': 'qr', 'lower': 0.1, 'upper': 0.5,
               'mode': 'notebook', 'coordinate_definition': 'q=2pi/d'}
    compare = {'left': 'profile:notebook', 'lower': float(2*np.pi*0.3), 'upper': float(2*np.pi*3)}
    calls = [
        ('read_evidence', {}), ('read_evidence', {'handle': 'context'}),
        ('read_evidence', {'handle': 'geometry'}), ('read_evidence', {'handle': 'raw', 'start': 2049000, 'limit': 12}),
        ('read_evidence', {'handle': 'mask', 'start': 2049000, 'limit': 12}),
        ('transform', transform | {'label': 'map_notebook'}),
        ('read_evidence', {'handle': 'map:map_notebook', 'start': 326168, 'limit': 12}),
        ('extract', extract | {'label': 'notebook'}),
        ('read_evidence', {'handle': 'profile:notebook', 'start': 550, 'limit': 12}),
        ('transform', transform | {'label': 'map_masked', 'mask_policy': 'supplied'}),
        ('extract', extract | {'label': 'masked', 'map_handle': 'map:map_masked'}),
        ('extract', extract | {'label': 'covered', 'map_handle': 'map:map_masked', 'mode': 'coverage_aware'}),
        ('compare', compare | {'label': 'mask_effect', 'right': 'profile:masked'}),
        ('extract', extract | {'label': 's_coordinate', 'coordinate_definition': 's=1/d'}),
        ('compare', compare | {'label': 'incompatible', 'right': 'profile:s_coordinate'}),
        ('extract', extract | {'label': 'empty_window', 'lower': 0.2, 'upper': 0.200001}),
        ('extract', extract | {'label': 'narrow', 'lower': 0.2, 'upper': 0.4}),
        ('compare', compare | {'label': 'window_effect', 'right': 'profile:narrow'}),
        ('read_evidence', {'handle': '../unregistered'}),
        ('transform', transform | {'label': 'map_repeat'}),
        ('extract', extract | {'label': 'repeat', 'map_handle': 'map:map_repeat'}),
        ('compare', compare | {'label': 'repeat_effect', 'right': 'profile:repeat'}),
        ('submit', {'status': 'partial', 'claims': [{'text': 'Reciprocal maps and declared profiles were computed; mask and window comparisons are recorded. Structural interpretation remains unresolved.',
            'evidence': ['map:map_notebook', 'profile:notebook', 'comparison:mask_effect', 'comparison:window_effect']}],
            'limitations': LIMITS, 'evidence_needed': NEEDED})]
    main_run = runner('main')
    # ponytail: only provider replies are scripted; all evidence, physics and accounting use production paths.
    replies = iter([response(call) for call in calls[:-1]] + [response(calls[-1],
        ('transform', transform | {'label': 'after_submit'}))])
    main_run.chat = lambda _messages: next(replies)
    with redirect_stdout(sys.stderr):
        require(main_run.run() == 0, 'main runner failed; inspect its durable receipt')
    expected = ['rejected' if i in (14, 15, 18) else 'success' for i in range(len(calls))]
    require([a['outcome'] for a in main_run.receipt['actions']] == expected, 'unexpected action outcome or missing action')
    require(main_run.receipt['workbench_import'] == 'reduction_bound', 'real reduction was not bound')
    require(not (main_run.output / 'after_submit').exists(), 'submission was not terminal')
    trace = strict_json((main_run.out / 'agent/trajectory.json').read_bytes())
    observations = [r for step in trace['steps'] for r in step.get('observation', {}).get('results', [])]
    require(len(observations) == len(calls), 'observations were lost')
    pages = [strict_json(observations[i]['content']) for i in (3, 4, 6, 8)]
    require(all(len(p['rows']) == 12 and p['next_start'] is not None for p in pages), 'native evidence paging failed')

    def artifact(handle):
        return main_run.evidence._bytes(handle)

    mapped, profile = arrays(artifact('map:map_notebook')), arrays(artifact('profile:notebook'))
    require(mapped['intensity'].shape == (2048, 2048), 'unexpected real map shape')
    for axis in ('qr', 'qz'):
        np.testing.assert_array_equal(mapped[axis], reference[axis])
    np.testing.assert_array_equal(profile['x'], reference['qr'])
    np.testing.assert_allclose(mapped['intensity'], reference['intensity'], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(profile['intensity'], reference['ip'], rtol=1e-6, atol=1e-5)
    repeated = arrays(artifact('map:map_repeat'))
    for key in mapped:
        np.testing.assert_array_equal(mapped[key], repeated[key])
    np.testing.assert_array_equal(profile['intensity'], arrays(artifact('profile:repeat'))['intensity'])
    sensitivities = {name: strict_json(artifact('comparison:' + name)) for name in ('mask_effect', 'window_effect', 'repeat_effect')}

    partial = runner('partial')
    partial.chat = lambda _messages: response(('submit', {'status': 'partial', 'claims': [],
        'limitations': ['No reduction was attempted; structural interpretation remains unresolved.', *LIMITS], 'evidence_needed': NEEDED}))
    with redirect_stdout(sys.stderr):
        require(partial.run() == 0, 'no-reduction partial failed')
    require(partial.receipt['workbench_import'] == 'unsupported_no_reduction', 'partial acquired a false reduction binding')
    broken = out / 'tampered-inputs'
    shutil.copytree(inputs, broken)
    with (broken / 's16_00401-00029.tif').open('ab') as stream:
        stream.write(b'TAMPERED development proof')
    failed = runner('failed', broken)
    failed.chat = lambda _messages: (_ for _ in ()).throw(RuntimeError('provider must not run after failed setup'))
    with redirect_stdout(sys.stderr):
        require(failed.run() == 1 and failed.receipt['ending']['kind'] == 'setup_failure', 'tampered input was not refused')
    require(not failed.receipt['actions'], 'failed setup executed an action')
    runs = (main_run, partial, failed)
    packets = {r.receipt['group']['episode_id']: freeze_packet(r.out, out / 'reviews' / r.receipt['group']['episode_id']) for r in runs}
    packet = strict_json(packets['main'].read_bytes())
    # Invalid synthetic requests exercise rejection only; no human/model scientific labels are created.
    invalid_review = {'packet_sha256': digest(packets['main'].read_bytes()), 'reviewer_source': 'human_unqualified',
        'reviewer_identity': 'SYNTHETIC invalid request; no human review', 'verdict': 'pass',
        'note': 'Rejection fixture only', 'evidence': ['invented-event']}
    for change in ({}, {'evidence': [packet['events'][0]['event_id']], 'scientific_reward': 1},
                   {'verdict': 'fail', 'evidence': [packet['events'][-1]['event_id']],
                    'earliest_error': packet['events'][14]['event_id'], 'repair_status': 'unrepaired'}):
        try:
            validate_annotation(packets['main'], invalid_review | change)
        except ValueError:
            continue
        raise ValueError('invalid method annotation was accepted')
    result = profile_group(group, [r.out / 'runner_receipt.json' for r in runs])
    require(result['counts']['scheduled'] == 4 and result['counts']['missing'] == 1 and result['counts']['invalid'] == 1,
            'attempt accounting lost failed or missing slots')
    require(result['scientific_reward'] == {'known': 0, 'missing': 4, 'mean': None, 'population_std': None},
            'scientific reward must remain unavailable')
    atomic_json(out / 'profile.json', result)
    require(digest(Path(__file__).read_bytes()) == script_sha256, 'proof script changed during execution')
    proof = {'schema_version': 1, 'recorded_at': now(), 'git_head': git_commit(),
        'command': [sys.executable, *sys.argv], 'proof_script_sha256': script_sha256,
        'mode': 'SCRIPTED provider replies, REAL isolated physics; no model sampling or optimizer',
        'group_purpose': 'Scripted plumbing/accounting coverage with contrasting branches, not repeated stochastic policy samples.',
        'archive_sha256': ARCHIVE_SHA256, 'baseline': {'sha256': args.baseline_sha256,
            'kind': 'prior local execution of unchanged deposited notebook scientific cells',
            'rtol': 1e-6, 'atol': 1e-5, 'axes_exact': True,
            'map_max_absolute_error': float(np.max(abs(mapped['intensity'] - reference['intensity']))),
            'profile_max_absolute_error': float(np.max(abs(profile['intensity'] - reference['ip'])))},
        'comparison_fingerprint': fingerprint, 'runtime': main_run.receipt['comparison']['identity']['giwaxs_runtime'],
        'map_shape': list(mapped['intensity'].shape), 'repeat_exact': True,
        'native_pages': [{'handle': p['handle'], 'sha256': p['sha256'], 'rows': len(p['rows']),
                          'start': p['start'], 'next_start': p['next_start']} for p in pages],
        'sensitivities': sensitivities, 'profile_selection': main_run.evidence.artifacts['profile:notebook']['parameters'],
        'masked_coverage': main_run.evidence.artifacts['profile:masked']['parameters'],
        'expected_rejected_actions': [calls[i][0] + ':' + str(calls[i][1]) for i in (14, 15, 18)],
        'terminal_submit': True, 'invalid_annotation_requests_rejected': 3, 'scientific_annotations_collected': 0,
        'counts': result['counts'], 'scientific_reward': result['scientific_reward'],
        'endings': {r.receipt['group']['episode_id']: r.receipt['ending']['kind'] for r in runs},
        'packets': {slot: {'path': str(path), 'sha256': digest(path.read_bytes())} for slot, path in packets.items()},
        'artifact_hashes': {handle: row[3] for handle, row in main_run.evidence.files.items()},
        'limitations': LIMITS + ['Numerical tolerances are engineering regressions, not scientific acceptance thresholds.',
            'All related crops, reductions and perturbations belong to one exposed development family.',
            'Raw/trial files remain externally hash-bound; keep the whole proof directory for later review.']}
    atomic_json(out / 'proof.json', proof)
    print(json.dumps({'proof': str(out / 'proof.json'), 'baseline': proof['baseline'],
                      'counts': result['counts'], 'scientific_reward': result['scientific_reward']}))


if __name__ == '__main__':
    main()
