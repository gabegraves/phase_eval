"""Development review integrity and chronology; no scientific judgment is validated."""
import json
from pathlib import Path

import pytest

from harness.agent import run_agent as runner_module
from harness.giwaxs.review import freeze_packet, validate_annotation
from harness.agent.episode import digest
from test_giwaxs import evidence
from test_giwaxs_runner import args, reply, PARTIAL, TRANSFORM


@pytest.fixture
def completed(args, monkeypatch):
    runner = runner_module.Runner(args)
    replies = iter([reply(('read_evidence', {'handle': 'context'})),
                    reply(('submit', PARTIAL | {'status': 'supported'})),
                    reply(('transform', TRANSFORM), ('submit', PARTIAL))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    assert runner.run() == 0
    return runner


def test_packet_is_readable_bound_and_unqualified(completed, tmp_path):
    packet = freeze_packet(completed.out, tmp_path / 'review')
    value = json.loads(packet.read_text())
    assert value['scientific_reward'] is None and value['scientific_qualified'] is False
    assert [e['outcome'] for e in value['events']] == ['success', 'rejected', 'success', 'success']
    report = packet.with_name('METHOD.md').read_text()
    assert 'Method summary' in report and 'supplied' in report and 'unassessed' in report
    with pytest.raises(FileExistsError):
        freeze_packet(completed.out, packet.parent)


def test_changed_source_refused(completed, tmp_path):
    (completed.out / 'agent/trajectory.json').write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        freeze_packet(completed.out, tmp_path / 'review')


def test_review_reuses_prior_evidence_and_later_repair_rules(completed, tmp_path):
    packet = freeze_packet(completed.out, tmp_path / 'review')
    events = json.loads(packet.read_text())['events']
    request = {'packet_sha256': digest(packet.read_bytes()), 'reviewer_source': 'human_unqualified',
               'reviewer_identity': 'synthetic test reviewer', 'verdict': 'pass', 'note': 'Synthetic structure check only',
               'evidence': [events[0]['event_id']], 'earliest_error': events[1]['event_id'],
               'observed_repair': events[3]['event_id'], 'repair_status': 'repaired'}
    assert validate_annotation(packet, request).scientific_reward is None
    assert validate_annotation(packet.parent / 'sources' / '..' / packet.name, request).scientific_reward is None
    for change in ({'evidence': ['invented']}, {'evidence': [events[3]['event_id']]},
                   {'observed_repair': events[1]['event_id']}, {'scientific_reward': 1},
                   {'earliest_error': 'message:0'}, {'observed_repair': 'message:0'},
                   {'earliest_error': events[2]['event_id'], 'observed_repair': events[3]['event_id']},
                   {'reviewer_source': 'expert'}, {'packet_sha256': '0' * 64}):
        with pytest.raises(ValueError):
            validate_annotation(packet, request | change)
    failed = request | {'verdict': 'fail', 'observed_repair': None, 'repair_status': 'unrepaired'}
    assert validate_annotation(packet, failed).verdict == 'fail'
    first_action = failed | {'evidence': ['message:0'], 'earliest_error': events[0]['event_id']}
    assert validate_annotation(packet, first_action).verdict == 'fail'


def deferred(packet):
    return {'packet_sha256': digest(packet.read_bytes()), 'reviewer_source': 'model_judge',
            'reviewer_identity': 'synthetic test fixture', 'reviewer_model': 'unit-test-model',
            'verdict': 'defer', 'note': 'No qualified scientific judgment in this mechanics test'}


@pytest.mark.parametrize('mode', ['partial', 'failed', 'no_submission'])
def test_no_reduction_and_failed_episodes_reviewable(args, monkeypatch, tmp_path, mode):
    if mode == 'failed':
        (args.input_dir / 'manifest.json').write_text('{}')
    runner = runner_module.Runner(args)
    monkeypatch.setattr(runner, 'chat', lambda _: reply(('submit', PARTIAL)) if mode == 'partial' else reply())
    assert runner.run() == (1 if mode == 'failed' else 0)
    packet = freeze_packet(runner.out, tmp_path / 'review', input_dir=args.input_dir)
    value = json.loads(packet.read_text())
    assert value['artifacts'] == {} and value['scientific_reward'] is None
    assert validate_annotation(packet, deferred(packet)).verdict == 'defer'
    with pytest.raises(ValueError, match='model'):
        validate_annotation(packet, deferred(packet) | {'reviewer_model': None})


def test_source_code_snapshot_survives_live_edits_but_not_snapshot_edits(completed, tmp_path, monkeypatch):
    packet = freeze_packet(completed.out, tmp_path / 'review')
    source = runner_module.REPO / 'harness/giwaxs/worker.py'
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: original(p) + (b'\nchanged' if p == source else b''))
    with pytest.raises(ValueError, match='digest'):
        freeze_packet(completed.out, tmp_path / 'after-edit')
    assert validate_annotation(packet, deferred(packet)).verdict == 'defer'
    (packet.parent / 'sources/harness/giwaxs/worker.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest'):
        validate_annotation(packet, deferred(packet))


def test_edited_artifact_and_forged_packet_event_rejected(completed, tmp_path):
    packet = freeze_packet(completed.out, tmp_path / 'review')
    value = json.loads(packet.read_text())
    value['events'][0]['event_id'] = 'invented'
    packet.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='bound source'):
        validate_annotation(packet, deferred(packet))
    (completed.output / 'map/data.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest'):
        freeze_packet(completed.out, tmp_path / 'after-edit')


def test_missing_event_rejected_even_if_new_trace_hash_is_recorded(completed, tmp_path):
    trace_path = completed.out / 'agent/trajectory.json'
    trace = json.loads(trace_path.read_text())
    trace['steps'][2]['tool_calls'][0]['event_id'] = 'invented'
    trace_path.write_text(json.dumps(trace))
    receipt_path = completed.out / 'runner_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['evidence_bindings']['agent/trajectory.json'] = digest(trace_path.read_bytes())
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='recorded call|event differ'):
        freeze_packet(completed.out, tmp_path / 'review')


def test_clipped_observation_retains_both_views_and_markdown_is_data(args, monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, 'SHOWN_CHARS', 300)
    runner = runner_module.Runner(args)
    replies = iter([reply(('read_evidence', {'handle': 'context'})), reply(('submit', PARTIAL | {
        'limitations': ['```\n<script>alert(1)</script>\n``` [link](https://example.org)']
    }))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    assert runner.run() == 0
    packet = freeze_packet(runner.out, tmp_path / 'review')
    value = json.loads(packet.read_text())
    event = value['events'][0]
    assert event['clipped'] and event['shown_observation'] != event['executor_observation']
    assert event['executor_sha256'] == digest(event['executor_observation'].encode())
    assert any(k.startswith('executor:') for k in value['bound_files'])
    report = packet.with_name('METHOD.md').read_text()
    assert '\n<script>' not in report and '\n```json\n' not in report.split('## Frozen context')[1]
    assert 'Executor observations may contain additional text' in report
    assert validate_annotation(packet, deferred(packet)).scientific_qualified is False


def test_schema_parity_and_roundtrip(completed, tmp_path):
    import jsonschema
    from phaseeval.giwaxs import MethodPacket, MethodAnnotation
    packet = freeze_packet(completed.out, tmp_path / 'review')
    for name, model, value in [
        ('giwaxs-method-packet.v1', MethodPacket, json.loads(packet.read_text())),
        ('giwaxs-method-annotation.v1', MethodAnnotation, validate_annotation(packet, deferred(packet)).model_dump()),
    ]:
        schema = json.loads((runner_module.REPO / 'schemas' / (name + '.json')).read_text())
        assert schema == model.model_json_schema()
        jsonschema.validate(value, schema)
        assert model.model_validate_json(json.dumps(value)).model_dump() == value


def test_owner_packet_cannot_escape_into_trial_with_dotdot(completed, tmp_path):
    output = tmp_path / 'outside' / '..' / completed.out.name / 'review'
    with pytest.raises(ValueError, match='outside'):
        freeze_packet(completed.out, output)
    assert not (completed.out / 'review').exists()


def test_rejected_non_object_arguments_and_unobserved_failure_are_preserved(args, monkeypatch, tmp_path):
    runner = runner_module.Runner(args)
    replies = iter([reply(('read_evidence', [])), reply(('transform', TRANSFORM))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    original = runner.call
    def failed(name, arguments):
        if name == 'transform':
            raise OSError('synthetic execution failure before observation')
        return original(name, arguments)
    monkeypatch.setattr(runner, 'call', failed)
    assert runner.run() == 1
    packet = freeze_packet(runner.out, tmp_path / 'review', input_dir=args.input_dir)
    events = json.loads(packet.read_text())['events']
    assert events[0]['arguments'] == [] and events[0]['outcome'] == 'rejected'
    assert events[1]['outcome'] == 'failure' and events[1]['shown_observation'] is None
    with pytest.raises(ValueError, match='recorded turn'):
        validate_annotation(packet, deferred(packet) | {'verdict': 'pass', 'evidence': [events[1]['event_id']]})
