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


def test_context_citation_survives_frozen_review_rebuild(args, monkeypatch, tmp_path):
    runner = runner_module.Runner(args)
    replies = iter([reply(('read_evidence', {'handle': 'context'})), reply(('submit', PARTIAL | {
        'claims': [{'text': 'Manifest supplies acquisition assumptions', 'evidence': ['context']}]
    }))])
    monkeypatch.setattr(runner, 'chat', lambda _: next(replies))
    assert runner.run() == 0
    packet = freeze_packet(runner.out, tmp_path / 'review')
    value = json.loads(packet.read_text())
    assert value['submission']['citations']['context'] == value['bound_files']['evidence:context']['sha256']
    assert validate_annotation(packet, deferred(packet)).scientific_qualified is False
    context = value['bound_files']['evidence:context']
    context_path = Path(context['root']) / context['file']
    context_path.unlink()
    with pytest.raises(ValueError, match='missing'):
        validate_annotation(packet, deferred(packet))
    assert not context_path.exists()


def test_provider_request_snapshots_bound_and_tampering_refused(completed, tmp_path):
    requests = completed.receipt['provider_requests']
    packet = freeze_packet(completed.out, tmp_path / 'review')
    value = json.loads(packet.read_text())
    for index, request in enumerate(requests):
        ref = value['bound_files'][f'provider_request:{index + 1}']
        assert ref['file'] == request['file'] and ref['sha256'] == request['sha256']
    assert 'Provider request context' in packet.with_name('METHOD.md').read_text()
    assert validate_annotation(packet, deferred(packet)).verdict == 'defer'
    (completed.out / requests[-1]['file']).write_text('{}')
    with pytest.raises(ValueError, match='digest'):
        freeze_packet(completed.out, tmp_path / 'tampered')
    with pytest.raises(ValueError, match='digest'):
        validate_annotation(packet, deferred(packet))


@pytest.mark.parametrize('pages', [0, 6])
def test_failed_provider_request_is_reviewable(args, monkeypatch, tmp_path, pages):
    runner = runner_module.Runner(args)
    def fail(_):
        if pages and not runner.receipt['provider_responses']:
            return reply(*[('read_evidence', {'handle': 'raw', 'limit': 1}) for _ in range(pages)])
        raise RuntimeError('synthetic provider failure')
    monkeypatch.setattr(runner, 'chat', fail)
    assert runner.run() == 1
    packet = freeze_packet(runner.out, tmp_path / 'review', input_dir=args.input_dir)
    value = json.loads(packet.read_text())
    assert len(value['receipt']['provider_requests']) == 1 + bool(pages)
    assert 'provider_request:1' in value['bound_files']
    if pages:
        assert value['receipt']['provider_requests'][-1]['omitted_message_indexes'] == []
        report = packet.with_name('METHOD.md').read_text()
        assert 'latest assistant/tool batch' in report
        assert 'does not prove the provider processed' in report
        assert isinstance(json.loads(value['events'][0]['shown_observation'])['rows'], list)
    assert validate_annotation(packet, deferred(packet)).verdict == 'defer'


def test_legacy_review_does_not_create_provider_logs(completed, tmp_path):
    receipt_path = completed.out / 'runner_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    for request in receipt.pop('provider_requests', []):
        (completed.out / request['file']).unlink()
    receipt_path.write_text(json.dumps(receipt))
    before = {p: p.read_bytes() for p in completed.out.rglob('*') if p.is_file()}
    packet = freeze_packet(completed.out, tmp_path / 'review')
    assert 'not recorded' in packet.with_name('METHOD.md').read_text()
    assert validate_annotation(packet, deferred(packet)).verdict == 'defer'
    assert before == {p: p.read_bytes() for p in completed.out.rglob('*') if p.is_file()}


def test_provider_request_requires_recorded_hash(completed, tmp_path):
    receipt_path = completed.out / 'runner_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['provider_requests'][0]['sha256'] = None
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='provider request'):
        freeze_packet(completed.out, tmp_path / 'review')


@pytest.mark.parametrize('mode', ['empty', 'truncated', 'extra'])
def test_provider_request_count_must_match_responses(completed, tmp_path, mode):
    receipt_path = completed.out / 'runner_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    requests = receipt['provider_requests']
    if mode == 'empty':
        requests.clear()
    elif mode == 'truncated':
        requests.pop()
    else:
        for _ in range(2):
            request = requests[-1] | {'file': f'agent/provider-request-{len(requests) + 1:04d}.json'}
            (completed.out / request['file']).write_bytes((completed.out / requests[-1]['file']).read_bytes())
            requests.append(request)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='provider request count'):
        freeze_packet(completed.out, tmp_path / 'review')
