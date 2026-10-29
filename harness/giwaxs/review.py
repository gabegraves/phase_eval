"""Freeze owner GIWAXS method evidence; annotations never qualify scientific rewards."""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'src')]
from harness.agent.evidence import checked_bytes
from harness.agent.episode import atomic_json, digest, strict_json
from harness.giwaxs.environment import GiwaxsEvidence
from phaseeval.giwaxs import EvidenceFile, EvidenceRegistry, MethodAnnotation, MethodEvent, MethodPacket, SubmissionReceipt
from phaseeval.workbench.labels import EventRef, validate_judgment_coherence, validate_judgment_events

BINDINGS = ('giwaxs-evidence.json', 'artifacts/workspace/output/analysis.json', 'agent/trajectory.json')
QUESTIONS = [
    'Is the supplied geometry and its calibration uncertainty adequate for the stated claim?',
    'Is the mask choice scientifically justified for the inspected detector regions?',
    'Do coverage, window selection and aggregation preserve the signal relevant to the claim?',
    'Were plausible reduction alternatives considered and their effects interpreted appropriately?',
    'Are the conclusion, stopping decision and requested additional evidence scientifically defensible?',
]


def _events(receipt, trajectory, trial, files):
    actions = {a['event_id']: a for a in receipt['actions']}
    if len(actions) != len(receipt['actions']):
        raise ValueError('duplicate runner event ID')
    events = []
    for turn_index, step in enumerate(trajectory['steps']):
        calls = step.get('tool_calls', [])
        results = step.get('observation', {}).get('results', [])
        by_id = {r['event_id']: r for r in results}
        if len(by_id) != len(results) or not set(by_id) <= {c['event_id'] for c in calls}:
            raise ValueError('observation names no unique recorded call')
        for call_index, call in enumerate(calls):
            event_id = call['event_id']
            action = actions.pop(event_id, None)
            if step['source'] != 'agent' or action is None or any(
                action[key] != call[key] for key in ('function_name', 'tool_call_id', 'outcome')
            ):
                raise ValueError('action and trajectory event differ')
            shown = full = sha = None
            result = by_id.get(event_id)
            clipped = bool(result and result.get('extra'))
            if result is not None:
                if result['source_call_id'] != call['tool_call_id'] or result['outcome'] != action['outcome']:
                    raise ValueError('observation and action differ')
                shown = full = result['content']
                if clipped:
                    extra = result['extra']
                    raw = checked_bytes(trial, extra['executor_output'], extra['executor_sha256'])
                    full = raw.decode('utf-8')
                    files['executor:' + event_id] = EvidenceFile(root=str(trial), file=extra['executor_output'],
                                                               kind='text', sha256=extra['executor_sha256'])
                sha = digest(full.encode())
            events.append(MethodEvent(event_id=event_id, turn_index=turn_index, call_index=call_index,
                function_name=call['function_name'], arguments=call['arguments'], transport_status=action['status'],
                outcome=action['outcome'], shown_observation=shown, executor_observation=full,
                executor_sha256=sha, clipped=clipped))
    if actions:
        raise ValueError('receipt action missing from chronology')
    return events


def _build_packet(trial, input_dir=None, source_root=REPO):
    trial = Path(trial).absolute()
    files = {}

    def bind(key, root, name, expected=None, kind='text'):
        raw = checked_bytes(root, name, expected)
        files[key] = EvidenceFile(root=str(root), file=name, kind=kind, sha256=digest(raw))
        return raw

    receipt = strict_json(bind('receipt', trial, 'runner_receipt.json'))
    if receipt.get('finished_at') is None or receipt.get('ending') is None:
        raise ValueError('episode must be finalized before review')
    config = strict_json(bind('config', trial, 'config.json'))
    if config['task'].get('task_kind') != 'giwaxs':
        raise ValueError('GIWAXS task configuration required')
    case_id = config['task']['case_id']
    bindings = receipt.get('evidence_bindings')
    if not isinstance(bindings, dict) or set(bindings) != set(BINDINGS):
        raise ValueError('final GIWAXS evidence bindings required')
    values = {}
    for name, sha in bindings.items():
        if sha is None:
            if (trial / name).exists() or (trial / name).is_symlink():
                raise ValueError('unexpected file at an absent evidence binding')
        else:
            values[name] = strict_json(bind(name, trial, name, sha))
    trajectory = values.get('agent/trajectory.json')
    if trajectory is None:
        raise ValueError('final chronology is required')
    if 'provider_requests' in receipt:
        requests = receipt['provider_requests']
        if not isinstance(requests, list):
            raise ValueError('provider requests must be a list')
        for index, request in enumerate(requests, 1):
            if (request['file'] != f'agent/provider-request-{index:04d}.json'
                    or not isinstance(request.get('sha256'), str)
                    or not re.fullmatch('[0-9a-f]{64}', request['sha256'])):
                raise ValueError('provider request sequence or hash is invalid')
            saved = strict_json(bind(f'provider_request:{index}', trial, request['file'], request['sha256']))
            if set(saved) != {'messages', 'tools'} or not all(isinstance(saved[k], list) for k in saved):
                raise ValueError('invalid provider input snapshot')
    registry = EvidenceRegistry.model_validate(values['giwaxs-evidence.json']) if 'giwaxs-evidence.json' in values else None
    comparison = receipt.get('comparison')
    case, artifacts = None, {}
    if comparison is not None:
        identity = comparison['identity']
        if identity.get('task_kind') != 'giwaxs' or digest(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()) != comparison['fingerprint']:
            raise ValueError('GIWAXS comparison identity mismatch')
        if identity['case_id'] != case_id:
            raise ValueError('configured case differs from comparison identity')
        for name, sha in identity['sources'].items():
            bind('source:' + name, source_root, name, sha)
        inputs = Path(input_dir or (registry.inputs_root if registry else '')).absolute()
        if input_dir is None and registry is None:
            raise ValueError('input_dir required to inspect a valid episode without a registry')
        evidence = GiwaxsEvidence(inputs, case_id, trial / 'artifacts/workspace/output', trial)
        if evidence.bound_inputs != identity['inputs']:
            raise ValueError('input identity changed')
        case = evidence.case
        for name, sha in evidence.bound_inputs.items():
            bind('input:' + name, inputs, name, sha)
        if registry is not None:
            if registry.inputs != evidence.bound_inputs or registry.manifest_sha256 != evidence.manifest_sha256 or registry.case_id != case_id:
                raise ValueError('registry input identity differs')
            for handle, ref in registry.files.items():
                root = Path(ref.root).absolute()
                if root != inputs and not root.is_relative_to(trial):
                    raise ValueError('registered evidence outside input and trial roots')
                evidence.files[handle] = (root, ref.file, ref.kind, ref.sha256)
                files['evidence:' + handle] = ref
            evidence.artifacts = {handle: value.model_dump() for handle, value in registry.artifacts.items()}
            for handle in registry.files:
                evidence._bytes(handle)
            for handle, artifact in registry.artifacts.items():
                ref = registry.files[handle]
                if artifact.handle != handle or artifact.case_id != case_id or artifact.inputs != evidence.bound_inputs or artifact.outputs != {ref.file: ref.sha256}:
                    raise ValueError('artifact identity differs')
                if strict_json(evidence._bytes(handle + ':receipt')) != artifact.model_dump():
                    raise ValueError('artifact receipt differs from registry')
            artifacts = registry.artifacts
    elif registry is not None or receipt['setup']['status'] == 'valid':
        raise ValueError('evidence requires a valid comparison identity')
    submission = SubmissionReceipt.model_validate(values['artifacts/workspace/output/analysis.json']) if 'artifacts/workspace/output/analysis.json' in values else None
    if submission is not None:
        if registry is None or submission.case_id != case_id:
            raise ValueError('submission has no matching evidence registry')
        cited = {handle for claim in submission.claims for handle in claim.evidence}
        if set(submission.citations) != cited or any(registry.files[h].sha256 != sha for h, sha in submission.citations.items()):
            raise ValueError('submission citation mismatch')
    events = _events(receipt, trajectory, trial, files)
    return MethodPacket(case_id=case_id, case=case, receipt=receipt, trajectory=trajectory,
        submission=submission, events=events, artifacts=artifacts, bound_files=files,
        message_events={f'message:{i}': i for i, step in enumerate(trajectory['steps'])
                        if isinstance(step.get('message'), str) and step['message'].strip()},
        integrity_checks=['Final receipt bindings and registered file hashes verified.',
                          'Runner actions and delivered observations have matching unique event identities.',
                          'Scientific judgment, calibration adequacy and reward validity remain unassessed.'],
        unassessed_questions=QUESTIONS)


def freeze_packet(trial: Path, output: Path, *, input_dir: Path | None = None) -> Path:
    """Export after finalization. input_dir is needed for a valid run with no registry."""
    trial, output = Path(trial).absolute(), Path(output).absolute()
    if any(p.is_symlink() for path in (trial, output) for p in (path, *path.parents)):
        raise ValueError('symlink review output')
    trial, output = trial.resolve(), output.resolve()
    if output.is_relative_to(trial):
        raise ValueError('owner packet must be outside the actor trial')
    if output.exists():
        raise FileExistsError(output)
    packet = _build_packet(trial, input_dir)
    output.mkdir(parents=True)
    # ponytail: snapshot code only; raw/trial files remain hash-bound until portable bundles are needed.
    for key, ref in packet.bound_files.items():
        if key.startswith('source:'):
            raw = checked_bytes(Path(ref.root), ref.file, ref.sha256)
            target = output / 'sources' / ref.file
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(raw)
            packet.bound_files[key] = ref.model_copy(update={'root': str(output / 'sources')})
    atomic_json(output / 'packet.json', packet.model_dump())
    (output / 'METHOD.md').write_text(render_method(packet), encoding='utf-8')
    return output / 'packet.json'


def _plain(value):
    return re.sub(r'([\\`*_{}\[\]()#+.!|~-])', r'\\\1', html.escape(str(value)).replace('\n', ' '))


def _fence(value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    fence = '`' * max(3, 1 + max((len(x) for x in re.findall(r'`+', text)), default=0))
    return f'{fence}json\n{text}\n{fence}\n'


def render_method(packet: MethodPacket) -> str:
    lines = ['# GIWAXS method record', '', 'Owner development review. Scientific validity and rewards are unassessed.',
             '', '## Method summary', '', f"Case: {_plain(packet.case_id)}. Ending: {_plain(packet.receipt['ending']['kind'])}.", '']
    for index, event in enumerate(packet.events, 1):
        a = event.arguments if isinstance(event.arguments, dict) else {}
        detail = {'transform': f"mask={a.get('mask_policy')}; incidence={a.get('incident_angle')} degrees; orientation={a.get('sample_orientation')}; tilt={a.get('tilt_angle')} degrees; distance scale={a.get('distance_scale', 1)}",
                  'extract': f"{a.get('axis', 'qr')} profile; requested strip=[{a.get('lower')}, {a.get('upper')}) q nm^-1; mode={a.get('mode')}; output={a.get('coordinate_definition', 'q=2pi/d')}",
                  'compare': f"{a.get('left')} versus {a.get('right')}; bounds={a.get('lower')}, {a.get('upper')}",
                  'submit': f"status={a.get('status')}", 'read_evidence': f"handle={a.get('handle', 'catalog')}"}.get(event.function_name, '')
        artifact = packet.artifacts.get(f"profile:{a.get('label')}") if event.function_name == 'extract' else None
        if artifact is not None:
            detail += f"; realized sample centers={artifact.parameters['realized_roi']}; selected rows={artifact.parameters['selected_rows']}"
        lines.append(f"{index}. {_plain(event.function_name)} — {_plain(detail)}. Outcome: {_plain(event.outcome)}.")
    if not packet.events:
        lines.append('No agent tool actions were recorded.')
    lines += ['', '## Provider request context', '']
    if 'provider_requests' not in packet.receipt:
        lines.append('Exact provider inputs were not recorded for this historical episode.')
    else:
        lines.append('The chronology below retains original delivered observations. Each saved request contains the exact '
                     'messages and tool schema passed to chat, including failed calls. Only the latest four numeric pages '
                     'are replayed in full; older rows and row_indexes are explicitly omitted, with metadata, summaries '
                     'and reread handles preserved. Earlier delivered evidence may therefore be absent from a later request.')
        for index, request in enumerate(packet.receipt['provider_requests'], 1):
            ref = packet.bound_files[f'provider_request:{index}']
            link = quote(str(Path(ref.root) / ref.file), safe='/')
            lines.append(f"- [Request {index}]({link}): omitted numeric payloads at zero-based message indexes "
                         f"{_plain(request['omitted_message_indexes'])}.")
    lines += ['', '## Submitted limitations', '']
    lines += ['- ' + _plain(x) for x in packet.submission.limitations] if packet.submission else ['No submitted conclusion.']
    lines += ['', '## Scientific questions — unassessed', ''] + ['- ' + q for q in packet.unassessed_questions]
    lines += ['', '## Reduction and extraction evidence', '']
    for artifact in packet.artifacts.values():
        ref = packet.bound_files['evidence:' + artifact.handle]
        link = quote(str(Path(ref.root) / ref.file), safe='/')
        lines += [f"### {_plain(artifact.handle)}", '', f'[Open artifact]({link})', '', _fence(artifact.model_dump())]
    lines += ['', '## Recorded message evidence', '']
    for event_id, index in packet.message_events.items():
        step = packet.trajectory['steps'][index]
        message = step['message']
        preview = message[:240] + ('… (full text in frozen context below)' if len(message) > 240 else '')
        lines.append(f"- {_plain(event_id)} ({_plain(step['source'])}): {_plain(preview)}")
    lines += ['', '## Exact event evidence', '', 'Shown observations were delivered to the actor. Executor observations may contain additional text that was not shown in that event.', '']
    for event in packet.events:
        lines += [f'### Event {_plain(event.event_id)}', '', _fence(event.model_dump())]
    lines += ['', '## Frozen context and integrity', '', _fence({'case': packet.case.model_dump() if packet.case else None,
                 'checks': packet.integrity_checks, 'receipt': packet.receipt, 'trajectory': packet.trajectory,
                 'message_event_ids_to_turn_indexes': packet.message_events,
                 'bound_files': {k: v.model_dump() for k, v in packet.bound_files.items()}})]
    return '\n'.join(lines) + '\n'


def validate_annotation(packet_path: Path, value: dict) -> MethodAnnotation:
    """Validate provenance and available-event chronology, never scientific correctness."""
    packet_path = Path(packet_path).absolute()
    raw = checked_bytes(packet_path.parent, packet_path.name)
    packet_path = packet_path.resolve()
    packet = MethodPacket.model_validate(strict_json(raw))
    annotation = MethodAnnotation.model_validate(value)
    if annotation.packet_sha256 != digest(raw):
        raise ValueError('annotation packet digest differs')
    for ref in packet.bound_files.values():
        checked_bytes(Path(ref.root), ref.file, ref.sha256)
    inputs = packet.bound_files.get('input:manifest.json')
    expected = _build_packet(Path(packet.bound_files['receipt'].root), Path(inputs.root) if inputs else None,
                             source_root=packet_path.parent / 'sources')
    if packet != expected:
        raise ValueError('packet differs from its bound source records')
    events = {event.event_id: event for event in packet.events}
    if len(events) != len(packet.events):
        raise ValueError('duplicate packet event ID')
    turns = []
    for index, step in enumerate(packet.trajectory['steps']):
        current = [e for e in packet.events if e.turn_index == index]
        turns.append({'source': step['source'], 'message': step.get('message') or '',
                      'calls': [{'call_index': e.call_index} for e in current]})
        turns.append({'source': 'tool', 'message': '', 'calls': [
            {'call_index': e.call_index} for e in current if e.shown_observation is not None]})

    def ref(event_id, observed=False):
        if event_id in packet.message_events:
            return EventRef(turn_index=2 * packet.message_events[event_id])
        if event_id not in events:
            raise ValueError('annotation names no recorded event')
        event = events[event_id]
        return EventRef(turn_index=2 * event.turn_index + int(observed), call_index=event.call_index)

    refs = {'evidence': tuple(ref(x, observed=True) for x in annotation.evidence),
            'earliest_error': ref(annotation.earliest_error) if annotation.earliest_error else None,
            'observed_repair': ref(annotation.observed_repair) if annotation.observed_repair else None}
    validate_judgment_coherence(verdict=annotation.verdict, repair_status=annotation.repair_status, **refs)
    validate_judgment_events(turns, **refs)
    return annotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--annotation', type=Path)
    args = parser.parse_args()
    if args.packet and args.annotation and not args.trial and not args.out:
        annotation = validate_annotation(args.packet, strict_json(checked_bytes(args.annotation.absolute().parent, args.annotation.name)))
        print(annotation.model_dump_json(indent=2))
    elif args.trial and args.out and not args.packet and not args.annotation:
        print(freeze_packet(args.trial, args.out, input_dir=args.input_dir))
    else:
        parser.error('use --trial/--out to freeze, or --packet/--annotation to validate an unqualified review')


if __name__ == '__main__':
    main()
