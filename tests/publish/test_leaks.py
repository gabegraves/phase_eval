import json
import importlib.util
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(('rule', 'value'), [
    ('ipv4', 'http://100.64.0.1/v1'),
    ('ipv4', '192.168.1.1'),
    ('ipv4', 'Host 192.168.1.1.'),
    ('ipv6', 'https://[2001:db8::1]/v1'),
    ('ipv6', '::1'),
    ('posix_home', '/Users/alice/run'),
    ('posix_home', '/home/alice/run'),
    ('posix_home', '/root/run'),
    ('windows_path', r'C:\Users\alice\run'),
    ('windows_path', 'C:/Users/alice/run'),
    ('windows_path', r'C:\n'),
    ('windows_path', r'\\server\share\run'),
    ('local_host', 'machine.local'),
    ('local_host', 'worker.tail123.ts.net'),
    ('local_host', 'server.internal'),
    ('local_host', 'localhost:8080'),
    ('url_with_port', 'https://example.com:8443/api'),
    ('email', 'alice@example.com'),
    ('bearer_like', 'Bearer abcdef0123456789abcdef0123456789'),
    ('bearer_like', 'Bearer\tabcdef0123456789abcdef0123456789'),
    ('bearer_like', 'Bearer\nabcdef0123456789abcdef0123456789'),
    ('bearer_like', 'sk-abcdefghijklmnopqrstuvwxyz'),
    ('bearer_like', 'hf_abcdefghijklmnopqrstuvwxyz'),
    ('bearer_like', 'ghp_abcdefghijklmnopqrstuvwxyz'),
    ('bearer_like', 'A7pQ9tY2uI6oP4sD8fG1hJ3kL5zX0cVbN'),
    ('custody_key', {'reference_answer': None}),
    ('custody_key', {'reference': None}),
    ('custody_key', {'truth': None}),
    ('custody_key', {'ground_truth': None}),
    ('custody_key', {'custody': None}),
    ('custody_key', {'nested': {'withheld_context': []}}),
    ('demo_run', 'u6-dev006-qwen-008'),
    ('demo_run', 'u6-dev102-qwen-006'),
])
def test_every_rule_catches_its_pattern(rule, value):
    from phaseeval.publish.leaks import scan

    assert rule in {f.rule for f in scan(value)}


def test_a_string_inside_a_list_is_scanned():
    from phaseeval.publish.leaks import scan

    findings = scan({'notes': [None, ['alice@example.com']]})
    assert [(f.rule, f.pointer) for f in findings] == [('email', '/notes/1/0')]
    assert all('alice' not in f.excerpt and len(f.excerpt) <= 80 for f in findings)


@pytest.mark.parametrize('value,in_list', [
    ('alice@example.com', False), ('alice@example.com', True),
    ('http://100.64.12.34:40000/v1', True),
    ('https://[2001:db8::1]/v1', False),
    (r'C:\Users\alice\run', False), ('/private/tmp/receipt', False),
    ('Bearer\tabcdef0123456789abcdef0123456789', False),
    ('Bearer\nabcdef0123456789abcdef0123456789', False),
])
def test_shared_scanner_closes_legacy_submission_gaps(tmp_path, value, in_list):
    record = json.loads((REPO / 'data/dataset-v0-public/episodes.jsonl').read_text().splitlines()[0])
    if in_list:
        record['runs'][0]['models_observed'] = [value]
    else:
        record['rights'][0]['redistribution']['observation'] = value

    if (REPO / 'src/phaseeval/publish/leaks.py').exists():
        from phaseeval.publish.leaks import PUBLIC_RULES, scan

        caught = bool(scan(record, rules=[r for r in PUBLIC_RULES if r.name != 'demo_run']))
    else:
        # main predates the shared scanner; exercise its public submission check.
        spec = importlib.util.spec_from_file_location('submit_run', REPO / 'scripts/submit_run.py')
        submit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submit)
        path = tmp_path / 'submission.jsonl'
        path.write_text(json.dumps(record) + '\n')
        assert any(v.is_valid(record) for v in submit.record_validators())
        try:
            submit.check(path, include_scans=False)
        except submit.Refused:
            caught = True
        else:
            caught = False
    assert caught, 'the public leak check accepted the injected private value'


def test_keys_are_scanned_and_pointers_are_escaped():
    from phaseeval.publish.leaks import scan

    assert scan({'a/b~c': {'alice@example.com': 1}})[0].pointer == '/a~1b~0c/alice@example.com'


def test_per_call_identity_is_literal_and_applies_to_keys_and_values():
    from phaseeval.publish.leaks import scan

    assert scan({'model': 'Astra+[v2]'}) == ()
    assert len(scan({'Astra+[v2]': ['Astra+[v2]']}, identity=['Astra+[v2]', ''])) == 2


@pytest.mark.parametrize('text', [
    'x[::-1]', r"with open('data') as f:\n    data = f.read()",
    r"if 'stages' in r:\n    print(r)",
    'Sussieck-Fornefeld_HEGEOR_1988_492.cif',
    'Tables 4.2.6.8 and 6.1.1.4', '12:34:56', 'version 1.2.3',
    'sha256: ' + 'a' * 64, 'cod_1010093', 'https://example.com/path',
    '999.999.999.999', '/home/coder/svn-repositories/cod/cif/1.cif',
    '/private-runs/jobs/demo/r1',
])
def test_scientific_text_and_documented_placeholders_are_allowed(text):
    from phaseeval.publish.leaks import scan, scan_bytes

    assert scan(text) == ()
    assert scan_bytes(text.encode()) == ()


@pytest.mark.parametrize('path', ['/private/var/run/x', '/private/tmp/x', '/private/etc/hosts'])
def test_private_system_paths_are_caught(path):
    from phaseeval.publish.leaks import scan

    assert 'posix_private' in {f.rule for f in scan(path)}


def test_the_private_runs_placeholder_is_allowed_and_documented():
    from phaseeval.publish.leaks import ALLOWED_PREFIXES, scan

    assert scan('/private-runs/jobs/demo') == ()
    assert '/private-runs/' in dict(ALLOWED_PREFIXES)
    assert '/private-runs/' in (REPO / 'launcher/demo/NOTICE.md').read_text()
    assert scan('/home/coder/svn-repositories/x /home/alice/x')
    assert scan('/private-runs/x /private/tmp/x')


def test_every_allowance_matches_a_shipped_file():
    from phaseeval.publish.leaks import ALLOWED_PREFIXES

    with tarfile.open(REPO / 'launcher/demo/phaseeval-demo.tar.xz') as archive:
        data = [archive.extractfile(m).read() for m in archive if m.isfile()]
    assert len(ALLOWED_PREFIXES) == 2
    for prefix, reason in ALLOWED_PREFIXES:
        assert reason
        assert any(prefix.encode() in blob for blob in data), prefix


def test_bytes_scan_json_jsonl_and_unstructured_text():
    from phaseeval.publish.leaks import scan_bytes

    assert scan_bytes(b'{"reference_answer": null}')[0].rule == 'custody_key'
    assert scan_bytes(b'{}\n{"notes": ["alice@example.com"]}\n')[0].rule == 'email'
    assert scan_bytes(b'\xff\x00log: /root/secret')[0].rule == 'posix_home'


def test_rules_can_be_selected_and_redaction_preserves_other_data():
    from phaseeval.publish.leaks import PUBLIC_RULES, redact, scan

    rules = [rule for rule in PUBLIC_RULES if rule.name == 'email']
    value = {'a/b~c': ['alice@example.com', 'fine', 2], 'alice@example.com': True}
    findings = scan(value, rules=rules)
    redacted = redact(value, findings)
    assert scan(redacted, rules=rules) == ()
    assert redacted['a/b~c'][1:] == ['fine', 2]
    assert value['a/b~c'][0] == 'alice@example.com'
    assert json.loads(json.dumps(redacted)) == redacted
    assert redact(value, ()) == value
    assert scan('/root/x', rules=rules) == ()
