import pytest

from phaseeval.api.auth import AuthorizationError
from phaseeval.kernel.errors import Denied
from phaseeval.workbench.labels import LabelStore
from phaseeval.workbench.store import ActionRefused, CustodyError
from tests.support.store import AGENT, OWNER, RATER_A


def test_a_validation_error_never_echoes_input(bench):
    secret = 'untrusted-input-sentinel'
    response = bench.post(OWNER, '/roots/' + bench.root + '/assignment', {'raters': [secret], 'snapshots': [secret]})
    assert response.status_code in (400, 422)
    assert secret not in response.text
    response = bench.post(OWNER, f'/{bench.root}/actions', {
        'steps': [{'idempotency_key': 'bad', 'action': {'verb': 'refine', 'recipe': secret}}],
    })
    assert response.status_code in (400, 422)
    assert secret not in response.text
    assert 'input_value' not in response.text
    response = bench.post(OWNER, f'/{bench.root}/actions', {
        'steps': [{'idempotency_key': 'unknown', 'action': {'verb': secret}}],
    })
    assert response.status_code == 422
    assert secret not in response.text


def test_unknown_and_ungranted_ids_get_the_same_403_body(bench):
    missing = bench.get(RATER_A, '/snapshots/' + 'f' * 64)
    existing = bench.get(RATER_A, f'/snapshots/{bench.snaps[0]}')
    assert missing.status_code == existing.status_code == 403
    assert missing.json() == existing.json()


@pytest.mark.parametrize('suffix', ['', '/path', '/review'])
def test_unknown_and_ungranted_attempts_get_the_same_403_body(bench, suffix):
    missing = bench.get(AGENT, '/' + 'f' * 64 + suffix)
    existing = bench.get(AGENT, f'/{bench.runs[0]}' + suffix)
    assert missing.status_code == existing.status_code == 403
    assert missing.json() == existing.json() == {'error': Denied.public_message}


@pytest.mark.parametrize('error,status', [
    (AuthorizationError, 403), (PermissionError, 403),
    (ActionRefused, 422), (CustodyError, 400), (ValueError, 400),
])
def test_attempt_errors_never_echo_exception_text(bench, monkeypatch, error, status):
    secret = 'private-exception-sentinel'

    def refused(*args):
        raise error(secret)

    monkeypatch.setattr(bench.store, 'roots', refused)
    response = bench.get(OWNER, '/roots')
    assert response.status_code == status
    assert secret not in response.text
    assert response.json()['error'] == (Denied.public_message if status == 403 else 'refused')


@pytest.mark.parametrize('error,status', [(PermissionError, 403), (ValueError, 422)])
def test_legacy_label_errors_never_echo_exception_text(bench, monkeypatch, error, status):
    secret = 'private-label-exception-sentinel'

    def refused(*args):
        raise error(secret)

    monkeypatch.setattr(LabelStore, 'snapshot', refused)
    response = bench.get(OWNER, f'/snapshots/{bench.snaps[0]}')
    assert response.status_code == status
    assert secret not in response.text
    assert response.json()['error'] == (Denied.public_message if status == 403 else 'refused')


def test_validation_messages_and_locations_never_echo_input(bench):
    secret = 'untrusted-validation-sentinel'
    bodies = [
        {'steps': [{'idempotency_key': 'bad', 'action': {'verb': 'refine'}}], secret: 'extra'},
        {'steps': [{'idempotency_key': 'bad', 'action': {'verb': 'refine'}, 'rationale': {secret: 'bad'}}]},
    ]
    for body in bodies:
        response = bench.post(OWNER, f'/{bench.root}/actions', body)
        assert response.status_code == 422
        assert secret not in response.text
        assert 'input' not in response.text
    response = bench.post(OWNER, f'/snapshots/{bench.snaps[0]}/labels', {
        'item': secret, 'answer': 'yes',
    })
    assert response.status_code == 422
    assert secret not in response.text
