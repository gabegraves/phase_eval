import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from phaseeval.api import auth
from phaseeval.api.app import create_app
from phaseeval.workbench import dashboard
from phaseeval.workbench.cli import Settings
from tests.support.demo import REPO
from tests.support.store import AGENT, IMPORTER, JUDGE, MIXED, OWNER, OWNER_RATER, RATER_A, _headers


DATA_ROUTES = (
    '/api/attempts/dashboard',
    '/api/attempts/scorecard',
    '/api/attempts/scorecard?model=qwen36-27b',
    '/api/attempts/scorecard/pairs.jsonl',
    '/api/attempts/scorecard/demo-pairs.jsonl',
)


@pytest.mark.parametrize('variable', ['PHASEEVAL_LOCAL_OWNER', 'AXA_WORKBENCH_LOCAL_OWNER'])
def test_local_owner_variables_are_ignored(variable):
    assert Settings.from_env({variable: '1'}) == Settings.from_env({})


@pytest.fixture
def client(bench):
    settings = Settings.from_env({
        'AXA_WORKBENCH_LOCAL_OWNER': '1',
        'PHASEEVAL_DATASET': str(REPO / 'data/dataset-v0-public'),
    })
    app = create_app(settings, store=bench.store, authenticator=bench.authenticator)
    with TestClient(app, base_url='http://127.0.0.1:18800', client=('127.0.0.1', 12345)) as client:
        yield client


@pytest.mark.parametrize('path', DATA_ROUTES)
def test_scorecard_data_without_a_token_is_401_even_with_the_local_owner_variable(client, path):
    response = client.get(path)
    assert response.status_code == 401, response.text


@pytest.mark.parametrize('path', DATA_ROUTES)
@pytest.mark.parametrize('authorization', ['Bearer invalid-token', 'Basic owner', 'Bearer '])
def test_invalid_credentials_are_401(client, path, authorization):
    assert client.get(path, headers={'Authorization': authorization}).status_code == 401


@pytest.mark.parametrize('path', DATA_ROUTES)
@pytest.mark.parametrize('principal', [RATER_A, OWNER_RATER, MIXED, AGENT, JUDGE, IMPORTER])
def test_non_admin_credentials_are_403(client, path, principal):
    response = client.get(path, headers=_headers(principal))
    assert response.status_code == 403, response.text
    assert 'qwen36-27b' not in response.text


@pytest.mark.parametrize('path', DATA_ROUTES)
def test_admin_credentials_succeed(client, path):
    response = client.get(path, headers=_headers(OWNER))
    assert response.status_code == 200, response.text
    if path in DATA_ROUTES[:2]:
        assert 'Outcomes per case, model and harness' in response.json()['html']


@pytest.mark.parametrize('path', DATA_ROUTES)
def test_the_dashboard_tables_need_the_admin_dependency(client, path):
    def denied():
        raise HTTPException(status_code=403, detail='admin dependency sentinel')

    client.app.dependency_overrides[dashboard._admin_owner] = denied
    response = client.get(path, headers=_headers(OWNER))
    assert response.status_code == 403, response.text
    assert response.json() == {'detail': 'admin dependency sentinel'}


@pytest.mark.parametrize('path', DATA_ROUTES)
def test_admin_data_requires_the_scorecard_rule(client, path, monkeypatch):
    monkeypatch.setitem(auth.RULES, auth.Op.SCORECARD, auth.OpRule(frozenset(), None))
    response = client.get(path, headers=_headers(OWNER))
    assert response.status_code == 403, response.text
