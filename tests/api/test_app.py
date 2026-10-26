import hashlib
import json
import os

from fastapi import APIRouter
from fastapi.testclient import TestClient

from phaseeval.api import app as application
from phaseeval.workbench import cli
from tests.support.store import BLOBS, CASE, OWNER, RATER_A, RATER_B, _headers


def test_two_apps_over_two_stores_do_not_share_state(tmp_path):
    apps = []
    for name in ('first', 'second'):
        registry = tmp_path / f'{name}.json'
        registry.write_text(json.dumps({
            'schema_version': 'axa.api_principals.v1',
            'principals': [{
                'principal_id': OWNER.principal_id, 'principal_type': 'user', 'roles': ['owner'],
                'credential_sha256': hashlib.sha256(name.encode()).hexdigest(),
            }],
        }))
        apps.append(application.create_app(cli.Settings(store=tmp_path / name, principals=registry)))
    first, second = apps
    root = first.state.workbench_store.load_case(OWNER, CASE, BLOBS)
    with TestClient(first) as a, TestClient(second) as b:
        assert a.get('/api/attempts/roots', headers={'Authorization': 'Bearer first'}).json() == [root]
        assert b.get('/api/attempts/roots', headers={'Authorization': 'Bearer second'}).json() == []
        assert a.get('/api/attempts/whoami', headers={'Authorization': 'Bearer second'}).status_code == 401
        assert b.get('/api/attempts/whoami', headers={'Authorization': 'Bearer first'}).status_code == 401
        first.add_api_route('/first-only', lambda: {'first': True})
        assert a.get('/first-only').status_code == 200
        assert b.get('/first-only').status_code == 404


def test_no_request_reads_the_environment(assigned, demo_store, monkeypatch):
    bench = assigned
    bench.sign_all(RATER_A)
    bench.sign_all(RATER_B)
    settings = cli.Settings.from_env({'PHASEEVAL_DATASET': str(demo_store.directory / 'dataset')})
    app = application.create_app(settings, store=bench.store, authenticator=bench.authenticator)
    reads = []
    original = type(os.environ).__getitem__

    def tracked(environment, name):
        if name.startswith(("AXA_", "PHASEEVAL_")):
            reads.append(name)
        return original(environment, name)

    with TestClient(app) as client:
        monkeypatch.setattr(type(os.environ), '__getitem__', tracked)
        for path in ['/api/health', '/api/attempts/whoami', '/api/attempts/roots',
                     '/api/attempts/scorecard', '/api/attempts/dashboard',
                     '/api/attempts/scorecard/pairs.jsonl', '/api/attempts/scorecard/demo-pairs.jsonl',
                     '/api/capabilities']:
            response = client.get(path, headers=_headers(OWNER))
            assert response.status_code == 200, (path, response.text)
        response = client.get(f"/api/attempts/snapshots/{bench.snaps[0]}/revealed", headers=_headers(RATER_A))
        assert response.status_code == 200, response.text
        assert reads == []


def test_environment_is_captured_before_startup(bench, demo_store, tmp_path, monkeypatch):
    from phaseeval.refinement.bgmn.installer import BGMNInstaller

    binary_dir = tmp_path / "bgmn" / "BGMNwin"
    binary_dir.mkdir(parents=True)
    names = BGMNInstaller.EXECUTABLES_WIN if os.name == "nt" else BGMNInstaller.EXECUTABLES_UNIX
    for name in names:
        (binary_dir / name).touch()
    monkeypatch.setenv("PHASEEVAL_BGMN_DIR", str(binary_dir.parent))
    monkeypatch.setenv('PHASEEVAL_DATASET', str(demo_store.directory / 'dataset'))
    app = application.create_app(store=bench.store, authenticator=bench.authenticator)
    monkeypatch.setenv('PHASEEVAL_DATASET', '/missing-dataset')
    monkeypatch.setenv('PHASEEVAL_BGMN_DIR', '/changed-after-open')
    with TestClient(app) as client:
        assert client.get('/api/attempts/dashboard', headers=_headers(OWNER)).status_code == 200
        assert client.get('/api/capabilities', headers=_headers(OWNER)).json()['refit']['available'] is True


def test_extra_routers_precede_the_pages():
    router = APIRouter()
    router.add_api_route('/workbench', lambda: {'extra': True})
    app = application.create_app(cli.Settings.from_env({}), extra_routers=(router,))
    assert TestClient(app).get('/workbench').json() == {'extra': True}


def test_a_refit_in_one_app_does_not_take_the_other_apps_slot(bench, tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tests.support.fake_bgmn import FakeRefine
    from tests.support.store import ADD, World

    entered, release = threading.Event(), threading.Event()
    fake = FakeRefine()

    def blocked(ctx, action):
        entered.set()
        assert release.wait(10)
        return fake(ctx, action)

    first, second = World(tmp_path / 'first', refine=blocked), World(tmp_path / 'second')
    parents = [world.act(OWNER, world.root, 'add', ADD).digest for world in (first, second)]
    clients = [TestClient(application.create_app(cli.Settings.from_env({}), store=world.store,
                                               authenticator=bench.authenticator)) for world in (first, second)]
    body = {'steps': [{'idempotency_key': 'fit', 'action': {'verb': 'refine'}}]}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(clients[0].post, f'/api/attempts/{parents[0]}/actions', json=body, headers=_headers(OWNER))
        try:
            assert entered.wait(10)
            response = clients[1].post(f'/api/attempts/{parents[1]}/actions', json=body, headers=_headers(OWNER))
            assert response.status_code == 201, response.text
            assert response.json()['attempts'][0]['status'] == 'completed'
        finally:
            release.set()
        assert pending.result().status_code == 201


def test_disabled_refits_stay_disabled_after_the_environment_changes(bench, tmp_path, monkeypatch):
    from tests.support.store import ADD

    settings = cli.Settings(bgmn=None)
    store = cli.open_store(tmp_path / 'disabled', tmp_path, settings=settings)
    root = store.load_case(OWNER, CASE, BLOBS)
    app = application.create_app(settings, store=store, authenticator=bench.authenticator)
    monkeypatch.setenv('PHASEEVAL_BGMN_DIR', '/changed-after-open')
    response = TestClient(app).post(f'/api/attempts/{root}/actions', headers=_headers(OWNER), json={
        'steps': [{'idempotency_key': 'add', 'action': ADD},
                  {'idempotency_key': 'fit', 'action': {'verb': 'refine'}}],
    })
    assert response.status_code == 201, response.text
    last = response.json()['attempts'][-1]
    assert last['status'] == 'infra_failed'
    assert last['failure_reason'] == 'executor_unavailable: BGMN refits are disabled'
