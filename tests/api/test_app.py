import hashlib
import json
import os

import pytest

from fastapi import APIRouter
from fastapi.testclient import TestClient

from phaseeval.api import app as application
from phaseeval.workbench import cli
from tests.support.store import ADD, BLOBS, CASE, OWNER, RATER_A, RATER_B, _headers


def _configured_app(environment=None, *, store=None, authenticator=None):
    # Exercise the pre-factory entry point too, so parent failures reach behavior assertions.
    if hasattr(application, 'create_app'):
        settings = cli.Settings.from_env(environment) if environment is not None else None
        return application.create_app(settings, store=store, authenticator=authenticator)
    if store is None:
        application.configure_workbench_from_environment(environment)
    else:
        application.configure_workbench(store, authenticator)
    return application.app


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
        apps.append(_configured_app({
            'AXA_WORKBENCH_ROOT': str(tmp_path / name),
            'PHASEEVAL_STORE': str(tmp_path / name),
            'AXA_API_PRINCIPALS_FILE': str(registry),
            'PHASEEVAL_PRINCIPALS': str(registry),
        }))
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
    dataset = str(demo_store.directory / 'dataset')
    monkeypatch.setenv('AXA_WORKBENCH_DATASET', dataset)
    app = _configured_app({'PHASEEVAL_DATASET': dataset}, store=bench.store, authenticator=bench.authenticator)
    reads = []
    original = type(os.environ).__getitem__

    def tracked(environment, name):
        if name.startswith(("AXA_", "PHASEEVAL_")) or name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
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
    monkeypatch.setenv('AXA_WORKBENCH_DATASET', str(demo_store.directory / 'dataset'))
    monkeypatch.setenv('AXA_BGMN_DIR', str(binary_dir.parent))
    app = _configured_app(store=bench.store, authenticator=bench.authenticator)
    monkeypatch.setenv('PHASEEVAL_DATASET', '/missing-dataset')
    monkeypatch.setenv('PHASEEVAL_BGMN_DIR', '/changed-after-open')
    monkeypatch.setenv('AXA_WORKBENCH_DATASET', '/missing-dataset')
    monkeypatch.setenv('AXA_BGMN_DIR', '/changed-after-open')
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


def test_refit_captures_thread_settings_before_requests(bench, tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from phaseeval.refinement.bgmn.runner import BGMNRunner
    from phaseeval.refinement.bgmn.installer import BGMNInstaller
    from phaseeval.refinement.protocol import RefinementOutput
    from phaseeval.workbench import executors

    binary = tmp_path / 'bgmn'
    binary.write_bytes(b'fixture binary')
    monkeypatch.setattr(BGMNInstaller, 'is_installed', lambda self: True)
    monkeypatch.setattr(BGMNInstaller, 'get_bgmn_path', lambda self: binary)
    native_threads = []
    original_run = subprocess.run

    def native_run(command, **kwargs):
        if command[0] == str(binary):
            native_threads.append([kwargs['env'].get(name) for name in executors.THREAD_VARIABLES])
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
        return original_run(command, **kwargs)

    def native_refine(backend, request, config):
        backend._ensure_runner().run(request.working_dir / 'probe.sav')
        return RefinementOutput(hypothesis_id=request.hypothesis_id, working_dir=request.working_dir)

    monkeypatch.setattr(subprocess, 'run', native_run)
    monkeypatch.setattr(BGMNRunner, '_prepare_shadow_bgmnwin', lambda self, workdir, path: binary.parent)
    monkeypatch.setattr(executors.BGMNBackend, 'refine', native_refine)
    for name, value in zip(executors.THREAD_VARIABLES, ('2', '3', '4'), strict=True):
        monkeypatch.setenv(name, value)
    store = cli.open_store(tmp_path / 'live', tmp_path)
    root = store.load_case(OWNER, CASE, BLOBS)
    app = _configured_app(store=store, authenticator=bench.authenticator)
    for name in executors.THREAD_VARIABLES:
        monkeypatch.setenv(name, '9')
    reads = []
    original = type(os.environ).__getitem__

    def tracked(environment, name):
        if name.startswith(('AXA_', 'PHASEEVAL_')) or name in (*executors.THREAD_VARIABLES, 'PATH'):
            reads.append(name)
        return original(environment, name)

    with TestClient(app) as client:
        monkeypatch.setattr(type(os.environ), '__getitem__', tracked)
        response = client.post(f'/api/attempts/{root}/actions', headers=_headers(OWNER), json={
            'steps': [{'idempotency_key': 'add', 'action': ADD},
                      {'idempotency_key': 'fit', 'action': {'verb': 'refine'}}],
        })
        assert response.status_code == 201, response.text
        attempt = store.lineage(OWNER, response.json()['attempts'][-1]['digest'])[-1][1]
        assert attempt.outcome.status == 'completed'
        assert attempt.outcome.origin.manifest.environment.threads.model_dump() == {
            'omp_num_threads': '2', 'openblas_num_threads': '3', 'mkl_num_threads': '4',
        }
        assert native_threads == [['2', '3', '4']]
        assert reads == []


@pytest.mark.parametrize('current', [False, True])
def test_capabilities_use_legacy_bgmn_path_unless_phaseeval_overrides_it(bench, tmp_path, monkeypatch, current):
    from phaseeval.refinement.bgmn.installer import BGMNInstaller

    installed = tmp_path / 'installed' / 'BGMNwin'
    installed.mkdir(parents=True)
    names = BGMNInstaller.EXECUTABLES_WIN if os.name == 'nt' else BGMNInstaller.EXECUTABLES_UNIX
    for name in names:
        (installed / name).touch()
    monkeypatch.setenv('AXA_BGMN_DIR', str(installed.parent))
    monkeypatch.delenv('PHASEEVAL_BGMN_DIR', raising=False)
    if current:
        monkeypatch.setenv('PHASEEVAL_BGMN_DIR', str(tmp_path / 'missing'))
    app = _configured_app(store=bench.store, authenticator=bench.authenticator)
    response = TestClient(app).get('/api/capabilities', headers=_headers(OWNER))
    assert response.status_code == 200, response.text
    assert response.json()['refit']['available'] is (not current)
