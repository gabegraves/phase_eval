"""The click-and-run launcher (src/phaseeval/launcher.py) as a user runs it: a real server process on a
free port over the bundled demo, signed in through the one-time nonce, stopped by Ctrl+C."""

from __future__ import annotations

import json
import http.cookiejar
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from phaseeval import launcher

REPO = Path(__file__).resolve().parents[2]


def _call(url: str, *, token: str | None = None, body: dict | None = None, headers: dict | None = None):
    sent = dict(headers or {})
    if token:
        sent["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body).encode()
    if data is not None:
        sent["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=sent, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _launch(config: Path, *extra: str) -> subprocess.Popen:
    # An inherited AXA_WORKBENCH_LOCAL_OWNER must not open the admin routes without a token (CR23).
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "AXA_WORKBENCH_LOCAL_OWNER": "1"}
    return subprocess.Popen(
        [sys.executable, "-m", "phaseeval.launcher", "--config", str(config), "--no-browser", *extra],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    state = tmp_path_factory.mktemp("phaseeval-state")
    config = state / "phaseeval.toml"
    config.write_text(
        'schema_version = "phaseeval-launcher.v1"\nport = 0\nstore = "demo"\ndataset = "demo"\n'
        f'tokens = "generate"\nbgmn = "none"\nstate_dir = "{state.as_posix()}"\n'
    )
    process = _launch(config)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"the launcher exited: {process.stdout.read()}")
        recorded = launcher.running_instance(state)
        if recorded:
            break
        time.sleep(0.5)
    else:
        process.kill()
        pytest.fail("the launcher did not come up within 180 s")
    tokens = {pid: (state / "session" / f"{pid}.token").read_text().strip() for pid in ("owner", "owner-rater")}
    yield {"base": f"http://127.0.0.1:{recorded['port']}", "port": recorded["port"], "tokens": tokens,
           "state": state, "config": config, "process": process}
    if process.poll() is None:
        process.kill()


def test_pages_and_health_load(served):
    for path in ("/api/health", "/dashboard", "/workbench", "/launcher"):
        assert _call(served["base"] + path)[0] == 200, path
    for path in ("/workbench", "/workbench/assets/workbench.js"):
        with urllib.request.urlopen(served["base"] + path) as response:
            assert response.headers["Cache-Control"] == "no-cache"


def test_auth_refuses_no_token_and_keeps_raters_off_admin_routes(served):
    base, tokens = served["base"], served["tokens"]
    assert _call(base + "/api/attempts/dashboard")[0] == 401
    assert _call(base + "/api/attempts/queue")[0] == 401
    assert _call(base + "/api/attempts/dashboard", token=tokens["owner-rater"])[0] == 403
    assert _call(base + "/api/attempts/dashboard", token=tokens["owner"])[0] == 200
    status, body = _call(base + "/api/attempts/queue", token=tokens["owner-rater"])
    assert status == 200 and len(json.loads(body)["items"]) == 6  # the six public demo answers


def test_scorecard_is_admin_only(served):
    base, tokens = served["base"], served["tokens"]
    if _call(base + "/scorecard")[0] == 404:
        pytest.skip("the scorecard (ovn/eval-scorecard) is not merged into this branch yet")
    assert _call(base + "/scorecard")[0] == 200
    assert _call(base + "/api/attempts/scorecard")[0] == 401
    assert _call(base + "/api/attempts/scorecard", token=tokens["owner-rater"])[0] == 403
    assert _call(base + "/api/attempts/scorecard", token=tokens["owner"])[0] == 200


def test_refit_is_off_without_bgmn(served):
    status, body = _call(served["base"] + "/api/capabilities", token=served["tokens"]["owner-rater"])
    refit = json.loads(body)["refit"]
    assert status == 200 and refit["available"] is False and "Profex-BGMN" in refit["note"]


def test_nonce_signs_in_once(served):
    base, tokens = served["base"], served["tokens"]
    assert _call(base + "/api/launcher/nonce", body={})[0] == 401
    assert _call(base + "/api/launcher/nonce", body={}, token=tokens["owner-rater"])[0] == 401
    status, body = _call(base + "/api/launcher/nonce", body={}, token=tokens["owner"])
    nonce = json.loads(body)["nonce"]
    assert status == 200
    status, body = _call(base + "/api/launcher/exchange", body={"nonce": nonce})
    assert status == 200
    landing = "/scorecard" if _call(base + "/scorecard")[0] == 200 else "/dashboard"
    assert json.loads(body)["next"] == landing
    assert json.loads(body)["session_storage"] == {
        "axa.dashboard.token": tokens["owner"],
        "axa.owner_token": tokens["owner-rater"],
    }
    assert _call(base + "/api/launcher/exchange", body={"nonce": nonce})[0] == 403  # spent
    assert _call(base + "/api/launcher/exchange", body={"nonce": "x" * 43})[0] == 403


def test_exchange_refuses_another_origin(served):
    base, tokens = served["base"], served["tokens"]
    nonce = json.loads(_call(base + "/api/launcher/nonce", body={}, token=tokens["owner"])[1])["nonce"]
    refused = _call(base + "/api/launcher/exchange", body={"nonce": nonce}, headers={"Origin": "http://evil.example"})
    assert refused[0] == 403


def test_new_tab_restores_only_review_access(served):
    base, tokens = served["base"], served["tokens"]
    jar = http.cookiejar.CookieJar()
    browser = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    nonce = json.loads(_call(base + "/api/launcher/nonce", body={}, token=tokens["owner"])[1])["nonce"]
    request = urllib.request.Request(base + "/api/launcher/exchange", data=json.dumps({"nonce": nonce}).encode(),
                                     headers={"Content-Type": "application/json"})
    browser.open(request).close()
    cookie, = jar
    assert cookie.name == f"phaseeval_review_{served['port']}"
    assert cookie.path == launcher.REVIEW_SESSION_PATH
    assert cookie.has_nonstandard_attr("HttpOnly")
    assert cookie.get_nonstandard_attr("SameSite") == "strict"
    assert cookie.value not in tokens.values()
    headers = {"Origin": base, "Cookie": f"{cookie.name}={cookie.value}"}
    for _ in range(2):
        status, body = _call(base + launcher.REVIEW_SESSION_PATH, body={}, headers=headers)
        assert status == 200 and json.loads(body) == {"token": tokens["owner-rater"]}
    assert _call(base + "/api/attempts/queue", headers=headers)[0] == 401
    assert _call(base + "/api/attempts/dashboard", token=json.loads(body)["token"])[0] == 403
    assert _call(base + "/api/launcher/nonce", body={}, headers=headers)[0] == 401
    for origin in ("http://evil.example", "http://127.0.0.1:3000", "null", ""):
        assert _call(base + launcher.REVIEW_SESSION_PATH, body={}, headers={**headers, "Origin": origin})[0] == 403
    assert _call(base + launcher.REVIEW_SESSION_PATH, body={}, headers={"Cookie": headers["Cookie"]})[0] == 403
    assert _call(base + launcher.REVIEW_SESSION_PATH, body={}, headers={"Origin": base})[0] == 401
    assert _call(base + launcher.REVIEW_SESSION_PATH, body={}, headers={**headers, "Cookie": f"{cookie.name}=wrong"})[0] == 401
    assert _call(base + launcher.REVIEW_SESSION_PATH, body={}, headers={**headers, "Host": "evil.example"})[0] == 403


def test_second_launch_reuses_the_running_server(served):
    second = _launch(served["config"])
    output, _ = second.communicate(timeout=120)
    assert second.returncode == 0
    assert f"already running on http://127.0.0.1:{served['port']}" in output
    assert launcher.running_instance(served["state"])["port"] == served["port"]


def test_review_session_expires_on_deadline_and_restart(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    base = "http://127.0.0.1:18828"
    session = launcher.Session(tmp_path / "principals.json", {"owner": "admin", "owner-rater": "rater"}, ())

    def server():
        app = FastAPI()
        nonces = launcher.Nonces()
        app.include_router(launcher.launcher_router(session, nonces, "test", 18828, lambda: "/workbench"))
        return TestClient(app, base_url=base, client=("127.0.0.1", 12345)), nonces

    client, nonces = server()
    response = client.post("/api/launcher/exchange", json={"nonce": nonces.mint()})
    assert response.status_code == 200
    headers = {"Origin": base}
    restored = client.post(launcher.REVIEW_SESSION_PATH, headers=headers)
    assert restored.status_code == 200 and restored.headers["cache-control"] == "no-store"
    restarted, _ = server()
    restarted.cookies.update(client.cookies)
    assert restarted.post(launcher.REVIEW_SESSION_PATH, headers=headers).status_code == 401
    clock = launcher.time.monotonic
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock() + launcher.REVIEW_SESSION_TTL + 1)
    assert client.post(launcher.REVIEW_SESSION_PATH, headers=headers).status_code == 401


def test_review_session_rejects_remote_clients(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    session = launcher.Session(tmp_path / "principals.json", {"owner": "admin", "owner-rater": "rater"}, ())
    app.include_router(launcher.launcher_router(session, launcher.Nonces(), "test", 18828, lambda: "/workbench"))
    client = TestClient(app, base_url="http://127.0.0.1:18828", client=("192.0.2.1", 12345))
    assert client.post(launcher.REVIEW_SESSION_PATH, headers={"Origin": "http://127.0.0.1:18828"}).status_code == 403


@pytest.mark.skipif(os.name == "nt", reason="Ctrl+C is a console event on Windows, not SIGINT to a pipe child")
def test_ctrl_c_stops_cleanly(served):
    process, state = served["process"], served["state"]
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=60) == 0
    assert not (state / "instance.json").exists()
    assert not any((state / "session").iterdir())  # this run's tokens are gone


def test_auto_bgmn_download_failure_leaves_refit_off(monkeypatch, tmp_path):
    from phaseeval.refinement.bgmn import installer

    def offline(self, force=False):
        raise RuntimeError("offline")

    monkeypatch.setattr(launcher, "bgmn_candidates", lambda: [])
    monkeypatch.setattr(launcher, "BGMN_HOME", tmp_path / "bgmn")
    monkeypatch.setattr(installer.BGMNInstaller, "install", offline)
    assert launcher.bgmn_dir("auto") is None


def test_bgmn_none_never_downloads(monkeypatch):
    from phaseeval.refinement.bgmn import installer

    monkeypatch.setattr(installer.BGMNInstaller, "install", lambda *_, **__: pytest.fail("downloaded BGMN"))
    assert launcher.bgmn_dir("none") is None


def test_an_arm64_mac_downloads_the_arm64_bgmn(monkeypatch):
    from phaseeval.refinement.bgmn import installer

    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    assert installer.BGMNInstaller().download_url().endswith("/bgmnwin_Darwin_m1.zip")
