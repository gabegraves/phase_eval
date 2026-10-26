"""The click-and-run launcher (src/phaseeval/launcher.py) as a user runs it: a real server process on a
free port over the bundled demo, signed in through the one-time nonce, stopped by Ctrl+C."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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


def _launch(config: Path, *extra: str, stdout=subprocess.PIPE) -> subprocess.Popen:
    # An inherited AXA_WORKBENCH_LOCAL_OWNER must not open the admin routes without a token (CR23).
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "AXA_WORKBENCH_LOCAL_OWNER": "1"}
    return subprocess.Popen(
        [sys.executable, "-m", "phaseeval.launcher", "--config", str(config), "--no-browser", *extra],
        cwd=REPO, env=env, stdout=stdout, stderr=subprocess.STDOUT, text=True,
    )


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    state = tmp_path_factory.mktemp("phaseeval-state")
    config = state / "phaseeval.toml"
    config.write_text(
        'schema_version = "phaseeval-launcher.v1"\nport = 0\nstore = "demo"\ndataset = "demo"\n'
        f'tokens = "generate"\nbgmn = "none"\nstate_dir = "{state.as_posix()}"\n'
    )
    log = state / "launcher.log"
    with log.open("w") as output:
        process = _launch(config, "--print-link", stdout=output)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"the launcher exited: {log.read_text()}")
        recorded = launcher.running_instance(state)
        if recorded and _call(f"http://127.0.0.1:{recorded['port']}/api/health")[0] == 200:
            break
        time.sleep(0.5)
    else:
        process.kill()
        pytest.fail("the launcher did not come up within 180 s")
    tokens = {pid: (state / "session" / f"{pid}.token").read_text().strip() for pid in ("owner", "owner-rater")}
    yield {"base": f"http://127.0.0.1:{recorded['port']}", "port": recorded["port"], "tokens": tokens,
           "state": state, "config": config, "process": process, "log": log}
    if process.poll() is None:
        process.kill()


def test_pages_and_health_load(served):
    for path in ("/api/health", "/dashboard", "/workbench", "/launcher"):
        assert _call(served["base"] + path)[0] == 200, path


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
    assert _call(base + "/scorecard")[0] == 200
    assert _call(base + "/api/attempts/scorecard")[0] == 401
    assert _call(base + "/api/attempts/scorecard", token=tokens["owner-rater"])[0] == 403
    assert _call(base + "/api/attempts/scorecard", token=tokens["owner"])[0] == 200


def test_refit_is_off_without_bgmn(served):
    status, body = _call(served["base"] + "/api/capabilities", token=served["tokens"]["owner-rater"])
    refit = json.loads(body)["refit"]
    assert status == 200 and refit["available"] is False and "Profex-BGMN" in refit["note"]
    assert "interactive prompt" in refit["note"]
    assert 'bgmn = "download"' in refit["note"]


def test_the_review_tab_holds_only_the_rater_token_and_lands_on_the_workbench(served):
    base, tokens = served["base"], served["tokens"]
    assert _call(base + "/api/launcher/nonce", body={})[0] == 401
    assert _call(base + "/api/launcher/nonce", body={}, token=tokens["owner-rater"])[0] == 401
    status, body = _call(base + "/api/launcher/nonce", body={}, token=tokens["owner"])
    nonce = json.loads(body)["nonce"]
    assert status == 200
    status, body = _call(base + "/api/launcher/exchange", body={"nonce": nonce})
    assert status == 200
    assert json.loads(body) == {
        "role": "rater",
        "next": "/workbench",
        "session_storage": {"phaseeval.rater_token": tokens["owner-rater"]},
    }
    tab = _page_script(served, "/launcher", {"phaseeval.admin_token": tokens["owner"]}, json.loads(body))
    assert tab["storage"] == {"phaseeval.rater_token": tokens["owner-rater"]}
    assert tab["destinations"] == ["/workbench"]
    assert _call(base + "/api/launcher/exchange", body={"nonce": nonce})[0] == 403  # spent
    assert _call(base + "/api/launcher/exchange", body={"nonce": "x" * 43})[0] == 403


@pytest.mark.parametrize("launch", ["fresh", "reuse"])
def test_the_admin_link_mints_its_own_code_for_a_separate_tab(served, launch):
    if launch == "reuse":
        second = _launch(served["config"], "--print-link")
        output, _ = second.communicate(timeout=120)
        assert second.returncode == 0
    else:
        deadline = time.monotonic() + 10
        while True:
            output = served["log"].read_text()
            if output.count("/launcher#n=") >= 2 or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    links = re.findall(r"http://127\.0\.0\.1:\d+/launcher#n=\S+", output)
    assert len(links) == 2, output
    assert len(set(links)) == 2
    sessions = []
    for link in links:
        nonce = parse_qs(urlsplit(link).fragment)["n"][0]
        status, body = _call(served["base"] + "/api/launcher/exchange", body={"nonce": nonce})
        assert status == 200
        signed_in = json.loads(body)
        sessions.append(signed_in)
        tab = _page_script(served, "/launcher", {}, signed_in)
        assert tab["storage"] == signed_in["session_storage"]
        assert tab["destinations"] == [signed_in["next"]]
        assert _call(served["base"] + "/api/launcher/exchange", body={"nonce": nonce})[0] == 403
    assert sorted(sessions, key=lambda s: s["role"]) == [
        {"role": "admin", "session_storage": {"phaseeval.admin_token": served["tokens"]["owner"]}, "next": "/scorecard"},
        {"role": "rater", "session_storage": {"phaseeval.rater_token": served["tokens"]["owner-rater"]}, "next": "/workbench"},
    ]


def _page_script(served, path, storage, response=None):
    status, body = _call(served["base"] + path)
    assert status == 200
    if not path.endswith(".js"):
        status, body = _call(served["base"] + re.search(r'<script src="([^"]+)"></script>', body.decode())[1])
        assert status == 200
    script = body.decode()
    result = subprocess.run(
        ["node", "-e", """
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const storage = new Map(Object.entries(input.storage));
const element = {classList: {add(){}, remove(){}, toggle(){}}, addEventListener(){}};
const calls = [], destinations = [];
vm.runInNewContext(input.script, {
  sessionStorage: {getItem: k => storage.get(k) || null, setItem: (k,v) => storage.set(k,v), removeItem: k => storage.delete(k)},
  document: {getElementById: () => element, addEventListener(){}, querySelectorAll: () => []},
  window: {addEventListener(){}}, location: {hash: '#n=code', pathname: '/launcher', search: '', replace: p => destinations.push(p)},
  history: {replaceState(){}}, URLSearchParams, setInterval(){},
  fetch: (url, options) => {
    calls.push({url, options});
    return input.response ? Promise.resolve({ok: true, json: async () => input.response}) : new Promise(() => {});
  },
});
setImmediate(() => process.stdout.write(JSON.stringify({storage: Object.fromEntries(storage), calls, destinations})));
"""],
        input=json.dumps({"script": script, "storage": storage, "response": response}),
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("role", ["rater", "admin"])
def test_sign_in_replaces_all_old_tab_credentials(served, role):
    storage = {"axa.owner_token": "old-rater", "axa.dashboard.token": "old-admin",
               "phaseeval.rater_token": "stale-rater", "phaseeval.admin_token": "stale-admin"}
    response = {"role": role, "session_storage": {f"phaseeval.{role}_token": "fresh"},
                "next": "/workbench" if role == "rater" else "/scorecard"}
    result = _page_script(served, "/launcher", storage, response)
    assert result["storage"] == response["session_storage"]
    assert result["destinations"] == [response["next"]]


@pytest.mark.parametrize("page, role, old_key", [
    ("/workbench/assets/workbench.js", "rater", "axa.owner_token"),
    ("/scorecard", "admin", "axa.dashboard.token"),
])
@pytest.mark.parametrize("current", [None, "current"])
def test_pages_migrate_only_their_own_legacy_token_once(served, page, role, old_key, current):
    key = f"phaseeval.{role}_token"
    storage = {"axa.owner_token": "old-rater", "axa.dashboard.token": "old-admin",
               "phaseeval.admin_token" if role == "rater" else "phaseeval.rater_token": "other-role"}
    if current:
        storage[key] = current
    result = _page_script(served, page, storage)
    expected = current or storage[old_key]
    assert result["storage"] == {key: expected}
    assert result["calls"][0]["options"]["headers"]["Authorization"] == f"Bearer {expected}"
    result["storage"][old_key] = "stale"
    assert _page_script(served, page, result["storage"])["storage"] == {key: expected}


def test_exchange_refuses_another_origin(served):
    base, tokens = served["base"], served["tokens"]
    nonce = json.loads(_call(base + "/api/launcher/nonce", body={}, token=tokens["owner"])[1])["nonce"]
    refused = _call(base + "/api/launcher/exchange", body={"nonce": nonce}, headers={"Origin": "http://evil.example"})
    assert refused[0] == 403
    assert _call(base + "/api/launcher/exchange", body={"nonce": nonce})[0] == 200


@pytest.mark.parametrize("role", ["rater", "admin"])
def test_expired_codes_are_refused_for_both_roles(monkeypatch, role):
    monkeypatch.setattr(launcher.time, "monotonic", lambda: 0)
    nonces = launcher.Nonces({launcher.OWNER: "admin", launcher.RATER: "rater"})
    code = nonces.mint(role)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: launcher.NONCE_TTL)
    with pytest.raises(launcher.HTTPException) as refused:
        nonces.exchange(code)
    assert refused.value.status_code == 403


def test_second_launch_reuses_the_running_server(served):
    second = _launch(served["config"])
    output, _ = second.communicate(timeout=120)
    assert second.returncode == 0
    assert f"already running on http://127.0.0.1:{served['port']}" in output
    assert launcher.running_instance(served["state"])["port"] == served["port"]


@pytest.mark.skipif(os.name == "nt", reason="Ctrl+C is a console event on Windows, not SIGINT to a pipe child")
def test_ctrl_c_stops_cleanly(served):
    process, state = served["process"], served["state"]
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=60) == 0
    assert not (state / "instance.json").exists()
    assert not any((state / "session").iterdir())  # this run's tokens are gone


def test_bgmn_download_failure_leaves_refit_off(monkeypatch, tmp_path):
    from phaseeval.refinement.bgmn import installer

    def offline(self, force=False):
        raise RuntimeError("offline")

    monkeypatch.setattr(launcher, "bgmn_candidates", lambda: [])
    monkeypatch.setattr(launcher, "BGMN_HOME", tmp_path / "bgmn")
    monkeypatch.setattr(installer.BGMNInstaller, "install", offline)
    assert launcher.bgmn_dir("download") is None


def test_bgmn_none_never_downloads(monkeypatch):
    from phaseeval.refinement.bgmn import installer

    monkeypatch.setattr(installer.BGMNInstaller, "install", lambda *_, **__: pytest.fail("downloaded BGMN"))
    assert launcher.bgmn_dir("none") is None


def test_an_arm64_mac_downloads_the_arm64_bgmn(monkeypatch):
    from phaseeval.refinement.bgmn import installer

    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    assert installer.BGMNInstaller().download_url().endswith("/bgmnwin_Darwin_m1.zip")


def test_bgmn_ask_without_a_tty_downloads_nothing(monkeypatch, tmp_path):
    from phaseeval.refinement.bgmn import installer

    config = tmp_path / "config.toml"
    config.write_text('schema_version = "phaseeval-launcher.v1"\n')
    downloads = []
    monkeypatch.setattr(launcher, "bgmn_candidates", lambda: [])
    monkeypatch.setattr(launcher, "BGMN_HOME", tmp_path / "bgmn")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(installer.BGMNInstaller, "install", lambda *_, **__: downloads.append(True))
    assert launcher.bgmn_dir(launcher.LaunchConfig.load(config).bgmn) is None
    assert downloads == [], "attempted a BGMN download without interactive consent"
    assert not (tmp_path / "bgmn").exists()


def test_bgmn_defaults_to_ask(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('schema_version = "phaseeval-launcher.v1"\n')
    assert launcher.LaunchConfig.load(config).bgmn == "ask"
    assert launcher.LaunchConfig.load(REPO / "launcher" / "phaseeval.toml").bgmn == "ask"


@pytest.mark.parametrize("answer, downloads", [("y", True), ("", False), ("no", False)])
def test_bgmn_ask_shows_the_pin_and_respects_the_answer(monkeypatch, tmp_path, capsys, answer, downloads):
    from phaseeval.refinement.bgmn import installer

    calls = []
    monkeypatch.setattr(launcher, "bgmn_candidates", lambda: [])
    monkeypatch.setattr(launcher, "BGMN_HOME", tmp_path / "bgmn")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    monkeypatch.setattr(installer.BGMNInstaller, "install", lambda *_, **__: calls.append(True))
    assert launcher.bgmn_dir("ask") is None
    assert bool(calls) == downloads
    row = installer.BGMNInstaller().archive
    output = capsys.readouterr().out
    assert row.url in output and row.sha256 in output and str(row.size_bytes) in output
