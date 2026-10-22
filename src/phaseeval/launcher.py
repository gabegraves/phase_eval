"""PhaseEval click-and-run on macOS, Linux and Windows.

Each OS entry point (PhaseEval.command, phaseeval.sh, PhaseEval.bat) installs uv if missing and
runs launcher/run.py from the locked launcher environment, which calls main() here. Everything
after that is OS-neutral and reads one LaunchConfig from launcher/phaseeval.toml.

Sign-in without pasting and without a token in a URL: the server keeps this run's tokens in
memory and in 0600 files under the state dir. The launcher opens
http://127.0.0.1:<port>/launcher#n=<nonce>. A URL fragment never reaches the server or its access
log; the page drops it from the address bar, then POSTs it to /api/launcher/exchange, which
answers a loopback client once, within NONCE_TTL seconds, with the admin owner token and the
label-mode owner-rater token under the sessionStorage keys the dashboard and the review page read.
A spent or expired nonce is worthless, so a copy left in browser history signs nobody in.

One server per state dir: instance.json names the running port. A second launch finds it
healthy, asks it for a fresh nonce with this run's admin token, and opens the browser on it.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import shutil
import signal
import socket
import sys
import tarfile
import threading
import time
import tomllib
import urllib.request
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "launcher" / "phaseeval.toml"
DEMO_ARCHIVE = REPO / "launcher" / "demo" / "phaseeval-demo.tar.xz"
ASSETS = Path(__file__).resolve().parent / "api" / "assets"
HOST = "127.0.0.1"
PORTS = range(18800, 18900)
NONCE_TTL = 120.0
OWNER, RATER = "owner", "owner-rater"
BGMN_HOME = Path.home() / ".axa" / "bgmn"  # where a first-use download goes
# The sessionStorage key each page reads its bearer token from.
SESSION_KEYS = {"axa.dashboard.token": OWNER, "axa.owner_token": RATER}
# The page a sign-in opens: the scorecard, which absorbs the dashboard, wherever the app serves it.
LANDINGS = ("/scorecard", "/dashboard")
INSTANCE_SCHEMA = "phaseeval-instance.v1"


# --- the launcher config ---------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchConfig:
    """launcher/phaseeval.toml. Every OS entry point reads this and only this."""

    port: int  # 0: the first free port in PORTS
    store: Path | Literal["demo"]  # "demo": the bundled public demo, unpacked under the state dir
    dataset: Path | Literal["demo"]  # the export the scorecard reads
    tokens: Path | Literal["generate"]  # a dir with principals.json, owner.token, owner-rater.token
    bgmn: Path | Literal["auto", "none"]  # a BGMN install dir or executable
    state_dir: Path

    @classmethod
    def load(cls, path: Path) -> LaunchConfig:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "phaseeval-launcher.v1":
            raise SystemExit(f"{path}: schema_version must be phaseeval-launcher.v1")

        def setting[K: str](name: str, keywords: tuple[K, ...]) -> Path | K:
            value = str(raw.get(name, keywords[0]))
            if value in keywords:
                return value  # type: ignore[return-value]
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                raise SystemExit(f"{path}: {name} must be one of {keywords} or an absolute path")
            return resolved

        port = int(raw.get("port", 0))
        if port and not 1024 <= port <= 65535:
            raise SystemExit(f"{path}: port must be 0 or 1024 to 65535")
        state = str(raw.get("state_dir", ""))
        return cls(
            port=port,
            store=setting("store", ("demo",)),
            dataset=setting("dataset", ("demo",)),
            tokens=setting("tokens", ("generate",)),
            bgmn=setting("bgmn", ("auto", "none")),
            state_dir=Path(state).expanduser() if state else default_state_dir(),
        )


def default_state_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PhaseEval"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "PhaseEval"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "phaseeval"


# --- store, tokens, BGMN ---------------------------------------------------------------------


def demo_paths(state: Path, archive: Path = DEMO_ARCHIVE) -> tuple[Path, Path]:
    """Unpack the bundled demo once per archive version; labels made in it stay in the state dir."""
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    target = state / f"demo-{digest[:12]}"
    if not (target / ".complete").is_file():
        print("Unpacking the demo store (first launch only)...", flush=True)
        staging = state / f".demo-{secrets.token_hex(4)}"
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(staging, filter="data")
        (staging / ".complete").write_text(digest + "\n")
        shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)
    return target / "store", target / "dataset"


@dataclass(frozen=True, repr=False)
class Session:
    principals: Path
    tokens: dict[str, str]  # principal id -> bearer token, for OWNER and RATER
    generated: tuple[Path, ...]  # files this run wrote and removes on exit


def _write_private(path: Path, text: str) -> None:
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def session_tokens(config: LaunchConfig) -> Session:
    if config.tokens != "generate":
        directory = config.tokens
        return Session(
            principals=directory / "principals.json",
            tokens={pid: (directory / f"{pid}.token").read_text().strip() for pid in (OWNER, RATER)},
            generated=(),
        )
    directory = config.state_dir / "session"
    directory.mkdir(parents=True, exist_ok=True)
    tokens = {OWNER: secrets.token_urlsafe(32), RATER: secrets.token_urlsafe(32)}
    roles = {OWNER: ["owner"], RATER: ["owner_rater"]}
    registry = {
        "schema_version": "axa.api_principals.v1",
        "principals": [
            {
                "principal_id": pid,
                "principal_type": "user",
                "roles": roles[pid],
                "credential_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }
            for pid, token in tokens.items()
        ],
    }
    files = [directory / "principals.json"] + [directory / f"{pid}.token" for pid in tokens]
    _write_private(files[0], json.dumps(registry, indent=2) + "\n")
    for pid, token in tokens.items():
        _write_private(directory / f"{pid}.token", token + "\n")
    return Session(principals=files[0], tokens=tokens, generated=tuple(files))


def bgmn_candidates() -> list[Path]:
    """Where a BGMN install may be: BGMNInstaller's own dir, then Profex-BGMN's layouts [U]."""
    found = [BGMN_HOME]
    if sys.platform == "darwin":
        for apps in (Path("/Applications"), Path.home() / "Applications"):
            found += [p / "Contents" / "Resources" for p in sorted(apps.glob("Profex*/Profex.app"))]
            found += [p / "Contents" / "Resources" for p in sorted(apps.glob("Profex*.app"))]
    elif os.name == "nt":
        for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                found += sorted(Path(base).glob("Profex*")) + sorted(Path(base).glob("Programs/Profex*"))
    on_path = shutil.which("bgmn")
    if on_path:
        found.append(Path(on_path).resolve().parent.parent)
    return found


def bgmn_dir(setting: Path | Literal["auto", "none"]) -> Path | None:
    """The install dir BGMNInstaller accepts (it holds BGMNwin/bgmn, teil, eflech, output).
    "auto" downloads BGMN once into ~/.axa/bgmn when no install is found; a failed download
    leaves refit off. BGMN is never bundled."""
    from phaseeval.refinement.bgmn.installer import BGMNInstaller

    if setting == "none":
        return None
    if setting == "auto":
        candidates = bgmn_candidates()
    else:  # an install dir, its BGMNwin dir, or the bgmn executable itself
        candidates = [setting, setting.parent, setting.parent.parent]
    found = next((c for c in candidates if BGMNInstaller(install_dir=c).is_installed()), None)
    if found is not None or setting != "auto":
        return found
    installer = BGMNInstaller(install_dir=BGMN_HOME)
    print(
        "Refit needs BGMN, third-party Rietveld software by J. Bergmann (bgmn.de), not part of "
        f"PhaseEval. Downloading it once from {installer.download_url()} into {BGMN_HOME} "
        '(set bgmn = "none" in launcher/phaseeval.toml to skip)...',
        flush=True,
    )
    try:
        installer.install()
    except Exception as error:  # any failure leaves refit off, never the server down
        print(f"BGMN download failed ({error}). Refit is off until a relaunch downloads it.", flush=True)
        return None
    return BGMN_HOME if installer.is_installed() else None


def bind(preferred: int) -> socket.socket:
    """Hold the listening socket from here on, so no other process takes the port first."""
    for port in [preferred] if preferred else PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((HOST, port))
        except OSError:
            sock.close()
            continue
        sock.listen(128)
        return sock
    raise SystemExit(f"no free port in {preferred or f'{PORTS.start} to {PORTS.stop - 1}'}")


# --- the nonce exchange ----------------------------------------------------------------------


class Nonces:
    """One-time sign-in codes: stored as hashes, spent on first use, dead after NONCE_TTL."""

    def __init__(self) -> None:
        self._live: dict[str, float] = {}
        self._lock = threading.Lock()

    def mint(self) -> str:
        nonce = secrets.token_urlsafe(32)
        with self._lock:
            now = time.monotonic()
            self._live = {h: t for h, t in self._live.items() if t > now}
            self._live[hashlib.sha256(nonce.encode()).hexdigest()] = now + NONCE_TTL
        return nonce

    def spend(self, nonce: str) -> bool:
        supplied = hashlib.sha256(nonce.encode()).hexdigest()
        with self._lock:
            match = next((h for h in self._live if hmac.compare_digest(h, supplied)), None)
            if match is None:
                return False
            return self._live.pop(match) > time.monotonic()


class Exchange(BaseModel):
    nonce: str


class Landing:
    """The page a sign-in lands on: the first of LANDINGS the running server answers 200 for.
    Probed over HTTP on first use, because FastAPI 0.161 keeps included routers lazy, so
    app.routes does not list their paths."""

    def __init__(self, port: int) -> None:
        self._port, self._path, self._lock = port, "", threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            if not self._path:
                self._path = next((p for p in LANDINGS if self._answers(p)), LANDINGS[-1])
            return self._path

    def _answers(self, path: str) -> bool:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{self._port}{path}", timeout=10) as response:
                return response.status == 200
        except OSError:
            return False


def launcher_router(session: Session, nonces: Nonces, instance: str, port: int, landing: Landing) -> APIRouter:
    router = APIRouter(include_in_schema=False)
    origin = f"http://{HOST}:{port}"
    no_store = {"Cache-Control": "no-store"}

    def local_only(request: Request) -> None:
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1"):
            raise HTTPException(status_code=403, detail="the launcher answers this computer only")
        if request.headers.get("origin") not in (None, origin):
            raise HTTPException(status_code=403, detail="cross-origin sign-in refused")

    @router.get("/launcher")
    def page() -> FileResponse:
        return FileResponse(ASSETS / "launcher.html", media_type="text/html", headers=no_store)

    @router.get("/api/launcher/instance")
    def get_instance(request: Request) -> dict[str, str]:
        local_only(request)
        return {"instance": instance}

    @router.post("/api/launcher/exchange")
    def exchange(body: Exchange, request: Request) -> JSONResponse:
        local_only(request)
        if not nonces.spend(body.nonce):
            raise HTTPException(status_code=403, detail="sign-in code already used or expired")
        storage = {key: session.tokens[pid] for key, pid in SESSION_KEYS.items()}
        return JSONResponse({"session_storage": storage, "next": landing()}, headers=no_store)

    @router.post("/api/launcher/nonce")
    def nonce(request: Request, authorization: str | None = Header(default=None)) -> dict[str, str]:
        local_only(request)
        expected = f"Bearer {session.tokens[OWNER]}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="the admin token of this run is required")
        return {"nonce": nonces.mint()}

    return router


# --- a running instance ----------------------------------------------------------------------


def _request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.loads(response.read())


def running_instance(state: Path) -> dict | None:
    """instance.json, if its server answers with the same instance id."""
    try:
        recorded = json.loads((state / "instance.json").read_text())
        answer = _request(f"http://{HOST}:{recorded['port']}/api/launcher/instance")
    except (OSError, ValueError, KeyError):
        return None
    return recorded if answer.get("instance") == recorded.get("instance") else None


def sign_in_url(port: int, nonce: str) -> str:
    return f"http://{HOST}:{port}/launcher#n={nonce}"


def open_browser(port: int, nonce: str, print_link: bool) -> None:
    url = sign_in_url(port, nonce)
    if print_link:
        print(f"One-time sign-in link (works once, for {int(NONCE_TTL)} s, on this computer): {url}", flush=True)
    elif not webbrowser.open(url):
        print("No browser could be opened. Run the launcher again with --print-link.", flush=True)


def reuse(config: LaunchConfig, recorded: dict, args: argparse.Namespace) -> int:
    tokens = config.state_dir / "session" if config.tokens == "generate" else config.tokens
    token = (tokens / f"{OWNER}.token").read_text().strip()
    answer = _request(
        f"http://{HOST}:{recorded['port']}/api/launcher/nonce",
        data=b"{}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    print(f"PhaseEval is already running on http://{HOST}:{recorded['port']}; opening it.", flush=True)
    if not args.no_browser:
        open_browser(recorded["port"], answer["nonce"], args.print_link)
    return 0


# --- serve -----------------------------------------------------------------------------------


def serve(config: LaunchConfig, args: argparse.Namespace) -> int:
    state = config.state_dir
    state.mkdir(parents=True, exist_ok=True)
    recorded = running_instance(state)
    if recorded is not None:
        return reuse(config, recorded, args)

    store, dataset = config.store, config.dataset
    if store == "demo" or dataset == "demo":
        demo_store, demo_dataset = demo_paths(state)
        store = demo_store if store == "demo" else store
        dataset = demo_dataset if dataset == "demo" else dataset
    bgmn = bgmn_dir(config.bgmn)
    session = session_tokens(config)
    environment = {
        "AXA_API_PRINCIPALS_FILE": str(session.principals),
        "AXA_WORKBENCH_ROOT": str(store),
        "AXA_WORKBENCH_DATASET": str(dataset),
    }
    os.environ.update(environment)
    os.environ.pop("AXA_WORKBENCH_CUSTODY", None)  # the launcher never reads custody
    # Tokenless owner access stays off: the launcher signs in through the nonce (CR23).
    os.environ.pop("AXA_WORKBENCH_LOCAL_OWNER", None)
    # BGMNInstaller reads AXA_BGMN_DIR; with no BGMN it names an empty dir, so a ~/.axa/bgmn the
    # config turned off ("none") stays off.
    os.environ["AXA_BGMN_DIR"] = str(bgmn if bgmn is not None else state / "no-bgmn")

    import uvicorn

    from phaseeval.api.app import app

    sock = bind(args.port or config.port)
    port = sock.getsockname()[1]
    landing = Landing(port)
    nonces, instance = Nonces(), secrets.token_hex(8)
    app.include_router(launcher_router(session, nonces, instance, port, landing))
    # Launcher routes go first, ahead of any catch-all static mount.
    launcher_paths = {"/launcher", "/api/launcher/instance", "/api/launcher/exchange", "/api/launcher/nonce"}
    app.router.routes.sort(key=lambda route: getattr(route, "path", "") not in launcher_paths)

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_keep_alive=600))
    instance_file = state / "instance.json"
    _write_private(
        instance_file,
        json.dumps({"schema_version": INSTANCE_SCHEMA, "instance": instance, "port": port, "pid": os.getpid()}) + "\n",
    )
    print(f"PhaseEval: http://{HOST}:{port}", flush=True)
    print(f"store: {store}", flush=True)
    print(f"refit: {'BGMN at ' + str(bgmn) if bgmn else 'off (BGMN not found)'}", flush=True)
    print("Close this window or press Ctrl+C to stop.", flush=True)

    def opened() -> None:
        while not server.started and not server.should_exit:
            time.sleep(0.05)
        if server.started:
            print(f"Signed-in page: http://{HOST}:{port}{landing()}", flush=True)
        if server.started and not args.no_browser:
            open_browser(port, nonces.mint(), args.print_link)

    def stop(*_: object) -> None:
        server.should_exit = True

    threading.Thread(target=opened, daemon=True).start()
    # uvicorn handles SIGINT and SIGTERM while serving, then re-raises the signal it caught into
    # the handler it found; this one lets the cleanup below run instead of dying mid-way. SIGHUP
    # (a closed terminal window) and Windows' SIGBREAK stop the server the same way.
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    try:
        server.run(sockets=[sock])
    finally:
        try:
            if json.loads(instance_file.read_text()).get("instance") == instance:
                instance_file.unlink()
        except (OSError, ValueError):
            pass
        for path in session.generated:
            path.unlink(missing_ok=True)
        print("PhaseEval stopped.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phaseeval", description="Run PhaseEval on this computer.")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("PHASEEVAL_CONFIG", CONFIG)))
    parser.add_argument("--port", type=int, default=0, help="override the config's port")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--print-link", action="store_true", help="print a one-time sign-in link instead of opening a browser"
    )
    args = parser.parse_args(argv)
    return serve(LaunchConfig.load(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
