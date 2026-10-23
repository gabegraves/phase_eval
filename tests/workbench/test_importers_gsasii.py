"""GSAS-II .gpx import through the bubblewrap sandbox.

The server process never unpickles a .gpx: every test here patches pickle in this process to
fail, so a passing import proves the load happened elsewhere. The sandbox's command line is
checked for what it shares (namespaces unshared, home and credentials not mounted, only /work
writable). Without bubblewrap the import refuses and the sandboxed tests skip. A real refined
project reads back when PHASEEVAL_GPX_FIXTURE names one and PHASEEVAL_GSASII_SOURCE names the
GSAS-II source; no project file is kept in this repository.
"""

from __future__ import annotations

import os
import pickle
import secrets
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from phaseeval.workbench.importers import gpx_reader, gsasii
from phaseeval.workbench.importers.gsasii import GpxImportError, GpxSandbox, SandboxUnavailable, read_gpx

FIXTURE = Path(os.environ.get("PHASEEVAL_GPX_FIXTURE", "")) if os.environ.get("PHASEEVAL_GPX_FIXTURE") else None
HAS_BWRAP = shutil.which("bwrap") is not None
HAS_GSASII = bool(os.environ.get("PHASEEVAL_GSASII_SOURCE"))
sandboxed = pytest.mark.skipif(not HAS_BWRAP, reason="bubblewrap not installed; see test_no_bubblewrap_refuses")
with_gsasii = pytest.mark.skipif(
    not (HAS_BWRAP and HAS_GSASII), reason="bubblewrap or PHASEEVAL_GSASII_SOURCE missing; the reader needs both"
)


@pytest.fixture
def server_never_unpickles(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a .gpx was unpickled in the server process")

    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    monkeypatch.setattr(pickle, "Unpickler", refuse)


@pytest.fixture
def sandbox_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GSAS-II source for the sandbox's command line; the tests that read a project need the real one."""
    monkeypatch.setenv("PHASEEVAL_GSASII_SOURCE", os.environ.get("PHASEEVAL_GSASII_SOURCE") or "/usr/share/gsasii")


def _binds(argv: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    ro, rw = [], []
    for i, arg in enumerate(argv):
        if arg in ("--ro-bind", "--ro-bind-try"):
            ro.append((argv[i + 1], argv[i + 2]))
        elif arg == "--bind":
            rw.append((argv[i + 1], argv[i + 2]))
    return ro, rw


@sandboxed
def test_the_sandbox_shares_only_the_runtime_the_input_and_one_temp_dir(tmp_path: Path, sandbox_env: None) -> None:
    inbox, work = tmp_path / "in", tmp_path / "work"
    argv = GpxSandbox.from_environment().argv(inbox, work)
    for flag in ("--unshare-all", "--clearenv", "--die-with-parent", "--new-session"):
        assert flag in argv  # no network, no inherited environment, no orphaned reader
    ro, rw = _binds(argv)
    assert rw == [(str(work), "/work")]  # the one writable path
    assert (str(inbox), "/in") in ro
    home = Path.home().resolve()
    private = [home / ".local/state", home / ".ssh", Path.cwd() / "data" / "custody", tmp_path]
    for source, _ in ro:
        src = Path(source).resolve()
        assert src != home, "the home directory is mounted"
        for path in private:
            assert not path.resolve().is_relative_to(src) or src == inbox.resolve(), f"{source} exposes {path}"


@with_gsasii
def test_a_file_that_is_not_a_project_is_refused(tmp_path: Path, server_never_unpickles: None) -> None:
    junk = tmp_path / "junk.gpx"
    junk.write_bytes(b"not a GSAS-II project")
    with pytest.raises(GpxImportError):
        read_gpx(junk)


def test_no_bubblewrap_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_never_unpickles: None, sandbox_env: None) -> None:
    monkeypatch.setattr(gsasii.shutil, "which", lambda name: None)
    project = tmp_path / "project.gpx"
    project.write_bytes(b"any bytes")
    with pytest.raises(SandboxUnavailable):
        read_gpx(project)


def test_without_a_gsasii_source_the_sandbox_is_not_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASEEVAL_GSASII_SOURCE", raising=False)
    with pytest.raises(GpxImportError, match="PHASEEVAL_GSASII_SOURCE"):
        GpxSandbox.from_environment().argv(tmp_path / "in", tmp_path / "work")


@pytest.mark.skipif(
    not (HAS_BWRAP and HAS_GSASII and FIXTURE and FIXTURE.is_file()),
    reason="no GSAS-II fixture project on this host (PHASEEVAL_GPX_FIXTURE, PHASEEVAL_GSASII_SOURCE)",
)
def test_a_refined_project_reads_back(server_never_unpickles: None) -> None:
    analysis = read_gpx(FIXTURE)
    assert analysis.saved_using  # the project's own record of what wrote it, apart from this reader
    assert analysis.phases
    for phase in analysis.phases:
        assert phase.space_group
        assert all(phase.cell[k] for k in ("length_a", "length_b", "length_c", "volume"))
        assert not phase.cell_refined or phase.cell_esd.get("length_a")  # s.u. only for a refined cell
    assert analysis.rwp_percent and analysis.gof
    (hist,) = analysis.histograms
    assert hist.x_quantity == "two_theta_deg" and hist.wavelength_angstrom
    n = len(hist.x)
    assert n > 100 and {len(hist.observed), len(hist.calculated), len(hist.background), len(hist.residual)} == {n}
    assert set(hist.mass_fractions) == {p.name for p in analysis.phases}
    derived = [f for f in hist.mass_fractions.values() if f.status == "derived"]
    assert derived and abs(sum(f.value for f in derived) - 1) < 1e-6
    assert len(analysis.source_sha256) == 64


def test_a_fixed_scale_keeps_the_su_its_normalized_fraction_inherits() -> None:
    """Scales .25/.75, only the first varied: the native propagation gives both normalized
    fractions s.u. .075, so the second keeps it."""
    scales = {"0:0:Scale", "1:0:Scale"}
    vary, dependent = {"0:0:Scale"}, set()
    first = gpx_reader.fraction_record(0.25, 0.075, "0:0:Scale", scales, vary, dependent, True)
    second = gpx_reader.fraction_record(0.75, 0.075, "1:0:Scale", scales, vary, dependent, True)
    assert (first["scale_status"], first["su"]) == ("varied", 0.075)
    assert (second["scale_status"], second["su"]) == ("fixed", 0.075)
    constrained = gpx_reader.fraction_record(0.75, 0.075, "1:0:Scale", scales, set(), {"1:0:Scale"}, True)
    assert (constrained["scale_status"], constrained["su"]) == ("dependent", 0.075)
    # every scale fixed: GSAS-II's zero s.u. is not reported as exactness
    frozen = gpx_reader.fraction_record(0.75, 0.0, "1:0:Scale", scales, set(), set(), True)
    assert frozen["su"] is None and frozen["value"] == 0.75
    # no saved covariance: no s.u. at all
    assert gpx_reader.fraction_record(0.75, 0.075, "1:0:Scale", scales, vary, dependent, False)["su"] is None


# --- a hostile .gpx ------------------------------------------------------------------------------
# A crafted .gpx runs code the moment GSAS-II unpickles it. The sandbox is judged by effects outside
# it: a secret planted outside the input is never read back, a loopback listener on the host never
# sees a connection, and no file appears outside the sandbox's temp directory. The identical file
# opened without the sandbox, in a child process, does all three, so the payload is live and the
# containment is what stops it. The payload touches only this test's canaries, listener and targets.

RAN = "phaseeval-hostile-gpx-payload-ran"

# Every probe reports on stderr, which read_gpx folds into its error, then the process exits 3 so
# the report is the last stderr line.
PAYLOAD = """
import os, socket, sys
spec = {spec!r}
read = []
for path in spec["canaries"]:
    try:
        with open(path) as fh:
            read.append(fh.read().strip())
    except OSError:
        pass
try:
    with socket.create_connection(("127.0.0.1", spec["port"]), timeout=3) as conn:
        conn.sendall(b"hello")
except OSError:
    pass
for path in spec["targets"]:
    try:
        with open(path, "w") as fh:
            fh.write("outside")
    except OSError:
        pass
sys.stderr.write({ran!r} + " read=" + ",".join(read) + "\\n")
sys.stderr.flush()
os._exit(3)
"""


class _Hostile:
    def __init__(self, code: str) -> None:
        self.code = code

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (exec, (self.code,))


@dataclass
class Bait:
    secret: str
    canaries: tuple[Path, ...]
    targets: tuple[Path, ...]
    listener: socket.socket
    gpx: Path

    def connected(self) -> bool:
        try:
            conn, _ = self.listener.accept()
        except TimeoutError:
            return False
        conn.close()
        return True


@pytest.fixture
def bait(tmp_path: Path) -> Iterator[Bait]:
    """Canaries in the test's tmp dir, beside the input and under $HOME; write targets in the same
    three places; a listener on the host's loopback. All removed afterwards."""
    secret, tag = secrets.token_hex(16), secrets.token_hex(6)
    source, cache = tmp_path / "source", Path.home() / ".cache"
    source.mkdir()
    cache.mkdir(exist_ok=True)
    canaries = (tmp_path / "canary.txt", source / "canary.txt", cache / f"phaseeval-gpx-canary-{tag}.txt")
    targets = (tmp_path / "escaped.txt", source / "escaped.txt", cache / f"phaseeval-gpx-escaped-{tag}.txt")
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(2)
    try:
        for canary in canaries:
            canary.write_text(secret)
        spec = {
            "canaries": [str(c) for c in canaries],
            "port": listener.getsockname()[1],
            "targets": [str(t) for t in targets],
        }
        gpx = source / "project.gpx"
        gpx.write_bytes(pickle.dumps(_Hostile(PAYLOAD.format(spec=spec, ran=RAN)), protocol=2))
        yield Bait(secret, canaries, targets, listener, gpx)
    finally:
        listener.close()
        for path in (*canaries, *targets):
            path.unlink(missing_ok=True)


@sandboxed
def test_unsandboxed_the_hostile_payload_reads_connects_and_writes(bait: Bait) -> None:
    """The positive control: without the sandbox the payload reaches every canary, the listener and
    every target, so the sandboxed test below cannot pass because the payload is inert."""
    run = subprocess.run(
        [sys.executable, "-I", "-c", "import pickle, sys; pickle.load(open(sys.argv[1], 'rb'))", str(bait.gpx)],
        capture_output=True, timeout=60, cwd=bait.gpx.parent, check=False,
    )
    report = run.stderr.decode("utf-8", "replace")
    assert run.returncode == 3 and RAN in report
    assert report.count(bait.secret) == len(bait.canaries)
    assert bait.connected()
    assert all(target.read_text() == "outside" for target in bait.targets)


@with_gsasii
def test_in_the_sandbox_the_hostile_payload_reads_connects_and_writes_nothing(
    bait: Bait, server_never_unpickles: None
) -> None:
    with pytest.raises(GpxImportError) as refused:
        read_gpx(bait.gpx)
    message = str(refused.value)
    assert "(exit 3)" in message and message.rstrip().endswith(f"{RAN} read=")  # it ran, in there
    assert bait.secret not in message
    assert not bait.connected()
    assert not any(target.exists() for target in bait.targets)
