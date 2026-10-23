"""GSAS-II .gpx import through the bubblewrap sandbox.

The server process never unpickles a .gpx: every test here patches pickle in this process to
fail, so a passing import proves the load happened elsewhere. The sandbox's command line is
checked for what it shares (namespaces unshared, home and credentials not mounted, only /work
writable on the host). Without bubblewrap the import refuses and the sandboxed tests skip. A real
refined project reads back when PHASEEVAL_GPX_FIXTURE names one and PHASEEVAL_GSASII_SOURCE names
the GSAS-II source; no project file is kept in this repository.

What a hostile file can cost the importing process is tested without bubblewrap, by standing a
flooding script in for the sandboxed reader: the read stops at the output cap with the parent's
memory still bounded, and the limits the sandbox sets on its processes hold.

Every test writes only under tmp_path: HOME, XDG_CACHE_HOME and TMPDIR point there.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import resource
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
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
MIB = 2**20


@pytest.fixture(autouse=True)
def only_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Nothing here writes outside tmp_path: the home directory, the cache directory and the
    temp directory read_gpx stages the project in all live under it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR
    yield home
    monkeypatch.setattr(tempfile, "tempdir", None)


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
def bait(tmp_path: Path, only_tmp_path: Path) -> Iterator[Bait]:
    """Canaries in the test's tmp dir, beside the input and under $HOME (itself under tmp_path);
    write targets in the same three places; a listener on the host's loopback."""
    secret, tag = secrets.token_hex(16), secrets.token_hex(6)
    source, cache = tmp_path / "source", only_tmp_path / ".cache"
    assert Path.home() == only_tmp_path
    source.mkdir()
    cache.mkdir()
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


# --- what a hostile file can cost the importing process -------------------------------------------
# Without bubblewrap, a script stands in for the sandboxed reader so the read loop and the limits
# are driven for real on any host: the sandbox command line is replaced, nothing else is.

STUB = """
import os, sys, time
open(sys.argv[1], "w").write(str(os.getpid()))
{body}
"""
FLOOD = "chunk = b'x' * (1 << 20)\nfor _ in range(600): os.write({fd}, chunk)\ntime.sleep(600)"
BIG_FILE = "with open(sys.argv[2], 'wb') as fh:\n    for _ in range(300): fh.write(b'z' * (1 << 20))\nos._exit(4)"
ALLOCATE = "try:\n    held = bytearray(2 << 30)\nexcept MemoryError:\n    os._exit(7)\nos._exit(0)"
WELL_BEHAVED = "sys.stderr.write('GSAS-II console noise\\n')\nsys.stdout.write(sys.argv[2])\nsys.stdout.flush()\nos._exit(0)"


@dataclass(frozen=True)
class _StubSandbox(GpxSandbox):
    """The same limits and read loop, with a script in place of bwrap and the reader."""

    body: str = ""
    pid_file: str = ""
    extra: str = ""

    def argv(self, inbox: Path, work: Path) -> list[str]:
        return [sys.executable, "-I", "-c", STUB.format(body=self.body), self.pid_file, self.extra]


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _stub(tmp_path: Path, body: str, extra: str = "", **limits: object) -> tuple[_StubSandbox, Path, Path]:
    pid_file = tmp_path / "reader.pid"
    project = tmp_path / "project.gpx"
    project.write_bytes(b"never unpickled here")
    sandbox = _StubSandbox(
        python=Path(sys.executable), gsasii_source=tmp_path, timeout_s=60.0,
        body=body, pid_file=str(pid_file), extra=extra, **limits,  # type: ignore[arg-type]
    )
    return sandbox, project, pid_file


def _gone(pid_file: Path) -> bool:
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_reader_that_floods_an_output_stream_is_killed_at_the_cap(
    tmp_path: Path, stream: str, server_never_unpickles: None
) -> None:
    """600 MB written to one stream: the read stops at 8 MiB, the stub (which would otherwise
    sleep for ten minutes) is gone at once, and this process has not held the flood."""
    fd = 1 if stream == "stdout" else 2
    sandbox, project, pid_file = _stub(
        tmp_path, FLOOD.format(fd=fd), max_output_bytes=8 * MIB, max_stderr_bytes=8 * MIB
    )
    before, started = _rss_bytes(), time.monotonic()
    with pytest.raises(GpxImportError, match=f"more than {8 * MIB} bytes to {stream}"):
        read_gpx(project, sandbox)
    elapsed, grown = time.monotonic() - started, _rss_bytes() - before
    assert elapsed < 20, f"the flood ran {elapsed:.1f} s; the cap did not stop it"
    assert grown < 64 * MIB, f"this process grew {grown / MIB:.0f} MiB holding the flood"
    assert _gone(pid_file)


def test_a_reader_cannot_write_a_file_past_the_limit(tmp_path: Path, server_never_unpickles: None) -> None:
    """RLIMIT_FSIZE holds inside: a 300 MB write is cut at 8 MiB and the reader dies for it."""
    target = tmp_path / "big"
    sandbox, project, _ = _stub(tmp_path, BIG_FILE, extra=str(target), max_file_bytes=8 * MIB)
    with pytest.raises(GpxImportError, match=r"could not be read \(exit"):
        read_gpx(project, sandbox)
    assert target.stat().st_size <= 8 * MIB


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux, where the sandbox runs")
def test_a_reader_cannot_allocate_past_the_memory_limit(tmp_path: Path, server_never_unpickles: None) -> None:
    sandbox, project, _ = _stub(tmp_path, ALLOCATE, max_memory_bytes=1 << 30)
    with pytest.raises(GpxImportError, match=r"\(exit 7\)"):
        read_gpx(project, sandbox)


def test_a_reader_under_the_caps_is_read_whole(tmp_path: Path, server_never_unpickles: None) -> None:
    """The bounded loop keeps every byte of an ordinary read: the stub's JSON validates into the
    analysis, with the digest taken from the file here, not from the reader."""
    payload = {
        "saved_using": "GSAS-II 5837", "reader_gsasii_version": "5837", "rwp_percent": 5.5, "gof": 1.2,
        "converged": True, "phases": [], "histograms": [],
    }
    sandbox, project, _ = _stub(tmp_path, WELL_BEHAVED, extra=json.dumps(payload))
    analysis = read_gpx(project, sandbox)
    assert analysis.rwp_percent == 5.5 and analysis.saved_using == "GSAS-II 5837"
    assert analysis.source_sha256 == hashlib.sha256(project.read_bytes()).hexdigest()


def test_the_sandbox_sizes_its_tmpfs_when_bubblewrap_can(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox_env: None) -> None:
    """With a bwrap that lists --size, every tmpfs is sized; the host-side writable path is still
    only /work, and the root and /dev are remounted read-only after setup."""
    fake = tmp_path / "bwrap"
    fake.write_text("#!/bin/sh\necho '  --size BYTES  Set size of next argument (only for --tmpfs)'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(gsasii.shutil, "which", lambda name: str(fake))
    gsasii._bwrap_sizes_tmpfs.cache_clear()
    sandbox = GpxSandbox.from_environment()
    monkeypatch.setattr(GpxSandbox, "runtime_paths", lambda self: (tmp_path / "runtime",))
    argv = sandbox.argv(tmp_path / "in", tmp_path / "work")
    tmpfs = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
    assert tmpfs == ["/dev/shm", "/tmp"]
    assert all(argv[i - 2 : i] == ["--size", str(sandbox.max_tmpfs_bytes)] for i, a in enumerate(argv) if a == "--tmpfs")
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--remount-ro"] == ["/dev", "/"]
    assert _binds(argv)[1] == [(str(tmp_path / "work"), "/work")]
    gsasii._bwrap_sizes_tmpfs.cache_clear()
