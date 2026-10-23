"""Finished GSAS-II projects as reviewable analyses.

A .gpx is a sequence of Python pickles (GSASIIscriptable.SaveDictToProjFile), so loading one runs
whatever code the file names. The server never unpickles one. read_gpx copies the file into a
fresh directory and runs gpx_reader.py under bubblewrap: every namespace unshared (no network),
only the Python runtime and the GSAS-II source mounted, read-only, the copy read-only at /in, and
one empty temp directory writable at /work (HOME and the working directory). Home, data/custody
and credentials are not mounted. Without bubblewrap the import refuses; it never falls back to an
in-process load. The reader's JSON is validated into GpxAnalysis before anything uses it.

What the file can cost this process is bounded too. Its stdout and stderr are read as they
arrive, each against a hard cap, and the sandbox is killed the moment either is passed, so a
flood never sits in memory here. Inside, every writable tmpfs has a size, each file the reader
writes has a size limit (RLIMIT_FSIZE), its address space is limited (RLIMIT_AS), and the read
has a deadline.

What the file can learn about this host is bounded as well. bwrap is started with an empty
environment (--clearenv only strips the reader's; bwrap's own process kept the importer's, and
/proc/1/environ showed it to the reader), and the reader is PID 1 of its namespace (--as-pid-1),
so no process of the sandbox carries the host's environment or bwrap's command line. What the
reader prints is not trusted either: a hostile project owns the reader's process, so every
string of its output has a length limit and a character whitelist here, and one past them
refuses the import. The only one of them that is stored, saved_using, is a short version string.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, ValidationError

from ..model import Digest, Record

READER = Path(__file__).with_name("gpx_reader.py")
# The GSAS-II source directory the sandbox mounts read-only (it holds the GSASII package). There is
# no default: the import refuses until the environment names it.
GSASII_SOURCE_VAR = "PHASEEVAL_GSASII_SOURCE"
GPX_PYTHON_VAR = "PHASEEVAL_GPX_PYTHON"
MAX_GPX_BYTES = 256 * 2**20
_CHUNK = 1 << 16


class GpxImportError(ValueError):
    """The project could not be read: not a GSAS-II project, a refused or failed load, or output
    the importer does not accept. The message carries the sandbox's stderr tail only."""


class SandboxUnavailable(RuntimeError):
    """bubblewrap is missing or cannot create its namespaces on this host."""


def _printable(value: str) -> str:
    """str.isprintable: letters, marks, numbers, punctuation, symbols and the space, nothing else
    (no control, format, separator or unassigned characters)."""
    if not value.isprintable():
        raise ValueError("holds a character that is not printable")
    return value


# Every string the reader returns, bounded. A version is ASCII, short, and never a path.
VersionText = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._+:()-]{0,63}$")]
NameText = Annotated[str, Field(min_length=1, max_length=200), AfterValidator(_printable)]  # a phase or histogram
SpaceGroupText = Annotated[str, Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9 /_:()-]+$")]
InstrumentText = Annotated[str, Field(min_length=1, max_length=16, pattern=r"^[A-Za-z0-9 _-]+$")]
DetailText = Annotated[str, Field(max_length=500), AfterValidator(_printable)]
BasisText = Annotated[str, Field(max_length=2000), AfterValidator(_printable)]
CellKey = Annotated[str, Field(pattern=r"^[a-z_]{1,32}$")]  # length_a..c, angle_alpha..gamma, volume


class GpxPhase(Record):
    name: NameText
    space_group: SpaceGroupText | None
    cell: dict[CellKey, float | None]  # GSAS-II keys: length_a..c (Angstrom), angle_alpha..gamma, volume
    cell_refined: bool  # a cell variable of this phase varied or depended on a varied one
    cell_esd: dict[CellKey, float | None]  # from the saved covariance; empty unless cell_refined
    # The phase's General doPawley: its intensities were extracted, not computed from the structure.
    pawley: bool


class GpxFraction(Record):
    """A mass fraction GSAS-II derives from the saved project state (ComputeMassFracs), normalized
    over the histogram's linked phases. Its s.u. is the native propagation through the saved
    covariance whenever any linked scale varied or depended on a varied parameter, so a fixed own
    scale can still have an uncertain fraction; with every scale fixed there is no s.u."""

    status: Literal["derived", "not_computable", "error"]
    scale_status: Literal["varied", "dependent", "fixed"] | None  # this phase's own scale parameter
    value: float | None
    su: float | None
    detail: DetailText = ""


class GpxHistogram(Record):
    name: NameText
    instrument_type: InstrumentText | None  # GSAS-II Type, e.g. PXC (X-ray, constant wavelength)
    # What x is: only two_theta_deg histograms are angle-dispersive patterns the residual check reads.
    x_quantity: Literal["two_theta_deg", "time_of_flight_us", "energy_kev", "unknown"]
    wavelength_angstrom: float | None
    wr_percent: float | None  # as GSAS-II recorded it in the saved project
    x: tuple[float | None, ...]
    observed: tuple[float | None, ...]
    calculated: tuple[float | None, ...]
    background: tuple[float | None, ...]
    residual: tuple[float | None, ...]
    fraction_basis: BasisText
    mass_fractions: dict[NameText, GpxFraction]  # phase name -> fraction, for phases linked to this histogram
    # Linked phases whose link to this histogram has LeBail set: intensities extracted, not computed.
    le_bail_phases: tuple[NameText, ...]


class GpxAnalysis(Record):
    """An imported project summary: statistics as the saved project recorded them and quantities
    derived from its saved state. Nothing here was refit by the importer."""

    schema_version: Literal["gpx-analysis.v3"] = "gpx-analysis.v3"
    source_sha256: Digest
    saved_using: VersionText | None  # the project's own Controls LastSavedUsing: the software that wrote it
    reader_gsasii_version: VersionText | None  # the GSAS-II installed here that read it
    rwp_percent: float | None
    gof: float | None
    converged: bool | None
    phases: tuple[GpxPhase, ...]
    histograms: tuple[GpxHistogram, ...]


@functools.lru_cache(maxsize=4)
def _bwrap_sizes_tmpfs(bwrap: str) -> bool:
    """Whether this bubblewrap takes `--size BYTES` before `--tmpfs` (0.5.0 and later)."""
    try:
        shown = subprocess.run([bwrap, "--help"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--size" in shown.stdout + shown.stderr


@dataclass(frozen=True)
class GpxSandbox:
    """What the sandbox may see: the Python runtime that has numpy (a venv and its base
    install) and the GSAS-II source, all read-only. And what the read may cost: a deadline, a cap
    on each output stream read here, a size for every writable tmpfs inside, and resource limits
    on the sandbox's processes (each written file, the address space)."""

    python: Path
    gsasii_source: Path | None  # None: not configured, so no sandbox is built
    timeout_s: float = 300.0
    max_output_bytes: int = 64 * 2**20  # stdout: the JSON the reader prints
    max_stderr_bytes: int = 4 * 2**20  # stderr: GSAS-II's console output, kept for the error tail
    max_tmpfs_bytes: int = 64 * 2**20  # each of /tmp and /dev/shm inside the sandbox (RAM)
    max_file_bytes: int = 64 * 2**20  # RLIMIT_FSIZE: any one file the reader writes, /work included
    max_memory_bytes: int = 8 * 2**30  # RLIMIT_AS of every process inside

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> GpxSandbox:
        env = os.environ if environ is None else environ
        source = env.get(GSASII_SOURCE_VAR)
        return cls(
            python=Path(env.get(GPX_PYTHON_VAR, sys.executable)),
            gsasii_source=Path(source) if source else None,
        )

    def source(self) -> Path:
        if self.gsasii_source is None:
            raise GpxImportError(f"set {GSASII_SOURCE_VAR} to the GSAS-II source directory the sandbox may read")
        return self.gsasii_source.resolve()

    def runtime_paths(self) -> tuple[Path, ...]:
        """Read-only mounts: the venv, the interpreter's base install, every other directory on
        the isolated interpreter's module path (a layered venv, as uv's --with builds, reaches its
        base through one), and the GSAS-II source."""
        probe = subprocess.run(
            [
                str(self.python), "-I", "-c",
                "import sys; print(sys.prefix); print(sys.base_prefix); print(*(p for p in sys.path if p), sep='\\n')",
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
        prefix, base_prefix, *module_path = (Path(line).resolve() for line in probe.stdout.splitlines() if line)
        mounts = {prefix, base_prefix, self.source()}
        for entry in module_path:
            if entry.is_dir() and not any(entry.is_relative_to(root) for root in mounts):
                mounts.add(entry)
        return tuple(sorted(mounts))

    def argv(self, inbox: Path, work: Path) -> list[str]:
        source = self.source()
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise SandboxUnavailable("bubblewrap (bwrap) is not installed; a .gpx is not opened without it")
        sized = ["--size", str(self.max_tmpfs_bytes)] if _bwrap_sizes_tmpfs(bwrap) else []
        args = [
            # --as-pid-1: the reader is PID 1 of the new pid namespace, so there is no bwrap init
            # inside whose /proc/1/{environ,cmdline} show the host's environment and these mounts
            bwrap, "--unshare-all", "--as-pid-1", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--proc", "/proc", "--dev", "/dev",
            *sized, "--tmpfs", "/dev/shm",
            *sized, "--tmpfs", "/tmp",
        ]
        for path in self.runtime_paths():
            args += ["--ro-bind", str(path), str(path)]
        args += [
            "--ro-bind", str(inbox), "/in",
            "--ro-bind", str(READER.resolve()), "/reader/gpx_reader.py",
            "--bind", str(work), "/work", "--chdir", "/work",
            "--remount-ro", "/dev", "--remount-ro", "/",
            "--setenv", "HOME", "/work", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "MPLBACKEND", "Agg",
            "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
            "--setenv", "GSASII_SOURCE", str(source),
            str(self.python), "-I", "/reader/gpx_reader.py", "/in/project.gpx",
        ]
        return args

    def limits(self) -> Callable[[], None]:
        """Applied in the sandbox process before it starts, and inherited by everything inside:
        no core dumps, no file past max_file_bytes, no address space past max_memory_bytes. The
        address-space limit is Linux's; the sandbox runs only there (macOS refuses to set it)."""
        bounds = [(resource.RLIMIT_CORE, 0), (resource.RLIMIT_FSIZE, self.max_file_bytes)]
        if sys.platform == "linux":
            bounds.append((resource.RLIMIT_AS, self.max_memory_bytes))

        def apply() -> None:
            for which, value in bounds:
                resource.setrlimit(which, (value, value))

        return apply


@dataclass(frozen=True)
class _Run:
    returncode: int
    stdout: bytes
    stderr: bytes


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """Stop the sandbox and everything it started. bwrap is PID 1 of its namespace, so its death
    ends every process inside; a reader started without bwrap leads its own session, so its
    process group is the same set."""
    for send in (lambda: os.killpg(proc.pid, signal.SIGKILL), proc.kill):
        try:
            send()
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - the kernel is still tearing it down
        pass


def _run_bounded(
    argv: list[str], *, timeout_s: float, caps: Mapping[str, int], preexec: Callable[[], None]
) -> _Run:
    """Run `argv`, reading stdout and stderr as they arrive. Passing a cap, or the deadline, kills
    the process group at once and raises GpxImportError; nothing past a cap is kept. The process
    starts with an empty environment: bwrap needs none, and the importer's (tokens, keys, paths)
    would otherwise be readable inside the sandbox through /proc."""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={},
        start_new_session=True,
        preexec_fn=preexec,
    )
    assert proc.stdout is not None and proc.stderr is not None
    streams = {"stdout": proc.stdout, "stderr": proc.stderr}
    buffers = {name: bytearray() for name in streams}
    deadline = time.monotonic() + timeout_s
    try:
        with selectors.DefaultSelector() as selector:
            for name, stream in streams.items():
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _kill(proc)
                    raise GpxImportError(f"the sandboxed read did not finish in {timeout_s:.0f} s")
                for key, _ in selector.select(timeout=min(remaining, 1.0)):
                    chunk = os.read(key.fd, _CHUNK)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    buffers[name] += chunk
                    if len(buffers[name]) > caps[name]:
                        _kill(proc)
                        raise GpxImportError(
                            f"the sandboxed read wrote more than {caps[name]} bytes to {name}, so it was stopped"
                        )
        try:
            returncode = proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
        except subprocess.TimeoutExpired as exc:
            _kill(proc)
            raise GpxImportError(f"the sandboxed read did not finish in {timeout_s:.0f} s") from exc
    finally:
        if proc.poll() is None:
            _kill(proc)
        proc.stdout.close()
        proc.stderr.close()
    return _Run(returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]))


def read_gpx(path: Path, sandbox: GpxSandbox | None = None) -> GpxAnalysis:
    """Read one finished GSAS-II project without unpickling it in this process."""
    sandbox = sandbox or GpxSandbox.from_environment()
    path = Path(path)
    if path.stat().st_size > MAX_GPX_BYTES:
        raise GpxImportError(f"the project is larger than {MAX_GPX_BYTES} bytes")
    data = path.read_bytes()
    if len(data) > MAX_GPX_BYTES:
        raise GpxImportError(f"the project is larger than {MAX_GPX_BYTES} bytes")
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="phaseeval-gpx-") as scratch:
        inbox, work = Path(scratch, "in"), Path(scratch, "work")
        inbox.mkdir(mode=0o700)
        work.mkdir(mode=0o700)
        (inbox / "project.gpx").write_bytes(data)
        os.chmod(inbox / "project.gpx", 0o400)
        run = _run_bounded(
            sandbox.argv(inbox, work),
            timeout_s=sandbox.timeout_s,
            caps={"stdout": sandbox.max_output_bytes, "stderr": sandbox.max_stderr_bytes},
            preexec=sandbox.limits(),
        )
    if run.returncode != 0:
        tail = run.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        if run.returncode in (1, 2) and any("bwrap:" in line for line in tail):
            raise SandboxUnavailable("; ".join(tail))
        raise GpxImportError(f"the project could not be read (exit {run.returncode}): " + " | ".join(tail))
    try:
        payload = json.loads(run.stdout)
        return GpxAnalysis.model_validate({**payload, "source_sha256": digest})
    except (ValueError, ValidationError) as exc:
        raise GpxImportError(f"the sandbox returned output the importer does not accept: {exc}") from exc
