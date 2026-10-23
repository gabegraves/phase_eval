"""Finished GSAS-II projects as reviewable analyses.

A .gpx is a sequence of Python pickles (GSASIIscriptable.SaveDictToProjFile), so loading one runs
whatever code the file names. The server never unpickles one. read_gpx copies the file into a
fresh directory and runs gpx_reader.py under bubblewrap: every namespace unshared (no network),
only the Python runtime and the GSAS-II source mounted, read-only, the copy read-only at /in, and
one empty temp directory writable at /work (HOME and the working directory). Home, data/custody
and credentials are not mounted. Without bubblewrap the import refuses; it never falls back to an
in-process load. The reader's JSON is validated into GpxAnalysis before anything uses it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from ..model import Digest, Record

READER = Path(__file__).with_name("gpx_reader.py")
# The GSAS-II source directory the sandbox mounts read-only (it holds the GSASII package). There is
# no default: the import refuses until the environment names it.
GSASII_SOURCE_VAR = "PHASEEVAL_GSASII_SOURCE"
GPX_PYTHON_VAR = "PHASEEVAL_GPX_PYTHON"
MAX_GPX_BYTES = 256 * 2**20


class GpxImportError(ValueError):
    """The project could not be read: not a GSAS-II project, a refused or failed load, or output
    the importer does not accept. The message carries the sandbox's stderr tail only."""


class SandboxUnavailable(RuntimeError):
    """bubblewrap is missing or cannot create its namespaces on this host."""


class GpxPhase(Record):
    name: str = Field(min_length=1, max_length=500)
    space_group: str | None
    cell: dict[str, float | None]  # GSAS-II keys: length_a..c (Angstrom), angle_alpha..gamma, volume
    cell_refined: bool  # a cell variable of this phase varied or depended on a varied one
    cell_esd: dict[str, float | None]  # from the saved covariance; empty unless cell_refined


class GpxFraction(Record):
    """A mass fraction GSAS-II derives from the saved project state (ComputeMassFracs), normalized
    over the histogram's linked phases. Its s.u. is the native propagation through the saved
    covariance whenever any linked scale varied or depended on a varied parameter, so a fixed own
    scale can still have an uncertain fraction; with every scale fixed there is no s.u."""

    status: Literal["derived", "not_computable", "error"]
    scale_status: Literal["varied", "dependent", "fixed"] | None  # this phase's own scale parameter
    value: float | None
    su: float | None
    detail: str = Field(default="", max_length=2000)


class GpxHistogram(Record):
    name: str = Field(min_length=1, max_length=500)
    instrument_type: str | None  # GSAS-II Type, e.g. PXC (X-ray, constant wavelength)
    # What x is: only two_theta_deg histograms are angle-dispersive patterns the residual check reads.
    x_quantity: Literal["two_theta_deg", "time_of_flight_us", "energy_kev", "unknown"]
    wavelength_angstrom: float | None
    wr_percent: float | None  # as GSAS-II recorded it in the saved project
    x: tuple[float | None, ...]
    observed: tuple[float | None, ...]
    calculated: tuple[float | None, ...]
    background: tuple[float | None, ...]
    residual: tuple[float | None, ...]
    fraction_basis: str = Field(max_length=2000)
    mass_fractions: dict[str, GpxFraction]  # phase name -> fraction, for phases linked to this histogram


class GpxAnalysis(Record):
    """An imported project summary: statistics as the saved project recorded them and quantities
    derived from its saved state. Nothing here was refit by the importer."""

    schema_version: Literal["gpx-analysis.v2"] = "gpx-analysis.v2"
    source_sha256: Digest
    saved_using: str | None  # the project's own Controls LastSavedUsing: the software that wrote it
    reader_gsasii_version: str | None  # the GSAS-II installed here that read it
    rwp_percent: float | None
    gof: float | None
    converged: bool | None
    phases: tuple[GpxPhase, ...]
    histograms: tuple[GpxHistogram, ...]


@dataclass(frozen=True)
class GpxSandbox:
    """What the sandbox may see: the Python runtime that has numpy (a venv and its base
    install) and the GSAS-II source, all read-only."""

    python: Path
    gsasii_source: Path | None  # None: not configured, so no sandbox is built
    timeout_s: float = 300.0
    max_output_bytes: int = 64 * 2**20

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
        """Read-only mounts: the venv, the interpreter's base install, the GSAS-II source."""
        probe = subprocess.run(
            [str(self.python), "-I", "-c", "import sys; print(sys.prefix); print(sys.base_prefix)"],
            capture_output=True, text=True, timeout=60, check=True,
        )
        prefixes = {Path(line).resolve() for line in probe.stdout.split()}
        return tuple(sorted(prefixes | {self.source()}))

    def argv(self, inbox: Path, work: Path) -> list[str]:
        source = self.source()
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise SandboxUnavailable("bubblewrap (bwrap) is not installed; a .gpx is not opened without it")
        args = [
            bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin",
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        ]
        for path in self.runtime_paths():
            args += ["--ro-bind", str(path), str(path)]
        args += [
            "--ro-bind", str(inbox), "/in",
            "--ro-bind", str(READER.resolve()), "/reader/gpx_reader.py",
            "--bind", str(work), "/work", "--chdir", "/work",
            "--setenv", "HOME", "/work", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "MPLBACKEND", "Agg",
            "--setenv", "GSASII_SOURCE", str(source),
            str(self.python), "-I", "/reader/gpx_reader.py", "/in/project.gpx",
        ]
        return args


def read_gpx(path: Path, sandbox: GpxSandbox | None = None) -> GpxAnalysis:
    """Read one finished GSAS-II project without unpickling it in this process."""
    sandbox = sandbox or GpxSandbox.from_environment()
    data = Path(path).read_bytes()
    if len(data) > MAX_GPX_BYTES:
        raise GpxImportError(f"the project is larger than {MAX_GPX_BYTES} bytes")
    digest = hashlib.sha256(data).hexdigest()
    with tempfile.TemporaryDirectory(prefix="phaseeval-gpx-") as scratch:
        inbox, work = Path(scratch, "in"), Path(scratch, "work")
        inbox.mkdir(mode=0o700)
        work.mkdir(mode=0o700)
        (inbox / "project.gpx").write_bytes(data)
        os.chmod(inbox / "project.gpx", 0o400)
        try:
            run = subprocess.run(
                sandbox.argv(inbox, work), capture_output=True, timeout=sandbox.timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise GpxImportError(f"the sandboxed read did not finish in {sandbox.timeout_s:.0f} s") from exc
    if len(run.stdout) > sandbox.max_output_bytes:
        raise GpxImportError("the sandboxed read produced more output than the importer accepts")
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
