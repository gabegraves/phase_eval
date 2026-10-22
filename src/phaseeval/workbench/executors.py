"""Executors: how a tool verb runs from a parent's derived state, and how its raw output
becomes a typed outcome with a replay manifest.

Target path: src/phaseeval/workbench/executors.py (public). `refine` is the only executing verb.

Reuse: `run_recipe` ports docs/research/2026-10-02-parity-experiment/physics/refine.py
(BGMNBackend + RefinementInput + RefinementConfig, no Ray, no DARA). U4 measured the DEV001
three-phase control at 35 to 40 s with 1 thread and the DEV006 corrected refit at 139 to
143 s with 8. Candidates are case-library ids resolved against digest-bound blobs; no request
carries a filesystem path (AXA-bev50.1).

The registry is derived from the action classes: construction fails when an executing verb has
no executor or an executor has no executing verb.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from phaseeval.refinement.bgmn import result_parser
from phaseeval.refinement.bgmn.backend import BGMNBackend
from phaseeval.refinement.bgmn.control_file import _DEVICES_DIR
from phaseeval.refinement.bgmn.installer import BGMNInstaller
from phaseeval.refinement.bgmn.str_cache import StrCache
from phaseeval.refinement.config import RefinementConfig
from phaseeval.refinement.protocol import RefinementInput, RefinementOutput

from .model import (
    ACTION_CLASSES,
    AnalysisState,
    ArtifactRef,
    BackendIdentity,
    BlobDigest,
    CaseManifest,
    Cell,
    Completed,
    Digest,
    ExecutionRecord,
    FitSummary,
    InfraFailed,
    LiveExecution,
    MediaType,
    PackageVersion,
    PhaseFit,
    PhaseInput,
    Radiation,
    Refine,
    RefineResult,
    Refused,
    ReplayEnvironment,
    ReplayManifest,
    ResolvedRecipe,
    StageName,
    StageSummary,
    ThreadSettings,
    ToolIdentity,
    TubeRadiation,
    canonical_bytes,
)

# BGMN takes its thread count only from NTHREADS in the .sav, so two slots hold 16 threads.
BGMN_SLOTS = threading.BoundedSemaphore(2)

# Every live refine, agent or expert, runs with this many BGMN threads (KTD3). The count is
# inside the recipe digest; imported runs replay with the recipe they recorded.
LIVE_BGMN_THREADS = 8

TOOL_NAME = "bgmn_refine"
TOOL_VERSION = "1"

ExecutionMode = Literal["live", "playback"]
RecipeName = Literal["staged_rietveld", "pawley_only"]

STAGES: tuple[StageName, ...] = get_args(StageName)
REPO_ROOT = Path(__file__).resolve().parents[3]
PARSER_PATH = "src/phaseeval/refinement/bgmn/result_parser.py"
# The pinned packages that turn a CIF into BGMN's .str and parse its output.
REPLAY_PACKAGES = ("numpy", "pymatgen", "spglib")
# The three thread variables U4 set; read by name, never by iterating os.environ (R4).
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
UNREPORTED_VERSION = "unreported"  # BGMN wrote no .lst, so it never stated its version

_LST_HEADER = re.compile(
    r"BGMN version (?P<version>\S+), (?P<points>\d+) measured points, "
    r"(?P<peaks>\d+) peaks, (?P<parameters>\d+) parameters"
)
_LST_GOAL = re.compile(r"^Q(?P<phase>\S+?)=(?P<value>[^+\s]+)\+-(?P<esd>\S+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ExecutionContext:
    case: CaseManifest
    state: AnalysisState  # the parent's derived state; context corrections flow in here
    read_blob: Callable[[BlobDigest], bytes]  # digest-checked by the store
    workdir: Path  # a fresh scratch directory owned by this one execution (KTD3)


ExecutorOutcome = Completed | Refused | InfraFailed
# Each executor also returns its tool identity, the blobs it read, and the blobs it produced.
ExecutorResult = tuple[ExecutorOutcome, ToolIdentity, tuple[BlobDigest, ...], Mapping[BlobDigest, bytes]]
Executor = Callable[[ExecutionContext, object], ExecutorResult]


def refinement_config(
    workdir: Path,
    *,
    recipe: RecipeName,
    n_threads: int,
    instrument_profile: str,
    wavelength: str | float,
) -> RefinementConfig:
    """The physics/refine.py config. BGMN reads the profile and wavelength only from the
    config (backend.py:888, control_file.py:530), never from RefinementInput."""
    return RefinementConfig(
        n_threads=n_threads,
        working_dir=workdir,
        prefer_tmpfs=False,
        preserve_temp_files=True,
        enable_ml_pawley_bypass=False,
        pawley_only=recipe == "pawley_only",
        instrument_profile=instrument_profile,
        wavelength=wavelength,
    )


def recipe_digest(config: Mapping[str, object]) -> str:
    """U4's recipe digest, over RefinementConfig.to_dict() or a recorded request.json config.
    working_dir differs per execution, so it is dropped. The recorded digests were taken with
    json's default separators, so compact separators would not reproduce them."""
    recipe = {key: value for key, value in config.items() if key != "working_dir"}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]


def bgmn_wavelength(radiation: Radiation) -> str | float:
    """Tube -> BGMN LAMBDA symbol (K-alpha doublet). Monochromatic -> SYNCHROTRON value.
    A Cu tube is "Cu", never the manifest's 1.540598: that float writes SYNCHROTRON, drops
    K-alpha2 and breaks every recorded replay."""
    if isinstance(radiation, TubeRadiation):
        return radiation.anode
    return radiation.wavelength_angstrom


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_recipe(
    recipe: RecipeName, *, n_threads: int, instrument_profile: str, wavelength: str | float
) -> ResolvedRecipe:
    """The recipe as BGMN will read it. Raises FileNotFoundError when the profile's .geq or
    .ger is not in src/phaseeval/refinement/data/devices/, where copy_instrument_profile looks."""
    config = refinement_config(
        Path("."), recipe=recipe, n_threads=n_threads, instrument_profile=instrument_profile,
        wavelength=wavelength,
    )
    return ResolvedRecipe(
        recipe=recipe,
        recipe_digest=recipe_digest(config.to_dict()),
        bgmn_threads=n_threads,
        instrument_profile=instrument_profile,
        geq_sha256=_sha256_file(_DEVICES_DIR / f"{instrument_profile}.geq"),
        ger_sha256=_sha256_file(_DEVICES_DIR / f"{instrument_profile}.ger"),
        bgmn_wavelength=wavelength,
    )


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_blob_sha(data: bytes) -> str:
    """What `git hash-object` prints for these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def parser_revision() -> str:
    """Git blob sha of the parser that turns BGMN's .lst into the compared numbers, taken
    from the file this process imported."""
    return git_blob_sha(Path(result_parser.__file__).read_bytes())


def _os_release() -> str:
    try:
        pretty = platform.freedesktop_os_release().get("PRETTY_NAME", platform.system())
    except OSError:
        pretty = platform.system()
    libc, libc_version = platform.libc_ver()
    return f"{pretty}; kernel {platform.release()}; {libc or 'libc'} {libc_version or 'unknown'}"


def capture_environment(backend: BackendIdentity) -> ReplayEnvironment:
    """Read the allowlisted fields only (R4): interpreter, OS, machine, git SHA and dirty flag,
    BGMN version and binary hash, pinned package versions, thread settings. Never iterates
    os.environ. Outside a git checkout the source counts as dirty, since nothing shows it
    is not."""
    git_sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no", "--", "src/axa")
    packages = []
    for name in REPLAY_PACKAGES:
        try:
            packages.append(PackageVersion(name=name, version=metadata.version(name)))
        except metadata.PackageNotFoundError:
            continue
    omp, openblas, mkl = (os.environ.get(name) for name in THREAD_VARIABLES)
    return ReplayEnvironment(
        python_version=platform.python_version(),
        os_release=_os_release(),
        machine=platform.machine(),
        axa_git_sha=git_sha or "unavailable",
        axa_source_dirty=status is None or status != "",
        bgmn=backend,
        packages=tuple(packages),
        threads=ThreadSettings(omp_num_threads=omp, openblas_num_threads=openblas, mkl_num_threads=mkl),
    )


# --- BGMN .lst facts the axa parser drops -------------------------------------------------------


@dataclass(frozen=True)
class LstFacts:
    bgmn_version: str | None
    measured_points: int | None
    peaks: int | None
    parameters: int | None
    # Q<phase>=v+-e GOAL lines: BGMN's mass-fraction esd as printed (a fraction, not percent).
    # A phase whose GOAL printed ERROR (Pawley) has no entry.
    weight_esd: Mapping[str, float]


def read_lst_facts(text: str) -> LstFacts:
    header = _LST_HEADER.search(text)
    esd = {match["phase"]: float(match["esd"]) for match in _LST_GOAL.finditer(text)}
    if header is None:
        return LstFacts(None, None, None, None, esd)
    return LstFacts(
        header["version"], int(header["points"]), int(header["peaks"]), int(header["parameters"]), esd
    )


def _stage_lst(fit_dir: Path, stage: StageName) -> Path | None:
    matches = sorted(fit_dir.glob(f"*_{stage}.lst"))
    return matches[0] if len(matches) == 1 else None


def _finite(value: Any) -> float | None:
    """A non-finite parse stays invalid as None: canonical bytes cannot hold NaN."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


FitFileRole = Literal["fit_curve", "native_lst", "native_dia", "native_par"]
NATIVE_FILES: tuple[tuple[str, FitFileRole], ...] = (
    (".lst", "native_lst"),
    (".dia", "native_dia"),
    (".par", "native_par"),
)


def _artifact(path: Path, media_type: MediaType, role: FitFileRole, stage: StageName) -> ArtifactRef:
    data = path.read_bytes()
    return ArtifactRef(
        sha256=BlobDigest(hashlib.sha256(data).hexdigest()),
        size_bytes=len(data),
        media_type=media_type,
        role=role,
        stage=stage,
    )


def _stage_summary(payload: Mapping[str, Any], facts: LstFacts | None, curve: ArtifactRef | None) -> StageSummary:
    """One stage from a StageResult or a recorded result.json stage, which share keys.
    Pawley fits every reflection's intensity, so its 0.0 weight means "no fraction" (None)."""
    stage: StageName = payload["stage"]
    pawley = stage == "pawley"
    esd = facts.weight_esd if facts is not None else {}
    phases = []
    for phase in payload["phases"]:
        lattice = phase.get("lattice_params") or {}
        error = esd.get(phase["phase_id"])
        phases.append(
            PhaseFit(
                candidate_id=phase["phase_id"],
                weight_percent=None if pawley else _finite(phase.get("weight_fraction")),
                weight_percent_error=None if pawley or error is None else float(Decimal(repr(error)) * 100),
                cell=Cell(**{key: _finite(lattice.get(key)) for key in ("a", "b", "c", "alpha", "beta", "gamma")}),
                spacegroup_no=phase.get("spacegroup_no"),
            )
        )
    rwp, rexp = _finite(payload.get("rwp")), _finite(payload.get("rexp"))
    return StageSummary(
        stage=stage,
        rwp_percent=rwp,
        gof=rwp / rexp if rwp is not None and rexp else None,
        rexp_percent=rexp,
        converged=payload.get("converged"),
        iterations=payload.get("iteration_count") or 0,
        measured_points=facts.measured_points if facts else None,
        peaks=facts.peaks if facts else None,
        parameters=facts.parameters if facts else None,
        phases=tuple(phases),
        curve=curve,
    )


def _stage_payload(output: RefinementOutput, stage: StageName) -> dict[str, Any] | None:
    result = getattr(output, stage)
    if result is None:
        return None
    return {
        "stage": stage,
        "rwp": result.rwp,
        "rexp": result.rexp,
        "converged": result.converged,
        "iteration_count": result.iteration_count,
        "phases": [
            {
                "phase_id": phase.phase_id,
                "weight_fraction": phase.weight_fraction,
                "lattice_params": phase.lattice_params,
                "spacegroup_no": phase.spacegroup_no,
            }
            for phase in result.phases
        ],
    }


def _backend_version(fit_dir: Path) -> str:
    for stage in STAGES:
        lst = _stage_lst(fit_dir, stage)
        if lst is not None:
            version = read_lst_facts(lst.read_text()).bgmn_version
            if version is not None:
                return version
    return UNREPORTED_VERSION


def _result(output_stages: list[tuple[Mapping[str, Any], LstFacts | None, ArtifactRef | None]], refinement_failed: bool, failure_reason: str | None) -> RefineResult:
    stages = tuple(_stage_summary(payload, facts, curve) for payload, facts, curve in output_stages)
    return RefineResult(
        stages=stages,
        final=None,  # the executor sets the quantification it knows from the case
        refinement_failed=refinement_failed,
        failure_reason=failure_reason,
    )


# --- one refine ---------------------------------------------------------------------------------


def _write_curve(workdir: Path, stage: StageName, dia_data: Mapping[str, Any]) -> Path:
    columns = [np.asarray(dia_data[key]) for key in ("two_theta", "observed", "calculated", "difference")]
    path = workdir / f"{stage}.csv"
    np.savetxt(path, np.column_stack(columns), delimiter=",", header="two_theta,observed,calculated,difference", comments="")
    return path


def run_recipe(
    pattern_sha256: Digest,
    phases: tuple[PhaseInput, ...],
    recipe: ResolvedRecipe,
    read_blob: Callable[[BlobDigest], bytes],
    workdir: Path,
) -> ExecutionRecord | InfraFailed:
    """One BGMN refine of `phases` in this order with this recipe, in `workdir`, which must be
    new or empty. The returned manifest is resolved here, so a replay whose profile files or
    binary drifted shows as a structural mismatch instead of being hidden.

    A nonzero exit, a signal or a timeout stays a crash (KTD8): a RefineResult with
    refinement_failed. A missing binary, a missing profile or an OS error while staging inputs
    is InfraFailed. The caller runs inside the bgmn slice (CLAUDE.md, CPU Allocation)."""
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise ValueError(f"execution workdir {workdir} is not fresh")
    installer = BGMNInstaller()
    if not installer.is_installed():
        return InfraFailed(cause="executor_unavailable", detail=f"BGMN is not installed under {installer.install_dir}")
    try:
        binary_sha256 = _sha256_file(installer.get_bgmn_path())
        resolved = resolve_recipe(
            recipe.recipe,
            n_threads=recipe.bgmn_threads,
            instrument_profile=recipe.instrument_profile,
            wavelength=recipe.bgmn_wavelength,
        )
        inputs = workdir / "input"
        inputs.mkdir()
        pattern_path = inputs / "pattern.xy"
        pattern_path.write_bytes(_checked_blob(read_blob, pattern_sha256))
        cif_paths = []
        for phase in phases:
            path = inputs / f"{phase.candidate_id}.cif"
            path.write_bytes(_checked_blob(read_blob, phase.cif_sha256))
            cif_paths.append(path)
    except OSError as error:
        return InfraFailed(cause="executor_unavailable", detail=f"{type(error).__name__}: {error}")

    config = refinement_config(
        workdir,
        recipe=resolved.recipe,
        n_threads=resolved.bgmn_threads,
        instrument_profile=resolved.instrument_profile,
        wavelength=resolved.bgmn_wavelength,
    )
    request = RefinementInput(
        pattern_path=pattern_path,
        cif_paths=cif_paths,
        phase_ids=[phase.candidate_id for phase in phases],
        working_dir=workdir,
        hypothesis_id="fit",
        instrument_profile=resolved.instrument_profile,
        wavelength=resolved.bgmn_wavelength,
    )
    with BGMN_SLOTS:
        output = BGMNBackend(str_cache=StrCache(workdir / "str_cache")).refine(request, config)
    if output.failure_reason == "BGMN not installed":
        return InfraFailed(cause="executor_unavailable", detail="BGMN not installed")

    fit_dir = workdir / "fit"
    stages = []
    for stage in STAGES:
        payload = _stage_payload(output, stage)
        if payload is None:
            continue
        lst = _stage_lst(fit_dir, stage)
        dia_data = getattr(output, stage).dia_data
        curve = None
        if dia_data is not None:
            curve = _artifact(_write_curve(workdir, stage, dia_data), "text/csv", "fit_curve", stage)
        stages.append((payload, read_lst_facts(lst.read_text()) if lst else None, curve))
    backend = BackendIdentity(version=_backend_version(fit_dir), binary_sha256=binary_sha256)
    manifest = ReplayManifest(
        pattern_sha256=pattern_sha256,
        phases=phases,
        recipe=resolved,
        parser_revision=parser_revision(),
        backend=backend,
        environment=capture_environment(backend),
    )
    return ExecutionRecord(
        manifest=manifest, result=_result(stages, output.refinement_failed, output.failure_reason)
    )


def _checked_blob(read_blob: Callable[[BlobDigest], bytes], digest: str) -> bytes:
    data = read_blob(BlobDigest(digest))
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"blob {digest} does not hash to its digest")
    return data


def native_artifacts(workdir: Path, result: RefineResult) -> dict[BlobDigest, tuple[ArtifactRef, bytes]]:
    """Every stage's curve and BGMN .lst, .dia and .par from one execution's workdir. Only
    named regular files are read, so the fit/bgmnwin/ symlinks never become blobs."""
    found: dict[BlobDigest, tuple[ArtifactRef, bytes]] = {}
    for summary in result.stages:
        paths: list[tuple[Path, MediaType, FitFileRole]] = []
        if summary.curve is not None:
            paths.append((workdir / f"{summary.stage}.csv", "text/csv", "fit_curve"))
        lst = _stage_lst(workdir / "fit", summary.stage)
        if lst is not None:
            for suffix, role in NATIVE_FILES:
                native = lst.with_suffix(suffix)
                if native.is_file() and not native.is_symlink():
                    paths.append((native, "text/plain", role))
        for path, media_type, role in paths:
            ref = _artifact(path, media_type, role, summary.stage)
            found[ref.sha256] = (ref, path.read_bytes())
    return found


def _final(result: RefineResult, *, restricts_quantification: bool) -> FitSummary | None:
    if not result.stages or result.stages[-1].rwp_percent is None:
        return None
    last = result.stages[-1]
    if last.stage == "pawley":
        quantification = "not_quantitative"
    elif restricts_quantification:
        quantification = "restricted_profile"
    else:
        quantification = "quantitative"
    return FitSummary(
        stage=last.stage,
        quantification=quantification,
        rwp_percent=last.rwp_percent,
        gof=last.gof,
        phases=last.phases,
        curve=last.curve,
    )


def refine(ctx: ExecutionContext, action: Refine) -> ExecutorResult:
    """The live refine of the parent's ordered phase set with LIVE_BGMN_THREADS (KTD3)."""
    tool = ToolIdentity(name=TOOL_NAME, version=TOOL_VERSION, axa_git_sha=_git("rev-parse", "HEAD") or "unavailable")
    hypothesis = ctx.state.hypothesis
    if hypothesis is None or not hypothesis.phases:
        return Refused(reason_code="no_phases", detail="the parent state has no phase set to refine"), tool, (), {}
    library = {entry.candidate_id: entry for entry in ctx.case.candidates}
    unknown = [candidate for candidate in hypothesis.phases if candidate not in library]
    if unknown:
        detail = f"not in this case's candidate library: {', '.join(unknown)}"
        return Refused(reason_code="candidate_not_in_case_library", detail=detail), tool, (), {}
    profile = ctx.state.measurement.profile.value
    if profile.instrument_profile is None:
        detail = "no BGMN instrument profile resolved for this measurement"
        return Refused(reason_code="no_instrument_profile", detail=detail), tool, (), {}

    phases = tuple(
        PhaseInput(candidate_id=candidate, cif_sha256=library[candidate].cif.sha256) for candidate in hypothesis.phases
    )
    inputs = (BlobDigest(ctx.case.pattern.sha256), *(BlobDigest(phase.cif_sha256) for phase in phases))
    try:
        recipe = resolve_recipe(
            action.recipe,
            n_threads=LIVE_BGMN_THREADS,
            instrument_profile=profile.instrument_profile,
            wavelength=bgmn_wavelength(ctx.state.measurement.radiation.value),
        )
    except FileNotFoundError as error:
        return InfraFailed(cause="executor_unavailable", detail=str(error)), tool, (), {}

    start = time.monotonic()
    record = run_recipe(ctx.case.pattern.sha256, phases, recipe, ctx.read_blob, ctx.workdir)
    elapsed = time.monotonic() - start
    if isinstance(record, InfraFailed):
        return record, tool, inputs, {}
    tool = tool.model_copy(update={"backend": record.manifest.backend})
    result = record.result.model_copy(
        update={"final": _final(record.result, restricts_quantification=profile.restricts_quantification)}
    )
    produced = native_artifacts(ctx.workdir, result)
    outcome = Completed(
        result=result,
        parent_state_sha256=hashlib.sha256(canonical_bytes(ctx.state)).hexdigest(),
        artifacts=tuple(ref for ref, _ in produced.values()),
        elapsed_seconds=elapsed,
        custody_scan="not_configured",
        origin=LiveExecution(manifest=record.manifest),
    )
    return outcome, tool, inputs, {digest: data for digest, (_, data) in produced.items()}


# --- recorded executions ------------------------------------------------------------------------


def _recorded_parser_revision(run_dir: Path) -> str | None:
    """The spike wrote `<label>.exec.json` beside each run with the git HEAD it ran at
    (data/custody/workbench/spike-u4/runs/<round>/<label>.exec.json). That commit names the
    parser blob. A run without it (a U6 Harbor trial) does not name its parser."""
    receipt = run_dir.parent / f"{run_dir.name}.exec.json"
    if not receipt.is_file():
        return None
    head = json.loads(receipt.read_text()).get("git_head")
    if not head:
        return None
    return _git("rev-parse", f"{head}:{PARSER_PATH}")


def read_recorded_execution(run_dir: Path, *, binary_sha256: str | None) -> ExecutionRecord:
    """A recorded physics/refine.py run directory (a U4 spike run or a U6 agent run) as the
    execution it recorded, so it replays with its own recipe (KTD3).

    request.json gives the pattern, the ordered candidates with their sha256 and the config;
    fit/<profile>.geq and .ger are the profile files BGMN read; result.json gives the parsed
    stages; each fit/*_<stage>.lst gives the counts, the BGMN version and the weight esd.
    `binary_sha256` is the BGMN binary the caller can cite for this run, or None. Nothing a
    run did not record is filled in (R11): the environment is None."""
    request = json.loads((run_dir / "request.json").read_text())
    output = json.loads((run_dir / "result.json").read_text())
    config = request["config"]
    fit_dir = run_dir / "fit"
    profile = config["instrument_profile"]
    recipe = ResolvedRecipe(
        recipe="pawley_only" if config["pawley_only"] else "staged_rietveld",
        recipe_digest=recipe_digest(config),
        bgmn_threads=config["n_threads"],
        instrument_profile=profile,
        geq_sha256=_sha256_file(fit_dir / f"{profile}.geq"),
        ger_sha256=_sha256_file(fit_dir / f"{profile}.ger"),
        bgmn_wavelength=config["wavelength"],
    )
    stages = []
    for stage in STAGES:
        payload = output.get(stage)
        if payload is None:
            continue
        lst = _stage_lst(fit_dir, stage)
        csv = run_dir / f"{stage}.csv"
        curve = _artifact(csv, "text/csv", "fit_curve", stage) if csv.is_file() else None
        stages.append(({**payload, "stage": stage}, read_lst_facts(lst.read_text()) if lst else None, curve))
    return ExecutionRecord(
        manifest=ReplayManifest(
            pattern_sha256=request["pattern_sha256"],
            phases=tuple(
                PhaseInput(candidate_id=row["candidate_id"], cif_sha256=row["sha256"]) for row in request["candidates"]
            ),
            recipe=recipe,
            parser_revision=_recorded_parser_revision(run_dir),
            backend=BackendIdentity(version=_backend_version(fit_dir), binary_sha256=binary_sha256),
            environment=None,
        ),
        result=_result(stages, output["refinement_failed"], output["failure_reason"]),
    )


# --- registry -----------------------------------------------------------------------------------


def replay_key(ctx: ExecutionContext, action: Refine) -> Digest:
    """What a recording is looked up by in playback mode: the phase set, recipe, measurement
    context digest and input blob digests."""
    raise NotImplementedError


class ExecutorRegistry:
    """Maps verbs to executors. `run` is the store's only way to execute.

    In playback mode a refine whose replay_key matches a recording in `recordings` returns the
    recorded outcome with origin RecordedPlayback; one that matches nothing is refused with
    no_recording, never run live by surprise (AE8)."""

    def __init__(
        self,
        executors: Mapping[str, Executor],
        *,
        mode: ExecutionMode = "live",
        recordings: Path | None = None,
    ) -> None:
        expected = {cls.model_fields["verb"].default for cls in ACTION_CLASSES if cls.executes}
        if set(executors) != expected:
            raise TypeError(f"executor registry {sorted(executors)} != executing verbs {sorted(expected)}")
        if (mode == "playback") != (recordings is not None):
            raise TypeError("playback mode needs a recordings directory, and only playback does")
        self._executors = dict(executors)
        self._mode = mode
        self._recordings = recordings

    def run(self, ctx: ExecutionContext, action: Refine) -> ExecutorResult:
        """Executors time themselves and never raise for science."""
        if self._mode == "playback":
            raise NotImplementedError("playback needs replay_key and the AE8 recordings")
        return self._executors[action.verb](ctx, action)


def default_registry(mode: ExecutionMode = "live", recordings: Path | None = None) -> ExecutorRegistry:
    return ExecutorRegistry({"refine": refine}, mode=mode, recordings=recordings)  # type: ignore[dict-item]
