"""A refinement finished in other software as the imported answer the review page reads.

A GSAS-II project (gsasii.GpxAnalysis) or saved BGMN or Profex outputs (bgmn.ImportedBgmnAnalysis)
enter the tree through the door a pre-generated agent trial uses, WorkbenchStore.import_run: one
ImportRun child under the case's root, completed with an ImportedRunResult whose basis fit is the
refinement and whose answer is its phase list and amounts. Nothing here pretends to be an agent
run: RecordedAgent.harness and the task reference's job name carry the source, GSASII_PROJECT or
BGMN_SAVED_OUTPUT, the model id names the software that wrote the file, the verdict is
NOT_RECORDED, the trajectory is empty, and wall time and cost are None. The task reference's
task_sha256 is the source file's digest (the .gpx, or the BGMN control file), and its trial id is
derived from it, so importing the same file twice yields one attempt.

Binding is by content, as for a trial (R31): the scan the refinement read must be the case's
pattern (every point of it a point of the case's .xy), and every phase must bind to one candidate
of the case, by candidate id, by BGMN's processed phase name (str_generator._process_phase_name,
which is how the harness names a candidate's phase), or by an explicit mapping. A phase that binds
to nothing refuses the import; nothing is dropped.

The stage is the one the saved refinement records. A GSAS-II phase with doPawley, or a
phase-histogram link with LeBail, makes the fit a Pawley-stage fit; BGMN's is read from its STRUC
files' LeBail lines (bgmn._stage). A Pawley-stage fit carries no phase amounts, whatever the
software printed for its scales.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from phaseeval.refinement.bgmn.str_generator import _process_phase_name

from ..importer import BindingError
from ..model import (
    ArtifactRef,
    BlobDigest,
    CandidateEntry,
    CaseManifest,
    Cell,
    Complete,
    FitSummary,
    ImportedAnswer,
    ImportedRunResult,
    ImportedSubmitted,
    ImportRun,
    Incomplete,
    PhaseFit,
    RecordedAgent,
    StageName,
)
from .bgmn import ImportedBgmnAnalysis
from .gsasii import GpxAnalysis, GpxHistogram

GSASII_PROJECT = "gsasii_project"
BGMN_SAVED_OUTPUT = "bgmn_saved_output"
Source = Literal["gsasii_project", "bgmn_saved_output"]
SOURCES: tuple[Source, ...] = (GSASII_PROJECT, BGMN_SAVED_OUTPUT)
# The software printed fractions (GSAS-II mass fractions, BGMN Q GOALs); they are kept as percent.
WEIGHT_UNIT = "fraction"
IMPORTER_VERSION = "1"
# The scan the refinement read is the case's pattern when every point of it is a case point:
# the angle to 1e-4 degrees, the count to 1e-3 or one part in a million.
X_TOLERANCE_DEG = 1e-4
Y_ATOL, Y_RTOL = 1e-3, 1e-6

Imported = tuple[ImportRun, ImportedRunResult, Mapping[BlobDigest, bytes]]


def parse_xy(data: bytes) -> np.ndarray:
    """A two-column pattern (angle, counts), as the case packages carry it."""
    rows = [line.split() for line in data.decode(errors="replace").splitlines() if line.strip()]
    try:
        return np.array([[float(r[0]), float(r[1])] for r in rows if not r[0].startswith(("#", "!", "%"))])
    except (IndexError, ValueError) as exc:
        raise BindingError(f"the pattern is not a two-column scan: {exc}") from exc


def check_pattern(case_pattern: bytes, x: Sequence[float], y: Sequence[float], what: str) -> None:
    """Every (angle, count) point of `what` is a point of the case's pattern, else BindingError."""
    case = parse_xy(case_pattern)
    if case.size == 0:
        raise BindingError("the case pattern holds no points")
    order = np.argsort(case[:, 0])
    cx, cy = case[order, 0], case[order, 1]
    points = np.array(list(zip(x, y, strict=True)), dtype=float).reshape(-1, 2)
    if points.shape[0] == 0:
        raise BindingError(f"{what} holds no points")
    idx = np.clip(np.searchsorted(cx, points[:, 0]), 1, len(cx) - 1)
    matched = np.zeros(points.shape[0], dtype=bool)
    for candidate in (idx - 1, idx):
        near = (np.abs(cx[candidate] - points[:, 0]) <= X_TOLERANCE_DEG) & np.isclose(
            cy[candidate], points[:, 1], rtol=Y_RTOL, atol=Y_ATOL
        )
        matched |= near
    if not matched.all():
        first = points[np.argmin(matched)]
        raise BindingError(
            f"{what} is not this case's pattern: {int((~matched).sum())} of {len(points)} points are not "
            f"in it, the first at {first[0]:.4f} deg ({first[1]:g} counts)"
        )


def bind_phases(
    names: Sequence[str], case: CaseManifest, explicit: Mapping[str, str] | None = None
) -> dict[str, CandidateEntry]:
    """Each refined phase name -> the one case candidate it is. Explicit mappings first, then the
    candidate id itself, then BGMN's processed form of it. Two phases on one candidate, or a phase
    with no candidate, refuse the import."""
    library = {c.candidate_id: c for c in case.candidates}
    processed: dict[str, list[str]] = {}
    for cid in library:
        processed.setdefault(_process_phase_name(cid), []).append(cid)
    explicit = dict(explicit or {})
    if unknown := sorted(set(explicit) - set(names)):
        raise BindingError(f"--phase names phases the refinement does not have: {unknown}")
    if outside := sorted(set(explicit.values()) - set(library)):
        raise BindingError(f"--phase names candidates that are not in {case.case_id}'s library: {outside}")
    bound: dict[str, CandidateEntry] = {}
    unbound: list[str] = []
    for name in names:
        if name in explicit:
            cid = explicit[name]
        elif name in library:
            cid = name
        elif len(processed.get(_process_phase_name(name), ())) == 1:
            cid = processed[_process_phase_name(name)][0]
        else:
            unbound.append(name)
            continue
        bound[name] = library[cid]
    if unbound:
        raise BindingError(
            f"no candidate of {case.case_id} for phase(s) {unbound}; candidates are {sorted(library)}. "
            "Name the binding with --phase NAME=CANDIDATE_ID."
        )
    if len({c.candidate_id for c in bound.values()}) != len(bound):
        raise BindingError("two refined phases bind to the same candidate")
    return bound


def curve_csv(
    two_theta: Sequence[float | None],
    observed: Sequence[float | None],
    calculated: Sequence[float | None],
    background: Sequence[float | None] | None,
) -> bytes:
    """An evidence.Curve CSV. A point the software masked (None) is left out; difference is
    observed minus calculated at every kept point."""
    columns = [two_theta, observed, calculated] + ([background] if background is not None else [])
    rows = [
        row
        for row in zip(*columns, strict=True)
        if all(v is not None and math.isfinite(v) for v in row)
    ]
    if len(rows) < 2:
        raise BindingError("the refinement holds fewer than two unmasked points")
    header = "two_theta,observed,calculated,difference" + (",background" if background is not None else "")
    lines = [header]
    for row in rows:
        tt, obs, calc, *rest = row
        lines.append(",".join(repr(float(v)) for v in (tt, obs, calc, obs - calc, *rest)))
    return ("\n".join(lines) + "\n").encode()


def _sha(data: bytes) -> BlobDigest:
    return BlobDigest(hashlib.sha256(data).hexdigest())


def _ref(data: bytes) -> ArtifactRef:
    return ArtifactRef(sha256=_sha(data), size_bytes=len(data), media_type="text/csv", role="fit_curve")


def _quantification(case: CaseManifest, phases: Sequence[PhaseFit]) -> str:
    if all(p.weight_percent is None for p in phases):
        return "not_quantitative"
    return "restricted_profile" if case.measurement.profile.value.restricts_quantification else "quantitative"


def _assemble(
    *,
    source: Source,
    source_bytes: bytes,
    case: CaseManifest,
    agent: RecordedAgent,
    bound: Mapping[str, CandidateEntry],
    phases: Sequence[PhaseFit],
    stage: StageName,
    rwp_percent: float | None,
    gof: float | None,
    curve: bytes,
    blobs: dict[BlobDigest, bytes],
    missing: list[str],
) -> Imported:
    digest = hashlib.sha256(source_bytes).hexdigest()
    if stage == "pawley":
        # Whatever the software printed for its scales, or failed to print, the stage is the
        # one reason there are no amounts; a parse-failure reason for them would be wrong.
        phases = [p.model_copy(update={"weight_percent": None, "weight_percent_error": None}) for p in phases]
        missing = [gap for gap in missing if not gap.startswith("weights:")]
        missing.append("weights: a Pawley or Le Bail fit extracts intensities and has no phase amounts")
    action = ImportRun(
        harbor={
            "job_name": source,
            "trial_id": f"{source}-{digest[:16]}",
            "task_sha256": digest,
            "pattern_sha256": case.pattern.sha256,
            "cif_sha256": {c.candidate_id: c.cif.sha256 for c in bound.values()},
        },
        external_source=source,
    )
    basis_fit = None
    if rwp_percent is None:
        missing.append("Rwp: the source records none, so there is no fit summary")
    else:
        blobs[_sha(curve)] = curve
        basis_fit = FitSummary(
            stage=stage,
            quantification=_quantification(case, phases),
            rwp_percent=rwp_percent,
            gof=gof,
            phases=tuple(phases),
            curve=_ref(curve),
        )
    weights = {p.candidate_id: p.weight_percent for p in phases}
    answer = ImportedAnswer(
        phases=tuple(p.candidate_id for p in phases),
        reported_weight_percent=weights if any(v is not None for v in weights.values()) else {},
        reported_weight_unit=WEIGHT_UNIT if any(v is not None for v in weights.values()) else None,
    )
    result = ImportedRunResult(
        agent=agent,
        budget_wall_seconds=None,
        wall_seconds=None,
        cost_usd=None,
        end=ImportedSubmitted(answer=answer, answer_sha256=None),
        basis_fit=basis_fit,
        trajectory=(),
        completeness=Incomplete(missing=tuple(missing)) if missing else Complete(),
    )
    return action, result, blobs


# --- GSAS-II -----------------------------------------------------------------------------------


def _gpx_cell(cell: Mapping[str, float | None]) -> Cell:
    return Cell(
        a=cell.get("length_a"),
        b=cell.get("length_b"),
        c=cell.get("length_c"),
        alpha=cell.get("angle_alpha"),
        beta=cell.get("angle_beta"),
        gamma=cell.get("angle_gamma"),
    )


def _histogram(analysis: GpxAnalysis, case: CaseManifest, pattern: bytes) -> GpxHistogram:
    """The one angle-dispersive histogram whose observed scan is the case's pattern."""
    angular = [h for h in analysis.histograms if h.x_quantity == "two_theta_deg"]
    if not angular:
        raise BindingError("the project holds no angle-dispersive (2-theta) histogram")
    bound, reasons = [], []
    for hist in angular:
        points = [(x, y) for x, y in zip(hist.x, hist.observed, strict=True) if x is not None and y is not None]
        try:
            check_pattern(pattern, [p[0] for p in points], [p[1] for p in points], f"histogram {hist.name!r}")
        except BindingError as exc:
            reasons.append(str(exc))
            continue
        bound.append(hist)
    if len(bound) != 1:
        raise BindingError(
            f"{len(bound)} histograms of the project read {case.case_id}'s pattern; exactly one must. "
            + " ".join(reasons)
        )
    return bound[0]


def from_gpx(
    analysis: GpxAnalysis,
    project: bytes,
    case: CaseManifest,
    pattern: bytes,
    phase_map: Mapping[str, str] | None = None,
) -> Imported:
    """One read GSAS-II project under `case`. `pattern` is the case's observed pattern bytes."""
    hist = _histogram(analysis, case, pattern)
    linked = [p for p in analysis.phases if p.name in hist.mass_fractions]
    if not linked:
        raise BindingError(f"no phase of the project is linked to histogram {hist.name!r}")
    bound = bind_phases([p.name for p in linked], case, phase_map)
    phases = []
    for phase in linked:
        fraction = hist.mass_fractions[phase.name]
        derived = fraction.status == "derived" and fraction.value is not None
        phases.append(
            PhaseFit(
                candidate_id=bound[phase.name].candidate_id,
                weight_percent=fraction.value * 100 if derived else None,
                weight_percent_error=fraction.su * 100 if derived and fraction.su is not None else None,
                cell=_gpx_cell(phase.cell),
                spacegroup_no=None,
            )
        )
    curve = curve_csv(hist.x, hist.observed, hist.calculated, hist.background)
    extracted = [p.name for p in linked if p.pawley or p.name in hist.le_bail_phases]
    agent = RecordedAgent(
        model_id="GSAS-II",
        model_version=analysis.saved_using,
        harness=GSASII_PROJECT,
        harness_version=analysis.schema_version,
    )
    missing: list[str] = []
    if analysis.converged is False:
        missing.append("convergence: the project records its last refinement as not converged")
    return _assemble(
        source=GSASII_PROJECT,
        source_bytes=project,
        case=case,
        agent=agent,
        bound=bound,
        phases=phases,
        stage="pawley" if extracted else "rietveld",
        rwp_percent=analysis.rwp_percent,
        gof=analysis.gof,
        curve=curve,
        blobs={_sha(project): project},
        missing=missing,
    )


# --- BGMN and Profex --------------------------------------------------------------------------


def from_bgmn(
    analysis: ImportedBgmnAnalysis,
    members: Mapping[BlobDigest, bytes],
    case: CaseManifest,
    pattern: bytes,
    phase_map: Mapping[str, str] | None = None,
) -> Imported:
    """One read BGMN or Profex bundle under `case`. `members` are read_bundle's blobs: every
    file of the bundle and the fit curve."""
    scans = [
        name for name, member in analysis.members.items() if any(k.startswith("VAL[") for k in member.named_by)
    ]
    if not scans:
        raise BindingError(
            f"the bundle holds no scan the control file names (VAL[n]); its references are "
            f"{analysis.missing_members or 'all present'}, so nothing binds it to {case.case_id}'s pattern"
        )
    for name in scans:
        scan = parse_xy(members[analysis.members[name].ref.sha256])
        check_pattern(pattern, scan[:, 0].tolist(), scan[:, 1].tolist(), f"scan {name!r}")
    fit = analysis.fit
    bound = bind_phases([p.name for p in fit.phases], case, phase_map)
    phases = [
        PhaseFit(
            candidate_id=bound[p.name].candidate_id,
            weight_percent=p.weight_percent,
            weight_percent_error=p.weight_percent_error,
            cell=p.cell,
            spacegroup_no=p.spacegroup_no,
        )
        for p in fit.phases
    ]
    control = members[analysis.members[analysis.control].ref.sha256]
    agent = RecordedAgent(
        model_id="BGMN",
        model_version=analysis.bgmn_version,
        harness=BGMN_SAVED_OUTPUT,
        harness_version=IMPORTER_VERSION,
    )
    missing: list[str] = []
    if fit.rwp.value is None:
        missing.append(f"Rwp: {fit.rwp.absent}")
    if fit.weight_basis == "unavailable":
        missing.append(f"weights: {fit.weight_detail}")
    return _assemble(
        source=BGMN_SAVED_OUTPUT,
        source_bytes=control,
        case=case,
        agent=agent,
        bound=bound,
        phases=phases,
        stage=fit.stage,
        rwp_percent=fit.rwp.value,
        gof=fit.gof,
        curve=members[fit.curve.sha256],
        blobs=dict(members),
        missing=missing,
    )


def describe(result: ImportedRunResult) -> dict[str, Any]:
    """What the CLI prints after an import: the source, the fit and the bound phases."""
    fit = result.basis_fit
    return {
        "source": result.agent.harness,
        "software": f"{result.agent.model_id} {result.agent.model_version or ''}".strip(),
        "stage": fit.stage if fit else None,
        "rwp_percent": fit.rwp_percent if fit else None,
        "phases": {p.candidate_id: p.weight_percent for p in (fit.phases if fit else ())},
        "missing": list(result.completeness.missing) if isinstance(result.completeness, Incomplete) else [],
    }
