"""Fit evidence over retained curves: side-by-side comparison and replay checks.

Target path: src/phaseeval/workbench/evidence.py (public). Pure functions; the store loads blobs.
Replaces the hard-coded `numerical_replay: Literal["unavailable","unverified"]`
(attempt_models.py:86) with a measured check appended to the attempt's ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from .model import (
    AttemptDigest,
    AttemptRef,
    BackendIdentity,
    CandidateId,
    ChildAttempt,
    Completed,
    Digest,
    ExecutionRecord,
    FitSummary,
    Hypothesis,
    Record,
    ReplayEnvironment,
    SafeID,
    StageName,
)


class Curve(Record):
    """A parsed fit_curve blob. Arrays come from the engine's .dia; nothing is synthesized."""

    two_theta: tuple[float, ...]
    observed: tuple[float, ...]
    calculated: tuple[float, ...]
    difference: tuple[float, ...]
    background: tuple[float, ...] | None = None  # recorded missing, never invented

    @classmethod
    def parse_csv(cls, data: bytes) -> Curve:
        """Header two_theta,observed,calculated,difference[,background] (executors._write_curve);
        equal lengths, finite, strictly increasing two_theta; ValueError otherwise."""
        lines = [line.strip() for line in data.decode("ascii").splitlines() if line.strip()]
        if not lines:
            raise ValueError("an empty curve")
        header = tuple(lines[0].split(","))
        if header not in (_CURVE_COLUMNS, (*_CURVE_COLUMNS, "background")):
            raise ValueError(f"unexpected curve header {lines[0]!r}")
        rows = [tuple(float(value) for value in line.split(",")) for line in lines[1:]]
        if not rows or any(len(row) != len(header) for row in rows):
            raise ValueError("curve rows disagree with the header")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("a curve value is not finite")
        columns = dict(zip(header, zip(*rows, strict=True), strict=True))
        two_theta = columns["two_theta"]
        if any(b <= a for a, b in zip(two_theta, two_theta[1:], strict=False)):
            raise ValueError("two_theta is not strictly increasing")
        return cls(**columns)


_CURVE_COLUMNS = ("two_theta", "observed", "calculated", "difference")


# --- side by side (R14) ----------------------------------------------------------------------


def fit_attempt_for_compare(
    digest: AttemptDigest,
    attempts: Mapping[AttemptDigest, ChildAttempt],
    children: Mapping[AttemptDigest, tuple[AttemptDigest, ...]],
) -> AttemptDigest | None:
    """The attempt whose fit stands for `digest` in a comparison: itself when it has a fit,
    otherwise its nearest descendant by the same principal that has one (R14). A phase edit
    compares through the refine posted with it."""
    start = attempts.get(digest)
    if start is None:
        return None
    principal = start.actor.principal_id
    level = [digest]
    while level:
        fitted = [d for d in level if fit_of(attempts[d]) is not None]
        if fitted:
            return min(fitted, key=lambda d: attempts[d].created_at_utc)
        level = [
            child
            for d in level
            for child in children.get(d, ())
            if child in attempts and attempts[child].actor.principal_id == principal
        ]
    return None


def fit_of(attempt: ChildAttempt) -> FitSummary | None:
    """The fit an attempt produced: a completed refine's final fit, or an imported run's basis."""
    outcome = attempt.outcome
    if not isinstance(outcome, Completed):
        return None
    result = outcome.result
    return result.final if result.kind == "refine" else result.basis_fit


class PhaseChanges(Record):
    added: tuple[CandidateId, ...]
    removed: tuple[CandidateId, ...]


class WeightDelta(Record):
    candidate_id: CandidateId
    a_percent: float | None
    b_percent: float | None


class ResidualChange(Record):
    two_theta_deg: float
    a_difference: float
    b_difference: float


class ResidualDelta(Record):
    a_rms: float
    b_rms: float
    largest_changes: tuple[ResidualChange, ...]


class Comparison(Record):
    a: AttemptRef
    b: AttemptRef
    a_fit_from: AttemptRef | None  # resolved by fit_attempt_for_compare
    b_fit_from: AttemptRef | None
    phases: PhaseChanges
    context_changes: tuple[str, ...]  # field paths, listed apart from phase changes
    a_fit: FitSummary | None
    b_fit: FitSummary | None
    delta_rwp_percent: float | None
    delta_gof: float | None
    weights: tuple[WeightDelta, ...]
    residual: ResidualDelta | None  # None when either curve is not retained
    like_for_like: bool
    caveats: tuple[str, ...]  # why it is not like-for-like, stated plainly


def phase_changes(a: Hypothesis | None, b: Hypothesis | None) -> PhaseChanges:
    a_phases = a.phases if a else ()
    b_phases = b.phases if b else ()
    return PhaseChanges(
        added=tuple(p for p in b_phases if p not in a_phases),
        removed=tuple(p for p in a_phases if p not in b_phases),
    )


def compare_fits(
    a: AttemptRef,
    b: AttemptRef,
    a_fit_from: AttemptRef | None,
    b_fit_from: AttemptRef | None,
    phases: PhaseChanges,
    context_changes: tuple[str, ...],
    a_fit: FitSummary | None,
    b_fit: FitSummary | None,
    a_curve: Curve | None,
    b_curve: Curve | None,
) -> Comparison:
    """Deltas are b minus a. The residual is compared on the shared grid only when both observed
    arrays are identical. Playback is caveated by the store, which knows each fit's origin."""
    caveats: list[str] = []
    if a_fit is None or b_fit is None:
        caveats.append("one side has no fit")
    else:
        if a_fit.stage != b_fit.stage:
            caveats.append(f"stages differ: {a_fit.stage} vs {b_fit.stage}")
        if a_fit.quantification != "quantitative" or b_fit.quantification != "quantitative":
            caveats.append(f"not both quantitative: {a_fit.quantification} vs {b_fit.quantification}")
    if context_changes:
        caveats.append(f"context differs: {', '.join(context_changes)}")
    a_weights = {p.candidate_id: p.weight_percent for p in (a_fit.phases if a_fit else ())}
    b_weights = {p.candidate_id: p.weight_percent for p in (b_fit.phases if b_fit else ())}
    residual = None
    if a_curve is not None and b_curve is not None:
        if a_curve.two_theta == b_curve.two_theta and a_curve.observed == b_curve.observed:
            residual = _residual_delta(a_curve, b_curve)
        else:
            caveats.append("the two curves do not share one observed pattern grid")
    return Comparison(
        a=a,
        b=b,
        a_fit_from=a_fit_from,
        b_fit_from=b_fit_from,
        phases=phases,
        context_changes=context_changes,
        a_fit=a_fit,
        b_fit=b_fit,
        delta_rwp_percent=b_fit.rwp_percent - a_fit.rwp_percent if a_fit and b_fit else None,
        delta_gof=b_fit.gof - a_fit.gof
        if a_fit and b_fit and a_fit.gof is not None and b_fit.gof is not None
        else None,
        weights=tuple(
            WeightDelta(candidate_id=c, a_percent=a_weights.get(c), b_percent=b_weights.get(c))
            for c in dict.fromkeys((*a_weights, *b_weights))
        ),
        residual=residual,
        like_for_like=not caveats,
        caveats=tuple(caveats),
    )


LARGEST_RESIDUAL_CHANGES = 5


def _residual_delta(a: Curve, b: Curve) -> ResidualDelta:
    def rms(values: tuple[float, ...]) -> float:
        return math.sqrt(sum(v * v for v in values) / len(values))

    order = sorted(
        range(len(a.two_theta)), key=lambda i: abs(b.difference[i] - a.difference[i]), reverse=True
    )
    return ResidualDelta(
        a_rms=rms(a.difference),
        b_rms=rms(b.difference),
        largest_changes=tuple(
            ResidualChange(
                two_theta_deg=a.two_theta[i], a_difference=a.difference[i], b_difference=b.difference[i]
            )
            for i in sorted(order[:LARGEST_RESIDUAL_CHANGES])
        ),
    )


# --- replay (R4) ------------------------------------------------------------------------------


class ReplayBounds(Record):
    """Numerical bounds justified before the event (R4). No defaults: a bound nobody justified
    cannot be constructed. GOF has no bound of its own: it is recomputed from the replayed Rwp
    and Rexp, so its tolerance is rwp_rexp_abs_percent * (1 + GOF) / Rexp."""

    version: SafeID
    justified_by: str = Field(min_length=1)
    # U4 found one public Linux build. Another build's first run is a new reference, not a
    # replay.
    build: BackendIdentity
    rwp_rexp_abs_percent: float = Field(gt=0)  # every stage
    weight_abs_percent: float = Field(gt=0)  # final stage
    cell_length_rel: float = Field(gt=0)  # final stage a, b, c
    cell_angle_abs_deg: float = Field(gt=0)  # final stage alpha, beta, gamma

    @model_validator(mode="after")
    def identified_build(self) -> Self:
        if self.build.binary_sha256 is None:
            raise ValueError("bounds name the BGMN binary they were measured on")
        return self


# The spread at a fixed recipe was 0 in every field across three environments on this build;
# each margin is the largest thread-count perturbation U4 measured.
U4_BOUNDS = ReplayBounds(
    version="u4-2026-10-19",
    justified_by="docs/research/2026-10-19-osi-hackathon/spike-findings.md, Replay bounds",
    build=BackendIdentity(
        version="4.2.23",
        binary_sha256="10cc9c1de0cca0eb6e12ff20abd44e959489c7d87082463639cfd297cfe9d23f",
    ),
    rwp_rexp_abs_percent=0.01,
    weight_abs_percent=0.03,
    cell_length_rel=2e-5,
    cell_angle_abs_deg=0.001,
)


class ReplayDelta(Record):
    stage: StageName
    quantity: str  # "rwp_percent", "gof", "weight_percent:CIF016107", "cell:CIF006427:a", ...
    retained: float
    replayed: float
    tolerance: float
    within: bool


ReplayVerdict = Literal[
    "within_bounds",
    "outside_bounds",
    "structural_mismatch",
    "invalid_value",
    "incomplete",
    "replay_failed",
]


class ReplayFindings(Record):
    verdict: ReplayVerdict
    # Manifest fields, stage names, phase ids per stage, point/peak/parameter counts,
    # refinement_failed and failure_reason that differ. Any one fails the replay outright.
    structural_mismatches: tuple[str, ...]
    # A zero, null or non-finite parsed value on either side: a finding, never a match.
    invalid_values: tuple[str, ...]
    # Structural fields the recorded side never captured (R11): neither a match nor a
    # mismatch, so the replay is incomplete. The deltas are still reported.
    missing: tuple[str, ...]
    deltas: tuple[ReplayDelta, ...]

    @model_validator(mode="after")
    def structure_gates_numbers(self) -> Self:
        expected: ReplayVerdict
        if self.verdict == "replay_failed":
            expected = "replay_failed"
        elif self.structural_mismatches:
            expected = "structural_mismatch"
        elif self.invalid_values:
            expected = "invalid_value"
        elif self.missing:
            expected = "incomplete"
        elif all(d.within for d in self.deltas):
            expected = "within_bounds"
        else:
            expected = "outside_bounds"
        if self.verdict != expected:
            raise ValueError(f"findings give {expected}, not {self.verdict}")
        return self


class ReplayCheck(Record):
    kind: Literal["replay"] = "replay"
    bounds_version: SafeID
    environment: ReplayEnvironment  # where the re-execution ran
    findings: ReplayFindings
    replay_outcome_sha256: Digest | None  # canonical bytes of the fresh outcome, kept as a blob
    created_at_utc: datetime


CELL_LENGTHS = ("a", "b", "c")
CELL_ANGLES = ("alpha", "beta", "gamma")
# Printed values sit on decimal quanta, so a delta of exactly one tolerance must count as
# within despite binary rounding (2.51 - 2.50 = 0.010000000000000231).
_ROUNDING = 1e-9


def _invalid(value: float | None) -> bool:
    return value is None or value == 0 or not math.isfinite(value)


class _Structure:
    """Collects structural findings. A None on the recorded side is missing, never a match."""

    def __init__(self) -> None:
        self.mismatches: list[str] = []
        self.missing: list[str] = []

    def exact(self, field: str, recorded: object, live: object) -> None:
        if recorded is None:
            self.missing.append(field)
        elif recorded != live:
            self.mismatches.append(f"{field}: recorded {recorded!r}, live {live!r}")


def _structure(recorded: ExecutionRecord, live: ExecutionRecord, bounds: ReplayBounds) -> _Structure:
    found = _Structure()
    a, b = recorded.manifest, live.manifest
    found.exact("pattern_sha256", a.pattern_sha256, b.pattern_sha256)
    a_ids = [p.candidate_id for p in a.phases]
    b_ids = [p.candidate_id for p in b.phases]
    if a_ids != b_ids:
        kind = "phases.order" if sorted(a_ids) == sorted(b_ids) else "phases.candidates"
        found.mismatches.append(f"{kind}: recorded {a_ids}, live {b_ids}")
    else:
        for x, y in zip(a.phases, b.phases, strict=True):
            found.exact(f"phases.{x.candidate_id}.cif_sha256", x.cif_sha256, y.cif_sha256)
    for field in type(a.recipe).model_fields:
        found.exact(f"recipe.{field}", getattr(a.recipe, field), getattr(b.recipe, field))
    found.exact("parser_revision", a.parser_revision, b.parser_revision)
    found.exact("backend.version", a.backend.version, b.backend.version)
    found.exact("backend.binary_sha256", a.backend.binary_sha256, b.backend.binary_sha256)
    if b.backend.binary_sha256 != bounds.build.binary_sha256:
        found.mismatches.append(
            f"backend.binary_sha256: live {b.backend.binary_sha256!r} is not the build the bounds "
            f"were measured on ({bounds.build.binary_sha256}); its first run is a new reference"
        )
    found.exact("refinement_failed", recorded.result.refinement_failed, live.result.refinement_failed)
    if recorded.result.failure_reason != live.result.failure_reason:
        found.mismatches.append(
            f"failure_reason: recorded {recorded.result.failure_reason!r}, live {live.result.failure_reason!r}"
        )
    a_stages = [s.stage for s in recorded.result.stages]
    b_stages = [s.stage for s in live.result.stages]
    if a_stages != b_stages:
        found.mismatches.append(f"stages: recorded {a_stages}, live {b_stages}")
    live_stages = {s.stage: s for s in live.result.stages}
    for x in recorded.result.stages:
        y = live_stages.get(x.stage)
        if y is None:
            continue
        x_ids = [p.candidate_id for p in x.phases]
        y_ids = [p.candidate_id for p in y.phases]
        if x_ids != y_ids:
            found.mismatches.append(f"{x.stage}.phases: recorded {x_ids}, live {y_ids}")
        for count in ("measured_points", "peaks", "parameters"):
            found.exact(f"{x.stage}.{count}", getattr(x, count), getattr(y, count))
    return found


def _invalid_values(record: ExecutionRecord, side: str) -> list[str]:
    """Only the fields a replay compares: Rwp and Rexp at every stage, weights and cells at
    the final stage. Intermediate cells are not replay fields (U4: a Pawley NiO cell parsed as
    4 Å in one run and 0 in another)."""
    found = []
    for stage in record.result.stages:
        for quantity in ("rwp_percent", "rexp_percent"):
            if _invalid(getattr(stage, quantity)):
                found.append(f"{side}.{stage.stage}.{quantity}={getattr(stage, quantity)!r}")
    if record.result.stages:
        final = record.result.stages[-1]
        for phase in final.phases:
            if _invalid(phase.weight_percent):
                found.append(f"{side}.{final.stage}.weight_percent:{phase.candidate_id}={phase.weight_percent!r}")
            for key in (*CELL_LENGTHS, *CELL_ANGLES):
                value = getattr(phase.cell, key)
                if _invalid(value):
                    found.append(f"{side}.{final.stage}.cell:{phase.candidate_id}:{key}={value!r}")
    return found


def _delta(stage: StageName, quantity: str, recorded: float | None, live: float | None, tolerance: float) -> ReplayDelta | None:
    if recorded is None or live is None or not (math.isfinite(recorded) and math.isfinite(live)):
        return None
    return ReplayDelta(
        stage=stage,
        quantity=quantity,
        retained=recorded,
        replayed=live,
        tolerance=tolerance,
        within=abs(live - recorded) - tolerance <= _ROUNDING,
    )


def _deltas(recorded: ExecutionRecord, live: ExecutionRecord, bounds: ReplayBounds) -> list[ReplayDelta]:
    deltas: list[ReplayDelta | None] = []
    live_stages = {s.stage: s for s in live.result.stages}
    pairs = [(x, live_stages[x.stage]) for x in recorded.result.stages if x.stage in live_stages]
    for x, y in pairs:
        deltas.append(_delta(x.stage, "rwp_percent", x.rwp_percent, y.rwp_percent, bounds.rwp_rexp_abs_percent))
        deltas.append(_delta(x.stage, "rexp_percent", x.rexp_percent, y.rexp_percent, bounds.rwp_rexp_abs_percent))
    if recorded.result.stages and live.result.stages and recorded.result.stages[-1].stage == live.result.stages[-1].stage:
        x, y = recorded.result.stages[-1], live.result.stages[-1]
        x_rwp, x_rexp, y_rwp, y_rexp = x.rwp_percent, x.rexp_percent, y.rwp_percent, y.rexp_percent
        if x_rwp and x_rexp and y_rwp and y_rexp and math.isfinite(x_rwp / x_rexp + y_rwp / y_rexp):
            gof_x, gof_y = x_rwp / x_rexp, y_rwp / y_rexp
            tolerance = bounds.rwp_rexp_abs_percent * (1 + gof_y) / y_rexp
            deltas.append(_delta(x.stage, "gof", gof_x, gof_y, tolerance))
        live_phases = {p.candidate_id: p for p in y.phases}
        for p in x.phases:
            q = live_phases.get(p.candidate_id)
            if q is None:
                continue
            deltas.append(
                _delta(x.stage, f"weight_percent:{p.candidate_id}", p.weight_percent, q.weight_percent, bounds.weight_abs_percent)
            )
            for key in CELL_LENGTHS:
                length = getattr(p.cell, key)
                if length is not None:
                    tolerance = bounds.cell_length_rel * abs(length)
                    deltas.append(_delta(x.stage, f"cell:{p.candidate_id}:{key}", length, getattr(q.cell, key), tolerance))
            for key in CELL_ANGLES:
                deltas.append(
                    _delta(x.stage, f"cell:{p.candidate_id}:{key}", getattr(p.cell, key), getattr(q.cell, key), bounds.cell_angle_abs_deg)
                )
    return [d for d in deltas if d is not None]


def compare_replay(recorded: ExecutionRecord, live: ExecutionRecord, bounds: ReplayBounds) -> ReplayFindings:
    """U4's replay rule. Structural fields match exactly; a field the recorded side never
    captured makes the replay incomplete. Then zero, null or non-finite compared values on
    either side. Then numbers: Rwp and Rexp at every stage; GOF (recomputed from Rwp and
    Rexp), weights and cells at the final stage. Iterations are never compared."""
    structure = _structure(recorded, live, bounds)
    invalid = _invalid_values(recorded, "recorded") + _invalid_values(live, "live")
    deltas = _deltas(recorded, live, bounds)
    verdict: ReplayVerdict
    if structure.mismatches:
        verdict = "structural_mismatch"
    elif invalid:
        verdict = "invalid_value"
    elif structure.missing:
        verdict = "incomplete"
    else:
        verdict = "within_bounds" if all(d.within for d in deltas) else "outside_bounds"
    return ReplayFindings(
        verdict=verdict,
        structural_mismatches=tuple(structure.mismatches),
        invalid_values=tuple(invalid),
        missing=tuple(structure.missing),
        deltas=tuple(deltas),
    )
