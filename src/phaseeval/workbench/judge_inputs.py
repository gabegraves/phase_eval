"""Shared blind/public fields and the exact bytes pinned by judge runs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import JsonValue

from phaseeval.kernel.ids import CandidateId, FreeText, Handle, SafeID
from phaseeval.kernel.records import Audience, AudienceRecord, canonical_bytes, check_audiences

StageName = Literal["pawley", "rietveld_lattice", "rietveld_profile", "rietveld"]
StageEnd = Literal["completed", "plateau_stopped", "timed_out", "crashed", "no_output", "skipped"]
CheckCode = Literal["K1", "K2", "K3", "K5"]
CheckStatus = Literal["pass", "flag", "not_computable", "not_applicable"]

B = Audience.RATER_BLIND


class BlindCase(AudienceRecord):
    task: Annotated[FreeText, B, Audience.PUBLIC]
    context: Annotated[JsonValue, B, Audience.PUBLIC]


class CandidateView(AudienceRecord):
    candidate_id: Annotated[CandidateId, B, Audience.PUBLIC]
    formula: Annotated[str | None, B, Audience.PUBLIC]
    space_group: Annotated[str | None, B, Audience.PUBLIC]


class BlindPhase(AudienceRecord):
    phase_id: Annotated[SafeID, B, Audience.PUBLIC]
    formula: Annotated[str | None, B, Audience.PUBLIC]
    space_group: Annotated[str | None, B, Audience.PUBLIC]
    weight_percent: Annotated[float | None, B, Audience.PUBLIC]
    weight_percent_error: Annotated[float | None, B, Audience.PUBLIC]
    reported_formula: Annotated[str | None, B, Audience.PUBLIC]


class BlindAnswer(AudienceRecord):
    phases: Annotated[tuple[BlindPhase, ...], B, Audience.PUBLIC]
    claims: Annotated[tuple[str, ...], B, Audience.PUBLIC]
    reported_answer: Annotated[JsonValue, B, Audience.PUBLIC]
    cited_fit: Annotated[Handle | None, B, Audience.PUBLIC]


class CurvePoints(AudienceRecord):
    two_theta: Annotated[tuple[float, ...], B, Audience.PUBLIC]
    observed: Annotated[tuple[float, ...], B, Audience.PUBLIC]
    calculated: Annotated[tuple[float, ...], B, Audience.PUBLIC]
    background: Annotated[tuple[float, ...] | None, B, Audience.PUBLIC]


class BlindFit(AudienceRecord):
    label: Annotated[str, B, Audience.PUBLIC]
    stage: Annotated[StageName | None, B, Audience.PUBLIC]
    stage_end: Annotated[StageEnd | None, B, Audience.PUBLIC]
    rwp_percent: Annotated[float | None, B, Audience.PUBLIC]
    rexp_percent: Annotated[float | None, B, Audience.PUBLIC]
    gof: Annotated[float | None, B, Audience.PUBLIC]
    parameters: Annotated[int | None, B, Audience.PUBLIC]
    points: Annotated[int | None, B, Audience.PUBLIC]
    boundary_hits: Annotated[int | None, B, Audience.PUBLIC]
    curve: Annotated[CurvePoints, B, Audience.PUBLIC]
    phase_curves: Annotated[Mapping[CandidateId, tuple[float, ...]], B, Audience.PUBLIC]


class BlindCheck(AudienceRecord):
    code: Annotated[CheckCode, B, Audience.PUBLIC]
    status: Annotated[CheckStatus, B, Audience.PUBLIC]
    legend: Annotated[str, B, Audience.PUBLIC]
    detail: Annotated[str, B, Audience.PUBLIC]


class Window(AudienceRecord):
    center_deg: Annotated[float, B, Audience.PUBLIC]
    rows: Annotated[CurvePoints, B, Audience.PUBLIC]


class BlindStep(AudienceRecord):
    step: Annotated[Handle, B, Audience.PUBLIC]
    kind: Annotated[Literal["refine", "segment", "tool_call", "attempt"], B, Audience.PUBLIC]
    label: Annotated[str, B, Audience.PUBLIC]


class BlindChemistry(AudienceRecord):
    phases: Annotated[tuple[tuple[CandidateId, str, str, str], ...], B, Audience.PUBLIC]
    element_inventory: Annotated[tuple[str, ...], B, Audience.PUBLIC]


class BlindView(AudienceRecord):
    schema_version: Annotated[Literal["phaseeval.blind-view.v2"], B, Audience.PUBLIC] = "phaseeval.blind-view.v2"
    item: Annotated[Handle, B, Audience.PUBLIC]
    case: Annotated[BlindCase, B, Audience.PUBLIC]
    candidates: Annotated[tuple[CandidateView, ...], B, Audience.PUBLIC]
    answer: Annotated[BlindAnswer, B, Audience.PUBLIC]
    fit: Annotated[BlindFit | None, B, Audience.PUBLIC]
    checks: Annotated[tuple[BlindCheck, ...], B, Audience.PUBLIC]
    policy_version: Annotated[SafeID, B, Audience.PUBLIC]
    residual_windows: Annotated[tuple[Window, ...], B, Audience.PUBLIC]
    phase_windows: Annotated[Mapping[CandidateId, tuple[Window, ...]], B, Audience.PUBLIC]
    steps: Annotated[tuple[BlindStep, ...], B, Audience.PUBLIC]
    chemistry: Annotated[BlindChemistry | None, B, Audience.PUBLIC]
    missing: Annotated[tuple[str, ...], B, Audience.PUBLIC]


class PublicBlindView(BlindView):
    schema_version: Annotated[Literal["phaseeval.public-blind-view.v1"], B, Audience.PUBLIC] = "phaseeval.public-blind-view.v1"


def public_blind_view(view: BlindView) -> PublicBlindView:
    data = view.model_dump(mode="json")
    data["schema_version"] = "phaseeval.public-blind-view.v1"
    return PublicBlindView.model_validate(data)


def judge_input_bytes(view: PublicBlindView) -> bytes:
    check_audiences(view, frozenset({Audience.PUBLIC}))
    return canonical_bytes(view)


def blind_view(snapshot, *, item: Handle, case: BlindCase, candidates=(), fit: BlindFit | None = None,
               steps=(), residual_windows=(), phase_windows=None) -> BlindView:
    chemistry = snapshot.chemistry
    return BlindView(
        item=item, case=case, candidates=candidates,
        answer=BlindAnswer(phases=tuple(BlindPhase(**{name: getattr(p, name) for name in BlindPhase.model_fields})
                                       for p in snapshot.phases),
                           claims=snapshot.claims, reported_answer=snapshot.reported_answer, cited_fit=None),
        fit=fit, checks=tuple(BlindCheck(code=c.code, status=c.status, legend=c.name, detail=c.detail)
                             for c in snapshot.checks),
        policy_version=snapshot.checks[0].policy_version if snapshot.checks else "unmeasured",
        residual_windows=residual_windows, phase_windows=phase_windows or {}, steps=steps,
        chemistry=BlindChemistry(
            phases=tuple((p.phase_id, p.charge_balance.status,
                          f"{p.hull.e_above_hull_ev_per_atom}: {p.hull.source}",
                          str(p.elements_not_in_precursors)) for p in chemistry.phases),
            element_inventory=chemistry.element_balance.declared_elements if chemistry.element_balance else (),
        ) if chemistry else None,
        missing=("fit evidence is not frozen in this snapshot",) if fit is None else (),
    )
