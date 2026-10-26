"""Who sees what, and who may record which judgment: projections, visibility, annotations.

Target path: src/phaseeval/workbench/review.py (public).

- Projection is derived from the authenticated principal, never from a request field.
- `VISIBILITY` decides, once, which audience may see each field of every stored record, and
  `ARTIFACT_AUDIENCE` does the same per artifact role [C3]. Import fails when a field or role
  has no row, and when an actor view type exposes a field its row withholds from agents.
- `ActorCaseView` refuses any Sourced value whose provenance is not actor-safe (R9).
- Annotations are the per-attempt ledger of facts that change no scientific state: notes,
  preferences, adjudications, replay checks, run closes, load events (R5). The expert's
  rubric judgment is a labels.py record on a frozen snapshot, not an annotation. Each is
  its own immutable file; nothing rewrites a shared file.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Annotated, Literal, Self, get_args

from pydantic import BaseModel, Field, JsonValue, model_validator

from phaseeval.api.auth import Principal, PrincipalType

from .evidence import ReplayCheck, fit_of
from .model import (
    FreeText,
    AnalysisState,
    ArtifactRef,
    ArtifactRole,
    Actor,
    Attempt,
    AttemptAction,
    AttemptDigest,
    AttemptRef,
    BlobRef,
    CaseManifest,
    CaseSource,
    ChildAttempt,
    Completed,
    Contamination,
    CustodyScan,
    Digest,
    DocumentChannel,
    DocumentSource,
    FitSummary,
    HumanActor,
    InfraFailed,
    MaterialContext,
    MeasurementContext,
    Record,
    Recorded,
    Refused,
    Rejected,
    RootAttempt,
    SafeID,
    ServiceActor,
    SplitAssignment,
    Sourced,
    ToolIdentity,
    ToolOutcome,
)

Audience = Literal["agent", "agent_own", "expert", "public"]
_BOTH: frozenset[Audience] = frozenset({"agent", "expert", "public"})
_OWN: frozenset[Audience] = frozenset({"agent_own", "expert"})
_EXPERT: frozenset[Audience] = frozenset({"expert"})
_PUBLIC_EXPERT: frozenset[Audience] = frozenset({"expert", "public"})
_NOBODY: frozenset[Audience] = frozenset()

# Provenance channels whose values may reach a model (R9, measurement plan R10). Free text and
# custody records stay expert-only; expert corrections are always visible as a named source.
ACTOR_SAFE_CHANNELS: frozenset[DocumentChannel] = frozenset(
    {
        "instrument_export",
        "instrument_registry",
        "xrf",
        "eds",
        "eln",
        "lims",
        "synthesis_recipe",
        "dataset_metadata",
        "packet",
    }
)

VISIBILITY: Mapping[tuple[str, str], frozenset[Audience]] = {
    ("CaseManifest", "schema_version"): _EXPERT,
    ("CaseManifest", "case_id"): _BOTH,
    ("CaseManifest", "source"): _PUBLIC_EXPERT,
    ("CaseManifest", "exposure"): _PUBLIC_EXPERT,
    ("CaseManifest", "split"): _PUBLIC_EXPERT,
    ("CaseManifest", "task_kind"): _PUBLIC_EXPERT,  # telling the agent "confirmation" is a hint
    ("CaseManifest", "task"): _BOTH,
    ("CaseManifest", "pattern"): _BOTH,
    ("CaseManifest", "measurement"): _BOTH,  # per-value provenance filter applies inside
    ("CaseManifest", "material"): _BOTH,  # per-value provenance filter applies inside
    ("CaseManifest", "candidates"): _BOTH,
    ("CaseManifest", "ingest"): _EXPERT,  # original filenames and raw headers
    ("CaseManifest", "follows_request"): _PUBLIC_EXPERT,
    ("ChildAttempt", "schema_version"): _PUBLIC_EXPERT,
    ("ChildAttempt", "kind"): _PUBLIC_EXPERT,
    ("ChildAttempt", "parent"): _BOTH,
    ("ChildAttempt", "root"): _BOTH,
    ("ChildAttempt", "depth"): _BOTH,
    ("ChildAttempt", "actor"): _PUBLIC_EXPERT,  # agents see only `own`
    ("ChildAttempt", "action"): _BOTH,
    ("ChildAttempt", "tool"): _BOTH,
    ("ChildAttempt", "inputs"): _BOTH,
    ("ChildAttempt", "rationale"): _OWN,  # a human's rationale may carry revealed knowledge
    ("ChildAttempt", "outcome"): _BOTH,
    ("ChildAttempt", "idempotency_key"): _OWN,
    ("ChildAttempt", "created_at_utc"): _BOTH,
}

ARTIFACT_AUDIENCE: Mapping[ArtifactRole, frozenset[Audience]] = {
    "observed_pattern": _BOTH,
    "candidate_cif": _BOTH,
    # Refinement output the model legitimately retrieves is evidence, not a leak (meas. R11).
    "fit_curve": _BOTH,
    "native_lst": _BOTH,
    "native_dia": _BOTH,
    "native_par": _BOTH,
    "raw_export": _PUBLIC_EXPERT,
    "context_document": _PUBLIC_EXPERT,
    "harbor_receipt": _PUBLIC_EXPERT,
    "custody_withheld": _NOBODY,
}


def visible(record: str, field: str, audience: Audience) -> bool:
    return audience in VISIBILITY[(record, field)]


def projection_for(principal: Principal) -> Literal["actor", "revealed"]:
    """USER and SERVICE -> revealed; AGENT and MODEL_JUDGE -> actor. A model judge reads only
    blind snapshots (labels.LabelStore), so it never gets the revealed tree."""
    revealed = principal.principal_type in (PrincipalType.USER, PrincipalType.SERVICE)
    return "revealed" if revealed else "actor"


# ---------------------------------------------------------------------------
# Actor projection: the R9 allowlist as closed types.
# ---------------------------------------------------------------------------


def _sources(value: object) -> Iterator[object]:
    if isinstance(value, Sourced):
        yield value.source
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _sources(getattr(value, name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _sources(item)


class ActorCandidateView(Record):
    candidate_id: str
    formula_sum: str
    elements: tuple[str, ...]
    cif: ArtifactRef


class ActorCaseView(Record):
    """Built from the effective context at the agent's run start. On a continuation from a
    human correction, the corrected fields appear with their expert source and nothing else
    of the human ancestor does: no rationale, judgment or preference (KTD12)."""

    root: AttemptRef
    case_id: SafeID
    task: str
    measurement: MeasurementContext
    material: MaterialContext
    pattern: ArtifactRef
    candidates: tuple[ActorCandidateView, ...]

    @model_validator(mode="after")
    def provenance_is_actor_safe(self) -> Self:
        for source in _sources((self.measurement, self.material)):
            if isinstance(source, DocumentSource) and source.channel not in ACTOR_SAFE_CHANNELS:
                raise ValueError(f"a {source.channel} value cannot reach an agent")
        if any("agent" not in ARTIFACT_AUDIENCE[a.role] for a in self._artifacts()):
            raise ValueError("an artifact role withheld from agents reached the case view")
        return self

    def _artifacts(self) -> tuple[ArtifactRef, ...]:
        return (self.pattern, *(c.cif for c in self.candidates))


class PublicCaseView(Record):
    root: AttemptRef
    case_id: SafeID
    source: CaseSource
    exposure: Literal["development_exposed", "unexposed"]
    split: SplitAssignment
    task_kind: Literal["identification", "confirmation"]
    task: str
    pattern: ArtifactRef
    measurement: MeasurementContext
    material: MaterialContext
    candidates: tuple[ActorCandidateView, ...]
    follows_request: AttemptRef | None


def public_case_view(root: AttemptDigest, case: CaseManifest) -> PublicCaseView:
    fields = {name: getattr(case, name) for name in type(case).model_fields if visible("CaseManifest", name, "public")}
    fields["candidates"] = tuple(
        ActorCandidateView.model_validate(c.model_dump(include=set(ActorCandidateView.model_fields)))
        for c in case.candidates
    )
    return PublicCaseView(root=root, **fields)


class PublicAttemptView(Record):
    schema_version: Literal["axa.attempt.v3"]
    kind: Literal["child"]
    parent: AttemptRef
    root: AttemptRef
    depth: int = Field(ge=1)
    actor: Actor
    action: AttemptAction
    tool: ToolIdentity
    inputs: tuple[BlobRef, ...]
    outcome: ToolOutcome
    created_at_utc: datetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        ChildAttempt.model_validate(self.model_dump() | {"rationale": "", "idempotency_key": "public-projection"})
        return self


def public_attempt_view(attempt: ChildAttempt) -> PublicAttemptView:
    return PublicAttemptView(**{
        name: getattr(attempt, name) for name in type(attempt).model_fields
        if visible("ChildAttempt", name, "public")
    })


class ActorOutcomeView(Record):
    status: Literal["completed", "recorded", "refused", "infra_failed", "rejected", "withheld"]
    result: JsonValue  # typed result as JSON, minus artifacts withheld from agents
    artifacts: tuple[ArtifactRef, ...]
    message: str | None  # refusal detail, rejection errors, or the withheld notice


class ActorAttemptView(Record):
    digest: AttemptRef
    parent: AttemptRef
    depth: int
    own: bool
    action: AttemptAction
    rationale: str | None  # own rationale only
    outcome: ActorOutcomeView


class ActorPath(Record):
    """The whole model context a harness may build from: the case view plus the steps of the
    agent's own run, from its RunStart on. Nothing before the RunStart is a step."""

    case: ActorCaseView
    steps: tuple[ActorAttemptView, ...]


# Fields an actor view adds that are not stored record fields.
_VIEW_EXTRAS = {"ActorCaseView": {"root"}, "ActorAttemptView": {"digest", "own"}}
_VIEW_SOURCES = {"ActorCaseView": "CaseManifest", "ActorAttemptView": "ChildAttempt"}


def _actor_safe(value: Sourced | None) -> Sourced | None:
    source = None if value is None else value.source
    if isinstance(source, DocumentSource) and source.channel not in ACTOR_SAFE_CHANNELS:
        return None
    return value


def _actor_material(material: MaterialContext) -> MaterialContext:
    """Optional values from channels that are not actor-safe drop to None, and a precursor whose
    formula is not actor-safe drops. Required fields stay, so the view's validator refuses a
    case whose required context is not actor-safe."""
    return material.model_copy(
        update={
            "precursors": tuple(
                p.model_copy(update={"amount": _actor_safe(p.amount)})
                for p in material.precursors
                if _actor_safe(p.formula) is not None
            ),
            "processing": tuple(
                step.model_copy(
                    update={
                        name: _actor_safe(getattr(step, name))
                        for name in type(step).model_fields
                        if name not in ("order", "operation")
                    }
                )
                for step in material.processing
            ),
            "declared_target": _actor_safe(material.declared_target),
        }
    )


def actor_case_view(root_digest: AttemptDigest, root: RootAttempt, state: AnalysisState) -> ActorCaseView:
    """`state` is the effective context at the agent's run start, so an expert correction reaches
    the agent as a value sourced to that correction and nothing else of the expert's attempt."""
    case = root.case
    return ActorCaseView(
        root=root_digest,
        case_id=case.case_id,
        task=case.task,
        measurement=state.measurement,
        material=_actor_material(state.material),
        pattern=case.pattern,
        candidates=tuple(
            ActorCandidateView(
                candidate_id=c.candidate_id, formula_sum=c.formula_sum, elements=c.elements, cif=c.cif
            )
            for c in case.candidates
        ),
    )


def _actor_outcome(outcome: ToolOutcome) -> ActorOutcomeView:
    match outcome:
        case Completed(contamination=Contamination()):
            return ActorOutcomeView(
                status="withheld", result=None, artifacts=(), message="withheld by the custody scan"
            )
        case Completed():
            return ActorOutcomeView(
                status="completed",
                result=outcome.result.model_dump(mode="json"),
                artifacts=tuple(a for a in outcome.artifacts if "agent" in ARTIFACT_AUDIENCE[a.role]),
                message=None,
            )
        case Recorded():
            message = None
        case Refused():
            message = f"{outcome.reason_code}: {outcome.detail}"
        case InfraFailed():
            message = f"{outcome.cause}: {outcome.detail}"
        case Rejected():
            message = "; ".join(outcome.errors)
    return ActorOutcomeView(status=outcome.status, result=None, artifacts=(), message=message)


def actor_attempt_view(digest: AttemptDigest, attempt: ChildAttempt, viewer: Principal) -> ActorAttemptView:
    own = attempt.actor.principal_id == viewer.principal_id
    return ActorAttemptView(
        digest=digest,
        parent=attempt.parent,
        depth=attempt.depth,
        own=own,
        action=attempt.action,
        rationale=attempt.rationale if own else None,
        outcome=_actor_outcome(attempt.outcome),
    )


class GestureStep(Record):
    """One attempt a gesture created, as the workbench page and the MCP tools receive it: the
    child's own verb, status and fit, and nothing from its parent, actor or annotations, so it
    carries no run identity before a rater's sign-off (D10, R32)."""

    digest: AttemptRef
    parent: AttemptRef
    verb: str
    status: Literal["completed", "recorded", "refused", "infra_failed", "rejected", "withheld"]
    fit: FitSummary | None  # stage, Rwp, GOF, phases with weight and esd
    failure_reason: str | None


def gesture_step(digest: AttemptDigest, attempt: ChildAttempt) -> GestureStep:
    outcome = attempt.outcome
    fit, reason = None, None
    match outcome:
        case Completed(contamination=Contamination()):
            status = "withheld"
        case Completed():
            status, fit = "completed", fit_of(attempt)
            reason = outcome.result.failure_reason if outcome.result.kind == "refine" else None
        case _:
            status, reason = outcome.status, _actor_outcome(outcome).message
    return GestureStep(
        digest=digest, parent=attempt.parent, verb=attempt.action.verb, status=status, fit=fit, failure_reason=reason
    )


# ---------------------------------------------------------------------------
# Revealed projection (the expert in intervene mode, services)
# ---------------------------------------------------------------------------


class RevealedAttemptView(Record):
    digest: AttemptRef
    attempt: Attempt
    state: AnalysisState  # derived on read: context with each field's source, hypothesis, fit
    children: tuple[AttemptRef, ...]
    annotations: tuple[Annotation, ...]


class TreeNodeSummary(Record):
    """One row of the lineage rail (R12)."""

    digest: AttemptRef
    parent: AttemptRef | None
    depth: int
    actor_kind: Literal["root", "agent", "human", "service"]
    principal_id: SafeID | None
    verb: str
    status: str | None
    origin: Literal["live", "imported", "playback"] | None  # playback is labeled (AE8)
    rwp_percent: float | None
    gof: float | None
    phases: tuple[str, ...]
    verdict: str | None
    created_at_utc: datetime | None


class TreeView(Record):
    root: AttemptRef
    nodes: tuple[TreeNodeSummary, ...]


def opening_attempt(tree: TreeView) -> AttemptRef:
    """A case opens on the agent's submitted attempt (R12): the latest agent submit or
    submitted import under the root; the root when there is none."""
    submitted = [
        node
        for node in tree.nodes
        if (node.actor_kind == "agent" and node.verb == "submit")
        or (node.verb == "import_run" and node.verdict is not None)
    ]
    if not submitted:
        return tree.root
    return max(submitted, key=lambda node: (node.created_at_utc, node.depth)).digest


# ---------------------------------------------------------------------------
# Annotations: facts that change no scientific state (R5)
# ---------------------------------------------------------------------------


class Note(Record):
    kind: Literal["note"] = "note"
    author: HumanActor
    created_at_utc: datetime
    text: FreeText = Field(min_length=1, max_length=8000)


class Preference(Record):
    """An explicit pairwise preference naming two digests: the only source of preference pairs
    (R5, R18)."""

    kind: Literal["preference"] = "preference"
    author: HumanActor
    chosen: AttemptRef
    rejected: AttemptRef
    reason: FreeText = Field(min_length=1, max_length=4000)
    created_at_utc: datetime

    @model_validator(mode="after")
    def two_digests(self) -> Self:
        if self.chosen == self.rejected:
            raise ValueError("a preference names two different attempts")
        return self


class SealedEvaluator(Record):
    """The measurement plan's code-scored verifier. Nothing at the hackathon writes one."""

    kind: Literal["sealed_evaluator"] = "sealed_evaluator"
    scorer_version: SafeID
    gate_verdict_sha256: Digest
    truth_receipt_sha256: Digest


class Adjudication(Record):
    """The one correctness-verdict fact. Written only by a sealed_evaluator service (R5)."""

    kind: Literal["adjudication"] = "adjudication"
    source: SealedEvaluator
    verdict: Literal["pass", "fail", "not_scorable"]
    private_detail_sha256: Digest | None
    created_at_utc: datetime


class AttestedCompletion(Record):
    idempotency_key: SafeID
    model_observed: str
    response_id: str
    cost_usd: float | None


class RunClose(Record):
    """Written by the harness at run close: the transcript's canary scan and the per-completion
    model identities an MCP server never sees (R9, R11, AE4)."""

    kind: Literal["run_close"] = "run_close"
    author: ServiceActor
    run_start: AttemptRef
    transcript_sha256: Digest
    completions: tuple[AttestedCompletion, ...]
    scan: CustodyScan
    contamination: Contamination | None
    created_at_utc: datetime

    @model_validator(mode="after")
    def match_names_the_canary(self) -> Self:
        if (self.scan == "match") != (self.contamination is not None):
            raise ValueError("a transcript match names its canary, and only a match does")
        return self


class LoadEvent(Record):
    kind: Literal["load_event"] = "load_event"
    principal_id: SafeID
    loaded_at_utc: datetime
    via: Literal["mcp_ingest", "script"]


Annotation = Annotated[
    Note | Preference | Adjudication | ReplayCheck | RunClose | LoadEvent,
    Field(discriminator="kind"),
]


class RevealEvent(Record):
    """First revealed view of a root by a human: the start of the expert's wall time (R20)."""

    principal_id: SafeID
    root: AttemptRef
    first_revealed_at_utc: datetime


# --- wire requests. None carries an author: the store stamps it from the principal. ----------


class NoteRequest(Record):
    kind: Literal["note"] = "note"
    text: FreeText = Field(min_length=1, max_length=8000)


class PreferenceRequest(Record):
    kind: Literal["preference"] = "preference"
    chosen: AttemptRef
    rejected: AttemptRef
    reason: FreeText = Field(min_length=1, max_length=4000)


RecordRequest = Annotated[
    NoteRequest | PreferenceRequest, Field(discriminator="kind")
]


def may_record(principal: Principal, request: NoteRequest | PreferenceRequest) -> str | None:
    """Return a refusal code, or None when allowed. Notes and preferences from agent principals
    are refused (R29); rubric labels and sign-offs are labels.py records with their own gate; services write adjudications and run closes through their
    own store methods."""
    if principal.principal_type is PrincipalType.AGENT:
        return f"agent_cannot_record_{request.kind}"
    if principal.principal_type is PrincipalType.SERVICE:
        return f"service_cannot_record_{request.kind}"
    if principal.principal_type is PrincipalType.MODEL_JUDGE:
        return f"model_judge_cannot_record_{request.kind}"
    return None


def _assert_visibility_complete() -> None:
    for model in (CaseManifest, ChildAttempt):
        missing = {f for f in model.model_fields if (model.__name__, f) not in VISIBILITY}
        if missing:
            raise RuntimeError(f"{model.__name__} fields without a visibility row: {sorted(missing)}")
    roles = set(get_args(ArtifactRole))
    if roles != set(ARTIFACT_AUDIENCE):
        raise RuntimeError(f"artifact roles without an audience row: {sorted(roles ^ set(ARTIFACT_AUDIENCE))}")
    for view in (ActorCaseView, ActorAttemptView):
        source = _VIEW_SOURCES[view.__name__]
        for field in set(view.model_fields) - _VIEW_EXTRAS[view.__name__]:
            row = VISIBILITY.get((source, field))
            if row is None or not row & {"agent", "agent_own"}:
                raise RuntimeError(f"{view.__name__}.{field} exposes a field withheld from agents")
    for view, source, extras in ((PublicCaseView, CaseManifest, {"root"}), (PublicAttemptView, ChildAttempt, set())):
        public = {name for name in source.model_fields if visible(source.__name__, name, "public")}
        if set(view.model_fields) - extras != public:
            raise RuntimeError(f"{view.__name__} differs from the public visibility column")
    if any("public" not in ARTIFACT_AUDIENCE[role] for role in ("observed_pattern", "candidate_cif")):
        raise RuntimeError("public case artifacts lack a public audience")


_assert_visibility_complete()
RevealedAttemptView.model_rebuild()
