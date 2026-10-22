"""Derived facts about paths: provenance, runs, episodes, exclusions, eligibility and rights.

Target path: src/phaseeval/workbench/policy.py (public). Nothing here is stored. Every function is a
pure transform from attempts and annotations plus the versioned inputs named in its signature.

The separate facts of R5, each its own type and owner:

    expert edit happened       the child attempt itself (HumanActor)    human_contributions(path)
    expert judgment            labels.Label and SignOff on a frozen snapshot, in the label store
    pairwise preference        review.Preference, naming two digests     on the annotation ledger
    adjudicated correctness    review.Adjudication, sealed evaluator     adjudication_for(...)
    eligibility per use        EligibilityDecision, versioned, default unreviewed
    rights per artifact        ArtifactRights, unknown kept as unknown

Replaces AttemptBundle.training_admission (Literal["unreviewed"]) and the
EpisodeTrace.training_eligible / promotion_eligible bools for attempt-derived data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Literal, Self, get_args

from pydantic import AfterValidator, Field, model_validator

from .model import (
    AgentActor,
    ArtifactRef,
    ArtifactRole,
    Attempt,
    AttemptDigest,
    AttemptRef,
    BlobRef,
    CaseManifest,
    ChildAttempt,
    Complete,
    Completed,
    Contamination,
    CustodyScan,
    Digest,
    Environment,
    FitSummary,
    Halt,
    HumanActor,
    ImportedInvalid,
    ImportedRecord,
    ImportedRunResult,
    ImportedSubmitted,
    ImportedUnsuccessful,
    ImportRun,
    InfraCause,
    InfraFailure,
    NextAction,
    Record,
    RefineResult,
    Rejected,
    RunStart,
    SafeID,
    Submit,
    TaskKind,
    Verdict,
    halt_validity,
)
from .review import Adjudication, Annotation, LoadEvent, Note, Preference, RunClose

AttemptPath = tuple[tuple[AttemptDigest, Attempt], ...]  # root first

# ---------------------------------------------------------------------------
# Provenance (R6; measurement plan R2). The tree can mint only TreeProvenance.
# ---------------------------------------------------------------------------

ProvenanceClass = Literal["expert", "weighed", "expert_reviewed_automated", "pipeline", "llm", "generated"]
TreeProvenance = Literal["expert_reviewed_automated", "pipeline", "llm", "generated"]
TruthProvenance = Literal["expert", "weighed"]
assert not set(get_args(TreeProvenance)) & set(get_args(TruthProvenance))
assert set(get_args(TreeProvenance)) | set(get_args(TruthProvenance)) == set(get_args(ProvenanceClass))


def _external_locator(value: str) -> str:
    lowered = value.lower()
    if lowered.startswith(("workbench:", "attempt:", "axa://")) or (
        len(value) == 64 and all(c in "0123456789abcdef" for c in lowered)
    ):
        raise ValueError("a primary source cannot be a workbench attempt")
    return value


class PrimarySourceCitation(Record):
    """Has no attempt field and refuses workbench locators [C3]."""

    kind: Literal["publication_table", "publication_figure", "lab_record", "weighed_preparation"]
    locator: Annotated[str, Field(min_length=1), AfterValidator(_external_locator)]
    sha256: Digest | None = None


class ReferencePhase(Record):
    formula: str
    space_group: int | None = Field(default=None, ge=1, le=230)
    weight_percent: float | None = Field(default=None, ge=0, le=100)


class SealedTruth(Record):
    """The sealed-truth shape the measurement lane admits. Expert or weighed provenance with a
    primary citation only, so an intervention output fails at the constructor (AE3) [C3]."""

    kind: Literal["sealed_truth"] = "sealed_truth"
    provenance: TruthProvenance
    source: PrimarySourceCitation
    phases: tuple[ReferencePhase, ...] = Field(min_length=1)


def _children(path: AttemptPath) -> tuple[tuple[AttemptDigest, ChildAttempt], ...]:
    return tuple((digest, attempt) for digest, attempt in path if isinstance(attempt, ChildAttempt))


def provenance(path: AttemptPath, case: CaseManifest) -> TreeProvenance:
    """synthetic case -> generated; any human attempt -> expert_reviewed_automated (a human acts
    only after the revealed view); any agent attempt or import -> llm; otherwise pipeline."""
    if case.source.kind == "synthetic":
        return "generated"
    children = [attempt for _, attempt in _children(path)]
    if any(isinstance(attempt.actor, HumanActor) for attempt in children):
        return "expert_reviewed_automated"
    if any(isinstance(attempt.actor, AgentActor) or isinstance(attempt.action, ImportRun) for attempt in children):
        return "llm"
    return "pipeline"


class HumanContribution(Record):
    attempt: AttemptRef
    principal_id: SafeID
    kind: Literal["phase_edit", "refine", "measurement_correction", "material_correction"]


_CONTRIBUTION_KIND: Mapping[str, Literal["phase_edit", "refine", "measurement_correction", "material_correction"]] = {
    "add_phases": "phase_edit",
    "remove_phases": "phase_edit",
    "replace_phase": "phase_edit",
    "refine": "refine",
    "correct_measurement": "measurement_correction",
    "correct_material": "material_correction",
}


def human_contributions(path: AttemptPath) -> tuple[HumanContribution, ...]:
    """Fact 1. Empty means no expert touched this path. A refused edit is still an expert's
    attempt on the path, so it counts."""
    return tuple(
        HumanContribution(
            attempt=digest, principal_id=attempt.actor.principal_id, kind=_CONTRIBUTION_KIND[attempt.action.verb]
        )
        for digest, attempt in _children(path)
        if isinstance(attempt.actor, HumanActor) and attempt.action.verb in _CONTRIBUTION_KIND
    )


class StepProvenance(Record):
    """Who acted, with what inputs, tool and version, from which parent state (R1)."""

    attempt: AttemptRef
    parent: AttemptRef | None
    actor_kind: Literal["root", "agent", "human", "service"]
    principal_id: SafeID | None
    model_observed: str | None  # inline receipt or RunClose attestation; None = unverified
    verb: str
    tool: str | None
    tool_version: str | None
    origin: Literal["live", "imported", "playback"] | None
    inputs: tuple[Digest, ...]
    created_at_utc: datetime | None


def _attested(annotations: Mapping[AttemptDigest, tuple[Annotation, ...]]) -> dict[str, str]:
    """Idempotency key -> model observed, from every RunClose on the path (AE4)."""
    return {
        completion.idempotency_key: completion.model_observed
        for notes in annotations.values()
        for note in notes
        if isinstance(note, RunClose)
        for completion in note.completions
    }


def step_provenance(path: AttemptPath, annotations: Mapping[AttemptDigest, tuple[Annotation, ...]]) -> tuple[StepProvenance, ...]:
    root_digest, root = path[0]
    loads = [note for note in annotations.get(root_digest, ()) if isinstance(note, LoadEvent)]
    steps = [
        StepProvenance(
            attempt=root_digest,
            parent=None,
            actor_kind="root",
            principal_id=loads[0].principal_id if loads else None,
            model_observed=None,
            verb="load_case",
            tool=None,
            tool_version=None,
            origin=None,
            inputs=(root.case.pattern.sha256, *(c.cif.sha256 for c in root.case.candidates)),
            created_at_utc=loads[0].loaded_at_utc if loads else None,
        )
    ]
    attested = _attested(annotations)
    for digest, attempt in _children(path):
        actor, outcome = attempt.actor, attempt.outcome
        observed = None
        if isinstance(actor, AgentActor):
            observed = actor.completion.model_observed if actor.completion else attested.get(attempt.idempotency_key)
        steps.append(
            StepProvenance(
                attempt=digest,
                parent=attempt.parent,
                actor_kind=actor.kind,
                principal_id=actor.principal_id,
                model_observed=observed,
                verb=attempt.action.verb,
                tool=attempt.tool.name,
                tool_version=attempt.tool.version,
                origin=outcome.origin.origin if isinstance(outcome, Completed) else None,
                inputs=attempt.inputs,
                created_at_utc=attempt.created_at_utc,
            )
        )
    return tuple(steps)


# ---------------------------------------------------------------------------
# Runs and episode status. One union, so an invalid run has no outcome [C3] (R11, AE2).
# ---------------------------------------------------------------------------


class Open(Record):
    status: Literal["open"] = "open"


class Submitted(Record):
    status: Literal["submitted"] = "submitted"
    answer_at: AttemptRef
    verdict: Verdict | Literal["NOT_RECORDED"]  # an imported answer records no verdict (R11)
    next_action: NextAction | str | None  # typed next action, or the requested data type


class Unsuccessful(Record):
    status: Literal["unsuccessful"] = "unsuccessful"
    reason: Literal["budget_exhausted", "malformed_tool_use", "no_submission"]


class Invalid(Record):
    status: Literal["invalid"] = "invalid"
    cause: InfraCause | Literal["abandoned", "preflight_failed", "unverified_model_identity", "cancelled"]


class Contaminated(Record):
    status: Literal["contaminated"] = "contaminated"
    contamination: Contamination
    at: AttemptRef


RunStatus = Annotated[Open | Submitted | Unsuccessful | Invalid | Contaminated, Field(discriminator="status")]


class BudgetUse(Record):
    model_actions: int
    wall_seconds: float | None
    executor_seconds: float
    cost_usd: float | None  # None when any completion lacked a cost: unknown, not zero


class RunReceipt(Record):
    start: AttemptRef  # the RunStart, or the ImportRun for an imported trial
    origin: Literal["live", "imported"]
    environment: Environment  # imported Harbor trials are harbor_container
    agent: str  # principal id for live runs, recorded model id for imports
    model_requested: str | None
    models_observed: tuple[str, ...]
    every_completion_verified: bool
    independent: bool  # starts at the root
    complete: bool  # False for an imported trial marked incomplete (R11)
    consumed: BudgetUse
    malformed_attempts: int
    status: RunStatus


def _import_receipt(digest: AttemptDigest, attempt: ChildAttempt) -> RunReceipt:
    """An imported trial: the recorded model id is its agent; nothing executed in the workbench,
    so its executor time here is zero and its own wall time and cost are what it recorded."""
    outcome = attempt.outcome
    assert isinstance(outcome, Completed) and isinstance(outcome.result, ImportedRunResult)
    result = outcome.result
    status: RunStatus
    match result.end:
        case ImportedSubmitted(answer=answer):
            next_action = answer.next_action or (answer.measurement.requested_data_type if answer.measurement else None)
            status = Submitted(answer_at=digest, verdict=answer.verdict, next_action=next_action)
        case ImportedUnsuccessful(reason=reason):
            status = Unsuccessful(reason=reason)
        case ImportedInvalid(cause=cause):
            status = Invalid(cause=cause)
    if outcome.contamination is not None:
        status = Contaminated(contamination=outcome.contamination, at=digest)
    return RunReceipt(
        start=digest,
        origin="imported",
        environment="harbor_container",
        agent=result.agent.model_id,
        model_requested=None,
        models_observed=(),
        every_completion_verified=False,
        independent=attempt.parent == attempt.root,
        complete=isinstance(result.completeness, Complete),
        consumed=BudgetUse(
            model_actions=len(result.trajectory),
            wall_seconds=result.wall_seconds,
            executor_seconds=0.0,
            cost_usd=result.cost_usd,
        ),
        malformed_attempts=0,
        status=status,
    )


def _live_receipt(
    children: tuple[tuple[AttemptDigest, ChildAttempt], ...],
    index: int,
    annotations: Mapping[AttemptDigest, tuple[Annotation, ...]],
    now: datetime,
) -> RunReceipt:
    start_digest, start = children[index]
    assert isinstance(start.action, RunStart) and isinstance(start.actor, AgentActor)
    run_start, agent = start.action, start.actor
    segment: list[tuple[AttemptDigest, ChildAttempt]] = [children[index]]
    status: RunStatus | None = None
    for digest, attempt in children[index + 1 :]:
        if isinstance(attempt.action, Halt):  # the agent's own halt, or the store guard's
            segment.append((digest, attempt))
            reason = attempt.action.reason
            if halt_validity(reason) == "unsuccessful":
                status = Unsuccessful(reason=reason.kind)
            else:
                status = Invalid(cause=reason.cause if isinstance(reason, InfraFailure) else "cancelled")
            break
        if attempt.actor.principal_id != agent.principal_id:
            break
        segment.append((digest, attempt))
        if isinstance(attempt.action, Submit):
            answer = attempt.action.answer
            next_action = getattr(answer, "next_action", None) or getattr(
                getattr(answer, "measurement", None), "requested_data_type", None
            )
            status = Submitted(answer_at=digest, verdict=answer.verdict, next_action=next_action)
            break
    if status is None:
        expired = (now - start.created_at_utc).total_seconds() > run_start.budget.max_wall_seconds
        status = Invalid(cause="abandoned") if expired else Open()
    for note in annotations.get(start_digest, ()):
        if isinstance(note, RunClose) and note.run_start == start_digest and note.contamination is not None:
            status = Contaminated(contamination=note.contamination, at=start_digest)
    for digest, attempt in segment:
        if isinstance(attempt.outcome, Completed) and attempt.outcome.contamination is not None:
            status = Contaminated(contamination=attempt.outcome.contamination, at=digest)
            break
    completions = [
        attempt.actor.completion
        for _, attempt in segment
        if isinstance(attempt.actor, AgentActor) and attempt.actor.completion is not None
    ]
    attested = [
        completion
        for note in annotations.get(start_digest, ())
        if isinstance(note, RunClose) and note.run_start == start_digest
        for completion in note.completions
    ]
    observed = sorted({c.model_observed for c in completions} | {c.model_observed for c in attested})
    costs = [c.cost_usd for c in completions] or [c.cost_usd for c in attested]
    return RunReceipt(
        start=start_digest,
        origin="live",
        environment=agent.environment,
        agent=agent.principal_id,
        model_requested=run_start.model_requested,
        models_observed=tuple(observed),
        every_completion_verified=bool(observed) and observed == [run_start.model_requested],
        independent=start.parent == start.root,
        complete=True,
        consumed=BudgetUse(
            model_actions=sum(1 for _, a in segment[1:] if isinstance(a.actor, AgentActor)),
            wall_seconds=(segment[-1][1].created_at_utc - start.created_at_utc).total_seconds(),
            executor_seconds=sum(a.outcome.elapsed_seconds for _, a in segment if isinstance(a.outcome, Completed)),
            cost_usd=None if not costs or any(c is None for c in costs) else sum(c for c in costs if c is not None),
        ),
        malformed_attempts=sum(1 for _, a in segment if isinstance(a.outcome, Rejected)),
        status=status,
    )


def runs_on(path: AttemptPath, annotations: Mapping[AttemptDigest, tuple[Annotation, ...]], now: datetime) -> tuple[RunReceipt, ...]:
    """Segments from each RunStart to its Submit or Halt, to the first attempt by someone else,
    or to the path's end; plus one receipt per ImportRun. A RunClose match makes a run
    Contaminated even after a submit (AE4). An open run past its wall cap is Invalid(abandoned)."""
    children = _children(path)
    receipts: list[RunReceipt] = []
    for index, (digest, attempt) in enumerate(children):
        if isinstance(attempt.action, ImportRun):
            receipts.append(_import_receipt(digest, attempt))
        elif isinstance(attempt.action, RunStart) and isinstance(attempt.actor, AgentActor):
            receipts.append(_live_receipt(children, index, annotations, now))
    return tuple(receipts)


class NoAgentRun(Record):
    status: Literal["no_agent_run"] = "no_agent_run"


EpisodeStatus = Annotated[
    Open | Submitted | Unsuccessful | Invalid | Contaminated | NoAgentRun, Field(discriminator="status")
]


def episode_status(runs: tuple[RunReceipt, ...]) -> EpisodeStatus:
    """Contaminated beats invalid beats the last run's status; no runs -> NoAgentRun."""
    if not runs:
        return NoAgentRun()
    for kind in (Contaminated, Invalid):
        for run in runs:
            if isinstance(run.status, kind):
                return run.status
    return runs[-1].status


class EpisodeEffort(Record):
    """Per-episode effort and cost for the dashboard (R20). None means unknown, shown as such."""

    expert_actions: int  # human attempts plus judgments, preferences and notes
    expert_wall_seconds: float | None  # first reveal to last human record
    agent_wall_seconds: float | None
    bgmn_seconds: float
    cost_usd: float | None


class Episode(Record):
    """One root-to-leaf path: the unit an export row and a dashboard row describe."""

    root: AttemptRef
    leaf: AttemptRef
    attempts: tuple[AttemptRef, ...]
    runs: tuple[RunReceipt, ...]
    status: EpisodeStatus
    provenance: TreeProvenance
    human_contributions: tuple[HumanContribution, ...]
    reference_exposed: bool  # any human attempt or judgment on the path with exposure "exposed" (R16)
    task_kind: TaskKind  # the R30 confirmation tag, copied from the root
    custody_scan: CustodyScan  # weakest scan over the path's completed outcomes and run closes
    effort: EpisodeEffort
    steps: tuple[StepProvenance, ...]


_SCAN_ORDER: tuple[CustodyScan, ...] = ("match", "not_configured", "clean")


def _effort(
    path: AttemptPath, annotations: Mapping[AttemptDigest, tuple[Annotation, ...]], runs: tuple[RunReceipt, ...]
) -> EpisodeEffort:
    """Expert wall time runs from the first to the last human record on the path; the store's
    first-reveal event is outside the path, so a single human record has no duration (None)."""
    children = [attempt for _, attempt in _children(path)]
    human_times = [a.created_at_utc for a in children if isinstance(a.actor, HumanActor)]
    human_notes = [
        note
        for digest, _ in path
        for note in annotations.get(digest, ())
        if isinstance(note, Preference | Note)
    ]
    human_times += [note.created_at_utc for note in human_notes]
    walls = [run.consumed.wall_seconds for run in runs]
    costs = [run.consumed.cost_usd for run in runs]
    return EpisodeEffort(
        expert_actions=sum(1 for a in children if isinstance(a.actor, HumanActor)) + len(human_notes),
        expert_wall_seconds=(max(human_times) - min(human_times)).total_seconds() if len(human_times) > 1 else None,
        agent_wall_seconds=None if not walls or None in walls else sum(w for w in walls if w is not None),
        bgmn_seconds=sum(
            a.outcome.elapsed_seconds
            for a in children
            if isinstance(a.outcome, Completed) and isinstance(a.outcome.result, RefineResult)
        ),
        cost_usd=None if not costs or None in costs else sum(c for c in costs if c is not None),
    )


def episode(path: AttemptPath, case: CaseManifest, annotations: Mapping[AttemptDigest, tuple[Annotation, ...]], now: datetime) -> Episode:
    runs = runs_on(path, annotations, now)
    children = _children(path)
    scans = [a.outcome.custody_scan for _, a in children if isinstance(a.outcome, Completed)]
    scans += [
        note.scan for digest, _ in path for note in annotations.get(digest, ()) if isinstance(note, RunClose)
    ]
    exposed = any(
        isinstance(a.actor, HumanActor) and a.actor.reference_exposure == "exposed" for _, a in children
    )
    return Episode(
        root=path[0][0],
        leaf=path[-1][0],
        attempts=tuple(digest for digest, _ in path),
        runs=runs,
        status=episode_status(runs),
        provenance=provenance(path, case),
        human_contributions=human_contributions(path),
        reference_exposed=exposed,
        task_kind=case.task_kind,
        custody_scan=next((s for s in _SCAN_ORDER if s in scans), "not_configured"),
        effort=_effort(path, annotations, runs),
        steps=step_provenance(path, annotations),
    )


ExclusionReason = Literal["contaminated", "invalid", "open", "no_agent_run", "development_session"]


def exclusion_reasons(episode: Episode) -> tuple[ExclusionReason, ...]:
    """Why an episode goes to excluded.jsonl instead of episodes.jsonl [C1]. A run from a
    host-side MCP client is a development session (R31)."""
    reasons: list[ExclusionReason] = []
    status = episode.status.status
    if status in ("contaminated", "invalid", "open", "no_agent_run"):
        reasons.append(status)
    if any(run.environment == "host_development" for run in episode.runs):
        reasons.append("development_session")
    return tuple(reasons)


def adjudication_for(annotations: tuple[Annotation, ...]) -> Adjudication | None:
    """The latest Adjudication on an answer attempt, or None. Nothing writes one this week."""
    found = [note for note in annotations if isinstance(note, Adjudication)]
    return max(found, key=lambda note: note.created_at_utc) if found else None


# ---------------------------------------------------------------------------
# Eligibility: a versioned decision per use and dataset version, default unreviewed (R5)
# ---------------------------------------------------------------------------

# Sealed test use is not a member: no attempt-derived record can be test truth (R6).
Use = Literal["sft_target", "preference", "judge_calibration", "dev_regression", "rl_train"]


class EligibilityDecision(Record):
    use: Use
    dataset_version: SafeID
    policy_version: SafeID
    status: Literal["unreviewed", "eligible", "ineligible"]
    decided_by: SafeID | None
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def decisions_have_deciders(self) -> Self:
        if (self.status == "unreviewed") != (self.decided_by is None):
            raise ValueError("a reviewed decision names who decided; an unreviewed one names nobody")
        return self


class EligibilityLedger(Record):
    """The owner's decisions as data, digested into the export manifest."""

    policy_version: SafeID
    decisions: Mapping[AttemptDigest, tuple[EligibilityDecision, ...]]  # keyed by episode leaf


def eligibility_for(episode: Episode, ledger: EligibilityLedger, dataset_version: SafeID) -> tuple[EligibilityDecision, ...]:
    """One decision per Use for this dataset version; an absent one is unreviewed."""
    given = {d.use: d for d in ledger.decisions.get(episode.leaf, ()) if d.dataset_version == dataset_version}
    return tuple(
        given.get(use)
        or EligibilityDecision(
            use=use,
            dataset_version=dataset_version,
            policy_version=ledger.policy_version,
            status="unreviewed",
            decided_by=None,
        )
        for use in get_args(Use)
    )


# ---------------------------------------------------------------------------
# Rights per artifact: training use and redistribution are separate rights [C1]
# ---------------------------------------------------------------------------


class Permitted(Record):
    right: Literal["permitted"] = "permitted"
    license_id: str
    attribution: str
    verified_from: Literal["source_page", "owner_statement"]


class Restricted(Record):
    right: Literal["restricted"] = "restricted"
    observation: str


class RightUnknown(Record):
    right: Literal["unknown"] = "unknown"
    observation: str


Right = Annotated[Permitted | Restricted | RightUnknown, Field(discriminator="right")]


class ArtifactRights(Record):
    artifact: BlobRef
    role: ArtifactRole
    training_use: Right
    redistribution: Right


class ContributorTerms(Record):
    """Per human contributor: expert traces need their own permission."""

    principal_id: SafeID
    training_use: Right
    redistribution: Right


class ModelOutputTerms(Record):
    """Per agent model: provider terms decide whether its outputs may be published (Q6)."""

    model_id: str
    training_use: Right
    redistribution: Right


class RightsRegistry(Record):
    version: SafeID
    artifacts: Mapping[BlobRef, ArtifactRights]
    contributors: Mapping[SafeID, ContributorTerms]
    model_outputs: Mapping[str, ModelOutputTerms]


def _fit_refs(fit: FitSummary | None) -> tuple[ArtifactRef, ...]:
    return () if fit is None or fit.curve is None else (fit.curve,)


def rights_for(episode: Episode, path: AttemptPath, registry: RightsRegistry) -> tuple[ArtifactRights, ...]:
    """One entry per artifact the episode references (pattern, each CIF, curves, receipts),
    RightUnknown for any artifact the registry does not list. An input whose role no record on
    the path names raises: an artifact is never exported without a role."""
    case = path[0][1].case
    roles: dict[str, ArtifactRole] = {case.pattern.sha256: "observed_pattern"}
    cifs = {c.cif.sha256 for c in case.candidates}
    for _, attempt in _children(path):
        outcome = attempt.outcome
        refs: list[ArtifactRef] = []
        if isinstance(outcome, Completed):
            refs += outcome.artifacts
            if isinstance(outcome.origin, ImportedRecord):
                refs += outcome.origin.receipts
            result = outcome.result
            if isinstance(result, RefineResult):
                refs += _fit_refs(result.final)
                refs += [stage.curve for stage in result.stages if stage.curve is not None]
            else:
                refs += _fit_refs(result.basis_fit)
        for ref in refs:
            roles[ref.sha256] = ref.role
        if isinstance(attempt.action, ImportRun):
            roles |= dict.fromkeys(attempt.action.harbor.cif_sha256.values(), "candidate_cif")
        for sha in attempt.inputs:
            if sha in cifs:
                roles[sha] = "candidate_cif"
            elif sha not in roles:
                raise ValueError(f"input {sha} of {attempt.action.verb} has no artifact role on the path")
    unknown = RightUnknown(observation=f"not listed in rights registry {registry.version}")
    return tuple(
        registry.artifacts.get(sha)
        or ArtifactRights(artifact=sha, role=role, training_use=unknown, redistribution=unknown)
        for sha, role in sorted(roles.items())
    )


# ---------------------------------------------------------------------------
# Fixed v0 inputs for the dev cases (overnight U6). Versioned data, digested into each export.
# ---------------------------------------------------------------------------

ELIGIBILITY_LEDGER_V0 = EligibilityLedger(policy_version="agentxrd-eligibility-v0", decisions={})  # all unreviewed

# The case collection's license as the U2 case package recorded it from the Zenodo record page.
COLLECTION_LICENSES_V0: Mapping[str, tuple[str, str]] = {
    "dara_zenodo_17410051": (
        "CC-BY-4.0",
        "Fei, McDermott, Rom, Wang, Ceder. DARA benchmark (precursor mixtures), doi:10.5281/zenodo.17410051",
    ),
    "precursor_genome_zenodo_21285546": (
        "CC-BY-4.0",
        "Walters, McDermott, Rendy, Fei, Ceder (LBNL). Precursor Genome, doi:10.5281/zenodo.21285546",
    ),
}

# Origin Q6: only the six local-model harness-v2 episodes are publishable until the owner decides
# on Astra outputs. v1 and v2 trials record the same agent identity, so the release names the jobs.
PUBLIC_JOBS_V0: frozenset[str] = frozenset(
    f"u6-{case}-qwen-{series}" for case in ("dev006", "dev102") for series in ("006", "007", "008")
)


def rights_registry_v0(cases: tuple[CaseManifest, ...]) -> RightsRegistry:
    """Each dev case's pattern under its collection license; every other artifact (COD CIFs,
    curves, receipts) stays unknown. Local-model outputs are publishable by the owner's Q6 rule;
    Astra outputs wait on the owner."""
    artifacts = {}
    for case in cases:
        license_entry = COLLECTION_LICENSES_V0.get(case.source.collection)
        if license_entry is None:
            continue
        permitted = Permitted(license_id=license_entry[0], attribution=license_entry[1], verified_from="source_page")
        artifacts[case.pattern.sha256] = ArtifactRights(
            artifact=case.pattern.sha256, role="observed_pattern", training_use=permitted, redistribution=permitted
        )
    q6 = Permitted(license_id="owner-rule-Q6", attribution="AgentXRD harness v2 local-model run", verified_from="owner_statement")
    pending = RightUnknown(observation="provider terms pending the owner's Q6 decision")
    return RightsRegistry(
        version="agentxrd-rights-v0",
        artifacts=artifacts,
        contributors={},
        model_outputs={
            "qwen36-27b": ModelOutputTerms(model_id="qwen36-27b", training_use=pending, redistribution=q6),
            "gpt-6-astra": ModelOutputTerms(model_id="gpt-6-astra", training_use=pending, redistribution=pending),
        },
    )
