"""Review records, cohort stages and atomic exposure stamps."""
from __future__ import annotations

import enum
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from phaseeval.kernel.errors import Conflict, Denied
from phaseeval.kernel.ids import HEX64, SAFE_ID, Family, Handle, SnapshotRef
from phaseeval.kernel.ledger import KeyShape, Ledger, Mode

from phaseeval.api.auth import (
    LABEL_MODE_ROLES,
    AuthorizationError,
    CaseOperation,
    Principal,
    PrincipalType,
)

from . import snapshot as snapshots
from .model import (
    FreeText,
    Attempt,
    AttemptRef,
    ChildAttempt,
    Completed,
    Digest,
    HumanActor,
    ImportedRunResult,
    ImportRun,
    Record,
    RootAttempt,
    SafeID,
    canonical_bytes,
)
from .store import (
    AttemptAccessError,
    CustodyError,
    WorkbenchStore,
)

RUBRIC_VERSION = "agentxrd-rubric-v1"
GUIDANCE_VERSION = "agentxrd-guidance-v1"
PERMISSION_TEXT_VERSION = "agentxrd-permission-v1"

# The frozen rubric (origin Q9, Periodic's process): two per-phase items, two per-analysis items.
RubricItemId = Literal[
    "phase_supported",
    "phase_plausible",
    "amounts_defensible",
    "fit_supports_conclusion",
]
PHASE_ITEMS: frozenset[RubricItemId] = frozenset({"phase_supported", "phase_plausible"})
ANALYSIS_ITEMS: frozenset[RubricItemId] = frozenset({"amounts_defensible", "fit_supports_conclusion"})
RUBRIC_TEXT: Mapping[RubricItemId, str] = {
    "phase_supported": "Is this phase supported by the pattern?",
    "phase_plausible": "Is this phase chemically plausible for this sample?",
    "amounts_defensible": "Are the reported amounts defensible on their basis?",
    "fit_supports_conclusion": "Does the fit support the conclusion?",
}

RubricAnswer = Literal["yes", "no", "cannot_assess"]
PeakCause = Literal["missing_phase", "already_fitted", "noise_or_profile", "cannot_assess"]
SignOffVerdict = Literal["would_sign", "would_not_sign", "unresolved"]
RaterKind = Literal["expert", "owner", "model_judge"]
Exposure = Literal["blind", "revealed"]
StepMark = Literal["wrong_move", "went_wrong_here", "good_move", "note"]


class JudgeIdentity(Record):
    """Present only on model_judge records: which model answered under which frozen prompt.
    Each judge label comes from a fresh tool-less invocation whose whole input is the frozen
    prompt plus one snapshot; `input_sha256` is the sha256 of that exact input payload, and the
    judge script refuses a payload carrying any custody field (PI security directive)."""

    model: str = Field(min_length=1, max_length=200)
    prompt_sha256: Digest
    input_sha256: Digest
    run_id: SafeID
    judge_is_actor_model: bool  # the judge's model produced the episode it judges (KTD11)


class JudgeRunManifest(Record):
    schema_version: Literal["phaseeval.judge-run.v1"] = "phaseeval.judge-run.v1"
    run_id: SafeID
    model: str = Field(min_length=1, max_length=200)
    prompt_sha256: Digest
    input_scope: Annotated[str, Field(pattern=r"^dataset:[A-Za-z0-9._-]{1,64}$")]
    input_sha256: Mapping[Handle, Digest] = Field(min_length=1)


class Rater(Record):
    """Stamped by the LabelStore from the authenticated principal, never from the request: the
    kind derives from the principal type and roles (rater_kind_for). No request type has a
    rater or rater_kind field, and extra="forbid" rejects a body that supplies one."""

    principal_id: SafeID
    kind: RaterKind
    judge: JudgeIdentity | None = None

    @model_validator(mode="after")
    def judge_iff_model(self) -> Self:
        if (self.kind == "model_judge") != (self.judge is not None):
            raise ValueError("a model_judge rater names its judge identity, and only it does")
        return self


class Label(Record):
    schema_version: Literal["label.v1"] = "label.v1"
    snapshot: Digest
    item: RubricItemId
    phase_id: SafeID | None  # the snapshot's stable phase id; required iff item is a phase item
    answer: RubricAnswer
    reason: FreeText = Field(default="", max_length=4000)  # required when answer is cannot_assess
    note: FreeText = Field(default="", max_length=4000)
    rater: Rater
    rubric_version: SafeID = RUBRIC_VERSION
    guidance_version: SafeID = GUIDANCE_VERSION
    exposure: Exposure
    created_at_utc: datetime

    @model_validator(mode="after")
    def shape(self) -> Self:
        if (self.item in PHASE_ITEMS) != (self.phase_id is not None):
            raise ValueError("phase items name a phase_id; analysis items do not")
        if self.answer == "cannot_assess" and not self.reason.strip():
            raise ValueError("cannot_assess stores its reason")
        return self


class PeakCall(Record):
    """The rater's call on one unexplained residual peak of the snapshot (evidence stage)."""

    schema_version: Literal["peak-call.v1"] = "peak-call.v1"
    snapshot: Digest
    peak_id: SafeID  # the snapshot's stable residual peak id
    cause: PeakCause
    note: FreeText = Field(default="", max_length=4000)
    rater: Rater
    exposure: Exposure
    created_at_utc: datetime


class SignOff(Record):
    schema_version: Literal["signoff.v1"] = "signoff.v1"
    snapshot: Digest
    root: AttemptRef  # the pattern; reveal is per rater and root
    verdict: SignOffVerdict
    comment: FreeText = Field(default="", max_length=8000)
    rater: Rater
    exposure: Exposure
    created_at_utc: datetime


class StepRef(Record):
    """One step of a trajectory. `attempt` holds the step: an imported run (ImportRun) for a
    recorded trajectory, or a tree child attempt. `call_index` is the RecordedToolCall.index of
    the step's first call in an imported trajectory; None when the attempt itself is the step."""

    attempt: AttemptRef
    call_index: int | None = Field(default=None, ge=0)
    step_kind: Literal["refine", "segment", "tool_call", "attempt"]
    label: str = Field(default="", max_length=300)  # display text, e.g. the refine run label


class StepFeedbackV1(Record):
    schema_version: Literal["step-feedback.v1"] = "step-feedback.v1"
    step: StepRef
    mark: StepMark
    note: FreeText = Field(default="", max_length=4000)
    rater: Rater
    exposure: Exposure
    created_at_utc: datetime

    @model_validator(mode="after")
    def note_carries_a_note(self) -> Self:
        if self.mark == "note" and not self.note.strip():
            raise ValueError("a note mark carries text")
        return self


class StepFeedback(Record):
    schema_version: Literal["phaseeval.step-feedback.v2"] = "phaseeval.step-feedback.v2"
    snapshot: SnapshotRef
    attempt: AttemptRef
    call_index: int | None = Field(default=None, ge=0)
    mark: Literal["wrong_move", "went_wrong_here", "good_move", "note"]
    note: FreeText = ""
    rater: Rater
    exposure: Exposure
    created_at_utc: datetime

    @model_validator(mode="after")
    def note_carries_a_note(self) -> Self:
        if self.mark == "note" and not self.note.strip():
            raise ValueError("a note mark carries text")
        return self


AnyStepFeedback = Annotated[StepFeedback | StepFeedbackV1, Field(discriminator="schema_version")]


class PermissionRecord(Record):
    """R33: separate choices, one record per decision; the latest per principal governs.
    A withdrawal is a new record with participation_and_storage false."""

    schema_version: Literal["permission.v1"] = "permission.v1"
    principal_id: SafeID
    permission_text_version: SafeID = PERMISSION_TEXT_VERSION
    participation_and_storage: bool
    internal_evaluation_and_calibration: bool
    public_release: bool
    training: bool
    public_identity_disclosure: bool
    created_at_utc: datetime


class AssignmentV1(Record):
    """Which human raters review which agent answers of a pattern (root). Written by the admin
    owner. The snapshots are the review cohort the reveal waits on; a snapshot frozen later
    (export, refit, judge) never joins it (Build D12)."""

    schema_version: Literal["assignment.v1"] = "assignment.v1"
    root: AttemptRef
    raters: tuple[SafeID, ...] = Field(min_length=1)
    snapshots: tuple[Digest, ...] = Field(min_length=1)  # agent_answer snapshots of this root
    assigned_by: SafeID
    created_at_utc: datetime


class Assignment(Record):
    schema_version: Literal["phaseeval.assignment.v2"] = "phaseeval.assignment.v2"
    root: AttemptRef
    cohort: int = Field(ge=1)
    version: int = Field(ge=1)
    raters: tuple[SafeID, ...] = Field(min_length=1)
    snapshots: tuple[SnapshotRef, ...] = Field(min_length=1)  # the gating set at this version
    assigned_by: SafeID
    created_at_utc: datetime


AnyAssignment = Annotated[Assignment | AssignmentV1, Field(discriminator="schema_version")]


class RevealState(Record):
    root: AttemptRef
    principal_id: SafeID
    signed_off: bool
    pending_raters: tuple[SafeID, ...]  # assigned raters who have not signed off yet
    revealed: bool  # reference and other raters' labels visible to this rater


class RevealOpened(Record):
    """The first time the store saw this rater's reveal condition hold for the root (W2 labels)."""

    root: AttemptRef
    principal_id: SafeID
    opened_at_utc: datetime


class ExposureDeclaration(Record):
    """An owner's statement that a principal has seen a root's reference outside the workbench
    (the hackathon owner has seen every demo reference). Append-only under
    exposure-declarations/<root>/, latest governs. Written by LabelStore.declare_exposure."""

    schema_version: Literal["exposure-declaration.v1"] = "exposure-declaration.v1"
    principal_id: SafeID
    root: AttemptRef
    exposed: Literal[True] = True
    reason: FreeText = Field(min_length=1, max_length=2000)
    declared_by: SafeID
    created_at_utc: datetime


class ReviewStage(enum.IntEnum):
    NONE = 0
    BLIND = 1
    SIGNED = 2
    REVEALED = 3


class Cohort(Record):
    root: AttemptRef
    number: int
    raters: tuple[SafeID, ...]
    snapshots: tuple[SnapshotRef, ...]
    closed: bool
    legacy_reference_raters: tuple[SafeID, ...] | None = None


def cohorts_of(records: Sequence[AnyAssignment], signoffs: Sequence[SignOff]) -> tuple[Cohort, ...]:
    result = []
    for root in sorted({r.root for r in records}):
        history = sorted((r for r in records if r.root == root and r.schema_version == "assignment.v1"),
                         key=lambda r: r.created_at_utc)
        signs = [s for s in signoffs if s.root == root]
        eligible = set()
        signed: dict[str, set[str]] = {}
        active = None
        events = [(r.created_at_utc, 0, r) for r in records if r.root == root] + [(s.created_at_utc, 1, s) for s in signs]
        for _, kind, event in sorted(events, key=lambda e: (e[0], e[1])):
            if kind == 0:
                active = event
            else:
                signed.setdefault(event.rater.principal_id, set()).add(event.snapshot)
            if active and all(set(active.snapshots) <= signed.get(r, set()) for r in active.raters):
                eligible.update(active.raters)
        cumulative: set[str] = set()
        previous: set[str] = set()
        current: set[str] = set()
        raters = ()
        number = 1
        for r in history:
            added = set(r.snapshots) - cumulative
            reduced = bool(previous - set(r.snapshots))
            signed_before = any(s.snapshot in current and s.created_at_utc < r.created_at_utc for s in signs)
            if current and (reduced or signed_before and (added or r.raters != raters)):
                result.append(Cohort(root=root, number=number, raters=raters, snapshots=tuple(sorted(current)),
                                     closed=True, legacy_reference_raters=tuple(sorted(eligible))))
                number += 1
                current = set(r.snapshots) if reduced or not added else set()
            current.update(added)
            cumulative.update(r.snapshots)
            previous = set(r.snapshots)
            raters = r.raters
        if history:
            result.append(Cohort(root=root, number=number, raters=raters, snapshots=tuple(sorted(current)),
                                 closed=any(s.snapshot in current for s in signs),
                                 legacy_reference_raters=tuple(sorted(eligible))))
        versions = {}
        for r in records:
            if r.root == root and r.schema_version != "assignment.v1":
                if r.cohort not in versions or versions[r.cohort].version < r.version:
                    versions[r.cohort] = r
        if history and versions:
            first_v2 = min(r.created_at_utc for r in records if r.root == root and isinstance(r, Assignment))
            # The first v2 assignment absorbs an unfinished legacy gating set; it cannot open on the old subset.
            if not any(s.snapshot in current and s.created_at_utc < first_v2 for s in signs):
                result.pop()
        for n, r in sorted(versions.items()):
            if history and n <= number:
                raise ValueError("v2 cohort ids must follow reconstructed legacy cohorts")
            result.append(Cohort(root=root, number=n, raters=r.raters, snapshots=r.snapshots,
                                 closed=any(s.snapshot in r.snapshots for s in signs)))
    return tuple(result)


class FamilyFacts(Record):
    family: Family
    cohorts: tuple[Cohort, ...]
    signoffs: tuple[SignOff, ...]
    declarations: tuple[ExposureDeclaration, ...]
    legacy_reveals: tuple[SafeID, ...]


def identity_open(rater: SafeID, snapshot: Digest, facts: FamilyFacts) -> bool:
    return any(rater in c.raters and snapshot in c.snapshots and
               all(any(s.root == c.root and s.rater.principal_id == rater and s.snapshot == sid
                       for s in facts.signoffs) for sid in c.snapshots) for c in facts.cohorts)


def reference_open(rater: SafeID, facts: FamilyFacts) -> bool:
    if rater in facts.legacy_reveals or any(d.principal_id == rater for d in facts.declarations):
        return True
    return any(bool(c.snapshots) and
               (rater in c.legacy_reference_raters if c.legacy_reference_raters is not None else
                rater in c.raters and
                all(all(any(s.root == c.root and s.rater.principal_id == who and s.snapshot == sid
                            for s in facts.signoffs) for sid in c.snapshots) for who in c.raters))
               for c in facts.cohorts)


def stage_of(rater: SafeID, snapshot: Digest, facts: FamilyFacts) -> ReviewStage:
    if identity_open(rater, snapshot, facts):
        return ReviewStage.REVEALED if reference_open(rater, facts) else ReviewStage.SIGNED
    if any(rater in c.raters and snapshot in c.snapshots for c in facts.cohorts):
        return ReviewStage.BLIND
    return ReviewStage.NONE


def exposure_of(rater: SafeID, facts: FamilyFacts) -> Exposure:
    seen_identity = any(identity_open(rater, sid, facts) for c in facts.cohorts for sid in c.snapshots)
    return "revealed" if reference_open(rater, facts) or seen_identity else "blind"


_MINT = object()


@dataclass(frozen=True)
class IdentityProof:
    principal_id: str
    snapshot: str
    _mint: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise TypeError("IdentityProof is issued by review.stage.require_identity only")

    def check(self, snapshot: str) -> None:
        if snapshot != self.snapshot:
            raise Denied()


@dataclass(frozen=True)
class ReferenceProof:
    principal_id: str
    family: str
    _mint: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._mint is not _MINT:
            raise TypeError("ReferenceProof is issued by review.stage.require_reference only")

    def check(self, family: str) -> None:
        if family != self.family:
            raise Denied()


# --- wire requests. None carries a rater or exposure: the store stamps both. ------------------


class LabelRequest(Record):
    snapshot: Digest
    item: RubricItemId
    phase_id: SafeID | None = None
    answer: RubricAnswer
    reason: FreeText = Field(default="", max_length=4000)
    note: FreeText = Field(default="", max_length=4000)


class PeakCallRequest(Record):
    snapshot: Digest
    peak_id: SafeID
    cause: PeakCause
    note: FreeText = Field(default="", max_length=4000)


class SignOffRequest(Record):
    snapshot: Digest
    verdict: SignOffVerdict
    comment: FreeText = Field(default="", max_length=8000)


class StepFeedbackRequest(Record):
    snapshot: Digest | None = None
    step: StepRef
    mark: StepMark
    note: FreeText = Field(default="", max_length=4000)


class PermissionRequest(Record):
    participation_and_storage: bool
    internal_evaluation_and_calibration: bool
    public_release: bool
    training: bool
    public_identity_disclosure: bool


ExportUse = Literal["internal_evaluation", "public_release", "training"]
# Roles that carry a label-mode credential, defined once (Build D11). The raw attempt routes
# refuse a principal holding any of them.


_USE_FIELD: Mapping[ExportUse, str] = {
    "internal_evaluation": "internal_evaluation_and_calibration",
    "public_release": "public_release",
    "training": "training",
}
_DENIED = "label access denied"


def rater_kind_for(principal: Principal) -> RaterKind:
    """MODEL_JUDGE -> model_judge; USER with role owner_rater -> owner; USER with role rater ->
    expert. Raises PermissionError for every other principal: agents, services, the admin
    owner (role "owner" without a label-mode role), and a USER that holds a label-mode role
    together with "owner" (Build D11: label mode has its own credentials)."""
    if principal.principal_type is PrincipalType.MODEL_JUDGE:
        return "model_judge"
    if principal.principal_type is PrincipalType.USER and "owner" not in principal.roles:
        if "owner_rater" in principal.roles:
            return "owner"
        if "rater" in principal.roles:
            return "expert"
    raise AuthorizationError(f"{principal.principal_id} holds no label-mode credential")


def is_admin_owner(principal: Principal) -> bool:
    """The admin credential: assigns raters and never labels (Build D11)."""
    return (
        principal.principal_type is PrincipalType.USER
        and "owner" in principal.roles
        and not principal.roles & LABEL_MODE_ROLES
    )


def _check_target(snapshot: snapshots.AnalysisSnapshot, phase_id: str | None) -> None:
    if phase_id is not None and phase_id not in {p.phase_id for p in snapshot.phases}:
        raise ValueError("phase_id names no phase of this snapshot")


class LabelStore:
    """Review operations share the tree's grants and ledger transaction.

    Human records require participation; model judges write through judge_label only.
    Identity is per snapshot, reference access and exposure span the pattern family.
    """

    def __init__(self, root: WorkbenchStore) -> None:
        self._store = root
        self._root = root.root
        self._clock = root._clock
        self._projection = StageProjection(root.root)
        with root.ledger.transaction():
            if not hasattr(root, '_label_collections'):
                root._label_collections = {
                    name: root.ledger.collection(name, kind, Mode.LOG, KeyShape((shape,)), owner=__name__)
                    for name, kind, shape in (
                        ('labels', Label, 'hex64'), ('peak-calls', PeakCall, 'hex64'),
                        ('signoffs', SignOff, 'hex64'), ('step-feedback', AnyStepFeedback, 'hex64'),
                        ('permissions', PermissionRecord, 'safe_id'), ('assignments', AnyAssignment, 'hex64'),
                        ('exposure-declarations', ExposureDeclaration, 'hex64'),
                    )
                }
                root._snapshot_collection = root.ledger.collection('snapshots', snapshots.AnalysisSnapshot, Mode.ONCE, KeyShape(('hex64',)), owner=__name__)
                root._rater_items = root.ledger.markers('rater-items', KeyShape(('safe_id', 'hex64')), owner=__name__)
                for (family,) in root._label_collections['assignments'].keys():
                    for assignment in root._label_collections['assignments'].entries((family,)):
                        for rater in assignment.raters:
                            for digest in assignment.snapshots:
                                root._rater_items.add((rater, digest))
        self._collections = root._label_collections

    # ----- writes --------------------------------------------------------------------------

    def put_snapshot(self, snapshot: snapshots.AnalysisSnapshot) -> Digest:
        """Store an AnalysisSnapshot (snapshot.py) content-addressed; idempotent. Its origin is
        an attempt under its root, it names that root's case and pattern, and an attempt is
        frozen once per kind (KTD10), so a label's digest names the exact evidence shown."""
        path = self._store._path(snapshot.origin)
        root_digest, root = path[0]
        if len(path) < 2 or root_digest != snapshot.root:
            raise ValueError("a snapshot's origin is an attempt under its root")
        if root.case.case_id != snapshot.case_id or root.case.pattern.sha256 != snapshot.pattern_sha256:
            raise ValueError("a snapshot names its root's case and pattern")
        data = canonical_bytes(snapshot)
        digest = hashlib.sha256(data).hexdigest()
        with self._store.ledger.transaction():
            if any(s.origin == snapshot.origin and s.kind == snapshot.kind and d != digest for d, s in self._snapshots()):
                raise ValueError("this attempt is already frozen as another snapshot of this kind")
            self._store._snapshot_collection.put((digest,), snapshot)
        return digest

    def freeze(self, principal: Principal, origin: AttemptRef, kind: snapshots.SnapshotKind) -> Digest:
        """POST /snapshots: a human rater with a grant on the root freezes an agent answer or
        an expert correction. A reference snapshot is frozen only by a trusted script through
        put_snapshot, so no HTTP caller ever builds one."""
        self._rater(principal)
        if kind == "reference":
            raise AuthorizationError("a reference snapshot is frozen only by a trusted script")
        self._store.lineage(principal, _hex(origin))  # the rater projection, before anything is built
        built = snapshots.build_snapshot(self._store, principal, origin, kind)
        if built.origin != origin or built.kind != kind:
            raise ValueError("the built snapshot names another origin or kind")
        return self.put_snapshot(built)

    def snapshot(self, principal: Principal, digest: Digest) -> snapshots.AnalysisSnapshot:
        with self._store.ledger.transaction():
            return self._snapshot_locked(principal, digest)

    def _snapshot_locked(self, principal: Principal, digest: Digest) -> snapshots.AnalysisSnapshot:
        self._store.ledger.assert_locked()
        kind = rater_kind_for(principal)
        found = self._snapshot(digest)
        if kind == "model_judge":
            if not self._judge_reads(found):
                raise AuthorizationError("a model judge reads only an agent's imported answer")
            return found
        self._lineage_locked(principal, found.origin)
        if found.kind == "reference" and not reference_open(principal.principal_id, self._projection._facts_locked(found.root)):
            raise AttemptAccessError(_DENIED)
        return found

    def record_permission(self, principal: Principal, request: PermissionRequest) -> PermissionRecord:
        self._rater(principal)
        record = PermissionRecord(
            principal_id=principal.principal_id, created_at_utc=self._clock(), **request.model_dump()
        )
        self._append(self._root / "permissions" / principal.principal_id, record, record.created_at_utc)
        return record

    def assign(self, principal: Principal, root: AttemptRef, raters: tuple[SafeID, ...], snapshots: tuple[Digest, ...], *, cohort: int | None = None) -> Assignment:
        with self._store.ledger.transaction():
            return self._assign_locked(principal, root, raters, snapshots, cohort=cohort)

    def _assign_locked(
        self, principal: Principal, root: AttemptRef, raters: tuple[SafeID, ...], snapshots: tuple[Digest, ...], *, cohort: int | None = None
    ) -> Assignment:
        if not is_admin_owner(principal):
            raise AuthorizationError("only the admin owner assigns raters")
        self._store._authorize(principal, _hex(root), CaseOperation.INSPECT)  # acl.json: roots only
        if len(set(snapshots)) != len(snapshots):
            raise ValueError("a cohort names each snapshot once")
        for digest in snapshots:
            found = self._snapshot(digest)
            if found.root != root or found.kind != "agent_answer":
                raise ValueError(f"cohort snapshot {digest} is not an agent_answer snapshot of this root")
        self._store.ledger.assert_locked()
        records = _records(self._root / "assignments" / root, AnyAssignment)
        signoffs = _records(self._root / "signoffs" / root, SignOff)
        cohorts = cohorts_of(records, signoffs)
        latest = cohorts[-1] if cohorts else None
        if cohort is not None and (latest is None or latest.number != cohort or latest.closed):
            raise Conflict("the requested cohort is closed or absent")
        if latest is not None and not latest.closed:
            number = latest.number
            raters = tuple(dict.fromkeys((*latest.raters, *raters)))
            previous = latest.snapshots
            if not any(isinstance(r, Assignment) for r in records):
                previous = max(records, key=lambda r: r.created_at_utc).snapshots
            snapshots = tuple(dict.fromkeys((*previous, *snapshots)))
        else:
            number = latest.number + 1 if latest else 1
        at = self._clock()
        if latest is not None and not any(isinstance(r, Assignment) for r in records):
            number = latest.number + 1
            # Legacy replay has only timestamps to order the bridge after retained facts.
            boundary = max(r.created_at_utc for r in (*records, *signoffs))
            if at <= boundary:
                raise Conflict("the legacy bridge timestamp must follow its history")
        versions = [r.version for r in records if isinstance(r, Assignment) and r.cohort == number]
        assignment = Assignment(
            root=root, cohort=number, version=max(versions, default=0) + 1,
            raters=raters, snapshots=snapshots, assigned_by=principal.principal_id,
            created_at_utc=at,
        )
        for rater in assignment.raters:
            for digest in assignment.snapshots:
                self._store._rater_items.add((rater, digest))
        self._append_locked(self._root / "assignments" / root, assignment, assignment.created_at_utc)
        return assignment

    def latest_assignment(self, root: AttemptRef) -> Assignment | AssignmentV1 | None:
        records = _records(self._root / "assignments" / _hex(root), AnyAssignment)
        return max(records, key=lambda r: (getattr(r, 'cohort', 0), getattr(r, 'version', 0), r.created_at_utc)) if records else None

    def declare_exposure(self, principal: Principal, principal_id: SafeID, root: AttemptRef, reason: str) -> ExposureDeclaration:
        with self._store.ledger.transaction():
            return self._declare_exposure_locked(principal, principal_id, root, reason)

    def _declare_exposure_locked(
        self, principal: Principal, principal_id: SafeID, root: AttemptRef, reason: str
    ) -> ExposureDeclaration:
        """Record reference access acquired outside the workbench."""
        if not is_admin_owner(principal):
            raise AuthorizationError("only the admin owner declares exposure")
        self._store._authorize(principal, _hex(root), CaseOperation.INSPECT)
        declaration = ExposureDeclaration(
            principal_id=principal_id,
            root=root,
            reason=reason,
            declared_by=principal.principal_id,
            created_at_utc=self._clock(),
        )
        self._append_locked(self._root / "exposure-declarations" / root, declaration, declaration.created_at_utc)
        return declaration

    def label(self, principal: Principal, request: LabelRequest) -> Label:
        with self._store.ledger.transaction():
            return self._label_locked(principal, request)

    def _label_locked(self, principal: Principal, request: LabelRequest) -> Label:
        self._store.ledger.assert_locked()
        rater = self._rater(principal)
        found = self._snapshot_locked(principal, request.snapshot)
        self._require_participation(principal.principal_id)
        label = Label(
            **request.model_dump(),
            rater=rater,
            exposure=exposure_of(principal.principal_id, self._projection._facts_locked(found.root)),
            created_at_utc=self._clock(),
        )
        _check_target(found, label.phase_id)
        self._append_locked(self._root / "labels" / request.snapshot, label, label.created_at_utc)
        return label

    def peak_call(self, principal: Principal, request: PeakCallRequest) -> PeakCall:
        with self._store.ledger.transaction():
            return self._peak_call_locked(principal, request)

    def _peak_call_locked(self, principal: Principal, request: PeakCallRequest) -> PeakCall:
        self._store.ledger.assert_locked()
        rater = self._rater(principal)
        found = self._snapshot_locked(principal, request.snapshot)
        if request.peak_id not in {p.peak_id for p in found.residual_peaks}:
            raise ValueError("peak_id names no residual peak of this snapshot")
        self._require_participation(principal.principal_id)
        call = PeakCall(
            **request.model_dump(),
            rater=rater,
            exposure=exposure_of(principal.principal_id, self._projection._facts_locked(found.root)),
            created_at_utc=self._clock(),
        )
        self._append_locked(self._root / "peak-calls" / request.snapshot, call, call.created_at_utc)
        return call

    def sign_off(self, principal: Principal, request: SignOffRequest) -> SignOff:
        with self._store.ledger.transaction():
            return self._sign_off_locked(principal, request)

    def _sign_off_locked(self, principal: Principal, request: SignOffRequest) -> SignOff:
        self._store.ledger.assert_locked()
        rater = self._rater(principal)
        found = self._snapshot_locked(principal, request.snapshot)
        self._require_participation(principal.principal_id)
        signoff = SignOff(
            **request.model_dump(),
            root=found.root,
            rater=rater,
            exposure=exposure_of(principal.principal_id, self._projection._facts_locked(found.root)),
            created_at_utc=self._clock(),
        )
        self._append_locked(self._root / "signoffs" / found.root, signoff, signoff.created_at_utc)
        return signoff

    def step_feedback(self, principal: Principal, request: StepFeedbackRequest) -> StepFeedback:
        with self._store.ledger.transaction():
            return self._step_feedback_locked(principal, request)

    def _step_feedback_locked(self, principal: Principal, request: StepFeedbackRequest) -> StepFeedback:
        rater = self._rater(principal)
        root = self._step_root(principal, request.step)
        self._require_participation(principal.principal_id)
        self._store.ledger.assert_locked()
        if request.snapshot is not None:
            shown = self._snapshot_locked(principal, request.snapshot)
            if shown.root != root:
                raise ValueError("step and snapshot must share a root")
            snapshot = request.snapshot
        else:
            candidates = self._cohort_locked(principal, root)
            if not candidates and reference_open(principal.principal_id, self._projection._facts_locked(root)):
                candidates = tuple(sid for sid, snap in self._snapshots() if snap.root == root and snap.kind != "reference")
            if not candidates:
                raise ValueError("step feedback requires a review snapshot")
            matches = tuple(sid for sid in candidates if self._snapshot(sid).origin == request.step.attempt)
            candidates = matches or candidates
            if len(candidates) != 1:
                raise ValueError("step feedback requires an explicit snapshot when multiple answers are possible")
            snapshot = candidates[0]
        feedback = StepFeedback(
            snapshot=snapshot, attempt=request.step.attempt, call_index=request.step.call_index,
            mark=request.mark,
            note=request.note,
            rater=rater,
            exposure=exposure_of(principal.principal_id, self._projection._facts_locked(root)),
            created_at_utc=self._clock(),
        )
        self._append_locked(self._root / "step-feedback" / request.step.attempt, feedback, feedback.created_at_utc)
        return feedback

    def judge_label(self, principal: Principal, label: Label) -> Label:
        """The judge script's write. The rater id, exposure and time are stamped here; only the
        judge identity and the answer come from the caller."""
        if rater_kind_for(principal) != "model_judge":
            raise AuthorizationError("only a model judge principal writes judge labels")
        if label.rater.judge is None:
            raise ValueError("a judge label names its judge identity")
        found = self.snapshot(principal, label.snapshot)
        rater = Rater(principal_id=principal.principal_id, kind="model_judge", judge=label.rater.judge)
        stamped = label.model_copy(update={"rater": rater, "exposure": "blind", "created_at_utc": self._clock()})
        stamped = Label.model_validate(stamped.model_dump())
        _check_target(found, stamped.phase_id)
        self._append(self._root / "labels" / label.snapshot, stamped, stamped.created_at_utc)
        return stamped

    # ----- reads ---------------------------------------------------------------------------

    def cohort(self, principal: Principal, root: AttemptRef) -> tuple[Digest, ...]:
        """Every historical cohort of this root that names the rater."""
        with self._store.ledger.transaction():
            return self._cohort_locked(principal, root)

    def _cohort_locked(self, principal: Principal, root: AttemptRef) -> tuple[Digest, ...]:
        self._store.ledger.assert_locked()
        facts = self._projection._facts_locked(root)
        return tuple(dict.fromkeys(sid for c in facts.cohorts if c.root == root and principal.principal_id in c.raters for sid in c.snapshots))

    def reveal_state(self, principal: Principal, root: AttemptRef) -> RevealState:
        self._rater(principal)
        self._store._authorize(principal, _hex(root), CaseOperation.INSPECT)
        with self._store.ledger.transaction():
            return self._reveal_state_locked(principal.principal_id, root)

    def own_signoff(self, principal: Principal, snapshot: Digest) -> SignOff | None:
        """This rater's latest sign-off on this snapshot: the rater's own record, so the page can
        restore its sign-off, critique and analysis report after a reload (AXA-43zwk.18)."""
        found = self.snapshot(principal, snapshot)
        mine = [
            s
            for s in _records(self._root / "signoffs" / found.root, SignOff)
            if s.snapshot == snapshot and s.rater.principal_id == principal.principal_id
        ]
        return mine[-1] if mine else None

    def visible_labels(self, principal: Principal, snapshot: Digest) -> tuple[Label, ...]:
        """Own labels always; other raters' labels only after this rater's reveal."""
        with self._store.ledger.transaction():
            found = self._snapshot_locked(principal, snapshot)
            return self._visible_locked(principal, found.root, _records(self._root / "labels" / snapshot, Label))

    def visible_step_feedback(self, principal: Principal, attempt: AttemptRef) -> tuple[AnyStepFeedback, ...]:
        """Own step feedback on one attempt always; other raters' only after this rater's reveal."""
        self._rater(principal)
        with self._store.ledger.transaction():
            root = self._lineage_locked(principal, attempt)[0][0]
            found = _records(self._root / "step-feedback" / attempt, AnyStepFeedback)
            return self._visible_locked(principal, root, found)

    def export_records(
        self, use: ExportUse
    ) -> tuple[tuple[Label, ...], tuple[PeakCall, ...], tuple[SignOff, ...], tuple[AnyStepFeedback, ...]]:
        """Every record whose rater's latest permission allows `use`; model_judge records always.
        A withdrawn rater's records are absent (R33). Rater ids are not pseudonymized here; the
        exporter does that with the private identity map."""
        field = _USE_FIELD[use]
        allowed: dict[str, bool] = {}

        def admits(rater: Rater) -> bool:
            if rater.kind == "model_judge":
                return True
            if rater.principal_id not in allowed:
                latest = self._permission(rater.principal_id)
                allowed[rater.principal_id] = (
                    latest is not None and latest.participation_and_storage and getattr(latest, field)
                )
            return allowed[rater.principal_id]

        def every[R: Label | PeakCall | SignOff | StepFeedback | StepFeedbackV1](name: str, kind: type[R]) -> tuple[R, ...]:
            return tuple(r for d in self._subdirs(name) for r in _records(d, kind) if admits(r.rater))

        return (
            every("labels", Label),
            every("peak-calls", PeakCall),
            every("signoffs", SignOff),
            every("step-feedback", AnyStepFeedback),
        )

    # ----- internals -----------------------------------------------------------------------

    def _rater(self, principal: Principal) -> Rater:
        """The human rater of a request: agents, services and model judges are refused."""
        kind = rater_kind_for(principal)
        if kind == "model_judge":
            raise AuthorizationError("a model judge writes only judge labels")
        return Rater(principal_id=principal.principal_id, kind=kind)

    def _judge_reads(self, found: snapshots.AnalysisSnapshot) -> bool:
        """A model judge reads and labels only an agent_answer snapshot whose origin is an agent's
        imported run (KTD11, CR-08): never a correction, by a rater or by a reference-exposed owner,
        never a reference, and never an answer frozen from a human attempt."""
        if found.kind != "agent_answer":
            return False
        origin = self._store._path(found.origin)[-1][1]
        return isinstance(origin, ChildAttempt) and isinstance(origin.action, ImportRun)

    def _visible_locked[R: Label | StepFeedback | StepFeedbackV1](self, principal: Principal, root: str, records: tuple[R, ...]) -> tuple[R, ...]:
        self._store.ledger.assert_locked()
        own = principal.principal_id
        everyone = principal.principal_type is PrincipalType.USER and reference_open(own, self._projection._facts_locked(root))
        return tuple(
            r for r in records if r.rater.principal_id == own or (everyone and self._participates(r.rater))
        )

    def _participates(self, rater: Rater) -> bool:
        if rater.kind == "model_judge":
            return True
        latest = self._permission(rater.principal_id)
        return latest is not None and latest.participation_and_storage

    def _require_participation(self, principal_id: str) -> None:
        latest = self._permission(principal_id)
        if latest is None or not latest.participation_and_storage:
            raise AuthorizationError("no permission record allows storing this rater's records (R33)")

    def _permission(self, principal_id: str) -> PermissionRecord | None:
        return _latest(self._root / "permissions" / principal_id, PermissionRecord)

    def _lineage_locked(self, principal: Principal, digest: AttemptRef) -> tuple[tuple[str, Attempt], ...]:
        self._store.ledger.assert_locked()
        path = self._store._path(_hex(digest))
        self._store._authorize(principal, path[0][0], CaseOperation.INSPECT)
        if not self._projection._admits_locked(principal, path):
            raise AttemptAccessError("attempt access denied")
        return path

    def _step_root(self, principal: Principal, step: StepRef) -> str:
        path = self._lineage_locked(principal, step.attempt)
        root = path[0][0]
        if step.call_index is not None:
            attempt = path[-1][1]
            result = (
                attempt.outcome.result
                if isinstance(attempt, ChildAttempt) and isinstance(attempt.outcome, Completed)
                else None
            )
            if not isinstance(result, ImportedRunResult) or step.call_index not in {
                call.index for call in result.trajectory
            }:
                raise ValueError("call_index names no recorded call of this attempt")
        return root

    def _snapshot(self, digest: str) -> snapshots.AnalysisSnapshot:
        return _read_snapshot(self._root, digest)

    def _snapshots(self) -> tuple[tuple[str, snapshots.AnalysisSnapshot], ...]:
        return _all_snapshots(self._root)

    def _subdirs(self, name: str) -> tuple[Path, ...]:
        top = self._root / name
        collection = self._collections[name]
        return tuple(top / key[0] for key in collection.keys())

    def _append(self, directory: Path, record: Record, at: datetime) -> None:
        with self._store.ledger.transaction():
            self._append_locked(directory, record, at)

    def _append_locked(self, directory: Path, record: Record, at: datetime) -> None:
        self._store.ledger.assert_locked()
        self._collections[directory.parent.name].append((directory.name,), record, at=at)

    def _reveal_state_locked(self, principal_id: str, root: str) -> RevealState:
        self._store.ledger.assert_locked()
        facts = self._projection._facts_locked(root)
        cohorts = [c for c in facts.cohorts if c.root == root]
        current = cohorts[-1] if cohorts else None
        def done(c, who):
            return all(any(s.root == root and s.rater.principal_id == who and s.snapshot == sid
                           for s in facts.signoffs) for sid in c.snapshots)
        return RevealState(
            root=root, principal_id=principal_id,
            signed_off=any(principal_id in c.raters and done(c, principal_id) for c in cohorts),
            pending_raters=tuple(who for who in current.raters if not done(current, who)) if current else (),
            revealed=reference_open(principal_id, facts),
        )

    def require_identity(self, principal: Principal, snapshot: Digest) -> IdentityProof:
        self._rater(principal)
        with self._store.ledger.transaction():
            found = self._snapshot_locked(principal, snapshot)
            return require_identity(principal.principal_id, snapshot, self._projection._facts_locked(found.root))

    def require_reference(self, principal: Principal, root: AttemptRef) -> ReferenceProof:
        self._rater(principal)
        self._store._authorize(principal, root, CaseOperation.INSPECT)
        with self._store.ledger.transaction():
            return require_reference(principal.principal_id, self._projection._facts_locked(root))


_HEX64 = re.compile(HEX64)
_SAFE_ID = re.compile(SAFE_ID)


def _records[R](directory: Path, kind: type[R]) -> tuple[R, ...]:
    ledger = Ledger(directory.parent.parent)
    collection = ledger.legacy(directory.parent.name, kind, reader=__name__, mode=Mode.LOG,
                               key=KeyShape(('safe_id' if directory.parent.name == 'permissions' else 'hex64',)))
    return collection.entries((directory.name,))


def _latest[R: Record](directory: Path, kind: type[R]) -> R | None:
    found = _records(directory, kind)
    return found[-1] if found else None


def _read_snapshot(store_root: Path, digest: str) -> snapshots.AnalysisSnapshot:
    collection = Ledger(store_root).legacy('snapshots', snapshots.AnalysisSnapshot, reader=__name__, key=KeyShape(('hex64',)))
    data = collection.get_bytes((_hex(digest),))
    if data is None:
        raise AttemptAccessError(_DENIED)
    if hashlib.sha256(data).hexdigest() != digest:
        raise CustodyError("snapshot bytes disagree with their digest")
    return snapshots.AnalysisSnapshot.model_validate_json(data)


def _all_snapshots(store_root: Path) -> tuple[tuple[str, snapshots.AnalysisSnapshot], ...]:
    collection = Ledger(store_root).legacy('snapshots', snapshots.AnalysisSnapshot, reader=__name__, key=KeyShape(('hex64',)))
    return tuple((key[0], _read_snapshot(store_root, key[0])) for key in collection.keys())


def require_identity(rater: SafeID, snapshot: Digest, facts: FamilyFacts) -> IdentityProof:
    if not identity_open(rater, snapshot, facts):
        raise Denied()
    return IdentityProof(rater, snapshot, _mint=_MINT)


def require_reference(rater: SafeID, facts: FamilyFacts) -> ReferenceProof:
    if not reference_open(rater, facts):
        raise Denied()
    return ReferenceProof(rater, facts.family, _mint=_MINT)


class StageProjection:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._ledger = Ledger(root)

    def facts(self, root: AttemptRef) -> FamilyFacts:
        with self._ledger.transaction():
            return self._facts_locked(root)

    def _facts_locked(self, root: AttemptRef) -> FamilyFacts:
        self._ledger.assert_locked()
        roots = self._ledger.legacy('attempts/{0}/attempt.json', RootAttempt, reader=__name__, key=KeyShape(('hex64',)))
        data = roots.get_bytes((_hex(root),))
        if data is None:
            raise AttemptAccessError(_DENIED)
        if hashlib.sha256(data).hexdigest() != root:
            raise CustodyError("root bytes disagree with their digest")
        attempt = RootAttempt.model_validate_json(data)
        family = attempt.case.pattern.sha256
        members = self._ledger.legacy('families/{0}/{1}', bytes, reader=__name__, key=KeyShape(('hex64', 'hex64')))
        family_roots = tuple(key[1] for key in members.keys((family,)))
        assignments = tuple(r for member in family_roots for r in _records(self._root / 'assignments' / member, AnyAssignment))
        signs = tuple(r for member in family_roots for r in _records(self._root / 'signoffs' / member, SignOff))
        declarations = tuple(r for member in family_roots for r in _records(self._root / 'exposure-declarations' / member, ExposureDeclaration))
        reveals = self._ledger.legacy('reveals/{0}/{1}.json', RevealOpened, reader=__name__, key=KeyShape(('hex64', 'safe_id')))
        opened = [(key, reveals.get(key)) for member in family_roots for key in reveals.keys((member,))]
        if any(record.root != key[0] or record.principal_id != key[1] for key, record in opened):
            raise CustodyError("a legacy reveal disagrees with its path")
        return FamilyFacts(family=family, cohorts=cohorts_of(assignments, signs), signoffs=signs,
                           declarations=declarations,
                           legacy_reveals=tuple(sorted({key[1] for key, _ in opened})))

    def exposure(self, principal_id: SafeID, root: AttemptRef) -> Literal['exposed', 'unexposed']:
        with self._ledger.transaction():
            return self._exposure_locked(principal_id, root)

    def _exposure_locked(self, principal_id: SafeID, root: AttemptRef) -> Literal['exposed', 'unexposed']:
        self._ledger.assert_locked()
        if not _SAFE_ID.fullmatch(principal_id):
            raise ValueError('principal_id must be a safe identifier')
        return 'exposed' if exposure_of(principal_id, self._facts_locked(root)) == 'revealed' else 'unexposed'

    def admits(self, principal: Principal, path: Sequence[tuple[str, Attempt]]) -> bool:
        with self._ledger.transaction():
            return self._admits_locked(principal, path)

    def _admits_locked(self, principal: Principal, path: Sequence[tuple[str, Attempt]]) -> bool:
        self._ledger.assert_locked()
        if principal.principal_type in (PrincipalType.AGENT, PrincipalType.MODEL_JUDGE):
            return False
        if not principal.roles & LABEL_MODE_ROLES:
            return True
        own = principal.principal_id
        facts = self._facts_locked(path[0][0])
        if reference_open(own, facts):
            return True
        cohort = {_read_snapshot(self._root, sid).origin for c in facts.cohorts
                  if c.root == path[0][0] and own in c.raters for sid in c.snapshots}
        return all(attempt.actor.principal_id == own or
                   (digest in cohort and not (isinstance(attempt.actor, HumanActor) and
                                              attempt.actor.reference_exposure == 'exposed'))
                   for digest, attempt in path[1:] if isinstance(attempt, ChildAttempt))


def _hex(digest: str) -> str:
    """Every path built from a request digest goes through here first."""
    if not _HEX64.fullmatch(digest):
        raise AttemptAccessError(_DENIED)
    return digest
