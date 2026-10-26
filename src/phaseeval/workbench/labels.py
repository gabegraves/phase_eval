"""Rubric labels, residual-peak calls, sign-offs, step feedback and permission records (R32, R33).

Target path: src/phaseeval/workbench/labels.py (public). Overnight contract commit (AXA-43zwk.7): the
record types and the LabelStore signatures are binding for every overnight unit; the bodies are
filled by the U3 labels worker.

- The labeled object is a frozen AnalysisSnapshot (snapshot.py), keyed by its digest. A label
  is one immutable record per (snapshot, item, target, rater) response; a later change is a new
  record, never a rewrite.
- rater_kind separates experts, the owner and model judges (plan KTD3). Agreement is computed
  within one kind only; a model-judge label is never presented or counted as an expert label.
- "cannot_assess" carries its reason and is excluded from agreement, never discarded.
- Blindness (R32): a rater sees no reference and no other rater's label until every rater
  assigned to that pattern (root) has signed off. A label made after the rater's reveal is
  stored with exposure "revealed". Until then the rater reads only its cohort and its own
  attempts, on every route: ExposureLedger.admits is that one rule (CR-07).
- Step feedback (N8, owner -14423) is its own record type, keyed to an attempt and a step of
  its trajectory, separate from the end-state rubric labels; both export.
- Nothing is stored for a human rater without a permission record whose participation choice
  is true (R33). The latest permission record per principal governs every later export.

Store layout (under the WorkbenchStore root, 0700, one file per record, nothing rewritten):

    snapshots/<digest>.json                          AnalysisSnapshot, sha256(bytes) == digest
    labels/<snapshot>/<utc>-<sha256>.json            Label
    peak-calls/<snapshot>/<utc>-<sha256>.json        PeakCall
    signoffs/<root>/<utc>-<sha256>.json              SignOff
    step-feedback/<attempt>/<utc>-<sha256>.json      StepFeedback
    permissions/<principal>/<utc>-<sha256>.json      PermissionRecord
    assignments/<root>/<utc>-<sha256>.json           Assignment (latest governs)
    reveals/<root>/<principal>.json                  RevealOpened, published once (W2 labels)
    exposure-declarations/<root>/<utc>-<sha256>.json ExposureDeclaration (architect D2)

The reveal marker makes exposure monotone: once the store has seen a rater's reveal condition
hold, a snapshot frozen later under the same root cannot close it again, so a record made after
the rater could see the reference is never stamped blind.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from phaseeval.kernel.ids import HEX64, SAFE_ID
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


class StepFeedback(Record):
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


class Assignment(Record):
    """Which human raters review which agent answers of a pattern (root). Written by the admin
    owner. The snapshots are the review cohort the reveal waits on; a snapshot frozen later
    (export, refit, judge) never joins it (Build D12)."""

    schema_version: Literal["assignment.v1"] = "assignment.v1"
    root: AttemptRef
    raters: tuple[SafeID, ...] = Field(min_length=1)
    snapshots: tuple[Digest, ...] = Field(min_length=1)  # agent_answer snapshots of this root
    assigned_by: SafeID
    created_at_utc: datetime


class RevealState(Record):
    root: AttemptRef
    principal_id: SafeID
    signed_off: bool  # this rater signed off every snapshot of the latest assignment's cohort
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


class ExposureLedger:
    """The read side of reveals/<root>/<principal>.json and exposure-declarations/, both written
    only by LabelStore. Implements store.ExposureReader. LabelStore stamps label exposure through
    the same reader, so labels (blind or revealed) and attempts (HumanActor.reference_exposure)
    answer from one source. Built from the store root path, not from a LabelStore, because
    LabelStore takes the WorkbenchStore (architect F5)."""

    def __init__(self, root: Path, clock: Callable[[], datetime] | None = None) -> None:
        self._root = root.resolve()
        self._clock = clock or (lambda: datetime.now(UTC))  # stamps a reveal marker admits opens

    def exposure(self, principal_id: SafeID, root: AttemptRef) -> Literal["exposed", "unexposed"]:
        """exposed iff the reveal marker exists or an ExposureDeclaration names the principal."""
        if not _SAFE_ID.fullmatch(principal_id):
            raise ValueError("principal_id must be a safe identifier")
        if _opened(self._root, _hex(root), principal_id) is not None:
            return "exposed"
        declared = _records(self._root / "exposure-declarations" / root, ExposureDeclaration)
        return "exposed" if any(d.principal_id == principal_id for d in declared) else "unexposed"

    def admits(self, principal: Principal, path: Sequence[tuple[str, Attempt]]) -> bool:
        """The rater projection (CR-07): may this principal read the attempt at the end of `path`
        (root first) and anything derived from it, or build on it? WorkbenchStore.lineage applies
        it to every review payload, snapshot build, freeze and step; the store applies it to a
        gesture's parent and a record's targets; LabelStore.snapshot to a snapshot's origin.

        A label-mode principal, until its reveal on the root opens, reads only a path whose every
        attempt after the root is its own, or the origin of a snapshot in its cohort made by an
        actor who had not seen the reference. Another rater's attempt, an owner's reference-
        informed correction and anything built on either stay hidden. An agent or a model judge
        reads no lineage: an agent reads its own ActorAttemptView and ActorPath, a judge the
        blind snapshot. Every other principal reads what its root grant allows. An opening reveal
        publishes its marker here as in LabelStore._reveal, so what a rater was shown keeps its
        later records stamped revealed."""
        if principal.principal_type in (PrincipalType.AGENT, PrincipalType.MODEL_JUDGE):
            return False
        if not principal.roles & LABEL_MODE_ROLES:
            return True
        own, root = principal.principal_id, path[0][0]
        if _reveal_state(self._root, own, root, self._clock).revealed:
            return True
        cohort = {_read_snapshot(self._root, digest).origin for digest in _cohort(self._root, own, root)}
        return all(
            attempt.actor.principal_id == own
            or (
                digest in cohort
                and not (isinstance(attempt.actor, HumanActor) and attempt.actor.reference_exposure == "exposed")
            )
            for digest, attempt in path[1:]
            if isinstance(attempt, ChildAttempt)
        )


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
    """The only writer of the records above. Shares the WorkbenchStore root and its custody
    collections; authorization uses the same root ACL.

    Principal rules: an AGENT or SERVICE principal is refused every method that writes (R29).
    A USER principal with role "rater" labels as kind "expert", one with role "owner_rater" as
    kind "owner"; nobody labels as another rater (R33). The admin owner (role "owner") only
    assigns raters and is refused every label-mode call, and a principal holding a label-mode
    role together with "owner" is refused everything (Build D11). A MODEL_JUDGE principal (new
    PrincipalType in phaseeval.api.auth) writes only through `judge_label`, always kind "model_judge"
    and exposure "blind", and is refused every other write.

    Reveal (R32, PI directive): run identity (model, harness, job, series) becomes visible to a
    rater once the store holds that rater's sign-off on the snapshot; the reference comparison
    and other raters' labels once every assigned rater of the root has signed off. Before that
    no response to the rater carries reference_match, sibling episodes or run identity.

    "Signed off" for the root means a sign-off on every snapshot of the cohort named by the
    root's latest Assignment (Build D12). A rater missing from that Assignment is never
    revealed. A human sees a snapshot whose origin the rater projection admits
    (ExposureLedger.admits, CR-07), and a reference snapshot only after the reveal. A model
    judge reads and labels only an agent_answer snapshot of an imported run (CR-08) and needs
    no root grant, because the only thing it reads is that blind snapshot (W2 labels).
    """

    def __init__(self, root: WorkbenchStore) -> None:
        self._store = root
        self._root = root.root
        self._clock = root._clock
        self._ledger = ExposureLedger(root.root, root._clock)
        with root.ledger.transaction():
            if not hasattr(root, '_label_collections'):
                root._label_collections = {
                    name: root.ledger.collection(name, kind, Mode.LOG, KeyShape((shape,)), owner=__name__)
                    for name, kind, shape in (
                        ('labels', Label, 'hex64'), ('peak-calls', PeakCall, 'hex64'),
                        ('signoffs', SignOff, 'hex64'), ('step-feedback', StepFeedback, 'hex64'),
                        ('permissions', PermissionRecord, 'safe_id'), ('assignments', Assignment, 'hex64'),
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
        kind = rater_kind_for(principal)
        found = self._snapshot(digest)
        if kind == "model_judge":
            if not self._judge_reads(found):
                raise AuthorizationError("a model judge reads only an agent's imported answer")
            return found
        self._store.lineage(principal, found.origin)
        if found.kind == "reference" and not self._reveal(principal.principal_id, found.root).revealed:
            raise AttemptAccessError(_DENIED)
        return found

    def record_permission(self, principal: Principal, request: PermissionRequest) -> PermissionRecord:
        self._rater(principal)
        record = PermissionRecord(
            principal_id=principal.principal_id, created_at_utc=self._clock(), **request.model_dump()
        )
        self._append(self._root / "permissions" / principal.principal_id, record, record.created_at_utc)
        return record

    def assign(
        self, principal: Principal, root: AttemptRef, raters: tuple[SafeID, ...], snapshots: tuple[Digest, ...]
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
        assignment = Assignment(
            root=root,
            raters=raters,
            snapshots=snapshots,
            assigned_by=principal.principal_id,
            created_at_utc=self._clock(),
        )
        self._append(self._root / "assignments" / root, assignment, assignment.created_at_utc)
        return assignment

    def latest_assignment(self, root: AttemptRef) -> Assignment | None:
        """The root's current cohort: its latest assignment, or None."""
        return _latest(self._root / "assignments" / _hex(root), Assignment)

    def declare_exposure(
        self, principal: Principal, principal_id: SafeID, root: AttemptRef, reason: str
    ) -> ExposureDeclaration:
        """The admin owner records that a principal has seen this root's reference outside the
        workbench. Exposure only: the reveal of other raters' labels still waits on the cohort."""
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
        self._append(self._root / "exposure-declarations" / root, declaration, declaration.created_at_utc)
        return declaration

    def label(self, principal: Principal, request: LabelRequest) -> Label:
        rater = self._rater(principal)
        found = self.snapshot(principal, request.snapshot)
        self._require_participation(principal.principal_id)
        label = Label(
            **request.model_dump(),
            rater=rater,
            exposure=self._exposure(principal.principal_id, found.root),
            created_at_utc=self._clock(),
        )
        _check_target(found, label.phase_id)
        self._append(self._root / "labels" / request.snapshot, label, label.created_at_utc)
        return label

    def peak_call(self, principal: Principal, request: PeakCallRequest) -> PeakCall:
        rater = self._rater(principal)
        found = self.snapshot(principal, request.snapshot)
        if request.peak_id not in {p.peak_id for p in found.residual_peaks}:
            raise ValueError("peak_id names no residual peak of this snapshot")
        self._require_participation(principal.principal_id)
        call = PeakCall(
            **request.model_dump(),
            rater=rater,
            exposure=self._exposure(principal.principal_id, found.root),
            created_at_utc=self._clock(),
        )
        self._append(self._root / "peak-calls" / request.snapshot, call, call.created_at_utc)
        return call

    def sign_off(self, principal: Principal, request: SignOffRequest) -> SignOff:
        rater = self._rater(principal)
        found = self.snapshot(principal, request.snapshot)
        self._require_participation(principal.principal_id)
        signoff = SignOff(
            **request.model_dump(),
            root=found.root,
            rater=rater,
            exposure=self._exposure(principal.principal_id, found.root),
            created_at_utc=self._clock(),
        )
        self._append(self._root / "signoffs" / found.root, signoff, signoff.created_at_utc)
        return signoff

    def step_feedback(self, principal: Principal, request: StepFeedbackRequest) -> StepFeedback:
        rater = self._rater(principal)
        root = self._step_root(principal, request.step)
        self._require_participation(principal.principal_id)
        feedback = StepFeedback(
            step=request.step,
            mark=request.mark,
            note=request.note,
            rater=rater,
            exposure=self._exposure(principal.principal_id, root),
            created_at_utc=self._clock(),
        )
        self._append(self._root / "step-feedback" / request.step.attempt, feedback, feedback.created_at_utc)
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
        """The snapshots of the root's latest assignment when it names this rater, else none
        (Build D12, D14): the reveal waits on them and lists them as siblings."""
        return _cohort(self._root, principal.principal_id, root)

    def reveal_state(self, principal: Principal, root: AttemptRef) -> RevealState:
        self._rater(principal)
        self._store._authorize(principal, _hex(root), CaseOperation.INSPECT)
        return self._reveal(principal.principal_id, root)

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

    def has_signed(self, principal: Principal, snapshot: Digest) -> bool:
        """The store holds this rater's own sign-off on this snapshot (triage.Triage.revealed)."""
        found = self.snapshot(principal, snapshot)
        return any(
            s.snapshot == snapshot and s.rater.principal_id == principal.principal_id
            for s in _records(self._root / "signoffs" / found.root, SignOff)
        )

    def visible_labels(self, principal: Principal, snapshot: Digest) -> tuple[Label, ...]:
        """Own labels always; other raters' labels only after this rater's reveal."""
        found = self.snapshot(principal, snapshot)
        return self._visible(principal, found.root, _records(self._root / "labels" / snapshot, Label))

    def visible_step_feedback(self, principal: Principal, attempt: AttemptRef) -> tuple[StepFeedback, ...]:
        """Own step feedback on one attempt always; other raters' only after this rater's reveal."""
        self._rater(principal)
        root = self._store.lineage(principal, _hex(attempt))[0][0]
        found = _records(self._root / "step-feedback" / attempt, StepFeedback)
        return self._visible(principal, root, found)

    def export_records(
        self, use: ExportUse
    ) -> tuple[tuple[Label, ...], tuple[PeakCall, ...], tuple[SignOff, ...], tuple[StepFeedback, ...]]:
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

        def every[R: Label | PeakCall | SignOff | StepFeedback](name: str, kind: type[R]) -> tuple[R, ...]:
            return tuple(r for d in self._subdirs(name) for r in _records(d, kind) if admits(r.rater))

        return (
            every("labels", Label),
            every("peak-calls", PeakCall),
            every("signoffs", SignOff),
            every("step-feedback", StepFeedback),
        )

    # ----- internals -----------------------------------------------------------------------

    def _rater(self, principal: Principal) -> Rater:
        """The human rater of a request: agents, services and model judges are refused."""
        kind = rater_kind_for(principal)
        if kind == "model_judge":
            raise AuthorizationError("a model judge writes only judge labels")
        return Rater(principal_id=principal.principal_id, kind=kind)

    def _reveal(self, principal_id: str, root: str) -> RevealState:
        return _reveal_state(self._root, principal_id, root, self._clock)

    def _exposure(self, principal_id: str, root: str) -> Exposure:
        """Read through the ExposureLedger: a reveal marker (published here when the reveal
        condition holds) or an owner declaration."""
        self._reveal(principal_id, root)
        return "revealed" if self._ledger.exposure(principal_id, root) == "exposed" else "blind"

    def _judge_reads(self, found: snapshots.AnalysisSnapshot) -> bool:
        """A model judge reads and labels only an agent_answer snapshot whose origin is an agent's
        imported run (KTD11, CR-08): never a correction, by a rater or by a reference-exposed owner,
        never a reference, and never an answer frozen from a human attempt."""
        if found.kind != "agent_answer":
            return False
        origin = self._store._path(found.origin)[-1][1]
        return isinstance(origin, ChildAttempt) and isinstance(origin.action, ImportRun)

    def _visible[R: Label | StepFeedback](self, principal: Principal, root: str, records: tuple[R, ...]) -> tuple[R, ...]:
        own = principal.principal_id
        everyone = principal.principal_type is PrincipalType.USER and self._reveal(own, root).revealed
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

    def _step_root(self, principal: Principal, step: StepRef) -> str:
        path = self._store.lineage(principal, step.attempt)
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
            self._collections[directory.parent.name].append((directory.name,), record, at=at)
            if isinstance(record, Assignment):
                for rater in record.raters:
                    for digest in record.snapshots:
                        self._store._rater_items.add((rater, digest))


_HEX64 = re.compile(HEX64)
_SAFE_ID = re.compile(SAFE_ID)


def _records[R: Record](directory: Path, kind: type[R]) -> tuple[R, ...]:
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


def _opened(store_root: Path, root: str, principal_id: str) -> RevealOpened | None:
    collection = Ledger(store_root).legacy('reveals/{0}/{1}.json', RevealOpened, reader=__name__, key=KeyShape(('hex64', 'safe_id')))
    return collection.get((root, principal_id))


def _cohort(store_root: Path, principal_id: str, root: str) -> tuple[Digest, ...]:
    """The snapshots of the root's latest assignment when it names this rater, else none."""
    assignment = _latest(store_root / "assignments" / root, Assignment)
    if assignment is None or principal_id not in assignment.raters:
        return ()
    return assignment.snapshots


def _reveal_state(store_root: Path, principal_id: str, root: str, clock: Callable[[], datetime]) -> RevealState:
    """The reveal condition, publishing its marker the first time it holds, so a reveal never
    closes again (LabelStore._reveal and ExposureLedger.admits)."""
    assignment = _latest(store_root / "assignments" / root, Assignment)
    assigned = assignment.raters if assignment is not None else ()
    gating = set(assignment.snapshots) if assignment is not None else set()
    signed: dict[str, set[str]] = {}
    for signoff in _records(store_root / "signoffs" / root, SignOff):
        signed.setdefault(signoff.rater.principal_id, set()).add(signoff.snapshot)

    def done(rater: str) -> bool:
        return bool(gating) and gating <= signed.get(rater, set())

    pending = tuple(rater for rater in assigned if not done(rater))
    revealed = _opened(store_root, root, principal_id) is not None
    if not revealed and principal_id in assigned and not pending:
        opened = RevealOpened(root=root, principal_id=principal_id, opened_at_utc=clock())
        ledger = Ledger(store_root)
        collection = ledger.collection('reveals/{0}/{1}.json', RevealOpened, Mode.ONCE, KeyShape(('hex64', 'safe_id')), owner=__name__)
        with ledger.transaction():
            if collection.get((root, principal_id)) is None:
                collection.put((root, principal_id), opened)
        revealed = True
    return RevealState(
        root=root, principal_id=principal_id, signed_off=done(principal_id), pending_raters=pending, revealed=revealed
    )


def _hex(digest: str) -> str:
    """Every path built from a request digest goes through here first."""
    if not _HEX64.fullmatch(digest):
        raise AttemptAccessError(_DENIED)
    return digest
