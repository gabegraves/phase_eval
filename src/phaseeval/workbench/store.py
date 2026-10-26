"""Content-addressed attempt tree store. The only writer of attempts and annotations.

Target path: src/phaseeval/workbench/store.py (public). Replaces src/phaseeval/api/attempt_store.py and
keeps its custody primitives verbatim (_read_bytes with O_NOFOLLOW and a size bound,
_atomic_write_bytes with fsync and rename, _fsync_directory, symlink refusal, the ACL file
and authorize_case_operation). The one stored legacy attempt under ~/.local/state/axa-inspector
migrates in the same wave as a root plus an ImportRun, or is recorded as unmigratable.

Layout under the store root (0700). Every writer writes its own files [KTD2]:

    attempts/<digest>/attempt.json                    immutable; sha256(bytes) == <digest>
    attempts/<digest>/annotations/<utc>-<sha256>.json one file per annotation
    children/<parent>/<child>                         empty marker, one per child
    idem/<slot>                                       IdempotencyClaim, published exclusively
    authored/<principal>/<digest>                     empty marker; agent read checks are O(1)
    blobs/<sha256>                                    artifact bytes, shared across attempts
    roots/<digest>/acl.json                           per-root grants, written by the owner
    roots/<digest>/reveals/<principal>.json           first revealed view (review.RevealEvent)
    scratch/<uuid>/                                   one isolated directory per execution

An attempt is visible only once it is referenced: a root by its acl.json, a child by its
children/<parent>/<child> marker. An orphan left by a crash is never read.

Write protocol for one action (a crash at any point converges):
    1. parse, authorize, guard          no disk writes
    2. execute in scratch               crash: scratch is garbage, nothing references it
    3. write blobs                      content-addressed: rewriting is a no-op
    4. stage and rename the attempt dir crash: an unreferenced attempt dir is an orphan
    5. exclusive idempotency claim      lost race: return the winner, leave ours orphaned
    6. children and authored markers    crash before: a retry finds the claim, re-creates them
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from phaseeval.api.auth import (
    LABEL_MODE_ROLES,
    AuthorizationError,
    CaseOperation,
    Principal,
    PrincipalType,
    authorize_case_operation,
)

from .evidence import (
    Comparison,
    Curve,
    ReplayBounds,
    ReplayCheck,
    compare_fits,
    fit_attempt_for_compare,
    fit_of,
    phase_changes,
)
from .executors import ExecutionContext, Executor
from .model import (
    Accept,
    Action,
    Actor,
    AddPhases,
    AgentActor,
    AnalysisState,
    ArtifactRef,
    Attempt,
    AttemptDigest,
    AttemptRef,
    BlobDigest,
    BudgetExhausted,
    CaseManifest,
    ChildAttempt,
    Completed,
    CompletionReceipt,
    Contamination,
    CorrectionSource,
    CorrectMaterial,
    CorrectMeasurement,
    Digest,
    FitSummary,
    Halt,
    HaltReason,
    HumanActor,
    ImportedRecord,
    ImportedRunResult,
    ImportedSubmitted,
    ImportRun,
    InfraFailure,
    MalformedToolUse,
    Record,
    Recorded,
    RecordedPlayback,
    ReferenceExposure,
    Refine,
    Refused,
    Rejected,
    RemovePhases,
    ReplacePhase,
    RootAttempt,
    RunStart,
    SafeID,
    ServiceActor,
    SetOfHypotheses,
    Step,
    Submit,
    ToolIdentity,
    ToolOutcome,
    Unparsed,
    actor_for,
    apply_material_patch,
    apply_measurement_patch,
    canonical_bytes,
    initial_state,
    state_at,
    state_sha256,
    transition,
)
from .review import (
    ARTIFACT_AUDIENCE,
    ActorAttemptView,
    ActorPath,
    Adjudication,
    Annotation,
    AttestedCompletion,
    GestureStep,
    LoadEvent,
    Note,
    NoteRequest,
    Preference,
    PreferenceRequest,
    RecordRequest,
    RevealedAttemptView,
    RevealEvent,
    RunClose,
    TreeNodeSummary,
    TreeView,
    actor_attempt_view,
    actor_case_view,
    gesture_step,
    may_record,
    projection_for,
)
from .seams import Seams


class ActionRequest(Record):
    """Wire type for one action. `action` stays raw JSON until the store parses it, because an
    agent's malformed call must still become an attempt. There is no actor field: a request
    that sends one is refused by extra="forbid" (R1, KTD12)."""

    idempotency_key: SafeID
    rationale: str = Field(default="", max_length=20_000)
    action: JsonValue
    completion: CompletionReceipt | None = None  # agents with inline receipts only


AttemptView = ActorAttemptView | RevealedAttemptView


class AttemptAccessError(PermissionError):
    """Unknown digest, missing grant, or a read outside the caller's projection. One error for
    all three so a caller cannot probe which digests exist."""


class CustodyError(ValueError):
    """Bytes on disk disagree with their digest, or a path is unsafe."""


class ActionRefused(ValueError):
    """A human or service request that fails to parse, or an action on a terminal parent
    (HTTP 422). Agents get an Unparsed attempt instead of a parse refusal."""


class IdempotencyConflict(ValueError):
    """A key reused in its scope with different arguments (HTTP 409) (KTD12)."""


# ---------------------------------------------------------------------------
# Scoped idempotency (KTD12). Pure; the store owns the claim files.
# ---------------------------------------------------------------------------


class IdempotencyScope(Record):
    """A key is unique per principal, episode and case. For an agent the episode is its run
    (the RunStart digest); a human's episode is the case itself."""

    principal_id: SafeID
    root: AttemptRef
    run: AttemptRef | None


class IdempotencyClaim(Record):
    child: AttemptRef
    request_sha256: Digest


def idempotency_slot(scope: IdempotencyScope, key: str) -> Digest:
    payload = json.dumps([scope.model_dump(mode="json"), key], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def request_digest(parent: AttemptDigest, action: JsonValue) -> Digest:
    """The arguments a key is bound to. Rationale is excluded: it is never part of replay."""
    payload = json.dumps([parent, action], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def check_claim(claim: IdempotencyClaim, request_sha256: Digest) -> AttemptRef:
    """A retry returns the recorded child; the same key with different arguments is refused."""
    if claim.request_sha256 != request_sha256:
        raise IdempotencyConflict("idempotency key reused with different arguments")
    return claim.child


# ---------------------------------------------------------------------------
# Path confinement for ingest and import (R29)
# ---------------------------------------------------------------------------


class InboxPathError(PermissionError):
    """A path outside the inbox root, absolute, or through a symlink."""


def resolve_inbox_path(inbox: Path, relative: str) -> Path:
    """The only way an ingest or import path becomes a file (R29, AE9). Ingest resolves inside
    the inbox root, the import route inside the import root."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InboxPathError("paths are relative to the inbox root")
    root = inbox.resolve(strict=True)
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise InboxPathError("paths do not pass through symlinks")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise InboxPathError("paths stay inside the inbox root")
    return resolved


# ---------------------------------------------------------------------------
# Custody primitives, kept from attempt_store.py
# ---------------------------------------------------------------------------

MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_BLOB_BYTES = 64 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DENIED = "attempt access denied"


def _read_bytes(path: Path, maximum: int = MAX_RECORD_BYTES) -> bytes:
    """Read the very bytes being checked, refusing symlinks and special files."""
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or getattr(before, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise CustodyError("custody file is not a regular file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not os.path.samestat(before, metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
                raise CustodyError("custody file is not a bounded regular file")
            data = handle.read(maximum + 1)
    except OSError as exc:
        raise CustodyError("custody file is unavailable") from exc
    if len(data) > maximum:
        raise CustodyError("custody file exceeds byte limit")
    return data


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temporary(directory: Path, name: str, data: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, raw_temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".partial", dir=directory)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return Path(raw_temporary)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = _write_temporary(path.parent, path.name, data)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_once(path: Path, data: bytes) -> bool:
    """O_EXCL with complete bytes: link a fsynced temporary into place, so a crash never leaves
    a partial file under the final name. False when the name already exists."""
    temporary = _write_temporary(path.parent, path.name, data)
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    return True


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600))
    _fsync_directory(path.parent)


def _directory(path: Path) -> None:
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise CustodyError("custody directory is unsafe")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

ACL_SCHEMA = "axa.workbench_acl.v1"
OWNER_OPERATIONS = ("inspect", "action", "export", "artifact")
# The identity the store writes its own guard halts under (budget, identity, malformed use).
GUARD = ServiceActor(principal_id="workbench-store", role="budget_guard")
_ACTION: TypeAdapter[Action] = TypeAdapter(Action)
_ATTEMPT: TypeAdapter[Attempt] = TypeAdapter(Attempt)
_ANNOTATION: TypeAdapter[Annotation] = TypeAdapter(Annotation)

Path_ = tuple[tuple[AttemptDigest, Attempt], ...]  # root first


@cache
def _axa_git_sha() -> str:
    found = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return found.stdout.strip() or "unknown"


def _tool(name: str) -> ToolIdentity:
    return ToolIdentity(name=name, version="1", axa_git_sha=_axa_git_sha())


class _Unparseable(Exception):
    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _errors(exc: ValidationError) -> tuple[str, ...]:
    return tuple(
        f"{'.'.join(str(part) for part in error['loc']) or 'action'}: {error['msg']}"
        for error in exc.errors()
    )


def _parse(actor: Actor, raw: JsonValue) -> Action:
    """One union for every actor (R2); the actor kind decides which verbs it may author."""
    try:
        action = _ACTION.validate_python(raw)
    except ValidationError as exc:
        raise _Unparseable(_errors(exc)) from exc
    allowed = actor.kind in type(action).authors and not (
        isinstance(actor, ServiceActor)
        and not (isinstance(action, Halt) and actor.role in ("budget_guard", "harness"))
    )
    if not allowed:
        raise _Unparseable((f"{actor.kind} may not author {action.verb}",))
    return action


def _fit_of(attempt: Attempt | None) -> FitSummary | None:
    return fit_of(attempt) if isinstance(attempt, ChildAttempt) else None


def _verdict(attempt: ChildAttempt) -> str | None:
    if isinstance(attempt.action, Submit):
        return attempt.action.answer.verdict
    if isinstance(attempt.outcome, Completed) and isinstance(attempt.outcome.result, ImportedRunResult):
        end = attempt.outcome.result.end
        return end.answer.verdict if isinstance(end, ImportedSubmitted) else None
    return None


def _context_changes(a: AnalysisState, b: AnalysisState) -> tuple[str, ...]:
    """Top-level context fields whose values differ, listed apart from phase changes (R14)."""
    changes = []
    for name, left, right in (("measurement", a.measurement, b.measurement), ("material", a.material, b.material)):
        left_json, right_json = left.model_dump(mode="json"), right.model_dump(mode="json")
        changes.extend(f"{name}.{field}" for field in left_json if left_json[field] != right_json.get(field))
    return tuple(changes)


def _run(path: Path_, principal_id: str) -> int | None:
    """Index of the nearest RunStart on the path authored by this principal."""
    for index in range(len(path) - 1, 0, -1):
        attempt = path[index][1]
        if isinstance(attempt.action, RunStart) and attempt.actor.principal_id == principal_id:
            return index
    return None


def _steps(path: Path_) -> tuple[Step, ...]:
    """A root-first path without its root, as the fold takes it."""
    return tuple((digest, attempt) for digest, attempt in path[1:] if isinstance(attempt, ChildAttempt))




def _refuse_label_mode(principal: Principal) -> None:
    """Label-mode credentials (R33) never read the raw views, which carry run identity and every
    annotation; a rater's page reads only the blind routes until sign-off (D10, D11)."""
    if LABEL_MODE_ROLES & principal.roles:
        raise AttemptAccessError(_DENIED)


def _attempt_refs(attempt: ChildAttempt) -> tuple[ArtifactRef, ...]:
    """Every artifact an attempt references itself: outcome artifacts, fit curves, receipts."""
    outcome = attempt.outcome
    if not isinstance(outcome, Completed):
        return ()
    result = outcome.result
    fits = (*result.stages, result.final) if result.kind == "refine" else (result.basis_fit,)
    curves = tuple(f.curve for f in fits if f is not None and f.curve is not None)
    receipts = outcome.origin.receipts if isinstance(outcome.origin, ImportedRecord) else ()
    return (*outcome.artifacts, *curves, *receipts)


class ExposureReader(Protocol):
    """Has this principal seen the root's reference, and may it read this lineage?
    labels.ExposureLedger answers from the reveal markers, the owner's declarations, the review
    cohort and the sign-offs, so a blind correction is told from a revealed one (architect D2,
    D3; R16) and a rater reads only its cohort and its own attempts before the reveal (CR-07)."""

    def exposure(self, principal_id: SafeID, root: AttemptRef) -> ReferenceExposure: ...

    def admits(self, principal: Principal, path: Path_) -> bool: ...


class WorkbenchStore:
    def __init__(
        self,
        root: Path,
        execute: Executor,
        seams: Seams,
        clock: Callable[[], datetime],
        *,
        import_root: Path,  # data/custody/workbench/runs; import paths resolve only inside it
        exposure: ExposureReader,  # labels.ExposureLedger(root) in production
    ) -> None:
        """`execute` is the one way the store runs a verb: `ExecutorRegistry.run` in
        production, a fake in tests."""
        if not root.is_absolute():
            raise ValueError("workbench store root must be absolute")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.import_root = import_root
        self._execute = execute
        self._seams = seams
        self._clock = clock
        self._exposure = exposure

    # ----- writes --------------------------------------------------------------------------

    def load_case(
        self,
        principal: Principal,
        case: CaseManifest,
        blobs: Mapping[BlobDigest, bytes],
        *,
        via: Literal["mcp_ingest", "script"] = "script",
    ) -> AttemptDigest:
        """Owner role only. Idempotent by construction: the root digest is the case digest, so a
        second load of the same evidence returns the existing root."""
        if not principal.can_create_cases:
            raise AuthorizationError("principal cannot create cases")
        refs = (case.pattern, *(c.cif for c in case.candidates), *(s.blob for s in case.ingest.sources))
        verified = self._verified(refs, blobs)
        scanner = self._seams.scanner
        if scanner is not None and any(
            "agent" in ARTIFACT_AUDIENCE[ref.role] and scanner.scan(verified[ref.sha256])
            for ref in refs
        ):
            raise CustodyError("a custody canary is in an actor-visible case file")
        data = canonical_bytes(RootAttempt(case=case))
        digest = AttemptDigest(hashlib.sha256(data).hexdigest())
        self._put_blobs(verified)
        acl = {
            "schema_version": ACL_SCHEMA,
            "root": digest,
            "access": {principal.principal_id: list(OWNER_OPERATIONS)},
        }
        _publish_once(self.root / "roots" / digest / "acl.json", _json_bytes(acl))
        self._put_attempt(digest, data)
        if not any(isinstance(a, LoadEvent) for a in self.annotations(digest)):
            event = LoadEvent(principal_id=principal.principal_id, loaded_at_utc=self._clock(), via=via)
            self._annotate(digest, event)
        return digest

    def grant(self, principal: Principal, root: AttemptDigest, grantee: SafeID, ops: tuple[CaseOperation, ...]) -> None:
        """Owner only: per-root grants, as the current ACL file. The owner is the one writer."""
        acl = self._acl(root)
        if not principal.can_create_cases or principal.principal_id not in acl["access"]:
            raise AttemptAccessError(_DENIED)
        if not _SAFE_ID.fullmatch(grantee):
            raise ValueError("grantee must be a safe identifier")
        access = dict(acl["access"])
        access[grantee] = sorted({*access.get(grantee, ()), *(op.value for op in ops)})
        _atomic_write_bytes(self.root / "roots" / root / "acl.json", _json_bytes(acl | {"access": access}))

    def act(self, principal: Principal, parent: AttemptDigest, request: ActionRequest) -> AttemptView:
        """The one action path for agents, the expert and services (R2), returning the caller's
        projection of the child."""
        return self._view(principal, self._path(self._act(principal, parent, request)))

    def gesture(
        self, principal: Principal, parent: AttemptDigest, requests: tuple[ActionRequest, ...]
    ) -> tuple[GestureStep, ...]:
        """One or two actions, each the child of the one before: a phase or context edit and the
        refine after it (R13). Step two runs only when step one was recorded or completed, so a
        refused edit is never refit. Returns blind summaries and records no reveal (D10)."""
        steps: list[GestureStep] = []
        for request in requests:
            digest = self._act(principal, parent, request)
            attempt = self._load(digest)
            assert isinstance(attempt, ChildAttempt)
            steps.append(gesture_step(digest, attempt))
            if steps[-1].status not in ("recorded", "completed"):
                break
            parent = digest
        return tuple(steps)

    def _act(self, principal: Principal, parent: AttemptDigest, request: ActionRequest) -> AttemptDigest:
        path = self._path(parent)
        root_digest, root = path[0]
        self._authorize(principal, root_digest, CaseOperation.ACTION)
        agent = principal.principal_type is PrincipalType.AGENT
        if not agent and not self._exposure.admits(principal, path):
            raise AttemptAccessError(_DENIED)  # nobody builds on what it may not read (CR-07)
        parent_attempt = path[-1][1]
        if isinstance(parent_attempt, ChildAttempt) and getattr(parent_attempt.action, "terminal", False):
            raise ActionRefused("the parent is terminal: nothing may follow it")
        run = _run(path, principal.principal_id) if agent else None
        scope = IdempotencyScope(
            principal_id=principal.principal_id,
            root=root_digest,
            run=path[run][0] if run is not None else None,
        )
        slot = idempotency_slot(scope, request.idempotency_key)
        req = request_digest(parent, request.action)
        claim = self._claim_at(slot)
        if claim is not None:
            child = check_claim(claim, req)
            self._mark(parent, child, principal.principal_id)
            return AttemptDigest(child)

        if request.completion is not None and not agent:
            raise ActionRefused("only agents send completion receipts")
        actor = actor_for(
            principal,
            completion=request.completion,
            reference_exposure=self._exposure.exposure(principal.principal_id, root_digest),
        )
        try:
            action: Action | Unparsed = _parse(actor, request.action)
            outcome: ToolOutcome | None = None
        except _Unparseable as exc:
            if not agent:
                raise ActionRefused(str(exc)) from exc
            action, outcome = Unparsed(raw=request.action), Rejected(errors=exc.errors)
        if agent and not isinstance(action, RunStart) and not self._authored(principal.principal_id, parent):
            raise AttemptAccessError(_DENIED)  # an agent continues only from its own attempts

        state = state_at(path[0], _steps(path))
        guarded = run is not None and not isinstance(action, RunStart)
        halt = self._run_halt(path[run:], request.completion) if guarded else None
        tool, inputs, produced = _tool("workbench_store"), (), {}
        if halt is not None and not (isinstance(action, Halt) and outcome is None):
            actor, action, outcome = GUARD, Halt(reason=halt), Recorded()
        elif outcome is None:
            outcome = self._refusal(actor, action, path, state)
        if outcome is None and type(action).executes:
            workdir = self.root / "scratch" / uuid.uuid4().hex
            workdir.mkdir(parents=True, mode=0o700)
            context = ExecutionContext(case=root.case, state=state, read_blob=self._read_blob, workdir=workdir)
            executed, tool, inputs, raw_blobs = self._execute(context, action)
            produced = dict(raw_blobs)
            for sha, data in produced.items():
                if hashlib.sha256(data).hexdigest() != sha:
                    raise CustodyError("an executor blob disagrees with its digest")
            outcome = self._sealed(executed, state, produced)
        attempt = self._child(
            path,
            actor,
            action,
            tool,
            inputs,
            outcome if outcome is not None else Recorded(),
            request,
            agent=agent,
        )
        self._put_blobs(produced)
        return self._write(attempt, principal.principal_id, slot, req)

    def import_run(
        self,
        principal: Principal,
        root: AttemptDigest,
        action: ImportRun,
        result: ImportedRunResult,
        blobs: Mapping[BlobDigest, bytes],
    ) -> AttemptDigest:
        """Importer service only (R28). Writes one child of the root whose outcome origin is
        imported; never runs a guard meant for live actions, never executes. Idempotent on the
        Harbor trial id; the same trial with a different parse is a conflict."""
        exposure = self._exposure.exposure(principal.principal_id, root)
        actor = actor_for(principal, completion=None, reference_exposure=exposure)
        if not (isinstance(actor, ServiceActor) and actor.role == "importer"):
            raise AttemptAccessError(_DENIED)
        path = self._path(root)
        if len(path) != 1:
            raise AttemptAccessError(_DENIED)
        # R31: a trial binds to its root by the hashes of the files it read, never by path.
        case, harbor = path[0][1].case, action.harbor
        library = {c.candidate_id: c.cif.sha256 for c in case.candidates}
        if harbor.pattern_sha256 != case.pattern.sha256 or any(
            library.get(candidate) != sha for candidate, sha in harbor.cif_sha256.items()
        ):
            raise ActionRefused("the trial read files that are not this root's case")
        receipts = tuple(
            ArtifactRef(sha256=sha, size_bytes=len(data), media_type="application/octet-stream", role="harbor_receipt")
            for sha, data in sorted(blobs.items())
        )
        verified = self._verified(receipts, blobs)
        end = result.end
        if isinstance(end, ImportedSubmitted) and end.answer_sha256 is not None and end.answer_sha256 not in verified:
            raise ActionRefused("the answer file the import names is not one of its receipts")
        key = action.harbor.trial_id
        slot = idempotency_slot(IdempotencyScope(principal_id=actor.principal_id, root=root, run=None), key)
        req = request_digest(
            root, {"action": action.model_dump(mode="json"), "result": result.model_dump(mode="json")}
        )
        claim = self._claim_at(slot)
        if claim is not None:
            child = check_claim(claim, req)
            self._mark(root, child, actor.principal_id)
            return child
        scanner = self._seams.scanner
        match = next((m for data in verified.values() if scanner and (m := scanner.scan(data))), None)
        outcome = Completed(
            result=result,
            parent_state_sha256=state_sha256(initial_state(*path[0])),
            elapsed_seconds=0.0,
            custody_scan="not_configured" if scanner is None else "match" if match else "clean",
            contamination=None
            if match is None
            else Contamination(canary_sha256=match.canary_sha256, channel="transcript"),
            origin=ImportedRecord(receipts=receipts),
        )
        attempt = ChildAttempt(
            parent=root,
            root=root,
            depth=1,
            actor=actor,
            action=action,
            tool=_tool("harbor_importer"),
            inputs=tuple(verified),
            outcome=outcome,
            idempotency_key=key,
            created_at_utc=self._clock(),
        )
        self._put_blobs(verified)
        return self._write(attempt, actor.principal_id, slot, req)

    def record(self, principal: Principal, target: AttemptDigest, request: RecordRequest) -> Annotation:
        """Notes and preferences, authored by a human principal (R5, R29)."""
        if (refusal := may_record(principal, request)) is not None:
            raise AuthorizationError(refusal)
        path = self._path(target)
        root_digest = path[0][0]
        self._authorize(principal, root_digest, CaseOperation.ACTION)
        if not self._exposure.admits(principal, path):
            raise AttemptAccessError(_DENIED)
        exposure = self._exposure.exposure(principal.principal_id, root_digest)
        author = actor_for(principal, completion=None, reference_exposure=exposure)
        assert isinstance(author, HumanActor)
        now = self._clock()
        annotation: Annotation
        match request:
            case NoteRequest():
                annotation = Note(author=author, created_at_utc=now, text=request.text)
            case PreferenceRequest():
                if request.chosen != target:
                    raise ActionRefused("a preference is recorded on its chosen attempt")
                if root_digest in (request.chosen, request.rejected):
                    raise ActionRefused("a preference never names the root")
                self._under_root(principal, request.rejected, root_digest)
                annotation = Preference(
                    author=author,
                    chosen=request.chosen,
                    rejected=request.rejected,
                    reason=request.reason,
                    created_at_utc=now,
                )
        self._annotate(target, annotation)
        return annotation

    def record_adjudication(self, principal: Principal, submit: AttemptDigest, adjudication: Adjudication) -> Adjudication:
        """sealed_evaluator service only. Nothing at the hackathon calls it (R5)."""
        raise NotImplementedError

    def close_run(
        self,
        principal: Principal,
        run_start: AttemptDigest,
        transcript: bytes,
        completions: tuple[AttestedCompletion, ...],
    ) -> RunClose:
        """Harness service only. Scans the full transcript, which also catches a leak through a
        channel the store never carried, and attests per-completion model identity (AE4)."""
        raise NotImplementedError

    def replay(self, principal: Principal, digest: AttemptDigest, bounds: ReplayBounds) -> ReplayCheck:
        """Re-execute one attempt's action from its parent's derived state in this environment,
        compare under bounds, append a ReplayCheck. Never creates a child (R4)."""
        raise NotImplementedError

    # ----- reads (projection derived from the principal) ----------------------------------

    def view(self, principal: Principal, digest: AttemptDigest) -> AttemptView:
        """Agent: only digests it authored, as ActorAttemptView. Human or service: the revealed
        view; a human's first view of a root records a RevealEvent. Label-mode principals are refused (D11)."""
        _refuse_label_mode(principal)
        path = self._path(digest)
        self._authorize(principal, path[0][0], CaseOperation.INSPECT)
        return self._view(principal, path)

    def path(self, principal: Principal, digest: AttemptDigest) -> ActorPath | tuple[RevealedAttemptView, ...]:
        """Root to digest. For an agent: review.ActorPath for its own run head, whose case view
        is the effective context at its RunStart (KTD12). Label-mode principals are refused (D11)."""
        _refuse_label_mode(principal)
        path = self._path(digest)
        root_digest, root = path[0]
        self._authorize(principal, root_digest, CaseOperation.INSPECT)
        if projection_for(principal) == "actor":
            start = _run(path, principal.principal_id)
            if start is None or not self._authored(principal.principal_id, digest):
                raise AttemptAccessError(_DENIED)
            state = state_at(path[0], _steps(path[: start + 1]))
            return ActorPath(
                case=actor_case_view(root_digest, root, state),
                steps=tuple(actor_attempt_view(d, a, principal) for d, a in path[start:]),
            )
        self._reveal(principal, root_digest)
        state = initial_state(*path[0])
        views = [self._revealed(root_digest, root, state)]
        for step in _steps(path):
            state = transition(state, step)
            views.append(self._revealed(*step, state))
        return tuple(views)

    def tree(self, principal: Principal, root: AttemptDigest) -> TreeView:
        """Humans and services only, label-mode principals excepted (D11). One summary row per referenced
        attempt, breadth first."""
        _refuse_label_mode(principal)
        if projection_for(principal) == "actor":
            raise AttemptAccessError(_DENIED)
        path = self._path(root)
        if len(path) != 1:
            raise AttemptAccessError(_DENIED)
        self._authorize(principal, root, CaseOperation.INSPECT)
        self._reveal(principal, root)
        root_attempt = path[0][1]
        state = initial_state(root, root_attempt)
        nodes = [
            TreeNodeSummary(
                digest=root,
                parent=None,
                depth=0,
                actor_kind="root",
                principal_id=None,
                verb="root",
                status=None,
                origin=None,
                rwp_percent=None,
                gof=None,
                phases=(),
                verdict=None,
                created_at_utc=None,
            )
        ]
        queue = deque((child, state) for child in self._children(root))
        while queue:
            digest, parent_state = queue.popleft()
            attempt = self._load(digest)
            assert isinstance(attempt, ChildAttempt)
            state = transition(parent_state, (digest, attempt))
            fit = _fit_of(attempt)
            outcome = attempt.outcome
            nodes.append(
                TreeNodeSummary(
                    digest=digest,
                    parent=attempt.parent,
                    depth=attempt.depth,
                    actor_kind=attempt.actor.kind,
                    principal_id=attempt.actor.principal_id,
                    verb=attempt.action.verb,
                    status=outcome.status,
                    origin=outcome.origin.origin if isinstance(outcome, Completed) else None,
                    rwp_percent=fit.rwp_percent if fit else None,
                    gof=fit.gof if fit else None,
                    phases=state.hypothesis.phases if state.hypothesis else (),
                    verdict=_verdict(attempt),
                    created_at_utc=attempt.created_at_utc,
                )
            )
            queue.extend((child, state) for child in self._children(digest))
        return TreeView(root=root, nodes=tuple(nodes))

    def compare(self, principal: Principal, a: AttemptDigest, b: AttemptDigest) -> Comparison:
        """Humans only, label-mode principals excepted (D11). Each side resolves through
        evidence.fit_attempt_for_compare (R14)."""
        _refuse_label_mode(principal)
        if principal.principal_type is not PrincipalType.USER:
            raise AttemptAccessError(_DENIED)
        a_path, b_path = self._path(a), self._path(b)
        root_digest = a_path[0][0]
        if b_path[0][0] != root_digest:
            raise ActionRefused("a comparison stays under one root")
        self._authorize(principal, root_digest, CaseOperation.INSPECT)
        self._reveal(principal, root_digest)
        attempts = {d: x for d, x in self.walk(root_digest) if isinstance(x, ChildAttempt)}
        children = {d: self._children(d) for d in attempts}
        sides = []
        for path in (a_path, b_path):
            digest = path[-1][0]
            source = fit_attempt_for_compare(digest, attempts, children)
            fit_path = self._path(source) if source is not None else path
            fit = _fit_of(fit_path[-1][1]) if source is not None else None
            sides.append((source, fit_path, fit, self._curve(fit)))
        (a_from, a_fit_path, a_fit, a_curve), (b_from, b_fit_path, b_fit, b_curve) = sides
        a_state = state_at(a_fit_path[0], _steps(a_fit_path))
        b_state = state_at(b_fit_path[0], _steps(b_fit_path))
        comparison = compare_fits(
            a,
            b,
            a_from,
            b_from,
            phase_changes(a_state.hypothesis, b_state.hypothesis),
            _context_changes(a_state, b_state),
            a_fit,
            b_fit,
            a_curve,
            b_curve,
        )
        playback = [
            side
            for side, (source, *_rest) in zip("ab", sides, strict=True)
            if source is not None
            and isinstance(outcome := attempts[source].outcome, Completed)
            and isinstance(outcome.origin, RecordedPlayback)
        ]
        if playback:
            caveat = f"playback fit on side {', '.join(playback)} (AE8)"
            comparison = comparison.model_copy(
                update={"like_for_like": False, "caveats": (*comparison.caveats, caveat)}
            )
        return comparison

    def blob(self, principal: Principal, digest: AttemptDigest, sha256: BlobDigest) -> tuple[str, bytes]:
        """Bytes referenced by an attempt the caller may view; agents get only roles whose
        review.ARTIFACT_AUDIENCE includes them. Digest re-checked on every read. Label-mode
        principals are refused (D11)."""
        _refuse_label_mode(principal)
        path = self._path(digest)
        root_digest, root = path[0]
        self._authorize(principal, root_digest, CaseOperation.ARTIFACT)
        attempt = path[-1][1]
        agent = projection_for(principal) == "actor"
        if agent and isinstance(attempt, ChildAttempt) and not self._authored(principal.principal_id, digest):
            raise AttemptAccessError(_DENIED)
        case = root.case
        case_refs = {r.sha256: r for r in (case.pattern, *(c.cif for c in case.candidates), *(s.blob for s in case.ingest.sources))}
        if isinstance(attempt, ChildAttempt):
            own = {r.sha256: r for r in _attempt_refs(attempt)}
            referenced = own | {sha: case_refs[sha] for sha in attempt.inputs if sha in case_refs and sha not in own}
        else:
            referenced = case_refs
        ref = referenced.get(sha256)
        audience = "agent" if agent else "expert"
        if ref is None or audience not in ARTIFACT_AUDIENCE[ref.role]:
            raise AttemptAccessError(_DENIED)
        data = self._read_blob(BlobDigest(sha256))
        if len(data) != ref.size_bytes:
            raise CustodyError("blob bytes disagree with their record")
        return ref.media_type, data

    def roots(self, principal: Principal) -> tuple[AttemptDigest, ...]:
        """Roots whose ACL grants this principal inspect."""
        directory = self.root / "roots"
        if not directory.is_dir():
            return ()
        found = []
        for name in sorted(os.listdir(directory)):
            if not _HEX64.fullmatch(name):
                continue
            try:
                self._authorize(principal, name, CaseOperation.INSPECT)
            except AttemptAccessError:
                continue
            found.append(AttemptDigest(name))
        return tuple(found)

    def lineage(self, principal: Principal, digest: AttemptDigest) -> tuple[tuple[AttemptDigest, Attempt], ...]:
        """Root to digest for a principal that may inspect the root and whose rater projection
        admits the whole path (ExposureReader.admits, CR-07), recording nothing (path records a
        reveal and returns views). Every review payload, snapshot build and freeze, snapshot
        read and step feedback reads through it and projects its own blind view (W3 view)."""
        path = self._path(digest)
        self._authorize(principal, path[0][0], CaseOperation.INSPECT)
        if not self._exposure.admits(principal, path):
            raise AttemptAccessError(_DENIED)
        return path

    def read_blob(self, sha256: BlobDigest) -> bytes:
        """Blob bytes by digest, re-checked on read. Trusted callers that already authorized the
        attempt referencing them, as for walk (W3 view)."""
        return self._read_blob(sha256)

    def walk(self, root: AttemptDigest) -> tuple[tuple[AttemptDigest, Attempt], ...]:
        """Every referenced attempt under a root, breadth first, for policy and export. Trusted
        callers."""
        found: list[tuple[AttemptDigest, Attempt]] = []
        queue = deque([self._path(root)[-1]])
        while queue:
            digest, attempt = queue.popleft()
            found.append((digest, attempt))
            queue.extend((child, self._load(child)) for child in self._children(digest))
        return tuple(found)

    def annotations(self, digest: AttemptDigest) -> tuple[Annotation, ...]:
        directory = self.root / "attempts" / digest / "annotations"
        if not directory.is_dir():
            return ()
        found = []
        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            data = _read_bytes(directory / name)
            if not name.endswith(f"-{hashlib.sha256(data).hexdigest()}.json"):
                raise CustodyError("annotation bytes disagree with their name")
            found.append(_ANNOTATION.validate_json(data))
        return tuple(found)

    def _under_root(self, principal: Principal, digest: str, root: str) -> None:
        """A digest a record names must be a referenced attempt under the same root, and one its
        author may read (CR-07)."""
        path = self._path(digest)
        if path[0][0] != root:
            raise ActionRefused("a record names only attempts under its own root")
        if not self._exposure.admits(principal, path):
            raise AttemptAccessError(_DENIED)

    def _curve(self, fit: FitSummary | None) -> Curve | None:
        """The retained curve of a fit, or None when it is not retained or does not parse."""
        if fit is None or fit.curve is None:
            return None
        try:
            return Curve.parse_csv(self._read_blob(fit.curve.sha256))
        except (CustodyError, ValueError):
            return None

    # ----- internals: decisions ------------------------------------------------------------

    def _run_halt(self, run: Path_, completion: CompletionReceipt | None) -> HaltReason | None:
        """Guards derived from the agent's open run (its RunStart first): model identity, budget
        and consecutive malformed calls. Each ends the run with a Halt instead of the action."""
        run_start = run[0][1]
        assert isinstance(run_start, ChildAttempt) and isinstance(run_start.action, RunStart)
        budget = run_start.action.budget
        steps = [attempt for _, attempt in run[1:] if isinstance(attempt, ChildAttempt)]
        if completion is not None and not completion.identity_verified:
            detail = f"requested {completion.model_requested}, observed {completion.model_observed}"
            return InfraFailure(cause="model_identity_mismatch", detail=detail)
        if len(steps) >= budget.max_model_actions:
            return BudgetExhausted(limit="model_actions")
        if (self._clock() - run_start.created_at_utc).total_seconds() >= budget.max_wall_seconds:
            return BudgetExhausted(limit="wall_seconds")
        executor_seconds = sum(a.outcome.elapsed_seconds for a in steps if isinstance(a.outcome, Completed))
        if executor_seconds >= budget.max_executor_seconds:
            return BudgetExhausted(limit="executor_seconds")
        cost = sum(
            a.actor.completion.cost_usd or 0.0
            for a in steps
            if isinstance(a.actor, AgentActor) and a.actor.completion is not None
        )
        if budget.max_cost_usd is not None and cost >= budget.max_cost_usd:
            return BudgetExhausted(limit="cost_usd")
        rejections = 0
        for attempt in reversed(steps):
            if not isinstance(attempt.outcome, Rejected):
                break
            rejections += 1
        if rejections >= budget.max_consecutive_rejections:
            return MalformedToolUse(consecutive_rejections=rejections)
        return None

    def _refusal(self, actor: Actor, action: Action, path: Path_, state: AnalysisState) -> Refused | None:
        """Valid actions the state does not admit. Each code names the observation."""
        case = path[0][1].case
        library = {c.candidate_id for c in case.candidates}
        phases = state.hypothesis.phases if state.hypothesis else ()
        if isinstance(action, AddPhases | RemovePhases | ReplacePhase):
            named = (action.old, action.new) if isinstance(action, ReplacePhase) else action.candidates
            if unknown := [c for c in named if c not in library]:
                return Refused(reason_code="candidate_not_in_case_library", detail=", ".join(unknown))
            if isinstance(action, AddPhases) and (present := [c for c in action.candidates if c in phases]):
                return Refused(reason_code="edit_not_applicable", detail=f"already present: {', '.join(present)}")
            if isinstance(action, RemovePhases) and (absent := [c for c in action.candidates if c not in phases]):
                return Refused(reason_code="edit_not_applicable", detail=f"not present: {', '.join(absent)}")
            if isinstance(action, ReplacePhase) and (action.old not in phases or action.new in phases):
                return Refused(reason_code="edit_not_applicable", detail=f"cannot replace {action.old} with {action.new}")
        if isinstance(action, Refine) and not phases:
            return Refused(reason_code="no_phases", detail="the phase set is empty")
        if isinstance(action, CorrectMeasurement | CorrectMaterial):
            source = CorrectionSource(attempt="0" * 64, principal_id=actor.principal_id)
            try:
                if isinstance(action, CorrectMeasurement):
                    apply_measurement_patch(state.measurement, action.patch, source)
                else:
                    apply_material_patch(state.material, action.patch, source)
            except ValueError as exc:
                return Refused(reason_code="correction_not_applicable", detail=str(exc))
        if isinstance(action, Submit) and isinstance(action.answer, Accept):
            fit = _fit_of(dict(path).get(action.answer.basis))
            if fit is None or fit.stage == "pawley":
                return Refused(reason_code="basis_not_rietveld_ancestor", detail=action.answer.basis)
            if state.measurement.profile.value.blocks_accept:
                return Refused(reason_code="profile_blocks_accept", detail="no usable instrument profile")
        if isinstance(action, Submit) and isinstance(action.answer, SetOfHypotheses):
            for ranked in action.answer.ranked:
                try:
                    attempt = self._path(ranked)[-1][1]
                except AttemptAccessError:
                    attempt = None
                if _fit_of(attempt) is None or attempt.root != path[0][0]:
                    return Refused(reason_code="ranked_not_fit_under_root", detail=ranked)
        return None

    def _sealed(self, outcome: ToolOutcome, state: AnalysisState, produced: Mapping[str, bytes]) -> ToolOutcome:
        """The store stamps what only it knows on a completed execution: the state it handed the
        executor, and the custody scan of every byte bound for the actor (R9)."""
        if not isinstance(outcome, Completed):
            return outcome
        scanner = self._seams.scanner
        update: dict[str, object] = {"parent_state_sha256": state_sha256(state)}
        if scanner is None:
            return outcome.model_copy(update=update | {"custody_scan": "not_configured"})
        hits = {sha: m for sha, data in produced.items() if (m := scanner.scan(data))}
        match = scanner.scan(canonical_bytes(outcome.result)) or next(iter(hits.values()), None)
        if match is None:
            return outcome.model_copy(update=update | {"custody_scan": "clean"})
        artifacts = tuple(
            a.model_copy(update={"role": "custody_withheld"}) if a.sha256 in hits else a
            for a in outcome.artifacts
        )
        contamination = Contamination(canary_sha256=match.canary_sha256, channel="tool_output")
        return outcome.model_copy(
            update=update | {"custody_scan": "match", "contamination": contamination, "artifacts": artifacts}
        )

    def _child(
        self,
        path: Path_,
        actor: Actor,
        action: Action | Unparsed,
        tool: ToolIdentity,
        inputs: tuple[BlobDigest, ...],
        outcome: ToolOutcome,
        request: ActionRequest,
        *,
        agent: bool,
    ) -> ChildAttempt:
        parent, parent_attempt = path[-1]
        fields = {
            "parent": parent,
            "root": path[0][0],
            "depth": len(path),
            "actor": actor,
            "tool": tool,
            "inputs": inputs,
            "rationale": "" if actor == GUARD else request.rationale,
            "idempotency_key": request.idempotency_key,
            "created_at_utc": self._clock(),
        }
        try:
            return ChildAttempt(**fields, action=action, outcome=outcome)
        except ValidationError as exc:
            if isinstance(outcome, Completed):
                raise  # the executor broke its contract; nothing is written
            if not agent:
                raise ActionRefused("; ".join(_errors(exc))) from exc
            return ChildAttempt(
                **fields, action=Unparsed(raw=request.action), outcome=Rejected(errors=_errors(exc))
            )

    # ----- internals: disk -----------------------------------------------------------------

    def _load(self, digest: str) -> Attempt:
        """Custody-checked read: the bytes re-hash to their digest."""
        if not _HEX64.fullmatch(digest):
            raise AttemptAccessError(_DENIED)
        directory = self.root / "attempts" / digest
        if not os.path.lexists(directory):
            raise AttemptAccessError(_DENIED)
        _directory(self.root / "attempts")
        _directory(directory)
        data = _read_bytes(directory / "attempt.json")
        if hashlib.sha256(data).hexdigest() != digest:
            raise CustodyError("attempt bytes disagree with their digest")
        return _ATTEMPT.validate_json(data)

    def _path(self, digest: str) -> Path_:
        """Root first. Every attempt is custody-checked and referenced; linkage is verified."""
        found: list[tuple[AttemptDigest, Attempt]] = []
        current = digest
        while True:
            attempt = self._load(current)
            found.append((AttemptDigest(current), attempt))
            if isinstance(attempt, RootAttempt):
                break
            if not os.path.lexists(self.root / "children" / attempt.parent / current):
                raise AttemptAccessError(_DENIED)
            current = attempt.parent
        if not os.path.lexists(self.root / "roots" / current / "acl.json"):
            raise AttemptAccessError(_DENIED)
        found.reverse()
        for depth, (_, attempt) in enumerate(found[1:], start=1):
            assert isinstance(attempt, ChildAttempt)
            if attempt.depth != depth or attempt.root != current:
                raise CustodyError("attempt linkage disagrees with its path")
        return tuple(found)

    def _acl(self, root: str) -> dict:
        try:
            _directory(self.root / "roots")
            _directory(self.root / "roots" / root)
            acl = json.loads(_read_bytes(self.root / "roots" / root / "acl.json"))
            if acl.get("schema_version") != ACL_SCHEMA or acl.get("root") != root:
                raise AuthorizationError(_DENIED)
            return acl
        except (OSError, ValueError, AttributeError):
            raise AttemptAccessError(_DENIED) from None

    def _authorize(self, principal: Principal, root: str, operation: CaseOperation) -> None:
        acl = self._acl(root)
        try:
            authorize_case_operation(principal, root, operation, {"case_id": root, "access": acl.get("access")})
        except AuthorizationError:
            raise AttemptAccessError(_DENIED) from None

    def _authored(self, principal_id: str, digest: str) -> bool:
        return os.path.lexists(self.root / "authored" / principal_id / digest)

    def _children(self, digest: str) -> tuple[AttemptDigest, ...]:
        directory = self.root / "children" / digest
        if not directory.is_dir():
            return ()
        return tuple(AttemptDigest(name) for name in sorted(os.listdir(directory)) if _HEX64.fullmatch(name))

    def _read_blob(self, sha256: BlobDigest) -> bytes:
        if not _HEX64.fullmatch(sha256):
            raise CustodyError("blob digests are 64 hex")
        data = _read_bytes(self.root / "blobs" / sha256, MAX_BLOB_BYTES)
        if hashlib.sha256(data).hexdigest() != sha256:
            raise CustodyError("blob bytes disagree with their digest")
        return data

    def _verified(self, refs: tuple[ArtifactRef, ...], blobs: Mapping[BlobDigest, bytes]) -> dict[BlobDigest, bytes]:
        verified: dict[BlobDigest, bytes] = {}
        for ref in refs:
            data = blobs.get(ref.sha256)
            if data is None:
                raise CustodyError(f"artifact {ref.sha256} has no bytes")
            if hashlib.sha256(data).hexdigest() != ref.sha256 or len(data) != ref.size_bytes:
                raise CustodyError(f"artifact {ref.sha256} bytes disagree with their record")
            verified[ref.sha256] = data
        return verified

    def _put_blobs(self, blobs: Mapping[str, bytes]) -> None:
        for sha256, data in blobs.items():
            if not os.path.lexists(self.root / "blobs" / sha256):
                _publish_once(self.root / "blobs" / sha256, data)

    def _put_attempt(self, digest: str, data: bytes) -> None:
        """Stage the directory, then rename it into place. The same digest already present holds
        the same bytes, so a second write is a no-op."""
        attempts = self.root / "attempts"
        target = attempts / digest
        if os.path.lexists(target):
            return
        stage = attempts / f".stage-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, mode=0o700)
        try:
            _atomic_write_bytes(stage / "attempt.json", data)
            try:
                os.rename(stage, target)
            except OSError:
                if not target.is_dir():
                    raise
            _fsync_directory(attempts)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _claim_at(self, slot: str) -> IdempotencyClaim | None:
        path = self.root / "idem" / slot
        if not os.path.lexists(path):
            return None
        return IdempotencyClaim.model_validate_json(_read_bytes(path))

    def _claim(self, slot: str, claim: IdempotencyClaim) -> IdempotencyClaim:
        """Publish the claim unless one exists; return whichever is on disk."""
        if _publish_once(self.root / "idem" / slot, canonical_bytes(claim)):
            return claim
        existing = self._claim_at(slot)
        assert existing is not None
        return existing

    def _mark(self, parent: str, child: str, principal_id: str) -> None:
        _touch(self.root / "children" / parent / child)
        _touch(self.root / "authored" / principal_id / child)

    def _write(self, attempt: ChildAttempt, principal_id: str, slot: Digest, request_sha256: Digest) -> AttemptDigest:
        """Steps 4 to 6 of the write protocol. Blobs are already written."""
        data = canonical_bytes(attempt)
        digest = AttemptDigest(hashlib.sha256(data).hexdigest())
        self._put_attempt(digest, data)
        winner = check_claim(self._claim(slot, IdempotencyClaim(child=digest, request_sha256=request_sha256)), request_sha256)
        self._mark(attempt.parent, winner, principal_id)
        return AttemptDigest(winner)

    def _annotate(self, digest: str, annotation: Annotation) -> None:
        data = canonical_bytes(annotation)
        name = f"{self._clock():%Y%m%dT%H%M%S%fZ}-{hashlib.sha256(data).hexdigest()}.json"
        _publish_once(self.root / "attempts" / digest / "annotations" / name, data)

    def _reveal(self, principal: Principal, root: AttemptDigest) -> None:
        if principal.principal_type is not PrincipalType.USER:
            return
        path = self.root / "roots" / root / "reveals" / f"{principal.principal_id}.json"
        if not os.path.lexists(path):
            event = RevealEvent(principal_id=principal.principal_id, root=root, first_revealed_at_utc=self._clock())
            _publish_once(path, canonical_bytes(event))

    # ----- internals: projections ----------------------------------------------------------

    def _revealed(self, digest: AttemptDigest, attempt: Attempt, state: AnalysisState) -> RevealedAttemptView:
        return RevealedAttemptView(
            digest=digest,
            attempt=attempt,
            state=state,
            children=self._children(digest),
            annotations=self.annotations(digest),
        )

    def _view(self, principal: Principal, path: Path_) -> AttemptView:
        digest, attempt = path[-1]
        if projection_for(principal) == "actor":
            if not isinstance(attempt, ChildAttempt) or not self._authored(principal.principal_id, digest):
                raise AttemptAccessError(_DENIED)
            return actor_attempt_view(digest, attempt, principal)
        self._reveal(principal, path[0][0])
        return self._revealed(digest, attempt, state_at(path[0], _steps(path)))
