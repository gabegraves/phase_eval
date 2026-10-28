"""The one episode format, written from the tree plus versioned policy inputs, and the eval
dataset built on it (R11, R18, R19, R33; overnight N4, N8, KTD10).

Target path: src/phaseeval/workbench/exports.py (public). Entry points:

    python -m phaseeval.workbench.exports --root <digest> --dataset-version <v> \
        --eligibility <ledger.json> --rights <registry.json> --out <dir>

writes episodes.jsonl, excluded.jsonl (typed reasons) and manifest.json. Re-export of the same
snapshot and inputs is byte-identical. `scripts/workbench/build_eval_dataset.py` adds the eval
dataset around that export: frozen analysis snapshots, labels split by rater kind, peak calls,
sign-offs, step feedback, judge inputs and a manifest with denominators. Every JSON Schema under
`schemas/` is generated from these types (`dataset_json_schemas`), so a schema and its type
cannot disagree.

An episode ends at every leaf of the tree and at every imported run that has children, so an
agent's own answer stays a row after an expert corrects it. SFT and RL-task export views are
deferred. Same-case replay proves capture and reproducibility, not learning.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from fastapi import APIRouter
from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator

from phaseeval.api.auth import Principal

from . import policy
from . import snapshot as snapshot_module
from .labels import (
    GUIDANCE_VERSION,
    RUBRIC_VERSION,
    ExportUse,
    Label,
    PeakCall,
    Rater,
    SignOff,
    StepFeedback,
    TrajectoryReview,
)
from .model import (
    AnalysisState,
    AttemptDigest,
    AttemptRef,
    ChildAttempt,
    Completed,
    Digest,
    HumanActor,
    ImportedRunResult,
    ImportedSubmitted,
    ImportRun,
    Incomplete,
    Record,
    RootAttempt,
    SafeID,
    canonical_bytes,
    digest_of,
    initial_state,
    transition,
)
from .policy import (
    ArtifactRights,
    EligibilityDecision,
    EligibilityLedger,
    EpisodeEffort,
    EpisodeStatus,
    ExclusionReason,
    HumanContribution,
    RightsRegistry,
    RunReceipt,
    TreeProvenance,
)
from .qualification import CaseExpectation, NotGradable, QualificationReport
from .review import Annotation, Note, Preference
from .snapshot import AnalysisSnapshot, SnapshotKind
from .store import WorkbenchStore

EPISODE_SCHEMA_VERSION = "axa.workbench.episode.v1"
EXPORTER_VERSION = "axa-workbench-exporter-v1"
DATASET_SCHEMA_VERSION = "axa.workbench.dataset.v1"


class EpisodeStep(Record):
    """One state transition: the attempt (action, actor, tool, inputs, outcome, origin and replay
    manifest) and the derived state after it."""

    digest: AttemptRef
    attempt: ChildAttempt
    state_after: AnalysisState
    annotations: tuple[Annotation, ...]  # notes, preferences, replay checks, run closes on it


class PreferencePair(Record):
    """Only from an explicit review.Preference (R18)."""

    chosen: AttemptRef
    rejected: AttemptRef
    fork: AttemptRef  # lowest common ancestor of the two paths
    reason: str
    author: HumanActor


class EpisodeRecord(Record):
    schema_version: Literal["axa.workbench.episode.v1"] = EPISODE_SCHEMA_VERSION
    record_id: Digest  # sha256 of the canonical bytes of the other fields
    root_digest: AttemptRef
    root: RootAttempt
    leaf: AttemptRef
    steps: tuple[EpisodeStep, ...]
    preference_pairs: tuple[PreferencePair, ...]
    status: EpisodeStatus
    runs: tuple[RunReceipt, ...]
    provenance: TreeProvenance  # never a sealed-truth class (R6)
    human_contributions: tuple[HumanContribution, ...]
    reference_exposed: bool
    task_kind: Literal["identification", "confirmation"]
    eligibility: tuple[EligibilityDecision, ...]
    rights: tuple[ArtifactRights, ...]
    effort: EpisodeEffort


class ExcludedEpisode(Record):
    root_digest: AttemptRef
    leaf: AttemptRef
    reasons: tuple[ExclusionReason, ...] = Field(min_length=1)


class ExportInputs(Record):
    dataset_version: SafeID
    eligibility: EligibilityLedger
    rights: RightsRegistry
    now: datetime


class ExportManifest(Record):
    exporter_version: SafeID
    schema_version: Literal["axa.workbench.episode.v1"] = EPISODE_SCHEMA_VERSION
    dataset_version: SafeID
    policy_version: SafeID
    rights_version: SafeID
    roots: tuple[AttemptRef, ...]
    leaves: tuple[AttemptRef, ...]  # the tree snapshot this export covers
    inputs_sha256: Digest
    episodes: int
    preference_pairs: int
    excluded: int
    excluded_by_reason: dict[str, int]


class ExportBundle(Record):
    manifest: ExportManifest
    episodes: tuple[EpisodeRecord, ...]
    excluded: tuple[ExcludedEpisode, ...]


def episode_json_schema() -> dict[str, Any]:
    """The published JSON Schema (R19), derived from the type."""
    schema = TypeAdapter(EpisodeRecord).json_schema()
    schema["$id"] = "schemas/workbench-episode.v1.json"
    return schema


Path_ = tuple[tuple[AttemptDigest, Any], ...]


def _paths(walked: tuple[tuple[AttemptDigest, Any], ...]) -> tuple[Path_, ...]:
    """Root-first paths to every episode end: each leaf, and each imported run with children."""
    attempts = dict(walked)
    parents = {digest: a.parent for digest, a in walked if isinstance(a, ChildAttempt)}
    with_children = set(parents.values())
    ends = [
        digest
        for digest, attempt in walked
        if digest not in with_children or (isinstance(attempt, ChildAttempt) and isinstance(attempt.action, ImportRun))
    ]
    paths = []
    for end in ends:
        chain = [end]
        while chain[-1] in parents:
            chain.append(parents[chain[-1]])
        paths.append(tuple((digest, attempts[digest]) for digest in reversed(chain)))
    return tuple(paths)


def _ancestors(digest: str, parents: Mapping[str, str]) -> list[str]:
    chain = [digest]
    while chain[-1] in parents:
        chain.append(parents[chain[-1]])
    return chain


def _record_id(fields: Mapping[str, Any]) -> str:
    draft = EpisodeRecord(record_id="0" * 64, **fields)
    body = draft.model_dump(mode="json", exclude={"record_id"})
    data = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def export_episodes(store: WorkbenchStore, roots: Iterable[AttemptDigest], inputs: ExportInputs) -> ExportBundle:
    """Pure over one walked snapshot plus inputs. Rows sorted by (root, leaf)."""
    episodes: list[EpisodeRecord] = []
    excluded: list[ExcludedEpisode] = []
    roots = tuple(sorted(roots))
    leaves: list[str] = []
    for root_digest in roots:
        walked = store.walk(root_digest)
        parents = {digest: a.parent for digest, a in walked if isinstance(a, ChildAttempt)}
        annotations = {digest: store.annotations(digest) for digest, _ in walked}
        root = walked[0][1]
        assert isinstance(root, RootAttempt)
        for path in _paths(walked):
            leaf = path[-1][0]
            leaves.append(leaf)
            ep = policy.episode(path, root.case, annotations, inputs.now)
            reasons = policy.exclusion_reasons(ep)
            if reasons:
                excluded.append(ExcludedEpisode(root_digest=root_digest, leaf=leaf, reasons=reasons))
                continue
            on_path = {digest for digest, _ in path}
            steps, state = [], initial_state(root_digest, root)
            for digest, attempt in path[1:]:
                state = transition(state, (digest, attempt))
                steps.append(
                    EpisodeStep(digest=digest, attempt=attempt, state_after=state, annotations=annotations[digest])
                )
            pairs = []
            for digest, _ in path:
                for note in annotations[digest]:
                    if not isinstance(note, Preference) or note.chosen not in on_path:
                        continue
                    if note.rejected not in parents and note.rejected != root_digest:
                        raise ValueError(f"preference on {note.chosen} names {note.rejected} outside its root")
                    chosen_line = set(_ancestors(note.chosen, parents))
                    fork = next(a for a in _ancestors(note.rejected, parents) if a in chosen_line)
                    pairs.append(
                        PreferencePair(
                            chosen=note.chosen, rejected=note.rejected, fork=fork, reason=note.reason, author=note.author
                        )
                    )
            fields = {
                "root_digest": root_digest,
                "root": root,
                "leaf": leaf,
                "steps": tuple(steps),
                "preference_pairs": tuple(pairs),
                "status": ep.status,
                "runs": ep.runs,
                "provenance": ep.provenance,
                "human_contributions": ep.human_contributions,
                "reference_exposed": ep.reference_exposed,
                "task_kind": ep.task_kind,
                "eligibility": policy.eligibility_for(ep, inputs.eligibility, inputs.dataset_version),
                "rights": policy.rights_for(ep, path, inputs.rights),
                "effort": ep.effort,
            }
            episodes.append(EpisodeRecord(record_id=_record_id(fields), **fields))
    episodes.sort(key=lambda e: (e.root_digest, e.leaf))
    excluded.sort(key=lambda e: (e.root_digest, e.leaf))
    manifest = ExportManifest(
        exporter_version=EXPORTER_VERSION,
        dataset_version=inputs.dataset_version,
        policy_version=inputs.eligibility.policy_version,
        rights_version=inputs.rights.version,
        roots=roots,
        leaves=tuple(sorted(leaves)),
        inputs_sha256=hashlib.sha256(canonical_bytes(inputs)).hexdigest(),
        episodes=len(episodes),
        preference_pairs=sum(len(e.preference_pairs) for e in episodes),
        excluded=len(excluded),
        excluded_by_reason=dict(sorted(Counter(r for e in excluded for r in e.reasons).items())),
    )
    return ExportBundle(manifest=manifest, episodes=tuple(episodes), excluded=tuple(excluded))


def _write_lines(path: Path, records: Iterable[Record]) -> None:
    path.write_bytes(b"".join(canonical_bytes(record) for record in records))


def write_bundle(bundle: ExportBundle, out_dir: Path) -> None:
    """episodes.jsonl, excluded.jsonl, manifest.json; one canonical JSON line per record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(out_dir / "episodes.jsonl", bundle.episodes)
    _write_lines(out_dir / "excluded.jsonl", bundle.excluded)
    (out_dir / "manifest.json").write_bytes(canonical_bytes(bundle.manifest))


def playback_states(record: EpisodeRecord) -> tuple[AnalysisState, ...]:
    """Rebuild every state from the record alone, without the store or the private checkout.
    This is what the public package plays back (R22, AE8)."""
    states, state = [], initial_state(record.root_digest, record.root)
    for step in record.steps:
        state = transition(state, (step.digest, step.attempt))
        states.append(state)
    return tuple(states)


# ---------------------------------------------------------------------------------------------
# The eval dataset (N4): snapshots, labels by rater kind, judge inputs, manifest with denominators
# ---------------------------------------------------------------------------------------------

RATER_PSEUDONYM = r"^r-[0-9a-f]{12}$"
# Field names that exist only in custody records (data/custody/workbench/<case>/custody.json).
CUSTODY_KEYS = frozenset(
    {
        "reference_answer",
        "reference_phase_coverage",
        "withheld_context",
        "split_check",
        "canary",
        "candidate_license_audit",
    }
)
# Run identity a blind rater or judge must not see (PI directive; labels.LabelStore reveal rules).
RUN_IDENTITY_KEYS = frozenset(
    {"reference_match", "job", "model", "harness", "series", "episode_id"}  # PI directive
    | {"job_name", "model_id", "harness_version", "trial_id", "reference", "siblings"}  # W3 view's set
)
# Text that names a run: a job directory spells the model and series (tests/workbench/test_snapshot.py).
RUN_IDENTITY_TEXT = re.compile(
    r"u6-|harbor-local|harbor-live|harbor-at|data/custody|series-\d|/jobs/|task-dev\d+__|qwen|astra|gpt-6",
    re.IGNORECASE,
)


class JudgeInput(Record):
    """One line of judge_inputs.jsonl, one per agent-answer snapshot (an expert correction never
    reaches a judge, CR-08): what a tool-less model judge reads for one snapshot, and nothing else. The frozen snapshot (phases with stable ids, checks, residual peaks, claims,
    chemistry) plus the expert's own blind view of it from snapshot.blind_payload: the brief the
    agent read, the answer with its cited fit, the fit curves with per-phase contributions, and
    the evidence the view names as missing, the fit's sampling facts (native points, stride,
    precision, curve sha256) and the native-resolution windows the view shares with the page,
    each with background and per-phase contributions: around every listed residual peak, and
    around each reported phase's strongest calculated peaks with the rule that chose them
    (snapshot.PHASE_WINDOW_RULE). Nothing reaches a judge that
    the expert's view does not carry (Science parity gate, AXA-43zwk.21 07:17 UTC).
    snapshot_digest and reported_phase_ids sit at the top level, as the frozen judge prompt and
    judge_labels.check_input require. Trajectory steps
    and the candidate library are left out: neither is rubric evidence, and the steps time the
    run. Each line is written as judge_labels.render_input writes it, so its sha256 is the
    judge's input_sha256."""

    schema_version: Literal["judge-input.v1"] = "judge-input.v1"
    snapshot_digest: Digest
    reported_phase_ids: tuple[SafeID, ...]
    # The frozen snapshot JSON verbatim: its canonical bytes hash to snapshot_digest. Re-serializing
    # through a later AnalysisSnapshot type would add defaulted fields and alter frozen evidence.
    snapshot: dict[str, JsonValue]
    brief: dict[str, JsonValue]
    answer: dict[str, JsonValue] | None
    fit: dict[str, JsonValue] | None
    residual_windows: tuple[dict[str, JsonValue], ...]
    phase_windows: tuple[dict[str, JsonValue], ...]
    phase_window_rule: str
    missing: tuple[str, ...]


JUDGE_VIEW_KEYS = ("brief", "answer", "fit", "residual_windows", "phase_windows", "phase_window_rule", "missing")
JUDGE_INPUT_KEYS = frozenset(JudgeInput.model_fields)


def judge_input_bytes(record: JudgeInput) -> bytes:
    """The exact bytes a judge reads: scripts/workbench/judge_labels.render_input."""
    return json.dumps(
        record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def frozen_json(data: bytes) -> dict[str, JsonValue]:
    """A frozen snapshot's stored canonical bytes as JSON; json.dumps(sort_keys, compact) + newline
    of the result reproduces the bytes exactly."""
    return json.loads(data)


def judge_input(digest: str, snap: AnalysisSnapshot, frozen: dict[str, JsonValue], view: Mapping[str, Any]) -> JudgeInput:
    return JudgeInput(
        snapshot_digest=digest,
        reported_phase_ids=tuple(p.phase_id for p in snap.phases),
        snapshot=frozen,
        **{key: view[key] for key in JUDGE_VIEW_KEYS},
    )


class DatasetCount(Record):
    name: SafeID
    count: int
    denominator: int
    denominator_name: str


class DatasetExclusion(Record):
    kind: Literal["episode", "run", "snapshot"]
    ref: str  # an attempt digest, or a run path from the load manifest
    reasons: tuple[str, ...] = Field(min_length=1)


class AgentEpisodes(Record):
    """Exported episodes per agent model and harness, for the dashboard and the card."""

    case_id: SafeID
    model: str
    harness: str
    episodes: int
    submitted: int
    unsuccessful: int
    incomplete: int


class DatasetFile(Record):
    path: str
    lines: int
    sha256: Digest


class JudgeManifestEntry(Record):
    snapshot_digest: Digest
    judged: bool  # a submitted answer present when the manifest froze (KTD11)
    actor_model: str | None  # the model whose run produced the snapshot; never sent to a judge
    judge_input_sha256: Digest  # sha256 of the judge_inputs.jsonl line (judge_input_bytes)


class JudgeManifest(Record):
    """KTD11: frozen once per dataset version, when the judges launch, with the exact judge input
    of every snapshot it lists, and never rewritten. A later build fails when a listed snapshot's
    judge input has changed, and lists a snapshot frozen later with judged false. Before the
    freeze a build writes a provisional manifest (frozen false) that nothing enforces.
    scripts/workbench/judge_labels.py renders the judged entries and reads actor_model for
    judge_is_actor_model."""

    schema_version: Literal["judge-manifest.v1"] = "judge-manifest.v1"
    dataset_version: SafeID
    rubric_version: SafeID
    frozen: bool
    frozen_at_utc: datetime  # when this manifest was computed; the freeze time when frozen
    entries: tuple[JudgeManifestEntry, ...]

    @model_validator(mode="after")
    def one_entry_per_snapshot(self) -> Self:
        digests = [e.snapshot_digest for e in self.entries]
        if len(set(digests)) != len(digests):
            raise ValueError("a judge manifest lists each snapshot once")
        return self


class SnapshotEntry(Record):
    digest: Digest
    origin: AttemptRef
    kind: SnapshotKind
    answered: bool
    judged: bool  # listed in the frozen judge manifest


Scope = Literal["private", "public"]


class QualificationSetEntry(Record):
    """One frozen qualification set whose expectations the private export carries."""

    set_digest: Digest
    roots: int  # its rows of expectations.jsonl, consecutive, in set order
    expectations_sha256: Digest  # over those rows' canonical bytes: QualificationReport.expectations_sha256


Expectation = CaseExpectation | NotGradable


class DatasetManifest(Record):
    schema_version: Literal["axa.workbench.dataset.v1"] = DATASET_SCHEMA_VERSION
    dataset_version: SafeID
    exporter_version: SafeID
    scope: Scope  # public: the origin-Q6 release only (policy.PUBLIC_JOBS_V0)
    export_use: ExportUse
    generated_at_utc: datetime
    rubric_version: SafeID
    guidance_version: SafeID
    episode_export: ExportManifest
    store_manifest_sha256: Digest | None  # the load manifest the store was built from
    judge_manifest_sha256: Digest
    snapshot_index: tuple[SnapshotEntry, ...]
    qualification_sets: tuple[QualificationSetEntry, ...]  # private scope only
    counts: tuple[DatasetCount, ...]
    agent_episodes: tuple[AgentEpisodes, ...]
    exclusions: tuple[DatasetExclusion, ...]
    files: tuple[DatasetFile, ...]


SCHEMA_FILES: Mapping[str, Any] = {
    "workbench-episode.v1.json": EpisodeRecord,
    "excluded-episode.v1.json": ExcludedEpisode,
    "snapshot.v1.json": AnalysisSnapshot,
    "label.v1.json": Label,
    "peak-call.v1.json": PeakCall,
    "signoff.v1.json": SignOff,
    "step-feedback.v1.json": StepFeedback,
    "trajectory-review.v1.json": TrajectoryReview,
    "judge-input.v1.json": JudgeInput,
    "judge-manifest.v1.json": JudgeManifest,
    "case-expectation.v2.json": Expectation,
    "qualification-report.v2.json": QualificationReport,
    "dataset-manifest.v1.json": DatasetManifest,
}
SCHEMA_FOR_FILE: Mapping[str, str] = {
    "episodes.jsonl": "workbench-episode.v1.json",
    "excluded.jsonl": "excluded-episode.v1.json",
    "snapshots.jsonl": "snapshot.v1.json",
    "labels.expert.jsonl": "label.v1.json",
    "labels.model_judge.jsonl": "label.v1.json",
    "peak_calls.jsonl": "peak-call.v1.json",
    "signoffs.jsonl": "signoff.v1.json",
    "step_feedback.jsonl": "step-feedback.v1.json",
    "judge_inputs.jsonl": "judge-input.v1.json",
}
OPTIONAL_SCHEMA_FOR_FILE: Mapping[str, str] = {
    "trajectory_reviews.jsonl": "trajectory-review.v1.json",
}
# The grader side of the private export (final contract row U6); a public bundle carries neither.
PRIVATE_SCHEMA_FOR_FILE: Mapping[str, str] = {
    "expectations.jsonl": "case-expectation.v2.json",
    "reports.jsonl": "qualification-report.v2.json",
}


def dataset_json_schemas() -> dict[str, dict[str, Any]]:
    """Every published schema, keyed by its file name under schemas/, derived from its type."""
    schemas = {}
    for name, model in SCHEMA_FILES.items():
        schema = episode_json_schema() if model is EpisodeRecord else TypeAdapter(model).json_schema()
        schema["$id"] = f"schemas/{name}"
        schemas[name] = schema
    return schemas


class Dataset(Record):
    manifest: DatasetManifest
    judge_manifest: JudgeManifest
    episodes: tuple[EpisodeRecord, ...]
    excluded: tuple[ExcludedEpisode, ...]
    snapshots: tuple[AnalysisSnapshot, ...]  # parsed under the current type, for reading values
    snapshot_json: tuple[dict[str, JsonValue], ...]  # the same snapshots as frozen, written verbatim
    labels_expert: tuple[Label, ...]
    labels_model_judge: tuple[Label, ...]
    peak_calls: tuple[PeakCall, ...]
    signoffs: tuple[SignOff, ...]
    step_feedback: tuple[StepFeedback, ...]
    judge_inputs: tuple[JudgeInput, ...]
    expectations: tuple[Expectation, ...]  # private scope only
    reports: tuple[QualificationReport, ...]  # private scope only
    trajectory_reviews: tuple[TrajectoryReview, ...] = ()


class RaterPseudonyms:
    """R33: rater ids are pseudonymous in every export file. The salt and the identity map live
    only under the private store root (0700 directory, 0600 files), never in a dataset dir."""

    def __init__(self, private_dir: Path) -> None:
        self.dir = private_dir
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.dir, 0o700)
        salt = self.dir / "rater-pseudonym-salt"
        if not salt.exists():
            fd = os.open(salt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(secrets.token_hex(32))
        self._salt = salt.read_text().strip().encode()
        self.identities: dict[str, str] = {}

    def __call__(self, principal_id: str) -> str:
        name = "r-" + hmac.new(self._salt, principal_id.encode(), hashlib.sha256).hexdigest()[:12]
        self.identities[name] = principal_id
        return name

    def write_map(self) -> Path:
        path = self.dir / "rater-identity-map.json"
        data = json.dumps(dict(sorted(self.identities.items())), indent=2) + "\n"
        tmp = self.dir / f".rater-identity-map.{secrets.token_hex(4)}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
        os.replace(tmp, path)
        return path


def _pseudonymous[T: (Label, PeakCall, SignOff, StepFeedback, TrajectoryReview)](
    record: T, pseudonym: Callable[[str], str]
) -> T:
    rater: Rater = record.rater
    return record.model_copy(update={"rater": rater.model_copy(update={"principal_id": pseudonym(rater.principal_id)})})


def _swap_ids(value: Any, ids: frozenset[str], pseudonym: Callable[[str], str]) -> Any:
    if isinstance(value, dict):
        return {
            key: pseudonym(item) if key == "principal_id" and item in ids else _swap_ids(item, ids, pseudonym)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_swap_ids(item, ids, pseudonym) for item in value]
    return value


def pseudonymous_episode(record: EpisodeRecord, pseudonym: Callable[[str], str]) -> EpisodeRecord:
    """Every human principal id in the episode (actors, correction sources, annotation authors)
    becomes the same pseudonym its labels carry (R33). Step digests still name the private
    store's attempts, which hold the real ids; record_id is recomputed over the export's bytes."""
    humans = {s.attempt.actor.principal_id for s in record.steps if isinstance(s.attempt.actor, HumanActor)}
    humans |= {
        note.author.principal_id
        for step in record.steps
        for note in step.annotations
        if isinstance(note, Note | Preference)
    }
    humans |= {c.principal_id for c in record.human_contributions}
    if not humans:
        return record
    body = _swap_ids(record.model_dump(mode="json", exclude={"record_id"}), frozenset(humans), pseudonym)
    fields = dict(EpisodeRecord.model_validate(body | {"record_id": "0" * 64}))
    fields.pop("record_id")
    return EpisodeRecord(record_id=_record_id(fields), **fields)


SnapshotBuilder = Callable[[WorkbenchStore, Principal, AttemptRef, SnapshotKind], AnalysisSnapshot]
ReviewPayload = Callable[[WorkbenchStore, Principal, AnalysisSnapshot, Digest], dict[str, Any]]


def blind_view(store: WorkbenchStore, principal: Principal, snap: AnalysisSnapshot, digest: Digest) -> dict[str, Any]:
    """snapshot.blind_payload for the trusted exporter: the same view derived from the frozen
    snapshot the exporter already read. blind_payload reads it through LabelStore.snapshot,
    which refuses the admin owner by design (CR-02); evidence() checks the root ACL."""
    ev = snapshot_module.evidence(store, principal, snap.origin, snap.kind)
    return snapshot_module._blind(store, ev, snap, digest)


def stored_snapshot_bytes(store_root: Path) -> dict[str, bytes]:
    """Every snapshot frozen in the store (labels.py layout snapshots/<digest>.json), its bytes
    re-hashed to its name: an expert-frozen correction is in the dataset even before a label."""
    found: dict[str, bytes] = {}
    directory = store_root / "snapshots"
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.stem:
            raise ValueError(f"snapshot bytes disagree with their name {path.name}")
        found[path.stem] = data
    return found


def stored_snapshots(store_root: Path) -> dict[str, AnalysisSnapshot]:
    """The frozen snapshots parsed under the current type, for reading values only."""
    return {d: AnalysisSnapshot.model_validate_json(b) for d, b in stored_snapshot_bytes(store_root).items()}


def stored_reports(store_root: Path) -> tuple[QualificationReport, ...]:
    """Every published qualification report (qualification-reports/<sha256>.json), re-hashed."""
    directory = store_root / "qualification-reports"
    if not directory.is_dir():
        return ()
    found = []
    for path in sorted(directory.glob("*.json")):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.stem:
            raise ValueError(f"report bytes disagree with their name {path.name}")
        found.append(QualificationReport.model_validate_json(data))
    return tuple(found)


def _snapshot_origin(episode: EpisodeRecord) -> tuple[AttemptDigest, SnapshotKind] | None:
    """KTD10 (CR-04): an agent episode freezes its imported answer only when the run submitted
    one. A run that ended without an answer (budget exhausted, no submission) keeps its episode
    evidence and gets no snapshot. An expert correction is frozen only through the label routes."""
    leaf = episode.steps[-1] if episode.steps else None
    if leaf is None or not isinstance(leaf.attempt.action, ImportRun):
        return None
    outcome = leaf.attempt.outcome
    if isinstance(outcome, Completed) and isinstance(outcome.result, ImportedRunResult):
        if isinstance(outcome.result.end, ImportedSubmitted):
            return leaf.digest, "agent_answer"
    return None


def publishable_v0(record: EpisodeRecord, registry: RightsRegistry) -> bool:
    """Origin Q6: every imported run of the episode is a released job, and every human who
    contributed to it has contributor terms that permit redistribution."""
    jobs = [s.attempt.action.harbor.job_name for s in record.steps if isinstance(s.attempt.action, ImportRun)]
    contributors = [registry.contributors.get(c.principal_id) for c in record.human_contributions]
    return (
        bool(jobs)
        and all(job in policy.PUBLIC_JOBS_V0 for job in jobs)
        and all(terms is not None and terms.redistribution.right == "permitted" for terms in contributors)
    )


def build_dataset(
    store: WorkbenchStore,
    labels: Any,  # labels.LabelStore over the same store root
    principal: Principal,
    roots: Iterable[AttemptDigest],
    inputs: ExportInputs,
    use: ExportUse,
    pseudonyms: RaterPseudonyms,
    *,
    build_snapshot: SnapshotBuilder,
    review_payload: ReviewPayload,
    scope: Scope = "private",
    judge_manifest: JudgeManifest | None = None,
    freeze_judge_manifest: bool = False,
    store_manifest: Mapping[str, Any] | None = None,
    store_manifest_sha256: str | None = None,
    qualifier: Any = None,  # qualification.Qualifier: expectations of every frozen set (private only)
) -> Dataset:
    """Freeze one snapshot per submitted agent answer, store it (idempotent), and
    gather every label record `use` permits. A snapshot that cannot be built is an exclusion
    with its reason, never a stand-in (R11). The public scope keeps only publishable episodes
    and the records on their snapshots and steps, under the public_release permission (R33).
    Given a judge manifest, the build enforces it; without one it computes a provisional manifest
    from its submitted-answer snapshots, marked frozen only when freeze_judge_manifest is set."""
    if scope == "public" and use != "public_release":
        raise ValueError("the public scope exports only what public_release permits")
    bundle = export_episodes(store, roots, inputs)
    exclusions = [DatasetExclusion(kind="episode", ref=e.leaf, reasons=e.reasons) for e in bundle.excluded]
    in_scope = bundle.episodes
    if scope == "public":
        in_scope = tuple(e for e in bundle.episodes if publishable_v0(e, inputs.rights))
    else:
        for row in (store_manifest or {}).get("rows", ()):  # load_store.py: one row per trial
            if row.get("error"):
                exclusions.append(DatasetExclusion(kind="run", ref=str(row["trial_dir"]), reasons=(str(row["error"]),)))
    scoped_attempts = {step.digest for e in in_scope for step in e.steps} | {e.root_digest for e in in_scope}

    snapshots: dict[str, AnalysisSnapshot] = {}
    stored_bytes = stored_snapshot_bytes(store.root)
    frozen_bytes: dict[str, bytes] = {}  # digest -> the exact bytes every export file carries
    stored = {d: AnalysisSnapshot.model_validate_json(b) for d, b in stored_bytes.items()}
    frozen_by_origin = {(s.origin, s.kind): (d, s) for d, s in stored.items()}
    candidates = 0
    for episode in in_scope:
        origin = _snapshot_origin(episode)
        if origin is None:
            continue
        candidates += 1
        if origin in frozen_by_origin:  # frozen evidence is never regenerated (KTD10)
            digest, snap = frozen_by_origin[origin]
        else:
            try:
                snap = build_snapshot(store, principal, origin[0], origin[1])
            except (ValueError, LookupError, PermissionError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                exclusions.append(DatasetExclusion(kind="snapshot", ref=origin[0], reasons=(reason,)))
                continue
            digest = labels.put_snapshot(snap)
            if digest != digest_of(snap):
                raise ValueError(f"the label store keyed snapshot {digest} differently from its canonical digest")
            stored_bytes[digest] = canonical_bytes(snap)
        snapshots[digest] = snap
        frozen_bytes[digest] = stored_bytes[digest]

    frozen = {d: s for d, s in stored.items() if d not in snapshots and s.origin in scoped_attempts}
    snapshots |= frozen
    frozen_bytes |= {d: stored_bytes[d] for d in frozen}

    judge_inputs = []
    for digest, snap in sorted(snapshots.items()):
        if snap.kind != "agent_answer":
            continue  # CR-08: only agent answers reach a judge; an expert correction may carry the reference
        try:
            view = review_payload(store, principal, snap, digest)
        except (ValueError, LookupError, PermissionError) as exc:
            exclusions.append(DatasetExclusion(kind="snapshot", ref=snap.origin, reasons=(f"{type(exc).__name__}: {exc}",)))
            del snapshots[digest]
            continue
        judge_inputs.append(judge_input(digest, snap, frozen_json(frozen_bytes[digest]), view))

    attempts = {step.digest: step.attempt for e in in_scope for step in e.steps}

    def entry(digest: str, judged: bool) -> JudgeManifestEntry:
        attempt = attempts.get(snapshots[digest].origin)
        result = attempt.outcome.result if attempt is not None and isinstance(attempt.outcome, Completed) else None
        return JudgeManifestEntry(
            snapshot_digest=digest,
            judged=judged,
            actor_model=result.agent.model_id if isinstance(result, ImportedRunResult) else None,
            judge_input_sha256=hashlib.sha256(judge_input_bytes(by_digest[digest])).hexdigest(),
        )

    by_digest = {j.snapshot_digest: j for j in judge_inputs}
    if judge_manifest is None:
        judge_manifest = JudgeManifest(
            dataset_version=inputs.dataset_version,
            rubric_version=RUBRIC_VERSION,
            frozen=freeze_judge_manifest,
            frozen_at_utc=inputs.now,
            entries=tuple(entry(d, judged=snapshots[d].answered) for d in sorted(by_digest)),
        )
    else:
        for frozen_entry in judge_manifest.entries:
            if frozen_entry.snapshot_digest in by_digest and (
                entry(frozen_entry.snapshot_digest, frozen_entry.judged).judge_input_sha256
                != frozen_entry.judge_input_sha256
            ):
                raise ValueError(
                    f"the judge input of snapshot {frozen_entry.snapshot_digest} changed after the judge manifest froze"
                )
        listed = {e.snapshot_digest for e in judge_manifest.entries}
        later = tuple(entry(d, judged=False) for d in sorted(by_digest) if d not in listed)
        judge_manifest = judge_manifest.model_copy(update={"entries": judge_manifest.entries + later})
    frozen_for_judging = {e.snapshot_digest for e in judge_manifest.entries if e.judged}

    label_rows, peak_rows, signoff_rows, step_rows = labels.export_records(use)
    on_snapshots = [r for r in label_rows if r.snapshot in snapshots]
    labels_out = sorted(
        (_pseudonymous(r, pseudonyms) for r in on_snapshots),
        key=lambda r: (r.snapshot, r.item, r.phase_id or "", r.rater.principal_id, r.created_at_utc),
    )
    expert = tuple(r for r in labels_out if r.rater.kind in ("expert", "owner"))
    judged = tuple(r for r in labels_out if r.rater.kind == "model_judge")
    peaks = tuple(
        sorted(
            (_pseudonymous(r, pseudonyms) for r in peak_rows if r.snapshot in snapshots),
            key=lambda r: (r.snapshot, r.peak_id, r.created_at_utc),
        )
    )
    signoffs = tuple(
        sorted(
            (_pseudonymous(r, pseudonyms) for r in signoff_rows if r.snapshot in snapshots),
            key=lambda r: (r.snapshot, r.created_at_utc),
        )
    )
    steps = tuple(
        sorted(
            (_pseudonymous(r, pseudonyms) for r in step_rows if scope == "private" or r.step.attempt in scoped_attempts),
            key=lambda r: (r.step.attempt, r.step.call_index or -1, r.created_at_utc),
        )
    )
    trajectory_reviews = tuple(
        sorted(
            (_pseudonymous(r, pseudonyms) for r in labels.export_trajectory_reviews(use) if r.snapshot in snapshots),
            key=lambda r: (r.snapshot, r.created_at_utc, r.submission_id),
        )
    )

    expectations: list[Expectation] = []
    set_entries = []
    reports: tuple[QualificationReport, ...] = ()
    if scope == "private":
        reports = stored_reports(store.root)
        for path in sorted((store.root / "qualification-sets").glob("*.json")) if qualifier is not None else ():
            qset = qualifier.qualification_set(principal, path.stem)
            by_root = qualifier.expectations(principal, path.stem, "internal_evaluation")
            rows = [by_root[root] for root in qset.roots]
            expectations += rows
            set_entries.append(
                QualificationSetEntry(
                    set_digest=path.stem,
                    roots=len(rows),
                    expectations_sha256=hashlib.sha256(b"".join(canonical_bytes(r) for r in rows)).hexdigest(),
                )
            )

    n_walked = bundle.manifest.episodes + bundle.manifest.excluded
    n_snap = len(snapshots)
    incomplete = sum(1 for e in in_scope if any(not run.complete for run in e.runs))
    counts = [
        DatasetCount(name="episodes_exported", count=len(in_scope), denominator=n_walked, denominator_name="episodes walked"),
        DatasetCount(name="episodes_excluded", count=bundle.manifest.excluded, denominator=n_walked, denominator_name="episodes walked"),
        DatasetCount(name="episodes_incomplete", count=incomplete, denominator=len(in_scope), denominator_name="episodes exported"),
        DatasetCount(
            name="agent_snapshots",
            count=sum(1 for s in snapshots.values() if s.kind == "agent_answer"),
            denominator=candidates,
            denominator_name="exported episodes with a submitted answer",
        ),
        DatasetCount(name="snapshots", count=n_snap, denominator=n_snap, denominator_name="snapshots"),
        DatasetCount(
            name="snapshots_with_chemistry",
            count=sum(1 for s in snapshots.values() if s.chemistry is not None),
            denominator=n_snap,
            denominator_name="snapshots",
        ),
        DatasetCount(
            name="snapshots_judged",
            count=len(frozen_for_judging & set(snapshots)),
            denominator=n_snap,
            denominator_name="snapshots",
        ),
        DatasetCount(
            name="snapshots_with_expert_labels",
            count=len({r.snapshot for r in expert}),
            denominator=n_snap,
            denominator_name="snapshots",
        ),
        DatasetCount(
            name="snapshots_with_model_judge_labels",
            count=len({r.snapshot for r in judged}),
            denominator=n_snap,
            denominator_name="snapshots",
        ),
        DatasetCount(name="labels_expert", count=len(expert), denominator=len(labels_out), denominator_name="labels"),
        DatasetCount(name="labels_model_judge", count=len(judged), denominator=len(labels_out), denominator_name="labels"),
        DatasetCount(name="peak_calls", count=len(peaks), denominator=len(peaks), denominator_name="peak calls"),
        DatasetCount(name="signoffs", count=len(signoffs), denominator=len(signoffs), denominator_name="sign-offs"),
        DatasetCount(name="step_feedback", count=len(steps), denominator=len(steps), denominator_name="step feedback records"),
        DatasetCount(
            name="trajectory_reviews",
            count=len(trajectory_reviews),
            denominator=len(trajectory_reviews),
            denominator_name="trajectory reviews",
        ),
        DatasetCount(
            name="gradable_expectations",
            count=sum(1 for e in expectations if isinstance(e, CaseExpectation)),
            denominator=len(expectations),
            denominator_name="roots of frozen qualification sets",
        ),
        DatasetCount(name="qualification_reports", count=len(reports), denominator=len(reports), denominator_name="reports"),
    ]
    runs = (store_manifest or {}).get("rows")
    if runs is not None and scope == "private":
        imported = sum(1 for row in runs if not row.get("error"))
        counts.append(DatasetCount(name="runs_imported", count=imported, denominator=len(runs), denominator_name="runs in the load manifest"))

    agents: Counter[tuple[str, str, str, str]] = Counter()
    for episode in in_scope:
        if episode.human_contributions:
            continue  # an expert's path past an answer is a correction, not another agent episode
        for run in episode.runs:
            step = next((s for s in episode.steps if s.digest == run.start), None)
            if step is None or not isinstance(step.attempt.outcome, Completed):
                continue
            result = step.attempt.outcome.result
            if not isinstance(result, ImportedRunResult):
                continue
            key = (episode.root.case.case_id, result.agent.model_id, result.agent.harness)
            agents[(*key, "episodes")] += 1
            agents[(*key, run.status.status)] += 1
            agents[(*key, "incomplete")] += isinstance(result.completeness, Incomplete)
    agent_rows = tuple(
        AgentEpisodes(
            case_id=case_id,
            model=model,
            harness=harness,
            episodes=agents[(case_id, model, harness, "episodes")],
            submitted=agents[(case_id, model, harness, "submitted")],
            unsuccessful=agents[(case_id, model, harness, "unsuccessful")],
            incomplete=agents[(case_id, model, harness, "incomplete")],
        )
        for case_id, model, harness in sorted({k[:3] for k in agents})
    )

    manifest = DatasetManifest(
        dataset_version=inputs.dataset_version,
        exporter_version=EXPORTER_VERSION,
        scope=scope,
        export_use=use,
        generated_at_utc=inputs.now,
        rubric_version=RUBRIC_VERSION,
        guidance_version=GUIDANCE_VERSION,
        episode_export=bundle.manifest,
        store_manifest_sha256=store_manifest_sha256,
        judge_manifest_sha256=hashlib.sha256(canonical_bytes(judge_manifest)).hexdigest(),
        snapshot_index=tuple(
            SnapshotEntry(digest=d, origin=s.origin, kind=s.kind, answered=s.answered, judged=d in frozen_for_judging)
            for d, s in sorted(snapshots.items())
        ),
        qualification_sets=tuple(set_entries),
        counts=tuple(counts),
        agent_episodes=agent_rows,
        exclusions=tuple(sorted(exclusions, key=lambda e: (e.kind, e.ref))),
        files=(),
    )
    return Dataset(
        manifest=manifest,
        judge_manifest=judge_manifest,
        episodes=tuple(pseudonymous_episode(e, pseudonyms) for e in in_scope),
        excluded=bundle.excluded,
        snapshots=tuple(snap for _, snap in sorted(snapshots.items())),
        snapshot_json=tuple(frozen_json(frozen_bytes[d]) for d in sorted(snapshots)),
        labels_expert=expert,
        labels_model_judge=judged,
        peak_calls=peaks,
        signoffs=signoffs,
        step_feedback=steps,
        judge_inputs=tuple(judge_inputs),
        expectations=tuple(expectations),
        reports=reports,
        trajectory_reviews=trajectory_reviews,
    )


def write_dataset(dataset: Dataset, out_dir: Path) -> DatasetManifest:
    """One canonical JSON line per record; manifest.json lists every file with its line count
    and sha256. The rater identity map is never written here (R33)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: Mapping[str, tuple[Any, ...]] = {
        "episodes.jsonl": dataset.episodes,
        "excluded.jsonl": dataset.excluded,
        "snapshots.jsonl": dataset.snapshot_json,
        "labels.expert.jsonl": dataset.labels_expert,
        "labels.model_judge.jsonl": dataset.labels_model_judge,
        "peak_calls.jsonl": dataset.peak_calls,
        "signoffs.jsonl": dataset.signoffs,
        "step_feedback.jsonl": dataset.step_feedback,
        "judge_inputs.jsonl": dataset.judge_inputs,
        "trajectory_reviews.jsonl": dataset.trajectory_reviews,
    }
    if dataset.manifest.scope == "private":
        rows = {**rows, "expectations.jsonl": dataset.expectations, "reports.jsonl": dataset.reports}
    files = []
    for name, records in rows.items():
        if name == "judge_inputs.jsonl":
            data = b"".join(judge_input_bytes(record) + b"\n" for record in records)
        elif name == "snapshots.jsonl":  # frozen JSON: these bytes are the ones each digest names
            data = b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n" for row in records
            )
        else:
            data = b"".join(canonical_bytes(record) for record in records)
        (out_dir / name).write_bytes(data)
        files.append(DatasetFile(path=name, lines=len(records), sha256=hashlib.sha256(data).hexdigest()))
    if dataset.manifest.scope == "private":
        (out_dir / "judge_manifest.json").write_bytes(canonical_bytes(dataset.judge_manifest))
    manifest = dataset.manifest.model_copy(update={"files": tuple(files)})
    (out_dir / "manifest.json").write_bytes(canonical_bytes(manifest))
    return manifest


def _keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def validate_dataset(out_dir: Path, schemas: Mapping[str, Mapping[str, Any]], canaries: Iterable[str] = ()) -> list[str]:
    """Every problem found, empty when the dataset is valid: each line validates against its
    schema; no custody key anywhere; no run-identity key in snapshots or judge inputs; label
    files never mix rater kinds; every label, peak call and sign-off names a snapshot in
    snapshots.jsonl; rater ids are pseudonyms; the manifest's counts and hashes match the files;
    no custody canary string appears in any byte (when canaries are given)."""
    from jsonschema import Draft202012Validator

    problems: list[str] = []
    validators = {name: Draft202012Validator(schema) for name, schema in schemas.items()}
    parsed: dict[str, list[dict[str, Any]]] = {}
    raw: dict[str, bytes] = {}
    manifest_path = out_dir / "manifest.json"
    manifest = None
    if not manifest_path.is_file():
        problems.append("manifest.json: missing")
    else:
        raw["manifest.json"] = manifest_path.read_bytes()
        try:
            manifest = DatasetManifest.model_validate_json(raw["manifest.json"])
        except ValidationError as exc:
            problems.append(f"manifest.json: {exc.errors()[0]['msg']}")
        if set(_keys(json.loads(raw["manifest.json"]))) & CUSTODY_KEYS:
            problems.append("manifest.json: custody field")
    expected = dict(SCHEMA_FOR_FILE)
    optional = {
        name: schema
        for name, schema in OPTIONAL_SCHEMA_FOR_FILE.items()
        if (out_dir / name).exists() or (manifest is not None and any(f.path == name for f in manifest.files))
    }
    expected |= optional
    if manifest is not None and manifest.scope == "private":
        expected |= PRIVATE_SCHEMA_FOR_FILE
    else:
        problems += [f"{name}: a public bundle carries no grader file" for name in PRIVATE_SCHEMA_FOR_FILE if (out_dir / name).exists()]
    for name, schema_name in expected.items():
        path = out_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        raw[name] = path.read_bytes()
        parsed[name] = []
        for number, line in enumerate(raw[name].splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{name}:{number}: not JSON ({exc})")
                continue
            parsed[name].append(row)
            for error in validators[schema_name].iter_errors(row):
                problems.append(f"{name}:{number}: schema {schema_name}: {error.message[:200]}")
                break
            keys = set(_keys(row))
            if leaked := keys & CUSTODY_KEYS:
                problems.append(f"{name}:{number}: custody field {sorted(leaked)}")
            if name in ("snapshots.jsonl", "judge_inputs.jsonl"):
                if named := keys & RUN_IDENTITY_KEYS:
                    problems.append(f"{name}:{number}: run identity field {sorted(named)}")
                if found := RUN_IDENTITY_TEXT.search(line.decode(errors="replace")):
                    problems.append(f"{name}:{number}: run identity text {found.group(0)!r}")
    if manifest is not None:
        listed = {f.path: f for f in manifest.files}
        for name in expected:
            entry = listed.get(name)
            if name not in raw:
                continue
            if entry is None:
                problems.append(f"manifest.json: {name} not listed")
            elif entry.lines != len(parsed[name]) or entry.sha256 != hashlib.sha256(raw[name]).hexdigest():
                problems.append(f"manifest.json: {name} line count or sha256 disagrees with the file")
    if manifest is not None and "expectations.jsonl" in raw:
        lines, start = raw["expectations.jsonl"].splitlines(keepends=True), 0
        for entry in manifest.qualification_sets:
            chunk = b"".join(lines[start : start + entry.roots])
            start += entry.roots
            if hashlib.sha256(chunk).hexdigest() != entry.expectations_sha256:
                problems.append(f"expectations.jsonl: rows of set {entry.set_digest[:12]} disagree with the manifest")
        if start != len(lines):
            problems.append("expectations.jsonl: rows outside every listed qualification set")
    if manifest is not None and manifest.scope == "public":  # final contract U6: no public file has the key
        for name in raw:
            if "reference" in set(_keys([json.loads(line) for line in raw[name].splitlines()])):
                problems.append(f"{name}: public file carries the key reference")
    for name, kinds in (("labels.expert.jsonl", {"expert", "owner"}), ("labels.model_judge.jsonl", {"model_judge"})):
        for number, row in enumerate(parsed.get(name, ()), start=1):
            if row.get("rater", {}).get("kind") not in kinds:
                problems.append(f"{name}:{number}: rater kind {row.get('rater', {}).get('kind')} does not belong here")
    snapshot_digests = {
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
        for row in parsed.get("snapshots.jsonl", ())
    }
    pseudonym = re.compile(RATER_PSEUDONYM)
    for name in (
        "labels.expert.jsonl",
        "labels.model_judge.jsonl",
        "peak_calls.jsonl",
        "signoffs.jsonl",
        "step_feedback.jsonl",
        "trajectory_reviews.jsonl",
    ):
        for number, row in enumerate(parsed.get(name, ()), start=1):
            if "snapshot" in row and row["snapshot"] not in snapshot_digests:
                problems.append(f"{name}:{number}: snapshot {row['snapshot']} is not in snapshots.jsonl")
            if not pseudonym.fullmatch(str(row.get("rater", {}).get("principal_id", ""))):
                problems.append(f"{name}:{number}: rater id is not a pseudonym")
    kinds = {
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest(): row.get("kind")
        for row in parsed.get("snapshots.jsonl", ())
    }
    judgeable = {d for d, kind in kinds.items() if kind == "agent_answer"}
    judged = set()
    for number, row in enumerate(parsed.get("judge_inputs.jsonl", ()), start=1):
        if (row.get("snapshot") or {}).get("kind") != "agent_answer":
            problems.append(f"judge_inputs.jsonl:{number}: a judge input that is not an agent answer (CR-08)")
        digest = row.get("snapshot_digest")
        body = json.dumps(row.get("snapshot"), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        if hashlib.sha256(body).hexdigest() != digest or digest not in snapshot_digests:
            problems.append(f"judge_inputs.jsonl:{number}: snapshot does not match its digest in snapshots.jsonl")
        if set(row) != JUDGE_INPUT_KEYS:
            problems.append(f"judge_inputs.jsonl:{number}: carries fields beyond the snapshot and its blind view")
        if row.get("reported_phase_ids") != [p.get("phase_id") for p in (row.get("snapshot") or {}).get("phases", [])]:
            problems.append(f"judge_inputs.jsonl:{number}: reported_phase_ids differ from the snapshot's phases")
        judged.add(digest)
    if judged != judgeable:
        problems.append(f"judge_inputs.jsonl: covers {len(judged)} of {len(judgeable)} agent-answer snapshots")
    if manifest is not None:
        if {e.digest for e in manifest.snapshot_index} != snapshot_digests:
            problems.append("manifest.json: snapshot_index disagrees with snapshots.jsonl")
        judge_path = out_dir / "judge_manifest.json"
        if judge_path.is_file():
            raw["judge_manifest.json"] = judge_path.read_bytes()
            frozen = JudgeManifest.model_validate_json(raw["judge_manifest.json"])
            if hashlib.sha256(canonical_bytes(frozen)).hexdigest() != manifest.judge_manifest_sha256:
                problems.append("judge_manifest.json: does not match the manifest's judge_manifest_sha256")
            entries = {e.snapshot_digest: e for e in frozen.entries}
            if any(e.judged != (e.digest in entries and entries[e.digest].judged) for e in manifest.snapshot_index):
                problems.append("manifest.json: a snapshot's judged flag disagrees with judge_manifest.json")
            if any(e.kind == "agent_answer" and e.digest not in entries for e in manifest.snapshot_index):
                problems.append("judge_manifest.json: an agent-answer snapshot is not listed")
            if any(e.kind != "agent_answer" and e.digest in entries for e in manifest.snapshot_index):
                problems.append("judge_manifest.json: lists a snapshot that is not an agent answer (CR-08)")
            lines_by_digest = {
                json.loads(line)["snapshot_digest"]: line for line in raw.get("judge_inputs.jsonl", b"").splitlines()
            }
            for digest, e in entries.items():
                line = lines_by_digest.get(digest)
                if e.judged and line is None:
                    problems.append(f"judge_manifest.json: judged snapshot {digest} has no judge input")
                elif line is not None and hashlib.sha256(line).hexdigest() != e.judge_input_sha256:
                    problems.append(f"judge_manifest.json: the judge input of {digest} differs from its frozen sha256")
        elif manifest.scope == "private":
            problems.append("judge_manifest.json: missing")
    for canary in canaries:
        needle = canary.encode()
        for name, data in raw.items():
            if needle and needle in data:
                problems.append(f"{name}: carries a custody canary")
    return problems


router = APIRouter(tags=["workbench-export"])


def get_export(root: str) -> Any:
    """GET /{root}/export (owner role): the bundle as a zip of the three files. Deferred: the
    overnight dataset is built by scripts/workbench/build_eval_dataset.py."""
    raise NotImplementedError


def main() -> None:  # pragma: no cover
    from .executors import default_registry
    from .labels import ExposureLedger
    from .seams import public_seams

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--import-root", type=Path, required=True)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--rights", type=Path, required=True)
    parser.add_argument("--now", type=datetime.fromisoformat, default=datetime.now(UTC))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    store = WorkbenchStore(
        args.store,
        default_registry().run,
        public_seams(),
        lambda: args.now,
        import_root=args.import_root,
        exposure=ExposureLedger(args.store),
    )
    inputs = ExportInputs(
        dataset_version=args.dataset_version,
        eligibility=EligibilityLedger.model_validate_json(args.eligibility.read_bytes()),
        rights=RightsRegistry.model_validate_json(args.rights.read_bytes()),
        now=args.now,
    )
    write_bundle(export_episodes(store, [AttemptDigest(r) for r in args.root], inputs), args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
