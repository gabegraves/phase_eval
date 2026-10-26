"""The one episode format, written from the tree plus versioned policy inputs, and the eval
dataset built on it (R11, R18, R19, R33; overnight N4, N8, KTD10).

`python -m phaseeval.workbench.cli export` writes the validated dataset. Re-export of the same
snapshot and inputs is byte-identical. `scripts/workbench/build_eval_dataset.py` also builds
the eval dataset: frozen analysis snapshots, labels split by rater kind, peak calls,
sign-offs, step feedback, judge inputs and a manifest with denominators. Every JSON Schema under
`schemas/` is generated from these types (`dataset_json_schemas`), so a schema and its type
cannot disagree.

An episode ends at every leaf of the tree and at every imported run that has children, so an
agent's own answer stays a row after an expert corrects it. SFT and RL-task export views are
deferred. Same-case replay proves capture and reproducibility, not learning.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from fastapi import APIRouter
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, model_validator

from phaseeval.api.auth import Principal
from phaseeval.publish.leaks import PUBLIC_RULES, redact, scan, scan_bytes

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
)
from .model import (
    AnalysisState,
    FREE_TEXT,
    FreeText,
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
from .review import (
    ARTIFACT_AUDIENCE,
    Annotation,
    Note,
    Preference,
    PublicAttemptView,
    PublicCaseView,
    public_attempt_view,
    public_case_view,
)
from .snapshot import AnalysisSnapshot, SnapshotKind
from .store import WorkbenchStore

EPISODE_SCHEMA_VERSION = "axa.workbench.episode.v1"
PUBLIC_EPISODE_SCHEMA_VERSION = "phaseeval.episode.v2"
EXPORTER_VERSION = "axa-workbench-exporter-v1"
DATASET_SCHEMA_VERSION = "phaseeval.dataset-manifest.v2"


class EpisodeStep(Record):
    """One state transition: the attempt (action, actor, tool, inputs, outcome, origin and replay
    manifest) and the derived state after it."""

    digest: AttemptRef
    attempt: ChildAttempt
    state_after: AnalysisState
    annotations: tuple[Annotation, ...]  # notes, preferences, replay checks, run closes on it


class PublicEpisodeStep(EpisodeStep):
    attempt: PublicAttemptView


class PublicRootView(Record):
    case: PublicCaseView


class PreferencePair(Record):
    """Only from an explicit review.Preference (R18)."""

    chosen: AttemptRef
    rejected: AttemptRef
    fork: AttemptRef  # lowest common ancestor of the two paths
    reason: FreeText
    author: HumanActor


class LegacyEpisodeRecord(Record):
    model_config = {"title": "EpisodeRecord"}
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


class EpisodeRecord(LegacyEpisodeRecord):
    schema_version: Literal["axa.workbench.episode.v1", "phaseeval.episode.v2"] = EPISODE_SCHEMA_VERSION
    root: RootAttempt | PublicRootView
    steps: tuple[EpisodeStep | PublicEpisodeStep, ...]

    @model_validator(mode="after")
    def versioned_projection(self) -> Self:
        public = self.schema_version == PUBLIC_EPISODE_SCHEMA_VERSION
        if isinstance(self.root, PublicRootView) != public or any(
            isinstance(step.attempt, PublicAttemptView) != public for step in self.steps
        ):
            raise ValueError("episode fields must match their schema version")
        return self


class PublicEpisodeRecord(EpisodeRecord):
    schema_version: Literal["phaseeval.episode.v2"] = PUBLIC_EPISODE_SCHEMA_VERSION
    root: PublicRootView
    steps: tuple[PublicEpisodeStep, ...]


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


class PublicExportManifest(ExportManifest):
    schema_version: Literal["phaseeval.episode.v2"] = PUBLIC_EPISODE_SCHEMA_VERSION


class ExportBundle(Record):
    manifest: PublicExportManifest
    episodes: tuple[EpisodeRecord, ...]
    excluded: tuple[ExcludedEpisode, ...]


def episode_json_schema() -> dict[str, Any]:
    """The published JSON Schema (R19), derived from the type."""
    schema = TypeAdapter(PublicEpisodeRecord).json_schema()
    schema["$id"] = "schemas/phaseeval-episode.v2.json"
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
            public_root = PublicRootView(case=public_case_view(root_digest, root.case))
            steps, state = [], initial_state(root_digest, public_root)
            for digest, attempt in path[1:]:
                attempt = public_attempt_view(attempt)
                state = transition(state, (digest, attempt))
                steps.append(
                    PublicEpisodeStep(
                        digest=digest, attempt=attempt,
                        state_after=state, annotations=annotations[digest],
                    )
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
                "schema_version": PUBLIC_EPISODE_SCHEMA_VERSION,
                "root_digest": root_digest,
                "root": public_root,
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
                "rights": tuple(right for right in policy.rights_for(ep, path, inputs.rights)
                                if "public" in ARTIFACT_AUDIENCE[right.role]),
                "effort": ep.effort,
            }
            episodes.append(EpisodeRecord(record_id=_record_id(fields), **fields))
    episodes.sort(key=lambda e: (e.root_digest, e.leaf))
    excluded.sort(key=lambda e: (e.root_digest, e.leaf))
    manifest = PublicExportManifest(
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
# Run identity a blind rater or judge must not see (PI directive; labels.LabelStore reveal rules).
RUN_IDENTITY_KEYS = frozenset(
    {"reference_match", "job", "model", "harness", "series", "episode_id"}  # PI directive
    | {"job_name", "model_id", "harness_version", "trial_id", "reference", "siblings"}  # W3 view's set
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


class LegacyDatasetManifest(Record):
    schema_version: Literal["axa.workbench.dataset.v1"] = "axa.workbench.dataset.v1"
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


class EpisodeExportV2(Record):
    exporter_version: SafeID
    schema_version: Literal["axa.workbench.episode.v1", "phaseeval.episode.v2"] = PUBLIC_EPISODE_SCHEMA_VERSION
    dataset_version: SafeID
    policy_version: SafeID
    rights_version: SafeID
    roots: tuple[AttemptRef, ...]
    leaves: tuple[AttemptRef, ...]  # the tree snapshot this export covers
    inputs_sha256: Digest
    episodes_walked: int = Field(ge=0)
    episodes_exported: int = Field(ge=0)
    preference_pairs: int
    excluded: int
    excluded_by_reason: dict[str, int]


class DatasetCountV2(DatasetCount):
    file: str


class DatasetManifestV2(LegacyDatasetManifest):
    schema_version: Literal["phaseeval.dataset-manifest.v2"] = DATASET_SCHEMA_VERSION
    episode_export: EpisodeExportV2
    counts: tuple[DatasetCountV2, ...]


class DatasetManifest(LegacyDatasetManifest):
    schema_version: Literal["axa.workbench.dataset.v1", "phaseeval.dataset-manifest.v2"] = DATASET_SCHEMA_VERSION
    episode_export: ExportManifest | EpisodeExportV2
    counts: tuple[DatasetCountV2 | DatasetCount, ...]


SCHEMA_FILES: Mapping[str, Any] = {
    "workbench-episode.v1.json": LegacyEpisodeRecord,
    "phaseeval-episode.v2.json": PublicEpisodeRecord,
    "excluded-episode.v1.json": ExcludedEpisode,
    "snapshot.v1.json": AnalysisSnapshot,
    "label.v1.json": Label,
    "peak-call.v1.json": PeakCall,
    "signoff.v1.json": SignOff,
    "step-feedback.v1.json": StepFeedback,
    "judge-input.v1.json": JudgeInput,
    "judge-manifest.v1.json": JudgeManifest,
    "case-expectation.v2.json": Expectation,
    "qualification-report.v2.json": QualificationReport,
    "dataset-manifest.v1.json": LegacyDatasetManifest,
    "phaseeval-dataset-manifest.v2.json": DatasetManifestV2,
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
# The grader side of the private export (final contract row U6); a public bundle carries neither.
PRIVATE_SCHEMA_FOR_FILE: Mapping[str, str] = {
    "expectations.jsonl": "case-expectation.v2.json",
    "reports.jsonl": "qualification-report.v2.json",
}


def dataset_json_schemas() -> dict[str, dict[str, Any]]:
    """Every published schema, keyed by its file name under schemas/, derived from its type."""
    schemas = {}
    for name, model in SCHEMA_FILES.items():
        schema = TypeAdapter(model).json_schema()
        if model is LegacyDatasetManifest:
            schema["title"] = "DatasetManifest"
        schema["$id"] = f"schemas/{name}"
        schemas[name] = schema
    return schemas


class Dataset(Record):
    manifest: DatasetManifestV2
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


def _pseudonymous[T: (Label, PeakCall, SignOff, StepFeedback)](record: T, pseudonym: Callable[[str], str]) -> T:
    rater: Rater = record.rater
    return record.model_copy(update={"rater": rater.model_copy(update={"principal_id": pseudonym(rater.principal_id)})})


def _public_text(value: Any, labels: Any, author: str | None = None) -> Any:
    if isinstance(value, BaseModel):
        for field in ("rater", "author", "actor"):
            person = getattr(value, field, None)
            if person is not None and hasattr(person, "principal_id"):
                author = person.principal_id
                break
        updates = {}
        for name, field in type(value).model_fields.items():
            item = getattr(value, name)
            if FREE_TEXT in field.metadata and item:
                permission = labels._permission(author) if author else None
                item = (redact(item, scan(item)) if permission is not None
                        and permission.participation_and_storage and permission.public_release else "[withheld]")
            else:
                item = _public_text(item, labels, author)
            updates[name] = item
        return value.model_copy(update=updates)
    if isinstance(value, tuple):
        return tuple(_public_text(item, labels, author) for item in value)
    if isinstance(value, list):
        return [_public_text(item, labels, author) for item in value]
    if isinstance(value, dict):
        return {key: _public_text(item, labels, author) for key, item in value.items()}
    return value


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


def _agent_episodes(episodes: Iterable[EpisodeRecord]) -> tuple[AgentEpisodes, ...]:
    agents: Counter[tuple[str, str, str, str]] = Counter()
    for episode in episodes:
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
    return tuple(
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

    label_rows, peak_rows, signoff_rows, step_rows = (
        _public_text(rows, labels) for rows in labels.export_records(use)
    )
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
            name="gradable_expectations",
            count=sum(1 for e in expectations if isinstance(e, CaseExpectation)),
            denominator=len(expectations),
            denominator_name="roots of frozen qualification sets",
        ),
        DatasetCount(name="qualification_reports", count=len(reports), denominator=len(reports), denominator_name="reports"),
    ]
    agent_rows = _agent_episodes(in_scope)

    manifest = DatasetManifestV2(
        dataset_version=inputs.dataset_version,
        exporter_version=EXPORTER_VERSION,
        scope=scope,
        export_use=use,
        generated_at_utc=inputs.now,
        rubric_version=RUBRIC_VERSION,
        guidance_version=GUIDANCE_VERSION,
        episode_export=EpisodeExportV2(**(bundle.manifest.model_dump(exclude={"episodes"}) | {
            "episodes_walked": n_walked, "episodes_exported": len(in_scope),
        })),
        store_manifest_sha256=store_manifest_sha256,
        judge_manifest_sha256=hashlib.sha256(canonical_bytes(judge_manifest)).hexdigest(),
        snapshot_index=tuple(
            SnapshotEntry(digest=d, origin=s.origin, kind=s.kind, answered=s.answered, judged=d in frozen_for_judging)
            for d, s in sorted(snapshots.items())
        ),
        qualification_sets=tuple(set_entries),
        counts=tuple(DatasetCountV2(**c.model_dump(), file=COUNT_FILES[c.name]) for c in counts),
        agent_episodes=agent_rows,
        exclusions=tuple(sorted(exclusions, key=lambda e: (e.kind, e.ref))),
        files=(),
    )
    return Dataset(
        manifest=manifest,
        judge_manifest=judge_manifest,
        episodes=tuple(pseudonymous_episode(_public_text(e, labels), pseudonyms) for e in in_scope),
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
    )



_ROW_COUNT_FILES_V1: Mapping[str, str] = {
    "episodes_exported": "episodes.jsonl",
    "episodes_excluded": "excluded.jsonl",
    "snapshots": "snapshots.jsonl",
    "labels_expert": "labels.expert.jsonl",
    "labels_model_judge": "labels.model_judge.jsonl",
    "peak_calls": "peak_calls.jsonl",
    "signoffs": "signoffs.jsonl",
    "step_feedback": "step_feedback.jsonl",
    "qualification_reports": "reports.jsonl",
}
_ROW_COUNT_FILES_V2: Mapping[str, str] = {**_ROW_COUNT_FILES_V1, "judge_inputs": "judge_inputs.jsonl"}


COUNT_FILES = {
    **_ROW_COUNT_FILES_V2,
    "episodes_incomplete": "episodes.jsonl",
    "agent_snapshots": "snapshots.jsonl",
    "snapshots_with_chemistry": "snapshots.jsonl",
    "snapshots_judged": "snapshots.jsonl",
    "snapshots_with_expert_labels": "snapshots.jsonl",
    "snapshots_with_model_judge_labels": "snapshots.jsonl",
    "gradable_expectations": "expectations.jsonl",
}


class ValidationProblem(Record):
    file: str
    line: int | None
    code: Literal["count_mismatch"]
    detail: str


ManifestVersion = Literal["axa.workbench.dataset.v1", "phaseeval.dataset-manifest.v2"]


def reconcile_counts(
    manifest_version: ManifestVersion,
    episode_export: Mapping[str, Any],
    counts: Sequence[Mapping[str, Any]],
    row_counts: Mapping[str, int],
) -> tuple[ValidationProblem, ...]:
    """Reconcile accepted/excluded counts in v1 and explicit walked/exported counts in v2."""
    problems: list[ValidationProblem] = []
    row_counts = {**dict.fromkeys(PRIVATE_SCHEMA_FOR_FILE, 0), **row_counts}

    def bad(detail: str) -> None:
        problems.append(ValidationProblem(file="manifest.json", line=None, code="count_mismatch", detail=detail))

    by_name = {row["name"]: row for row in counts}
    if len(by_name) != len(counts):
        bad("count names must be unique")
    for row in counts:
        if row["name"] not in COUNT_FILES:
            bad(f"{row['name']}: unsupported count")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("count", "denominator")):
            bad(f"{row.get('name')}: counts must be nonnegative integers")
    if problems:
        return tuple(problems)
    denominators = {
        "episodes_incomplete": "episodes.jsonl",
        "snapshots": "snapshots.jsonl",
        "snapshots_with_chemistry": "snapshots.jsonl",
        "snapshots_judged": "snapshots.jsonl",
        "snapshots_with_expert_labels": "snapshots.jsonl",
        "snapshots_with_model_judge_labels": "snapshots.jsonl",
        "peak_calls": "peak_calls.jsonl",
        "signoffs": "signoffs.jsonl",
        "step_feedback": "step_feedback.jsonl",
        "gradable_expectations": "expectations.jsonl",
        "qualification_reports": "reports.jsonl",
    }
    for name, file in denominators.items():
        if name in by_name and file in row_counts and by_name[name]["denominator"] != row_counts[file]:
            bad(f"{name}: denominator differs from {file}")
    if all(file in row_counts for file in ("labels.expert.jsonl", "labels.model_judge.jsonl")):
        labels = row_counts["labels.expert.jsonl"] + row_counts["labels.model_judge.jsonl"]
        for name in ("labels_expert", "labels_model_judge"):
            if name in by_name and by_name[name]["denominator"] != labels:
                bad(f"{name}: denominator differs from the total labels")
    files = _ROW_COUNT_FILES_V1 if manifest_version == "axa.workbench.dataset.v1" else _ROW_COUNT_FILES_V2
    for row in counts:
        if row["count"] > row["denominator"]:
            bad(f"{row['name']}: count {row['count']} exceeds its denominator {row['denominator']}")
    for name, file in files.items():
        if name in by_name and file in row_counts and by_name[name]["count"] != row_counts[file]:
            bad(f"{name}: manifest says {by_name[name]['count']}, {file} has {row_counts[file]} rows")
    if manifest_version == "axa.workbench.dataset.v1":
        accepted, excluded = episode_export.get("episodes"), episode_export.get("excluded", 0)
        if type(accepted) is not int or type(excluded) is not int or min(accepted, excluded) < 0:
            bad("v1 requires nonnegative accepted episodes and excluded counts")
        else:
            walked = accepted + excluded
            if "leaves" in episode_export and walked != len(episode_export["leaves"]):
                bad("accepted + excluded differs from the number of episode_export.leaves")
            for name in ("episodes_exported", "episodes_excluded"):
                row = by_name.get(name)
                if row and row["denominator"] != walked:
                    bad(f"{name}: denominator must equal accepted + excluded ({walked})")
            exported = by_name.get("episodes_exported")
            if exported and exported["count"] > accepted:
                bad("publicly exported episodes exceed accepted episodes")
            if row_counts.get("episodes.jsonl", 0) > accepted:
                bad("publicly exported episodes exceed accepted episodes")
            if "excluded.jsonl" in row_counts and row_counts["excluded.jsonl"] != excluded:
                bad("episode_export.excluded differs from excluded.jsonl")
            if "episodes_excluded" in by_name and by_name["episodes_excluded"]["count"] != excluded:
                bad("episodes_excluded differs from episode_export.excluded")

    else:
        walked, exported_n = episode_export.get("episodes_walked"), episode_export.get("episodes_exported")
        if any(type(n) is not int or n < 0 for n in (walked, exported_n)):
            bad("v2 episode_export names episodes_walked and episodes_exported")
        else:
            if "leaves" in episode_export and walked != len(episode_export["leaves"]):
                bad("episodes_walked differs from the number of episode_export.leaves")
            if "episodes.jsonl" in row_counts and exported_n != row_counts["episodes.jsonl"]:
                bad(f"episode_export.episodes_exported {exported_n} is not the episodes.jsonl row count")
            for name in ("episodes_exported", "episodes_excluded"):
                row = by_name.get(name)
                if row and row["denominator"] != walked:
                    bad(f"{name}: denominator must equal episodes_walked ({walked})")
            if "excluded.jsonl" in row_counts and episode_export.get("excluded", 0) != row_counts["excluded.jsonl"]:
                bad("episode_export.excluded differs from excluded.jsonl")
            if exported_n + episode_export.get("excluded", 0) > walked:
                bad("exported plus excluded episodes exceed episodes_walked")
            if walked < exported_n:
                bad(f"episodes_walked {walked} is below episodes_exported {exported_n}")
    return tuple(problems)


def write_dataset(dataset: Dataset, out_dir: Path, *, canaries: Iterable[str] = ()) -> DatasetManifestV2:
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_dir.name}.", dir=out_dir.parent) as directory:
        staging = Path(directory)
        manifest = _write_dataset(dataset, staging)
        problems = validate_dataset(staging, dataset_json_schemas(), canaries)
        if problems:
            raise ValueError("; ".join(problems))
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError(out_dir)
        staging.rename(out_dir)
    return manifest


def migrate_dataset(source: Path, target: Path, *, store: Any | None = None) -> list[str]:
    """Project episodes to v2; preserve all other record files verbatim."""
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError("migration target must be outside the source directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    schemas = dataset_json_schemas()
    if problems := validate_dataset(source, schemas):
        raise ValueError("; ".join(problems))
    rows = []
    for line in (source / "episodes.jsonl").read_bytes().splitlines():
        episode = EpisodeRecord.model_validate_json(line)
        public_root = PublicRootView(case=public_case_view(episode.root_digest, episode.root.case))
        steps, state = [], initial_state(episode.root_digest, public_root)
        for step in episode.steps:
            attempt = public_attempt_view(step.attempt)
            state = transition(state, (step.digest, attempt))
            steps.append(PublicEpisodeStep(
                digest=step.digest, attempt=attempt, state_after=state, annotations=step.annotations,
            ))
        fields = dict(episode)
        fields.pop("record_id")
        fields.update(
            schema_version=PUBLIC_EPISODE_SCHEMA_VERSION, root=public_root, steps=tuple(steps),
            rights=tuple(right for right in episode.rights if "public" in ARTIFACT_AUDIENCE[right.role]),
        )
        episode = EpisodeRecord(record_id=_record_id(fields), **fields)
        rows.append(episode)
    manifest = DatasetManifest.model_validate_json((source / "manifest.json").read_bytes())
    export = manifest.episode_export.model_dump()
    if "episodes" in export:
        export["episodes_walked"] = export.pop("episodes") + export["excluded"]
        export["episodes_exported"] = len(rows)
    export["schema_version"] = PUBLIC_EPISODE_SCHEMA_VERSION
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as directory:
        staging = Path(directory)
        shutil.copytree(source, staging, dirs_exist_ok=True)
        (staging / "episodes.jsonl").write_bytes(b"".join(canonical_bytes(row) for row in rows))
        data = (staging / "episodes.jsonl").read_bytes()
        fields = manifest.model_dump() | {
            "schema_version": DATASET_SCHEMA_VERSION,
            "episode_export": export,
            "counts": [row.model_dump() | {"file": COUNT_FILES[row.name]} for row in manifest.counts],
            "files": [
                entry.model_dump() | (
                    {"lines": len(rows), "sha256": hashlib.sha256(data).hexdigest()}
                    if entry.path == "episodes.jsonl" else {}
                ) for entry in manifest.files
            ],
        }
        (staging / "manifest.json").write_bytes(canonical_bytes(DatasetManifestV2.model_validate(fields)))
        if problems := validate_dataset(staging, schemas):
            raise ValueError("; ".join(problems))
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        staging.rename(target)
    return []


def _write_dataset(dataset: Dataset, out_dir: Path) -> DatasetManifestV2:
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
    typed: dict[str, list[Any]] = {}
    raw: dict[str, bytes] = {}
    judge_lines: dict[str, bytes] = {}
    manifest_path = out_dir / "manifest.json"
    manifest = None
    if not manifest_path.is_file():
        problems.append("manifest.json: missing")
    else:
        raw["manifest.json"] = manifest_path.read_bytes()
        try:
            manifest = TypeAdapter(LegacyDatasetManifest | DatasetManifestV2).validate_json(raw["manifest.json"])
        except ValidationError as exc:
            problems.append(f"manifest.json: {exc.errors()[0]['msg']}")

    rules = tuple(rule for rule in PUBLIC_RULES if rule.name != "demo_run")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            name = path.relative_to(out_dir).as_posix()
            if name not in raw:
                raw[name] = path.read_bytes()
            for finding in scan_bytes(raw[name], rules=rules):
                if (manifest is not None and manifest.scope == "private" and name == "expectations.jsonl"
                        and finding.rule == "custody_key" and finding.pointer.endswith("/reference")):
                    continue
                problems.append(f"{name}:{finding.pointer}: leak {finding.rule}")
                if finding.rule == "custody_key":
                    problems.append(f"{name}: custody field {[finding.pointer.rsplit('/', 1)[-1]]}")
    expected = dict(SCHEMA_FOR_FILE)
    if manifest is not None and manifest.scope == "private":
        expected |= PRIVATE_SCHEMA_FOR_FILE
    else:
        problems += [f"{name}: a public bundle carries no grader file" for name in PRIVATE_SCHEMA_FOR_FILE if (out_dir / name).exists()]
    for name, schema_name in expected.items():
        if name not in raw:
            problems.append(f"{name}: missing")
            continue
        parsed[name] = []
        typed[name] = []
        for number, line in enumerate(raw[name].splitlines(), start=1):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                problems.append(f"{name}:{number}: not JSON ({exc})")
                continue
            if not isinstance(row, dict):
                problems.append(f"{name}:{number}: model requires an object")
                continue
            try:
                json.dumps(row, allow_nan=False)
            except ValueError:
                problems.append(f"{name}:{number}: nonfinite number is not JSON compliant")
                continue
            if name == "episodes.jsonl":
                schema_name = (
                    "phaseeval-episode.v2.json" if row.get("schema_version") == PUBLIC_EPISODE_SCHEMA_VERSION
                    else "workbench-episode.v1.json"
                )
            for error in validators[schema_name].iter_errors(row):
                problems.append(f"{name}:{number}: schema {schema_name}: {error.message[:200]}")
                break
            keys = set(_keys(row))
            if name in ("snapshots.jsonl", "judge_inputs.jsonl"):
                if named := keys & RUN_IDENTITY_KEYS:
                    problems.append(f"{name}:{number}: run identity field {sorted(named)}")
                identities = (
                    "u6-", "harbor-local", "harbor-live", "harbor-at", "data/custody", "series-",
                    "/jobs/", "task-dev", "qwen", "astra", "gpt-6",
                ) + tuple(
                    identity
                    for episode in typed.get("episodes.jsonl", ()) for step in episode.steps
                    if isinstance(step.attempt.action, ImportRun) and step.attempt.action.external_source is None
                    and isinstance(step.attempt.outcome, Completed)
                    and isinstance(step.attempt.outcome.result, ImportedRunResult)
                    for identity in (step.attempt.action.harbor.job_name, step.attempt.action.harbor.trial_id,
                                     step.attempt.outcome.result.agent.model_id, step.attempt.outcome.result.agent.harness)
                )
                for finding in scan(row, rules=tuple(r for r in PUBLIC_RULES if r.name == "run_identity"), identity=identities):
                    problems.append(f"{name}:{number}: run identity text {finding.pointer}")
            try:
                record = (
                    EpisodeRecord.model_validate(row) if name == "episodes.jsonl"
                    else TypeAdapter(SCHEMA_FILES[schema_name]).validate_python(row)
                )
            except ValidationError as exc:
                problems.append(f"{name}:{number}: model: {exc.errors()[0]['msg']}")
                continue
            if name == "episodes.jsonl":
                body = {key: value for key, value in row.items() if key != "record_id"}
                data = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                if record.record_id != hashlib.sha256(data).hexdigest():
                    problems.append(f"{name}:{number}: record_id differs from its canonical body hash")
            parsed[name].append(row)
            typed[name].append(record)
            if name == "judge_inputs.jsonl":
                judge_lines[row["snapshot_digest"]] = line
    if manifest is not None:
        listed = {f.path: f for f in manifest.files}
        for name in expected:
            entry = listed.get(name)
            if name not in raw:
                continue
            if entry is None:
                problems.append(f"manifest.json: {name} not listed")
            elif entry.lines != len(raw[name].splitlines()) or entry.sha256 != hashlib.sha256(raw[name]).hexdigest():
                problems.append(f"manifest.json: {name} line count or sha256 disagrees with the file")
    if manifest is not None:
        problems += [f"{p.file}: {p.code}: {p.detail}" for p in reconcile_counts(
            manifest.schema_version, manifest.episode_export.model_dump(),
            [row.model_dump() for row in manifest.counts], {name: len(data.splitlines()) for name, data in raw.items()},
        )]
        if manifest.agent_episodes != _agent_episodes(typed.get("episodes.jsonl", ())):
            problems.append("manifest.json: count_mismatch: agent_episodes disagree with episodes.jsonl")
        subset_counts = {
            "episodes_incomplete": sum(any(not run.complete for run in episode.runs)
                                       for episode in typed.get("episodes.jsonl", ())),
            "agent_snapshots": sum(s.kind == "agent_answer" for s in typed.get("snapshots.jsonl", ())),
            "snapshots_with_chemistry": sum(s.chemistry is not None for s in typed.get("snapshots.jsonl", ())),
            "snapshots_judged": sum(entry.judged for entry in manifest.snapshot_index),
            "snapshots_with_expert_labels": len({r.snapshot for r in typed.get("labels.expert.jsonl", ())}),
            "snapshots_with_model_judge_labels": len({r.snapshot for r in typed.get("labels.model_judge.jsonl", ())}),
            "gradable_expectations": sum(isinstance(r, CaseExpectation) for r in typed.get("expectations.jsonl", ())),
        }
        snapshot_candidates = sum(_snapshot_origin(e) is not None for e in typed.get("episodes.jsonl", ()))
        for row in manifest.counts:
            if row.name in subset_counts and row.count != subset_counts[row.name]:
                problems.append(f"manifest.json: count_mismatch: {row.name} disagrees with its records")
            if row.name == "agent_snapshots" and row.denominator != snapshot_candidates:
                problems.append("manifest.json: count_mismatch: agent_snapshots: denominator differs from eligible episodes")
        if manifest.schema_version == DATASET_SCHEMA_VERSION:
            for row in manifest.counts:
                if row.file != COUNT_FILES.get(row.name):
                    problems.append(f"manifest.json: count_mismatch: {row.name} names the wrong file")
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
        for name, rows in parsed.items():
            if "reference" in set(_keys(rows)):
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
    for name in ("labels.expert.jsonl", "labels.model_judge.jsonl", "peak_calls.jsonl", "signoffs.jsonl", "step_feedback.jsonl"):
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
        if "judge_manifest.json" in raw:
            try:
                frozen = JudgeManifest.model_validate_json(raw["judge_manifest.json"])
            except ValidationError as exc:
                problems.append(f"judge_manifest.json: model: {exc.errors()[0]['msg']}")
                return problems
            if hashlib.sha256(canonical_bytes(frozen)).hexdigest() != manifest.judge_manifest_sha256:
                problems.append("judge_manifest.json: does not match the manifest's judge_manifest_sha256")
            entries = {e.snapshot_digest: e for e in frozen.entries}
            if any(e.judged != (e.digest in entries and entries[e.digest].judged) for e in manifest.snapshot_index):
                problems.append("manifest.json: a snapshot's judged flag disagrees with judge_manifest.json")
            if any(e.kind == "agent_answer" and e.digest not in entries for e in manifest.snapshot_index):
                problems.append("judge_manifest.json: an agent-answer snapshot is not listed")
            if any(e.kind != "agent_answer" and e.digest in entries for e in manifest.snapshot_index):
                problems.append("judge_manifest.json: lists a snapshot that is not an agent answer (CR-08)")
            for digest, e in entries.items():
                line = judge_lines.get(digest)
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


def check_submission(files: Sequence[Path]) -> list[str]:
    if len(files) != 1 or files[0].name != "episodes.jsonl":
        return ["only episodes.jsonl is submittable"]
    findings = scan_bytes(files[0].read_bytes())
    if findings:
        return [f"episodes.jsonl:{f.pointer}: leak {f.rule}" for f in findings]
    return validate_dataset(files[0].parent, dataset_json_schemas())


router = APIRouter(tags=["workbench-export"])


def get_export(root: str) -> Any:
    """GET /{root}/export (owner role): the bundle as a zip of the three files. Deferred: the
    overnight dataset is built by scripts/workbench/build_eval_dataset.py."""
    raise NotImplementedError


def main() -> None:  # pragma: no cover
    raise SystemExit("This export command is retired; use python -m phaseeval.workbench.cli export --out DIR")


if __name__ == "__main__":  # pragma: no cover
    main()
