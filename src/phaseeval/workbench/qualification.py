"""Qualification (C1c, C2): a lab's expert judgments are the oracle; anything that answers a
case, routes an analysis or labels one is a Subject graded against them.

Target path: src/phaseeval/workbench/qualification.py. Contract: overnight architect final design
(sketch/qualification.py, sketch/seams.py). One question, four subjects:

    subject         observation per root                     graded against
    Producer        its agent_answer snapshots of the root    accepted and rejected analyses
    Triager         its published routing decisions           the expert's sign-off (triage grader)
    JudgeVersion    its model_judge labels                    expert labels per rubric item,
                                                              held_out roots only
    Refiner         not built: needs CrossBuildBounds (design Q6)

Invariants
- Truth boundary. A CaseExpectation is what this lab's raters said, never TruthProvenance;
  policy.py is unchanged. A report claims agreement_with_lab_judgments, never accuracy.
- Model judges never enter an expectation: HumanRaterKind has no model_judge member, and
  RootLedger refuses a model-judge label or sign-off when it is constructed.
- Single source of truth. Expectations are derived on every read from the label ledger,
  filtered by the set's cut time, its counted rater kinds and each rater's latest permission.
  A frozen set pins each root's counted record set by hash, so a withdrawal or revision shows
  per case as not_gradable(records_changed) instead of silently moving the expectation.
- A judgment is a test case whichever way it went. An accepted analysis yields required phases;
  a rejected analysis yields a negative case, even on a root with no accepted analysis. No rule
  derives a forbidden phase until an explicit root-level exclusion judgment exists.
- The published reference never decides. It arrives only through seams.ReferenceReader and
  only as a per-phase stance (agrees | disputes | absent).
- Every root of a frozen set stays in every report. A root that cannot be graded is
  not_gradable with a reason, never dropped. Sets and reports are published once.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from functools import cache
from itertools import combinations
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, PositiveFloat, TypeAdapter, model_validator

from phaseeval.api.auth import Access, Op, CaseOperation, Principal

from .labels import ExportUse, Label, LabelStore, RubricItemId, SignOff
from .model import (
    AttemptRef,
    CandidateId,
    ChildAttempt,
    Completed,
    Digest,
    HumanActor,
    ImportedRunResult,
    Record,
    SafeID,
    canonical_bytes,
)
from .seams import NoReference, ReferenceReader, SealedReference, SealedReferencePhase
from .snapshot import AnalysisSnapshot, SnapshotKind, SnapshotPhase
from .store import CustodyError, WorkbenchStore
from phaseeval.kernel.ledger import KeyShape, Mode
from .triage import (
    RECALL_MIN_AUDIT_REVIEWS,
    CheckPolicy,
    RouteDecision,
    RoutingGrade,
    Triage,
    grade_routing,
)

EXPECTATION_RULE = "lab-expectation-v2"  # v2: rejected analyses become negative cases
GRADER_VERSION = "axa-grader-v1"
HELD_OUT_SHARE = 0.5  # share of a split set's roots assigned held_out by the seeded hash


def _sha256(data: bytes) -> Digest:
    return hashlib.sha256(data).hexdigest()


def record_sha256(record: Record) -> Digest:
    """The sha256 a ledger file is named by (LabelStore._append writes canonical_bytes)."""
    return _sha256(canonical_bytes(record))


# ---------------------------------------------------------------------------------------------
# The lab's boundary: policy and frozen set
# ---------------------------------------------------------------------------------------------

HumanRaterKind = Literal["expert", "owner"]  # model_judge is not a member: unrepresentable


def _sorted_unique[T: str](values: tuple[T, ...]) -> tuple[T, ...]:
    return tuple(sorted(set(values)))


class QualificationPolicy(Record):
    """The lab's rules for which judgments count and how strictly answers are graded. A tolerance
    has no default: None means amounts are never graded, and the report says so."""

    version: SafeID
    justified_by: str = Field(min_length=1)
    # A sorted tuple, not a frozenset: set order varies per process and this record is digested.
    rater_kinds: Annotated[tuple[HumanRaterKind, ...], AfterValidator(_sorted_unique)] = Field(min_length=1)
    # True: a root with any counted record made after its rater's reveal, or an accepted or
    # rejected snapshot whose author had seen the reference, is not_blind (fail closed).
    require_blind: bool
    # Assessing raters an item or sign-off needs before it counts. A lab that wants unanimity sets
    # min_raters to its panel size; any yes/no split is `disputed` regardless.
    min_raters: int = Field(ge=1)
    # COD carries duplicates (CIF012075, CIF012076 and CIF012077 are all NiO in Fm-3m), so a lab
    # compares reduced formula plus space group number. That is a declared coarse rule: it does
    # not establish structural equivalence, and two structures with one formula and space group
    # share a key.
    phase_identity: Literal["candidate_id", "formula_and_space_group"]
    amount_abs_wt_pct: PositiveFloat | None  # required; None: amounts never graded
    expectation_rule: Literal["lab-expectation-v2"]


Partition = Literal["tune", "held_out"]


class QualificationSet(Record):
    """The lab's frozen qualification boundary, written once by a lab owner at
    qualification-sets/<digest>.json. Holds content addresses, never copies of judgments."""

    schema_version: Literal["qualification-set.v2"] = "qualification-set.v2"
    lab: SafeID
    name: SafeID
    roots: tuple[AttemptRef, ...] = Field(min_length=1)
    policy: QualificationPolicy
    # Records created after the cut never count. Freeze sets it to the latest counted record's
    # time, so the same request over the same ledger returns the same digest.
    cut_utc: datetime
    # sha256 over the sorted file digests of every counted label and sign-off of the root at
    # freeze time (CaseExpectation.source_records). Changed later -> not_gradable(records_changed).
    pins: Mapping[AttemptRef, Digest]
    # One role per root, so tune and held_out are root-disjoint inside a set by construction.
    split: Mapping[AttemptRef, Partition] | None
    split_seed: SafeID | None
    frozen_by: SafeID

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if len(set(self.roots)) != len(self.roots) or set(self.pins) != set(self.roots):
            raise ValueError("every root appears once and has one pin")
        if (self.split is None) != (self.split_seed is None):
            raise ValueError("a split comes with its seed")
        if self.split is not None and set(self.split) != set(self.roots):
            raise ValueError("a split assigns every root")
        return self


def partition_of(seed: SafeID, root: AttemptRef) -> Partition:
    """Seeded and independent of every judgment, so the split cannot be chosen after the fact."""
    draw = int(_sha256(f"{seed}:{root}".encode()), 16) / 2**256
    return "held_out" if draw < HELD_OUT_SHARE else "tune"


# ---------------------------------------------------------------------------------------------
# The counted ledger of one root
# ---------------------------------------------------------------------------------------------


def _not_model_judge[R: Label | SignOff](record: R) -> R:
    if record.rater.kind == "model_judge":
        raise ValueError("a model-judge record never enters an expectation")
    return record


HumanLabel = Annotated[Label, AfterValidator(_not_model_judge)]
HumanSignOff = Annotated[SignOff, AfterValidator(_not_model_judge)]
_HUMAN_LABELS = TypeAdapter(tuple[HumanLabel, ...])


class RootLedger(Record):
    """The counted human judgments of one root and every snapshot of it. Qualifier builds it
    after the cut, rater-kind and permission filters; constructing it refuses a model-judge
    record, so no path from a judge label to an expectation exists."""

    root: AttemptRef
    snapshots: tuple[AnalysisSnapshot, ...]
    labels: tuple[HumanLabel, ...]
    signoffs: tuple[HumanSignOff, ...]
    # Snapshots whose origin path carries a human actor stamped reference_exposure "exposed": a
    # correction made after its author had seen the reference.
    exposed: tuple[Digest, ...] = ()

    @model_validator(mode="after")
    def one_root(self) -> Self:
        if any(s.root != self.root for s in (*self.snapshots, *self.signoffs)):
            raise ValueError("a ledger holds one root")
        known = {snapshot_digest(s) for s in self.snapshots}
        if any(r.snapshot not in known for r in (*self.labels, *self.signoffs)) or not set(self.exposed) <= known:
            raise ValueError("a counted record names a snapshot outside the ledger")
        return self


def snapshot_digest(snapshot: AnalysisSnapshot) -> Digest:
    return record_sha256(snapshot)  # == LabelStore.put_snapshot's digest


def source_records(ledger: RootLedger) -> tuple[Digest, ...]:
    return tuple(sorted(record_sha256(r) for r in (*ledger.labels, *ledger.signoffs)))


def pin_of(ledger: RootLedger) -> Digest:
    return _sha256("\n".join(source_records(ledger)).encode())


# ---------------------------------------------------------------------------------------------
# Phase identity
# ---------------------------------------------------------------------------------------------


@cache
def reduced_formula(formula: str) -> str:
    from pymatgen.core import Composition

    try:
        return Composition(formula).reduced_formula
    except Exception:  # noqa: BLE001 - an unparseable formula keeps its own spelling
        return formula.replace(" ", "")


@cache
def space_group_number(space_group: str) -> int | None:
    """'Pna2_1 (33)' (the snapshot's form), '33', or a Hermann-Mauguin symbol."""
    found = re.search(r"\((\d{1,3})\)\s*$", space_group) or re.fullmatch(r"\s*(\d{1,3})\s*", space_group)
    if found:
        return int(found.group(1))
    from pymatgen.symmetry.groups import SpaceGroup

    try:
        return SpaceGroup(space_group.strip()).int_number
    except Exception:  # noqa: BLE001 - unknown symbol: no number
        return None


# The grading identity of a phase: "NiO|225" by formula and space group, "id:CIF012075" by
# candidate id or when the snapshot lacks either fact (it then matches only itself).
PhaseKey = str


def phase_key(phase: SnapshotPhase, identity: Literal["candidate_id", "formula_and_space_group"]) -> PhaseKey:
    if identity == "formula_and_space_group" and phase.formula and phase.space_group:
        number = space_group_number(phase.space_group)
        if number is not None:
            return f"{reduced_formula(phase.formula)}|{number}"
    return f"id:{phase.phase_id}"


def _names(reference: SealedReferencePhase, phases: Iterable[SnapshotPhase]) -> bool:
    """A reference phase names a snapshot phase by shipped candidate, else by reduced formula
    and (when both sides carry one) space group number."""
    for phase in phases:
        if phase.phase_id in reference.candidates:
            return True
        if phase.formula and reduced_formula(phase.formula) == reduced_formula(reference.formula):
            number = space_group_number(phase.space_group) if phase.space_group else None
            if number is None or reference.space_group is None or number == reference.space_group:
                return True
    return False


# ---------------------------------------------------------------------------------------------
# Expectations: derived from the ledger, never stored
# ---------------------------------------------------------------------------------------------

Consensus = Literal["yes", "no", "disputed", "unassessable"]
SignOffConsensus = Literal["would_sign", "would_not_sign", "unresolved", "disputed", "none"]


class ItemConsensus(Record):
    """One rubric item on one snapshot over the counted raters, each rater's latest answer.
    cannot_assess is counted and excluded; fewer than min_raters assessing is unassessable."""

    item: RubricItemId
    phase_id: CandidateId | None
    consensus: Consensus
    yes: int
    no: int
    cannot_assess: int


class JudgedSnapshot(Record):
    snapshot: Digest
    kind: SnapshotKind
    phases: tuple[CandidateId, ...]
    items: tuple[ItemConsensus, ...]
    signoff: SignOffConsensus
    exposure: Literal["blind", "revealed", "mixed"]


Target = tuple[Digest, RubricItemId, str | None]


def latest_labels(labels: Iterable[Label]) -> dict[Target, dict[str, Label]]:
    """target -> rater -> that rater's latest label. A later change is a new record."""
    out: dict[Target, dict[str, Label]] = defaultdict(dict)
    for label in sorted(labels, key=lambda r: (r.created_at_utc, record_sha256(r))):
        out[(label.snapshot, label.item, label.phase_id)][label.rater.principal_id] = label
    return out


def item_consensus(labels: Iterable[Label], min_raters: int) -> dict[Target, ItemConsensus]:
    out: dict[Target, ItemConsensus] = {}
    for target, by_rater in latest_labels(labels).items():
        answers = [label.answer for label in by_rater.values()]
        yes, no = answers.count("yes"), answers.count("no")
        out[target] = ItemConsensus(
            item=target[1],
            phase_id=target[2],
            consensus="disputed" if yes and no else "unassessable" if yes + no < min_raters else "yes" if yes else "no",
            yes=yes,
            no=no,
            cannot_assess=answers.count("cannot_assess"),
        )
    return out


def signoff_consensus(signoffs: Iterable[SignOff], min_raters: int) -> dict[Digest, SignOffConsensus]:
    latest: dict[Digest, dict[str, SignOff]] = defaultdict(dict)
    for signoff in sorted(signoffs, key=lambda r: (r.created_at_utc, record_sha256(r))):
        latest[signoff.snapshot][signoff.rater.principal_id] = signoff
    out: dict[Digest, SignOffConsensus] = {}
    for snapshot, by_rater in latest.items():
        verdicts = [s.verdict for s in by_rater.values()]
        sign, refuse = verdicts.count("would_sign"), verdicts.count("would_not_sign")
        out[snapshot] = (
            "disputed"
            if sign and refuse
            else "would_sign"
            if sign >= min_raters
            else "would_not_sign"
            if refuse >= min_raters
            else "unresolved"
        )
    return out


# agrees: the reference names the phase iff the accepted analysis has it (on a rejected-only
# root, iff the phase is not forbidden). disputes: it does not. absent: no reference was read.
ReferenceStance = Literal["agrees", "disputes", "absent"]


class PhaseExpectation(Record):
    candidate_id: CandidateId  # the accepted analysis's candidate for the key, else the first seen
    key: PhaseKey
    # No rule produces "forbidden" today: a single fit's negative plus a signed alternative is not
    # an explicit root-level exclusion (science review, AXA-43zwk.26). It waits for a scoped
    # root-level judgment; until then grade_answer reaches it only on a hand-built expectation.
    expect: Literal["required", "forbidden", "free"]
    because: Literal[
        "supported_in_accepted",  # in the accepted analysis: supported yes, plausible not no
        "unsupported",  # phase_supported consensus no on a judged snapshot
        "implausible",  # phase_plausible consensus no on a judged snapshot
        "disputed",  # raters disagree, or one snapshot says yes where another says no
        "unassessable",  # too few assessing raters, or only cannot_assess
        "inconsistent",  # in the signed analysis while judged no: reported, never graded
        "outside_accepted",  # supported on another snapshot, absent from the accepted analysis
        # judged no on a fit nobody rejected, or on a root with no accepted analysis: evidence
        # about that fit, not a root-wide exclusion, so naming the phase needs review
        "negative_in_one_fit",
    ]
    weight_percent: float | None  # from the accepted fit, only when amounts are graded
    reference: ReferenceStance  # reported beside the verdict; never changes it


class RejectedAnalysis(Record):
    """A would_not_sign snapshot: a judgment about that exact analysis. Presenting the same
    snapshot again fails repeats_rejected_analysis. Anything else with the same phase keys needs
    review, a shared fit curve included: the rejection may concern fractions, quantity basis,
    uncertainty or prose, which neither the phase set nor the curve encodes."""

    snapshot: Digest
    kind: SnapshotKind
    keys: tuple[PhaseKey, ...]  # sorted


class CaseExpectation(Record):
    schema_version: Literal["case-expectation.v2"] = "case-expectation.v2"
    root: AttemptRef
    case_id: SafeID
    # Who judged (expert, owner, or both) and whether any counted record or deciding snapshot was
    # made with the reference seen. Owner judgments are never presented as expert judgments.
    provenance: Literal[
        "lab_expert_blind",
        "lab_expert_revealed",
        "lab_owner_blind",
        "lab_owner_revealed",
        "lab_mixed_blind",
        "lab_mixed_revealed",
    ]
    # The snapshot the raters would sign: an agent answer or an expert correction. None on a
    # rejected-only root, which is still a gradable negative case.
    accepted: Digest | None
    rejected: tuple[RejectedAnalysis, ...]
    phases: tuple[PhaseExpectation, ...]
    amounts_graded: bool
    judged: tuple[JudgedSnapshot, ...]  # every snapshot of the root with a counted record
    source_records: tuple[Digest, ...]  # sha256 of every counted label and sign-off, sorted
    rule: Literal["lab-expectation-v2"] = EXPECTATION_RULE


class NotGradable(Record):
    root: AttemptRef
    reason: Literal[
        "no_counted_judgment",  # nobody counted (permission, rater kind, cut) judged the root
        "no_resolved_analysis",  # no snapshot with sign-off consensus would_sign or would_not_sign
        "no_phase_judgments",  # the accepted analysis has no counted phase label
        "signoff_disputed",  # counted raters split on every signed analysis
        "not_blind",  # require_blind and a counted record or deciding snapshot saw the reference
        "records_changed",  # the counted record set no longer hashes to the set's pin
    ]


def derive_expectation(
    ledger: RootLedger,
    case_id: SafeID,
    policy: QualificationPolicy,
    reference: SealedReference | None,
) -> CaseExpectation | NotGradable:
    """Pure. Rule lab-expectation-v2 over an already filtered ledger.

    1. Item consensus per (snapshot, item, phase); sign-off consensus per snapshot.
    2. accepted: a would_sign analysis, an expert correction before an agent answer, then the one
       signed last. rejected: every would_not_sign analysis. Neither: not gradable.
    3. Phases of the accepted analysis: supported yes and plausible not no -> required; disputed
       -> free; judged no while signed -> free(inconsistent); otherwise free(unassessable).
    4. A no on a fit is evidence about that fit, not a root-wide exclusion: a phase absent from
       the accepted analysis and judged no elsewhere is free(negative_in_one_fit), and naming it
       needs review. A no on another fit of a required phase, or a yes and a no on two fits, is
       free(disputed). No rule here produces forbidden.
    5. Amounts are graded only on a quantitative accepted fit with amounts_defensible yes and a
       policy tolerance. 6. Reference stance per phase. 7. Provenance from record exposure and
       the correction authors' reference exposure; require_blind fails closed on either.
    """
    if not ledger.labels and not ledger.signoffs:
        return NotGradable(root=ledger.root, reason="no_counted_judgment")
    records = (*ledger.labels, *ledger.signoffs)
    seen = any(r.exposure != "blind" for r in records) or bool(set(ledger.exposed) & {r.snapshot for r in records})
    if policy.require_blind and seen:
        return NotGradable(root=ledger.root, reason="not_blind")
    snaps = {snapshot_digest(s): s for s in ledger.snapshots}
    items = item_consensus(ledger.labels, policy.min_raters)
    signs = signoff_consensus(ledger.signoffs, policy.min_raters)
    judged = [d for d in sorted(snaps) if snaps[d].kind != "reference" and d in signs]
    signed = [d for d in judged if signs[d] == "would_sign"]
    refused = [d for d in judged if signs[d] == "would_not_sign"]
    if not signed and not refused:
        reason = "signoff_disputed" if any(signs[d] == "disputed" for d in judged) else "no_resolved_analysis"
        return NotGradable(root=ledger.root, reason=reason)
    last_signed = {d: max(s.created_at_utc for s in ledger.signoffs if s.snapshot == d) for d in signed}
    accepted = max(signed, key=lambda d: (snaps[d].kind == "expert_correction", last_signed[d], d)) if signed else None
    phase_items = ("phase_supported", "phase_plausible")
    if accepted and not any(x.snapshot == accepted and x.item in phase_items for x in ledger.labels):
        return NotGradable(root=ledger.root, reason="no_phase_judgments")

    def keyed(digest: Digest) -> dict[PhaseKey, SnapshotPhase]:
        return {phase_key(p, policy.phase_identity): p for p in snaps[digest].phases}

    def stance(digest: Digest, phase: SnapshotPhase) -> tuple[Consensus | None, Consensus | None]:
        sup = items.get((digest, "phase_supported", phase.phase_id))
        pla = items.get((digest, "phase_plausible", phase.phase_id))
        return (sup.consensus if sup else None, pla.consensus if pla else None)

    amounts_item = items.get((accepted, "amounts_defensible", None)) if accepted else None
    amounts_graded = (
        accepted is not None
        and policy.amount_abs_wt_pct is not None
        and snaps[accepted].quantification == "quantitative"
        and amounts_item is not None
        and amounts_item.consensus == "yes"
    )
    accepted_phases = keyed(accepted) if accepted else {}

    decided: dict[PhaseKey, tuple[str, str]] = {}
    for key, phase in accepted_phases.items():
        sup, pla = stance(accepted, phase)  # type: ignore[arg-type]
        if "disputed" in (sup, pla):
            decided[key] = ("free", "disputed")
        elif "no" in (sup, pla):
            decided[key] = ("free", "inconsistent")
        elif sup == "yes":
            decided[key] = ("required", "supported_in_accepted")
        else:
            decided[key] = ("free", "unassessable")

    members: dict[PhaseKey, list[SnapshotPhase]] = defaultdict(list)
    elsewhere: dict[PhaseKey, list[tuple[Consensus | None, Consensus | None]]] = defaultdict(list)
    for digest in sorted(snaps):
        for key, phase in keyed(digest).items():
            members[key].append(phase)
            if digest != accepted:
                elsewhere[key].append(stance(digest, phase))
    for key, stances in elsewhere.items():
        sups, plas = {s for s, _ in stances}, {p for _, p in stances}
        said_no = "no" in sups or "no" in plas
        if key in decided:
            if said_no and decided[key][0] == "required":
                decided[key] = ("free", "disputed")
        elif said_no and "yes" in sups:
            decided[key] = ("free", "disputed")
        elif said_no:
            decided[key] = ("free", "negative_in_one_fit")
        elif "disputed" in sups or "disputed" in plas:
            decided[key] = ("free", "disputed")
        elif "yes" in sups:
            decided[key] = ("free", "outside_accepted")
        elif any(s is not None or p is not None for s, p in stances):
            decided[key] = ("free", "unassessable")
        # never judged on any snapshot: no expectation, so naming it is needs_review

    def reference_stance(key: PhaseKey, expect: str) -> ReferenceStance:
        if reference is None:
            return "absent"
        named = any(_names(r, members[key]) for r in reference.phases)
        expected = key in accepted_phases if accepted else expect != "forbidden"
        return "agrees" if named == expected else "disputes"

    phases = tuple(
        PhaseExpectation(
            candidate_id=(accepted_phases[key] if key in accepted_phases else members[key][0]).phase_id,
            key=key,
            expect=expect,  # type: ignore[arg-type]
            because=because,  # type: ignore[arg-type]
            weight_percent=accepted_phases[key].weight_percent if amounts_graded and expect == "required" else None,
            reference=reference_stance(key, expect),
        )
        for key, (expect, because) in sorted(decided.items())
    )

    def judged_snapshot(digest: Digest) -> JudgedSnapshot:
        mine = [r for r in (*ledger.labels, *ledger.signoffs) if r.snapshot == digest]
        exposures = {r.exposure for r in mine}
        return JudgedSnapshot(
            snapshot=digest,
            kind=snaps[digest].kind,
            phases=tuple(p.phase_id for p in snaps[digest].phases),
            items=tuple(c for t, c in sorted(items.items(), key=lambda kv: (kv[0][1], kv[0][2] or "")) if t[0] == digest),
            signoff=signs.get(digest, "none"),
            exposure="mixed" if len(exposures) > 1 else exposures.pop(),  # type: ignore[arg-type]
        )

    counted = {r.snapshot for r in records}
    kinds = {r.rater.kind for r in records}
    who = "mixed" if len(kinds) > 1 else kinds.pop()
    return CaseExpectation(
        root=ledger.root,
        case_id=case_id,
        provenance=f"lab_{who}_{'revealed' if seen else 'blind'}",  # type: ignore[arg-type]
        accepted=accepted,
        rejected=tuple(
            RejectedAnalysis(
                snapshot=d,
                kind=snaps[d].kind,
                keys=tuple(sorted(keyed(d))),
            )
            for d in refused
        ),
        phases=phases,
        amounts_graded=amounts_graded,
        judged=tuple(judged_snapshot(d) for d in sorted(counted)),
        source_records=source_records(ledger),
    )


# ---------------------------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------------------------


class Producer(Record):
    """Anything that answers a case: an agent (model + harness), the pipeline, or imported
    software output. Matched on the import's RecordedAgent."""

    kind: Literal["producer"] = "producer"
    model_id: str = Field(min_length=1)
    harness: SafeID
    harness_version: str = Field(min_length=1)


class CrossBuildBounds(Record):
    """How far a new BGMN build or parser may move an accepted expert fit and still pass. No
    defaults and no instance: U4_BOUNDS measured one build (design Q6)."""

    version: SafeID
    justified_by: str = Field(min_length=1)
    rwp_abs_points: float = Field(gt=0)
    weight_abs_wt_pct: float = Field(gt=0)
    cell_length_rel: float = Field(gt=0)


class Refiner(Record):
    """The installed BGMN build and parser, re-running every accepted expert refit."""

    kind: Literal["refiner"] = "refiner"
    bounds: CrossBuildBounds


class Triager(Record):
    """A routing policy, graded on the decisions Triage.sweep published under it for the set's
    signed agent answers, by triage.grade_routing against the counted sign-offs that reach the
    lab's min_raters quorum. Eligibility rests on sign-offs, not phase expectations. To back-test a
    candidate policy, sweep with it first. Nothing here routes or grades routes a second way."""

    kind: Literal["triager"] = "triager"
    policy: CheckPolicy


class TunedOn(Record):
    """One partition a prompt revision was tuned against. Declared when the prompt changes."""

    set_digest: Digest
    partition: Partition


class JudgeVersion(Record):
    """A model judge as a versioned subject. `tuned_on` grows each time the prompt is revised
    against a partition; Qualifier.run refuses to score it on a held_out root that sits in a
    partition it was tuned on, in this set or any other."""

    kind: Literal["judge"] = "judge"
    model: str = Field(min_length=1)
    prompt_sha256: Digest
    tuned_on: tuple[TunedOn, ...]


Subject = Annotated[Producer | Refiner | Triager | JudgeVersion, Field(discriminator="kind")]

# ---------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------

CaseVerdict = Literal[
    "pass",
    "fail",
    "needs_review",  # the subject named a phase no counted rater judged: route it to an expert
    "no_answer",  # the producer ran and wrote no answer (abstention, timeout)
    "not_run",  # no observation for this root yet
    "not_gradable",  # the expectation is NotGradable, or the root is outside the graded partition
]
SideVerdict = Literal["pass", "fail", "needs_review"]


class CaseResult(Record):
    root: AttemptRef
    partition: Partition | None
    expectation_sha256: Digest | None
    observed: Digest | None  # the graded snapshot, or the digest of the judge's labels on the root
    verdict: CaseVerdict
    # Producer only. accepted_side: required present, forbidden absent, amounts within
    # tolerance; None on a rejected-only root. rejected_side: repeats no rejected analysis; None
    # when the root has none. An abstention fails the accepted side and passes the rejected side.
    accepted_side: SideVerdict | None = None
    rejected_side: SideVerdict | None = None
    # Producer only: the sides the expectation defines, whatever happened to the answer, so a
    # not_run root still counts as applicable and unassessed.
    applicable: tuple[Literal["accepted", "rejected"], ...] = ()
    # Stable codes: "missing_required:CIF012075", "forbidden_present:CIF001772",
    # "repeats_rejected_analysis:<snapshot>", "unjudged_extra:CIF005806",
    # "amount_outside:CIF001783:+1.21", "disputed_skipped:CIF001772",
    # "reference_disputes:CIF001772", "records_changed", "tune_partition", "disagrees:...".
    reasons: tuple[str, ...]


class Interval(Record):
    k: int
    n: int
    rate: float | None  # None when n == 0
    wilson95: tuple[float, float] | None


def interval(k: int, n: int) -> Interval:
    if n == 0:
        return Interval(k=0, n=0, rate=None, wilson95=None)
    z, p = 1.959963984540054, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Interval(k=k, n=n, rate=p, wilson95=(max(0.0, centre - half), min(1.0, centre + half)))


def _by_verdict(cases: Iterable[CaseResult]) -> dict[CaseVerdict, int]:
    counts: dict[CaseVerdict, int] = defaultdict(int)
    for case in cases:
        counts[case.verdict] += 1
    return dict(counts)


def _pass_rate(sides: Iterable[str | None]) -> Interval:
    decided = [s for s in sides if s in ("pass", "fail")]
    return interval(decided.count("pass"), len(decided))


class ProducerSummary(Record):
    """Two rates, never pooled. A subject that always abstains scores 0 on the accepted side and
    1 on the rejected side, so neither rate alone can hide it. Every rate is conditional: its n
    counts only the assessed (pass or fail) roots. Show it beside `roots`, the side's applicable
    count and by_verdict; 1 pass beside 9 needs_review is 1/1 assessed of 10 applicable, never
    a 100% qualification."""

    kind: Literal["producer"] = "producer"
    roots: int  # every root of the set
    accepted_applicable: int  # roots with an accepted analysis
    rejected_applicable: int  # roots with a rejected analysis
    by_verdict: Mapping[CaseVerdict, int]
    pass_rate_of_gradable: Interval  # pass / (pass + fail); needs_review and no_answer beside it
    accepted_pass_rate: Interval  # accepted_side pass / (pass + fail), abstentions as fail
    rejected_pass_rate: Interval  # rejected_side pass / (pass + fail), abstentions as pass


def summarize_producer(cases: Iterable[CaseResult]) -> ProducerSummary:
    rows = list(cases)
    return ProducerSummary(
        roots=len(rows),
        accepted_applicable=sum("accepted" in r.applicable for r in rows),
        rejected_applicable=sum("rejected" in r.applicable for r in rows),
        by_verdict=_by_verdict(rows),
        pass_rate_of_gradable=_pass_rate(r.verdict for r in rows),
        accepted_pass_rate=_pass_rate(r.accepted_side for r in rows),
        rejected_pass_rate=_pass_rate(r.rejected_side for r in rows),
    )


class ItemAgreement(Record):
    """Periodic's comparison per rubric item on held_out roots: the judge against each expert next
    to expert against expert, and TPR/TNR against the expert consensus. Agreement and TPR/TNR are
    conditional on the judge answering; read them beside coverage. The intervals are descriptive:
    pairs share snapshots, raters and runs, so they are not independent trials."""

    item: RubricItemId
    eligible: int  # targets on held_out roots where at least one counted expert said yes or no
    coverage: Interval  # eligible targets the judge answered yes or no; an abstaining judge reads 0
    n: int  # judge yes/no answers on targets where at least one counted expert assessed
    judge_expert_pairwise: Interval
    expert_expert_pairwise: Interval | None  # None with fewer than two assessing experts anywhere
    tpr: Interval  # judge yes | consensus yes; disputed and unassessable excluded
    tnr: Interval  # judge no | consensus no
    judge_abstained: int  # judge cannot_assess where an expert committed: counted, never an error


class JudgeSummary(Record):
    kind: Literal["judge"] = "judge"
    partition: Literal["held_out"] = "held_out"
    items: tuple[ItemAgreement, ...]
    judge_is_actor_model_excluded: int  # labels set apart because the judge produced the episode


class TriagerSummary(Record):
    """triage.grade_routing over every gradable root of the set: its counts, listed misses and
    recall estimate (withheld below RECALL_MIN_AUDIT_REVIEWS). Sign-offs collected while the page
    shows K statuses are router-exposed; see RoutingGrade."""

    kind: Literal["triager"] = "triager"
    grade: RoutingGrade


Summary = Annotated[ProducerSummary | JudgeSummary | TriagerSummary, Field(discriminator="kind")]


class QualificationReport(Record):
    schema_version: Literal["qualification-report.v2"] = "qualification-report.v2"
    set_digest: Digest
    subject: Subject
    claim: Literal["agreement_with_lab_judgments"] = "agreement_with_lab_judgments"
    rule: Literal["lab-expectation-v2"] = EXPECTATION_RULE
    grader: SafeID = GRADER_VERSION
    expectations_sha256: Digest  # over the canonical expectations, in set order
    cases: tuple[CaseResult, ...]  # one per root of the set, in set order
    # Producer: every matching episode graded, oldest first per root. `cases` grades the latest.
    episodes: tuple[CaseResult, ...] = ()
    summary: Summary


# ---------------------------------------------------------------------------------------------
# Pure graders. Public: a model lab runs grade_answer on its own outputs (C2 export).
# ---------------------------------------------------------------------------------------------


def grade_answer(
    expectation: CaseExpectation | NotGradable, answer: AnalysisSnapshot | None, policy: QualificationPolicy
) -> CaseResult:
    sha = record_sha256(expectation)
    base = {"root": expectation.root, "partition": None, "expectation_sha256": sha}
    if isinstance(expectation, NotGradable):
        return CaseResult(**base, observed=None, verdict="not_gradable", reasons=(expectation.reason,))
    accepted = expectation.accepted is not None
    rejected = bool(expectation.rejected)
    base["applicable"] = (*(("accepted",) if accepted else ()), *(("rejected",) if rejected else ()))
    if answer is None:
        return CaseResult(**base, observed=None, verdict="not_run", reasons=())
    observed = snapshot_digest(answer)
    if not answer.answered:
        return CaseResult(
            **base,
            observed=observed,
            verdict="no_answer",
            accepted_side="fail" if accepted else None,
            rejected_side="pass" if rejected else None,
            reasons=(),
        )
    given = {phase_key(p, policy.phase_identity): p for p in answer.phases}
    tolerance = policy.amount_abs_wt_pct
    misses: list[str] = []
    forbidden: list[str] = []
    reviews: list[str] = []  # both sides: a phase a rater judged no on one fit only
    notes: list[str] = []
    for phase in expectation.phases:
        present = given.get(phase.key)
        if phase.expect == "required" and present is None:
            misses.append(f"missing_required:{phase.candidate_id}")
        elif phase.expect == "forbidden" and present is not None:
            forbidden.append(f"forbidden_present:{present.phase_id}")
        elif phase.expect == "free" and present is not None and phase.because == "negative_in_one_fit":
            reviews.append(f"negative_in_one_fit:{present.phase_id}")
        elif phase.expect == "free" and present is not None:
            notes.append(f"{phase.because}_skipped:{present.phase_id}")
        if phase.reference == "disputes":
            notes.append(f"reference_disputes:{phase.candidate_id}")
        if phase.weight_percent is not None and tolerance is not None and present is not None:
            if answer.quantification != "quantitative" or present.weight_percent is None:
                misses.append(f"amount_not_comparable:{present.phase_id}")
            elif abs(present.weight_percent - phase.weight_percent) > tolerance:
                misses.append(f"amount_outside:{present.phase_id}:{present.weight_percent - phase.weight_percent:+.2f}")
    # The rejected analysis itself, presented again. The accepted analysis itself is never
    # reviewed against a rejection it shares phases with: the raters signed it.
    repeats = [f"repeats_rejected_analysis:{r.snapshot}" for r in expectation.rejected if observed == r.snapshot]
    same_phases = [
        f"same_phases_as_rejected:{r.snapshot}"
        for r in expectation.rejected
        if observed not in (r.snapshot, expectation.accepted) and set(given) == set(r.keys)
    ]
    # An accepted analysis with no required phase constrains nothing, so it cannot pass anything.
    unconstrained = ["no_required_phase"] if accepted and not any(p.expect == "required" for p in expectation.phases) else []
    known = {p.key for p in expectation.phases}
    extras = [f"unjudged_extra:{p.phase_id}" for key, p in sorted(given.items()) if key not in known]

    def side(failures: list[str], review: list[str]) -> SideVerdict:
        return "fail" if failures else "needs_review" if review or extras else "pass"

    accepted_side = side(misses + forbidden, reviews + unconstrained) if accepted else None
    rejected_side = side(repeats + forbidden, reviews + same_phases) if rejected else None
    sides = (accepted_side, rejected_side)
    verdict: CaseVerdict = "fail" if "fail" in sides else "needs_review" if "needs_review" in sides else "pass"
    return CaseResult(
        **base,
        observed=observed,
        verdict=verdict,
        accepted_side=accepted_side,
        rejected_side=rejected_side,
        reasons=(
            *sorted(misses + forbidden),
            *repeats,
            *sorted(reviews + same_phases + unconstrained),
            *extras,
            *sorted(set(notes)),
        ),
    )


class NoFrozenSplit(ValueError):
    """Judge calibration over a set frozen without a split."""


class TunedOnHeldOut(ValueError):
    """The judge was tuned on a partition that shares a root with this set's held_out roots."""


def grade_judge(
    judge: JudgeVersion,
    judge_labels: Iterable[Label],
    expert_labels: Iterable[Label],
    expectations: Mapping[AttemptRef, CaseExpectation | NotGradable],
    held_out: frozenset[AttemptRef],
) -> tuple[tuple[CaseResult, ...], JudgeSummary]:
    """One CaseResult per root of `expectations`, in its order; agreement on held_out roots only.
    The caller has refused tuned-on roots (Qualifier.run). Only labels under this judge's model
    and prompt count. cannot_assess on the expert side drops the pair; on the judge side it counts
    as judge_abstained."""
    experts = _HUMAN_LABELS.validate_python(tuple(expert_labels))
    mine = [
        x
        for x in judge_labels
        if x.rater.judge is not None and (x.rater.judge.model, x.rater.judge.prompt_sha256) == (judge.model, judge.prompt_sha256)
    ]
    stats: dict[RubricItemId, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    actor_model = 0
    answered: dict[RubricItemId, set[Target]] = defaultdict(set)
    cases: list[CaseResult] = []
    for root, expectation in expectations.items():
        base = {"root": root, "partition": "held_out" if root in held_out else "tune", "expectation_sha256": record_sha256(expectation)}
        if isinstance(expectation, NotGradable):
            cases.append(CaseResult(**base, observed=None, verdict="not_gradable", reasons=(expectation.reason,)))
            continue
        if root not in held_out:
            cases.append(CaseResult(**base, observed=None, verdict="not_gradable", reasons=("tune_partition",)))
            continue
        snaps = {j.snapshot for j in expectation.judged}
        consensus = {(j.snapshot, c.item, c.phase_id): c.consensus for j in expectation.judged for c in j.items}
        by_target = latest_labels(x for x in experts if x.snapshot in snaps)
        for (_, item, _), by_rater in by_target.items():
            assessing = [x.answer for x in by_rater.values() if x.answer != "cannot_assess"]
            for a, b in combinations(assessing, 2):
                stats[item]["ee_n"] += 1
                stats[item]["ee_k"] += a == b
        runs: dict[tuple[Target, str, str], Label] = {}
        for x in sorted((x for x in mine if x.snapshot in snaps), key=lambda r: (r.created_at_utc, record_sha256(r))):
            runs[((x.snapshot, x.item, x.phase_id), x.rater.principal_id, x.rater.judge.run_id)] = x  # type: ignore[union-attr]
        for target, by_rater in by_target.items():
            if any(x.answer != "cannot_assess" for x in by_rater.values()):
                stats[target[1]]["eligible"] += 1
        if not runs:
            cases.append(CaseResult(**base, observed=None, verdict="not_run", reasons=()))
            continue
        wrong: list[str] = []
        compared = 0
        for (target, _, _), x in runs.items():
            s = stats[x.item]
            if x.rater.judge.judge_is_actor_model:  # type: ignore[union-attr]
                actor_model += 1
                continue
            assessing = [e.answer for e in by_target.get(target, {}).values() if e.answer != "cannot_assess"]
            if not assessing:
                continue
            if x.answer == "cannot_assess":
                s["abstained"] += 1
                continue
            s["n"] += 1
            answered[x.item].add(target)
            s["je_n"] += len(assessing)
            s["je_k"] += assessing.count(x.answer)
            truth = consensus.get(target)
            if truth in ("yes", "no"):
                compared += 1
                s[f"{truth}_n"] += 1
                s[f"{truth}_k"] += x.answer == truth
                if x.answer != truth:
                    wrong.append(f"disagrees:{x.item}:{x.phase_id or '-'}:{target[0][:12]}")
        observed = _sha256("\n".join(sorted(record_sha256(x) for x in runs.values())).encode())
        verdict: CaseVerdict = "fail" if wrong else "pass" if compared else "not_gradable"
        reasons = tuple(sorted(set(wrong))) if compared else ("no_decided_target",)
        cases.append(CaseResult(**base, observed=observed, verdict=verdict, reasons=reasons))
    items = tuple(
        ItemAgreement(
            item=item,
            eligible=s["eligible"],
            coverage=interval(len(answered[item]), s["eligible"]),
            n=s["n"],
            judge_expert_pairwise=interval(s["je_k"], s["je_n"]),
            expert_expert_pairwise=interval(s["ee_k"], s["ee_n"]) if s["ee_n"] else None,
            tpr=interval(s["yes_k"], s["yes_n"]),
            tnr=interval(s["no_k"], s["no_n"]),
            judge_abstained=s["abstained"],
        )
        for item in ("phase_supported", "phase_plausible", "amounts_defensible", "fit_supports_conclusion")
        for s in (stats[item],)
    )
    return tuple(cases), JudgeSummary(items=items, judge_is_actor_model_excluded=actor_model)


# ---------------------------------------------------------------------------------------------
# The facade
# ---------------------------------------------------------------------------------------------


class Qualifier:
    """The only writer of qualification-sets/ and qualification-reports/. Reads the attempt tree,
    the label ledger and the sealed reference as a service; never serves any of it to a rater.
    `reference` is the custody reader on the owner's host and NoReference everywhere else."""

    def __init__(self, store: WorkbenchStore, labels: LabelStore, reference: ReferenceReader) -> None:
        self._store = store
        self._labels = labels
        self._reference = reference
        with store.ledger.transaction():
            if not hasattr(store, '_qualification_collections'):
                store._qualification_collections = (
                    store.ledger.collection('qualification-sets', QualificationSet, Mode.ONCE, KeyShape(('hex64',)), owner=__name__),
                    store.ledger.collection('qualification-reports', QualificationReport, Mode.ONCE, KeyShape(('hex64',)), owner=__name__),
                )
        self._sets, self._reports = store._qualification_collections

    def freeze(
        self,
        principal: Principal,
        *,
        lab: SafeID,
        name: SafeID,
        roots: tuple[AttemptRef, ...],
        policy: QualificationPolicy,
        split_seed: SafeID | None,
    ) -> Digest:
        """Lab owner only, on roots it can inspect. Pins each root's counted records and writes
        the set once; the same request over the same ledger returns the same digest. A root with
        no counted judgment is accepted and stays in every report."""
        self._owner(principal, roots)
        ledgers, _ = self._ledgers(policy, None, "internal_evaluation")
        times = [r.created_at_utc for root in roots for r in (*_ledger(ledgers, root).labels, *_ledger(ledgers, root).signoffs)]
        qset = QualificationSet(
            lab=lab,
            name=name,
            roots=roots,
            policy=policy,
            cut_utc=max(times, default=datetime(1970, 1, 1, tzinfo=UTC)),
            pins={root: pin_of(_ledger(ledgers, root)) for root in roots},
            split={root: partition_of(split_seed, root) for root in roots} if split_seed else None,
            split_seed=split_seed,
            frozen_by=principal.principal_id,
        )
        data = canonical_bytes(qset)
        digest = _sha256(data)
        self._sets.put((digest,), qset)
        return digest

    def qualification_set(self, principal: Principal, set_digest: Digest) -> QualificationSet:
        qset = self._read_set(set_digest)
        self._owner(principal, qset.roots)
        return qset

    def expectations(
        self, principal: Principal, set_digest: Digest, use: ExportUse = "internal_evaluation"
    ) -> Mapping[AttemptRef, CaseExpectation | NotGradable]:
        """Derived on read under the set's cut, rater kinds and each rater's latest permission for
        `use`. Every use other than internal_evaluation derives with NoReference, so no
        reference stance leaves the lab."""
        qset = self.qualification_set(principal, set_digest)
        ledgers, _ = self._ledgers(qset.policy, qset.cut_utc, use)
        reader = self._reference if use == "internal_evaluation" else NoReference()
        return self._expect(qset, ledgers, reader)

    def run(self, principal: Principal, set_digest: Digest, subject: Subject) -> QualificationReport:
        """Grade `subject` over every root of the set and publish the report once.

        Producer: per root, every agent_answer snapshot whose import's RecordedAgent matches;
          the latest grades the root and every one is listed in `episodes`. Running the producer
          is outside this layer (`plan` lists roots with no observation).
        JudgeVersion: NoFrozenSplit without a split; TunedOnHeldOut when a held_out root of this
          set sits in a partition the judge was tuned on.
        Triager: each root's judged agent answers, with the decisions published under the policy,
          graded by triage.grade_routing; a root fails on a listed miss or a false alarm and is
          not_run while any judged answer has no decision.
        Refiner raises NotImplementedError naming what it waits for.
        """
        qset = self.qualification_set(principal, set_digest)
        if isinstance(subject, Refiner):
            raise NotImplementedError(
                "Refiner needs CrossBuildBounds justified by a spread measured across two BGMN "
                "builds (design Q6); U4_BOUNDS measured one build"
            )
        if isinstance(subject, JudgeVersion):
            self._refuse_leaks(qset, subject)
        ledgers, judge_labels = self._ledgers(qset.policy, qset.cut_utc, "internal_evaluation")
        expectations = self._expect(qset, ledgers, self._reference)
        episodes: tuple[CaseResult, ...] = ()
        summary: ProducerSummary | JudgeSummary | TriagerSummary
        if isinstance(subject, Triager):
            cases, summary = self._triage(principal, qset, ledgers, expectations, subject.policy)
        elif isinstance(subject, Producer):
            graded = [self._produce(qset, expectations[root], subject) for root in qset.roots]
            cases = tuple(latest for latest, _ in graded)
            episodes = tuple(e for _, every in graded for e in every)
            summary = summarize_producer(cases)
        else:
            held_out = frozenset(r for r, p in (qset.split or {}).items() if p == "held_out")
            experts = tuple(x for ledger in ledgers.values() for x in ledger.labels)
            ordered = {root: expectations[root] for root in qset.roots}
            cases, summary = grade_judge(subject, judge_labels, experts, ordered, held_out)
        if qset.split is not None:
            cases = tuple(c.model_copy(update={"partition": qset.split[c.root]}) for c in cases)
        report = QualificationReport(
            set_digest=set_digest,
            subject=subject,
            expectations_sha256=_sha256(b"".join(canonical_bytes(expectations[root]) for root in qset.roots)),
            cases=cases,
            episodes=episodes,
            summary=summary,
        )
        data = canonical_bytes(report)
        self._reports.put((_sha256(data),), report)
        return report

    def plan(self, principal: Principal, set_digest: Digest, subject: Producer) -> tuple[AttemptRef, ...]:
        """Roots of the set with no observation for this producer: the launcher's work list."""
        qset = self.qualification_set(principal, set_digest)
        return tuple(root for root in qset.roots if not self._answers(root, subject))

    # ----- internals ---------------------------------------------------------------------------

    def _owner(self, principal: Principal, roots: Iterable[AttemptRef]) -> None:
        Access.for_principal(principal).require(Op.SCORECARD)
        for root in roots:
            self._store._authorize(principal, root, CaseOperation.INSPECT)

    def _read_set(self, set_digest: Digest) -> QualificationSet:
        if not re.fullmatch(r"[0-9a-f]{64}", set_digest):
            raise KeyError(f"no qualification set {set_digest[:12]}")
        data = self._sets.get_bytes((set_digest,))
        if data is None:
            raise KeyError(f"no qualification set {set_digest[:12]}")
        if _sha256(data) != set_digest:
            raise CustodyError("qualification set bytes disagree with their digest")
        return QualificationSet.model_validate_json(data)

    def _refuse_leaks(self, qset: QualificationSet, judge: JudgeVersion) -> None:
        if qset.split is None:
            raise NoFrozenSplit("judge calibration needs a set frozen with a split")
        held_out = {r for r, p in qset.split.items() if p == "held_out"}
        for tuned in judge.tuned_on:
            other = self._read_set(tuned.set_digest)
            seen = {r for r, p in (other.split or {}).items() if p == tuned.partition}
            if held_out & seen:
                raise TunedOnHeldOut(
                    f"judge prompt {judge.prompt_sha256[:12]} was tuned on {tuned.partition} of set "
                    f"{tuned.set_digest[:12]}, which shares {len(held_out & seen)} root(s) with this held_out"
                )

    def _ledgers(
        self, policy: QualificationPolicy, cut: datetime | None, use: ExportUse
    ) -> tuple[dict[AttemptRef, RootLedger], tuple[Label, ...]]:
        """Counted human records per root and the model-judge labels, both before the cut. The
        permission filter is LabelStore.export_records: a withdrawn rater's records are absent."""
        labels, _, signoffs, _ = self._labels.export_records(use)
        snapshots = dict(self._labels._snapshots())

        def before(record: Label | SignOff) -> bool:
            return cut is None or record.created_at_utc <= cut

        by_root: dict[str, tuple[list[Label], list[SignOff]]] = defaultdict(lambda: ([], []))
        for label in labels:
            if label.rater.kind in policy.rater_kinds and before(label):
                by_root[snapshots[label.snapshot].root][0].append(label)
        for signoff in signoffs:
            if signoff.rater.kind in policy.rater_kinds and before(signoff):
                by_root[signoff.root][1].append(signoff)
        def exposed(snapshot: AnalysisSnapshot) -> bool:
            return any(
                isinstance(a, ChildAttempt) and isinstance(a.actor, HumanActor) and a.actor.reference_exposure == "exposed"
                for _, a in self._store._path(snapshot.origin)
            )

        ledgers = {
            root: RootLedger(
                root=root,
                snapshots=tuple(s for _, s in sorted(snapshots.items()) if s.root == root),
                labels=tuple(found[0]),
                signoffs=tuple(found[1]),
                exposed=tuple(d for d, s in sorted(snapshots.items()) if s.root == root and exposed(s)),
            )
            for root, found in by_root.items()
        }
        judges = tuple(x for x in labels if x.rater.kind == "model_judge" and before(x))
        return ledgers, judges

    def _expect(
        self, qset: QualificationSet, ledgers: Mapping[AttemptRef, RootLedger], reader: ReferenceReader
    ) -> dict[AttemptRef, CaseExpectation | NotGradable]:
        out: dict[AttemptRef, CaseExpectation | NotGradable] = {}
        for root in qset.roots:
            ledger = _ledger(ledgers, root)
            if pin_of(ledger) != qset.pins[root]:
                out[root] = NotGradable(root=root, reason="records_changed")
            else:
                case_id = self._store._path(root)[0][1].case.case_id
                out[root] = derive_expectation(ledger, case_id, qset.policy, reader.reference(case_id))
        return out

    def _answers(self, root: AttemptRef, producer: Producer) -> list[AnalysisSnapshot]:
        """The producer's agent_answer snapshots of the root, oldest import first."""
        found = []
        for _, snapshot in self._labels._snapshots():
            if snapshot.root != root or snapshot.kind != "agent_answer":
                continue
            attempt = self._store._load(snapshot.origin)
            if not isinstance(attempt, ChildAttempt) or not isinstance(attempt.outcome, Completed):
                continue
            result = attempt.outcome.result
            if isinstance(result, ImportedRunResult) and (
                result.agent.model_id,
                result.agent.harness,
                result.agent.harness_version,
            ) == (producer.model_id, producer.harness, producer.harness_version):
                found.append((attempt.created_at_utc, snapshot.origin, snapshot))
        return [snapshot for _, _, snapshot in sorted(found, key=lambda row: row[:2])]

    def _triage(
        self,
        principal: Principal,
        qset: QualificationSet,
        ledgers: Mapping[AttemptRef, RootLedger],
        expectations: Mapping[AttemptRef, CaseExpectation | NotGradable],
        policy: CheckPolicy,
    ) -> tuple[tuple[CaseResult, ...], TriagerSummary]:
        published = {d.snapshot: d for d in Triage(self._store, self._labels, policy).decisions(principal, policy.version)}
        cases: list[CaseResult] = []
        decisions: list[RouteDecision] = []
        signoffs: list[SignOff] = []
        for root in qset.roots:
            expectation = expectations[root]
            base = {"root": root, "partition": None, "expectation_sha256": record_sha256(expectation)}
            # Routing is graded on sign-offs, not on phase expectations: only the safeguards that
            # bind every grade (pin, blindness, a counted record) can make a root ineligible.
            if isinstance(expectation, NotGradable) and expectation.reason in _ROUTING_BLOCKERS:
                cases.append(CaseResult(**base, observed=None, verdict="not_gradable", reasons=(expectation.reason,)))
                continue
            ledger = _ledger(ledgers, root)
            kinds = {snapshot_digest(s): s.kind for s in ledger.snapshots}
            signed = {x.snapshot for x in ledger.signoffs if kinds[x.snapshot] == "agent_answer"}
            unrouted = tuple(f"unrouted:{d}" for d in sorted(signed) if d not in published)
            if unrouted:
                cases.append(CaseResult(**base, observed=None, verdict="not_run", reasons=unrouted))
                continue
            # Every decision on the root; one without a counted sign-off stays unreviewed.
            routed = tuple(d for _, d in sorted(published.items()) if d.root == root)
            if not routed:
                cases.append(CaseResult(**base, observed=None, verdict="not_gradable", reasons=("no_routed_agent_answer",)))
                continue
            # One sign-off population for quorum and grader: triage counts blind sign-offs only, so
            # a revealed one cannot supply quorum either. A sign-off counts only on a snapshot whose
            # blind raters reach min_raters (or disagree); below it the decision stays unreviewed.
            blind = tuple(x for x in ledger.signoffs if x.exposure == "blind")
            quorum = signoff_consensus(blind, qset.policy.min_raters)
            counted = tuple(x for x in blind if quorum[x.snapshot] != "unresolved")
            grade = grade_routing(routed, counted, policy, min_audit_reviews=RECALL_MIN_AUDIT_REVIEWS)
            reviewed = sum(t.wrong + t.right for t in grade.by_route.values())
            wrong = (*(f"missed:{m.snapshot}" for m in grade.misses), *((f"false_alarms:{grade.false_alarms}",) if grade.false_alarms else ()))
            observed = _sha256(b"".join(canonical_bytes(d) for d in routed))
            verdict: CaseVerdict = "not_gradable" if not reviewed else "fail" if wrong else "pass"
            cases.append(CaseResult(**base, observed=observed, verdict=verdict, reasons=wrong if reviewed else ("no_reviewed_decision",)))
            decisions.extend(routed)
            signoffs.extend(counted)
        overall = grade_routing(decisions, signoffs, policy, min_audit_reviews=RECALL_MIN_AUDIT_REVIEWS)
        return tuple(cases), TriagerSummary(grade=overall)

    def _produce(
        self, qset: QualificationSet, expectation: CaseExpectation | NotGradable, producer: Producer
    ) -> tuple[CaseResult, tuple[CaseResult, ...]]:
        answers = self._answers(expectation.root, producer)
        every = tuple(grade_answer(expectation, a, qset.policy) for a in answers)
        return (every[-1] if every else grade_answer(expectation, None, qset.policy)), every


# NotGradable reasons that also bar routing grades. The others concern phase expectations.
_ROUTING_BLOCKERS = frozenset({"no_counted_judgment", "not_blind", "records_changed"})


def _ledger(ledgers: Mapping[AttemptRef, RootLedger], root: AttemptRef) -> RootLedger:
    found = ledgers.get(root)
    return found if found is not None else RootLedger(root=root, snapshots=(), labels=(), signoffs=())


__all__ = [
    "CaseExpectation",
    "CaseResult",
    "CrossBuildBounds",
    "HumanLabel",
    "Interval",
    "ItemAgreement",
    "JudgeSummary",
    "JudgeVersion",
    "NoFrozenSplit",
    "NotGradable",
    "PhaseExpectation",
    "Producer",
    "ProducerSummary",
    "QualificationPolicy",
    "QualificationReport",
    "QualificationSet",
    "Qualifier",
    "Refiner",
    "RejectedAnalysis",
    "RootLedger",
    "Subject",
    "Triager",
    "TunedOn",
    "TriagerSummary",
    "TunedOnHeldOut",
    "derive_expectation",
    "grade_answer",
    "grade_judge",
    "partition_of",
    "phase_key",
    "pin_of",
    "record_sha256",
    "snapshot_digest",
    "summarize_producer",
]
