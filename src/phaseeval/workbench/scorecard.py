"""Episode scorecard and RL view: every agent episode scored by workflow step, the failure modes
behind each failed step, a declared reward spec and the explicit preference pairs
(AXA-43zwk.11; owner request 2026-10-22: "how do we easily eval what the agent does right and
what it does wrong").

Target path: src/phaseeval/workbench/scorecard.py (public). Owner only. GET /scorecard serves the page
shell (assets/scorecard.html, no data). GET /api/attempts/scorecard returns the view and its
rendered tables, and GET /api/attempts/scorecard/pairs.jsonl the review-eligible pairs, to the admin
owner only: the view names models, jobs and the published reference, so label-mode, rater,
agent, judge and service credentials get 403. Like the dashboard it stores nothing; every number
is recomputed from the files exports.write_dataset wrote, plus the custody reference that the
reveal route reads (snapshot._reference), so the page cannot disagree with the export.

The data shape is one EpisodeScore per agent episode: the model, case and ending, and one
StepOutcome per workflow step in STEPS. A step outcome is pass, fail or unknown, decided by the
first source in its StepDefinition that reads pass or fail; every other reading stays on the
outcome. A mechanical fact (a submission, amounts copied from the fit, no residual peak) decides
only on fail: it never stands in for the support or defensibility a step asks about (CR15).
Raters of one kind who disagree read unknown. Sources:
- deterministic_check: K1, K2 and K5 frozen on the snapshot under its CheckPolicy, and whether
  the run ended with a submitted answer.
- reference_match: the case's published reference phase list (formula level, every polymorph
  counts). Rendered "matches published reference", never "correct": DEV006's reference omits a
  phase the DARA authors call correct.
- model_judge: initial blind model-judge labels. Never rendered as expert labels.
- expert_label: expert-kind labels and step feedback only. Owner labels are revealed demo labels
  and never count.

Failure codes are seeded from the S1 to S11 step legend of failures-trajectories.md
(osi/eval-direction cf6e18b44). Each failed step gets the code its deciding check implies. The
doc's own first-divergence reading of the 14 U6 runs is shown beside it as analyst coding: an
agent's custody-informed reading, not an expert label.

The RL view declares reward v0 and scores each episode under it. Preference pairs come only from
explicit preferences the export carries (review.Preference, R18), each with its exposure and
eligibility; only pairs eligible for review reach the JSONL, and none is marked trainable, since
training permission (rights, contributor and model-output terms) is not resolved here. A correction is shown as an unordered before
and after until someone records a preference, and a path that left the answer unchanged
(a refused or no-op edit) is counted, never listed (CR16). No model was trained on any of it.
"""

from __future__ import annotations

import html
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from phaseeval.api.auth import Principal
from phaseeval.api.routes.attempts import authenticate_principal

from . import snapshot as snapshot_module
from .exports import DatasetManifest, EpisodeRecord
from .labels import Label, SignOff, StepFeedback, is_admin_owner
from .model import (
    Completed,
    Hypothesis,
    ImportedRunResult,
    ImportedSubmitted,
    ImportRun,
    Record,
    digest_of,
)
from .snapshot import AnalysisSnapshot

ASSETS = Path(__file__).resolve().parents[1] / "api" / "assets"
DEFAULT_DATASET = Path.home() / ".local/state/axa-workbench/dataset-v0"

StepName = Literal[
    "phase_search", "phase_selection", "refinement", "quantification", "final_answer"
]
Source = Literal["expert_label", "reference_match", "deterministic_check", "model_judge"]
Outcome = Literal["pass", "fail", "unknown"]

STEPS: tuple[StepName, ...] = (
    "phase_search",
    "phase_selection",
    "refinement",
    "quantification",
    "final_answer",
)
SOURCES: tuple[Source, ...] = (
    "expert_label",
    "reference_match",
    "deterministic_check",
    "model_judge",
)
SOURCE_TEXT: Mapping[Source, str] = {
    "expert_label": "expert label",
    "reference_match": "published reference",  # a reading says "matches" or "does not match published reference"
    "deterministic_check": "deterministic check",
    "model_judge": "model judge",
}


# --- the step table ----------------------------------------------------------------------------


class Decider(Record):
    source: Source
    check: str
    rule: str  # what pass and fail mean, in words
    # "fail_only": a mechanical fact (a submission, amounts copied from the fit, no residual peak)
    # that can sink a step but never stands in for the support or defensibility it asks about.
    decides: Literal["pass_or_fail", "fail_only"] = "pass_or_fail"


class StepDefinition(Record):
    step: StepName
    title: str
    question: str
    deciders: tuple[Decider, ...]  # precedence order: the first reading of pass or fail decides
    trace_anchor: str  # the recorded call the trace opens at


EXPERT_STEP_RULE = "an expert-kind step-feedback mark on a call in this step: wrong move or went wrong here fails, good move passes"
STEP_DEFINITIONS: tuple[StepDefinition, ...] = (
    StepDefinition(
        step="phase_search",
        title="Phase search",
        question="Did the search reach every phase the sample holds?",
        deciders=(
            Decider(source="expert_label", check="step_feedback", rule=EXPERT_STEP_RULE),
            Decider(
                source="reference_match",
                check="reference.tried",
                rule="every published reference formula has a shipped candidate in at least one recorded refinement call"
                " (parsed from call summaries, so a miss may come from a truncated summary)",
            ),
        ),
        trace_anchor="the first candidate screen, else the first refinement",
    ),
    StepDefinition(
        step="phase_selection",
        title="Phase selection",
        question="Is the reported phase list right?",
        deciders=(
            Decider(
                source="expert_label",
                check="phase_supported",
                rule="each expert that answers says yes on every reported phase; experts who disagree read unknown",
            ),
            Decider(
                source="reference_match",
                check="reference.phase_list",
                rule="the answer's phase list equals the published reference list at formula level; every polymorph counts",
            ),
            Decider(
                source="deterministic_check",
                check="K2",
                rule="an unexplained residual peak at or above the policy height fails; no peak does not show the list is right",
                decides="fail_only",
            ),
            Decider(
                source="model_judge",
                check="phase_supported, phase_plausible",
                rule="each judge that answers says yes on every reported phase; judges that disagree read unknown",
            ),
        ),
        trace_anchor="the refinement whose candidates equal the answer, else the last refinement",
    ),
    StepDefinition(
        step="refinement",
        title="Refinement setup and convergence",
        question="Did the cited refinement run to a Rietveld fit without degrading?",
        deciders=(
            Decider(source="expert_label", check="step_feedback", rule=EXPERT_STEP_RULE),
            Decider(
                source="deterministic_check",
                check="K1",
                rule="the cited fit is a Rietveld stage and K1 passes (Rietveld did not degrade after a good Pawley fit)",
            ),
        ),
        trace_anchor="the refinement whose candidates equal the answer, else the last refinement",
    ),
    StepDefinition(
        step="quantification",
        title="Quantification",
        question="Are the reported amounts the cited fit's amounts, and defensible?",
        deciders=(
            Decider(
                source="expert_label",
                check="amounts_defensible",
                rule="each expert that answers says yes; experts who disagree read unknown",
            ),
            Decider(
                source="deterministic_check",
                check="K5",
                rule="a reported amount off the cited fit by more than the policy's wt% fails; copied amounts are not defensibility",
                decides="fail_only",
            ),
            Decider(
                source="model_judge",
                check="amounts_defensible",
                rule="each judge that answers says yes; judges that disagree read unknown",
            ),
        ),
        trace_anchor="the last call that touches analysis.json",
    ),
    StepDefinition(
        step="final_answer",
        title="Final answer",
        question="Does the fit support the submitted conclusion?",
        deciders=(
            Decider(
                source="expert_label",
                check="fit_supports_conclusion",
                rule="each expert that answers says yes; experts who disagree read unknown",
            ),
            Decider(
                source="deterministic_check",
                check="submitted",
                rule="no submitted answer (budget exhaustion, no submission) fails; a submission is not support",
                decides="fail_only",
            ),
            Decider(
                source="model_judge",
                check="fit_supports_conclusion",
                rule="each judge that answers says yes; judges that disagree read unknown",
            ),
        ),
        trace_anchor="the last call that touches analysis.json, else the last call",
    ),
)
DEFINITION: Mapping[StepName, StepDefinition] = {d.step: d for d in STEP_DEFINITIONS}


# --- failure codes, seeded from failures-trajectories.md -----------------------------------------

FAILURE_DOC = "docs/research/2026-10-21-eval-direction/failures-trajectories.md (osi/eval-direction cf6e18b44)"


class FailureCode(Record):
    code: str
    title: str
    step: StepName


FAILURE_CODES: tuple[FailureCode, ...] = (
    FailureCode(code="S3", title="material context misread before the search", step="phase_search"),
    FailureCode(code="S4", title="candidate search missed a reference phase", step="phase_search"),
    FailureCode(code="S5", title="phase identification", step="phase_selection"),
    FailureCode(code="S6", title="polymorph choice or label", step="phase_selection"),
    FailureCode(code="S8", title="refinement execution and convergence", step="refinement"),
    FailureCode(code="S9", title="quantification", step="quantification"),
    FailureCode(
        code="S10", title="verdict: the fit does not support the conclusion", step="final_answer"
    ),
    FailureCode(
        code="S11",
        title="orchestration: budget, context or tool misuse ended the run",
        step="final_answer",
    ),
)


class AnalystCoding(Record):
    """The doc's first divergence of one U6 run: an agent's reading against custody, not an
    expert label. `call_index` is the doc's call number #n minus one (RecordedToolCall order)."""

    code: str
    call_index: int | None
    note: str
    disputed: bool = False


# Analyst coding of individual runs is private evaluation data and is not shipped. No run in
# the public export had any.
ANALYST_CODING: Mapping[str, AnalystCoding] = {}
ANALYST_NONE: frozenset[str] = frozenset()

# DEV006: the published reference lists the three precursors; DARA's first solution and Jade also
# report Bi2CO5, and the DARA authors mark it correct (CONTEXT.md). An expert call, not a scored fix.
DISPUTED: frozenset[tuple[str, str]] = frozenset({("DEV006", "Bi2CO5")})


# --- reward spec v0 ----------------------------------------------------------------------------


class RewardComponent(Record):
    id: str
    source: Source
    check: str
    weight: float
    rule: str


class RewardSpec(Record):
    version: str
    components: tuple[RewardComponent, ...]
    rules: tuple[str, ...]


REWARD_SPEC_V0 = RewardSpec(
    version="reward-v0-2026-10-22",
    components=(
        RewardComponent(
            id="answered",
            source="deterministic_check",
            check="submitted",
            weight=0.20,
            rule="1 if the run submitted an answer citing a saved fit",
        ),
        RewardComponent(
            id="k2_residual",
            source="deterministic_check",
            check="K2",
            weight=0.15,
            rule="1 if K2 passes",
        ),
        RewardComponent(
            id="k1_refinement",
            source="deterministic_check",
            check="K1",
            weight=0.10,
            rule="1 if K1 passes on a Rietveld fit",
        ),
        RewardComponent(
            id="k5_amounts",
            source="deterministic_check",
            check="K5",
            weight=0.15,
            rule="1 if K5 passes",
        ),
        RewardComponent(
            id="reference_phase_list",
            source="reference_match",
            check="reference.phase_list",
            weight=0.20,
            rule="1 if the phase list matches the published reference (owner only, DEV cases)",
        ),
        RewardComponent(
            id="judge_phase_supported",
            source="model_judge",
            check="phase_supported",
            weight=0.10,
            rule="share of yes among yes and no answers, all judges pooled",
        ),
        RewardComponent(
            id="judge_fit_supports",
            source="model_judge",
            check="fit_supports_conclusion",
            weight=0.10,
            rule="share of yes among yes and no answers, all judges pooled",
        ),
    ),
    rules=(
        "An episode with no answer scores 0 on every component.",
        "A check that is flagged or not computable scores 0 (fail closed, as triage routes it).",
        "A component with no reading (no reference, no judge answer) scores 0 and is listed as missing.",
        "Weights sum to 1.0: 0.80 on verifiable checks, 0.20 on model-judge items.",
    ),
)


# --- the record --------------------------------------------------------------------------------


class Reading(Record):
    source: Source
    check: str
    outcome: Outcome
    detail: str


class StepOutcome(Record):
    step: StepName
    outcome: Outcome
    source: Source | None  # the deciding source; None when every reading is unknown
    check: str | None
    detail: str
    call_index: int | None  # the recorded call the trace opens at
    readings: tuple[Reading, ...]
    failure_code: str | None


class RewardTerm(Record):
    id: str
    weight: float
    value: float
    missing: bool
    detail: str


class RewardBreakdown(Record):
    spec_version: str
    total: float
    verifiable: float
    judge: float
    terms: tuple[RewardTerm, ...]


class EpisodeScore(Record):
    episode: str  # the imported run's attempt digest
    leaf: str
    case_id: str
    model: str
    harness: str
    job: str
    trial: str
    answered: bool
    end: str  # submitted, or the unsuccessful or invalid reason
    snapshot: str | None  # the frozen answer snapshot's digest
    rwp_percent: float | None
    tool_calls: int
    steps: tuple[StepOutcome, ...]
    analyst: AnalystCoding | None
    analyst_none: bool  # the doc read this run and found no divergence
    reward: RewardBreakdown


class TraceRow(Record):
    index: int
    tool: str
    summary: str
    marks: tuple[str, ...]  # "screen", "refine", "answer"


class Count(Record):
    passed: int
    failed: int
    unknown: int
    denominator: int


class MatrixRow(Record):
    model: str
    source: Source | Literal["decided"]
    episodes: int
    cells: dict[str, Count]  # step -> counts over the model's episodes


class FailureRow(Record):
    code: str
    title: str
    step: StepName
    source: Source | Literal["analyst"]
    check: str
    episodes: tuple[str, ...]  # attempt digests
    denominator: int  # episodes this source coded: every episode, or the runs the analyst read


class JudgeStat(Record):
    model: str
    labels: int
    yes: int
    no: int
    cannot_assess: int


class JudgePairStat(Record):
    model_a: str
    model_b: str
    paired_targets: int
    comparable: int  # neither answered cannot_assess
    agree: int
    kappa: float | None  # Cohen's kappa over the comparable pairs


class PairRow(Record):
    """An explicit review.Preference (R5, R18) the export carries as a PreferencePair: the only
    source of chosen and rejected. Both endpoints resolve across every exported path."""

    case_id: str
    model: str  # the producer on the episode's import, "-" for a host session
    rejected: str  # attempt digest
    chosen: str
    fork: str
    reason: str
    author: str
    rejected_rwp: float | None
    chosen_rwp: float | None
    rejected_phases: tuple[str, ...]
    chosen_phases: tuple[str, ...]
    resolved: bool  # both endpoints are steps of exported episodes; never an empty stand-in
    chosen_exposed: bool
    rejected_exposed: bool | None  # None when the rejected endpoint is not in the export
    chosen_use: str  # "preference" eligibility over every exported path through the endpoint
    rejected_use: str | None
    review_eligible: bool  # resolved, no reference exposure and preference use eligible on both
    withheld_because: str | None
    # Artifact rights, contributor and model-output terms are not resolved here (no complete
    # training-permission gate exists), so no pair is ever marked trainable (CR18).
    training_permission: Literal["not checked"] = "not checked"


DEMO_PAIR_RULE = (
    "A demo pair needs all of: a human path past an agent's answer whose every edit is by one rater; a refit at"
    " its end that completed with a fit and changed the answer; a later sign-off by that same rater on the agent"
    " answer's snapshot; and that rater signing with the owner account (owner kind, revealed). The agent's answer"
    " is rejected and the refit chosen. Expert-kind paths stay candidates until an explicit preference."
)


class DemoPairRow(Record):
    """PI D-sub-6 (b): the on-camera loop (a demo rater's refit, then sign-off) as one pair, held
    in the revealed demo bucket: never an expert record, never eligible, never in the main file."""

    case_id: str
    model: str
    rejected: str  # the agent's imported answer
    chosen: str  # the demo rater's refit
    rejected_rwp: float | None
    chosen_rwp: float | None
    rejected_phases: tuple[str, ...]
    chosen_phases: tuple[str, ...]
    author: str  # the rater's export pseudonym
    contributions: tuple[str, ...]
    signoff_verdict: str
    signed_at_utc: datetime
    refit_at_utc: datetime
    bucket: Literal["demo_revealed"] = "demo_revealed"


class CandidateRow(Record):
    """A human path past an agent's answer whose answer changed: shown before and after,
    unordered. A correction is not a preference until someone records one."""

    case_id: str
    model: str
    before: str  # the imported answer's attempt
    after: str  # the path's leaf
    contributions: tuple[str, ...]
    before_rwp: float | None
    after_rwp: float | None
    before_phases: tuple[str, ...]
    after_phases: tuple[str, ...]
    after_fit: bool  # False: an edit with no refit after it
    reference_exposed: bool
    preference_use: str
    has_preference: bool  # an explicit preference already names this path
    demo_pair: bool  # it meets DEMO_PAIR_RULE and is listed with the demo pairs


class ReconLine(Record):
    name: str
    page: int
    manifest: int
    manifest_field: str
    ok: bool


class ScorecardView(Record):
    generated_at_utc: datetime
    dataset_version: str
    scope: str
    steps: tuple[StepDefinition, ...]
    failure_codes: tuple[FailureCode, ...]
    episodes: tuple[EpisodeScore, ...]
    matrix: tuple[MatrixRow, ...]
    failures: tuple[FailureRow, ...]
    reward_spec: RewardSpec
    pairs: tuple[PairRow, ...]
    candidates: tuple[CandidateRow, ...]
    demo_pairs: tuple[DemoPairRow, ...]
    unchanged_paths: int  # human paths that left the answer as it was: refused or no-op edits
    judges: tuple[JudgeStat, ...]
    judge_pairs: tuple[JudgePairStat, ...]
    expert_labels: int
    owner_labels: int
    reference_cases: tuple[str, ...]  # cases with a published reference on this host
    excluded_episodes: int  # manifest counts.episodes_excluded
    traces: dict[str, tuple[TraceRow, ...]]  # episode -> its recorded calls
    reconciliation: tuple[ReconLine, ...]
    caveats: tuple[str, ...]


# --- readings ----------------------------------------------------------------------------------

CANDIDATE = re.compile(r"\b(?:CIF|COD)\d{6,7}\b")


def _marks(tool: str, summary: str) -> tuple[str, ...]:
    marks = []
    if "screen.py" in summary or "candidates.py" in summary:
        marks.append("screen")
    if "refine.py" in summary or tool == "refine" or summary.startswith("refine "):
        marks.append("refine")
    if "analysis.json" in summary:
        marks.append("answer")
    return tuple(marks)


def _aggregate(
    source: Source, check: str, labels: Iterable[Label], who: Callable[[Label], str]
) -> Reading:
    """Per rater: fail on any no, pass on at least one yes and no no. Across raters: the shared
    verdict when every rater that committed agrees; raters who disagree read unknown, with
    each rater's answers in the detail, so no one rater stands in for the others."""
    by: dict[str, Counter[str]] = defaultdict(Counter)
    for label in labels:
        by[who(label)][label.answer] += 1
    if not by:
        return Reading(source=source, check=check, outcome="unknown", detail="no answers")
    verdicts = {
        r: ("fail" if c["no"] else "pass" if c["yes"] else "unknown") for r, c in by.items()
    }
    committed = {v for v in verdicts.values() if v != "unknown"}
    outcome: Outcome = committed.pop() if len(committed) == 1 else "unknown"
    detail = ("raters disagree; " if len(committed) > 1 else "") + "; ".join(
        f"{r}: {c['yes']} yes, {c['no']} no, {c['cannot_assess']} cannot assess"
        for r, c in sorted(by.items())
    )
    return Reading(source=source, check=check, outcome=outcome, detail=detail)


def _initial(labels: Iterable[Label]) -> list[Label]:
    """The earliest blind label per (rater identity, target); revealed labels are revisions."""
    held: dict[tuple, Label] = {}
    for label in labels:
        if label.exposure != "blind":
            continue
        who = label.rater.judge.model if label.rater.judge else label.rater.principal_id
        key = (who, label.snapshot, label.item, label.phase_id)
        if key not in held or label.created_at_utc < held[key].created_at_utc:
            held[key] = label
    return list(held.values())


def _check(snap: AnalysisSnapshot | None, code: str) -> Reading:
    if snap is None:
        return Reading(
            source="deterministic_check",
            check=code,
            outcome="unknown",
            detail="no frozen answer snapshot",
        )
    for c in snap.checks:
        if c.code == code:
            outcome: Outcome = (
                "pass" if c.status == "pass" else "fail" if c.status == "flag" else "unknown"
            )
            return Reading(
                source="deterministic_check",
                check=code,
                outcome=outcome,
                detail=f"{c.status} ({c.policy_version}): {c.detail[:300]}",
            )
    return Reading(
        source="deterministic_check",
        check=code,
        outcome="unknown",
        detail=f"{code} not on the snapshot",
    )


def _expert_marks(feedback: Iterable[StepFeedback], episode: str, calls: set[int]) -> Reading:
    marks = [
        f
        for f in feedback
        if f.rater.kind == "expert" and f.step.attempt == episode and f.step.call_index in calls
    ]
    bad = sum(f.mark in ("wrong_move", "went_wrong_here") for f in marks)
    good = sum(f.mark == "good_move" for f in marks)
    outcome: Outcome = "fail" if bad else "pass" if good else "unknown"
    return Reading(
        source="expert_label",
        check="step_feedback",
        outcome=outcome,
        detail=f"{len(marks)} expert marks ({bad} wrong, {good} good)",
    )


def _decide(
    step: StepName, readings: list[Reading], call: int | None, code_for: Callable[[Reading], str]
) -> StepOutcome:
    by_key = {(r.source, r.check): r for r in readings}
    for decider in DEFINITION[step].deciders:
        reading = by_key.get((decider.source, decider.check))
        if reading is not None and (
            reading.outcome == "fail"
            or (reading.outcome == "pass" and decider.decides == "pass_or_fail")
        ):
            return StepOutcome(
                step=step,
                outcome=reading.outcome,
                source=reading.source,
                check=reading.check,
                detail=reading.detail,
                call_index=call,
                readings=tuple(readings),
                failure_code=code_for(reading) if reading.outcome == "fail" else None,
            )
    return StepOutcome(
        step=step,
        outcome="unknown",
        source=None,
        check=None,
        detail="no source reads pass or fail",
        call_index=call,
        readings=tuple(readings),
        failure_code=None,
    )


def _phase_text(h: Hypothesis | None) -> dict[str, Any]:
    if h is None:
        return {"phases": [], "fit": None}
    weights = {p.candidate_id: p.weight_percent for p in h.fit.phases} if h.fit else {}
    return {
        "phases": [{"candidate_id": p, "weight_percent": weights.get(p)} for p in h.phases],
        "fit": None
        if h.fit is None
        else {"stage": h.fit.stage, "rwp_percent": h.fit.rwp_percent, "gof": h.fit.gof},
    }


# --- the view ----------------------------------------------------------------------------------


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines() if path.is_file() else []


def scorecard_view(
    dataset: Path,
    reference: Callable[[str], Mapping[str, Any] | None] = snapshot_module._reference,
    now: datetime | None = None,
) -> ScorecardView:
    """`reference` is the reveal route's custody reader by default; it reads
    $AXA_WORKBENCH_CUSTODY on the owner's host and returns None elsewhere."""
    manifest = DatasetManifest.model_validate_json((dataset / "manifest.json").read_bytes())
    records = [
        EpisodeRecord.model_validate_json(line) for line in _lines(dataset / "episodes.jsonl")
    ]
    snapshots = [
        AnalysisSnapshot.model_validate_json(line) for line in _lines(dataset / "snapshots.jsonl")
    ]
    expert_file = [
        Label.model_validate_json(line) for line in _lines(dataset / "labels.expert.jsonl")
    ]
    judge_file = [
        Label.model_validate_json(line) for line in _lines(dataset / "labels.model_judge.jsonl")
    ]
    feedback = [
        StepFeedback.model_validate_json(line) for line in _lines(dataset / "step_feedback.jsonl")
    ]
    signoffs = [SignOff.model_validate_json(line) for line in _lines(dataset / "signoffs.jsonl")]

    answer_of = {s.origin: (digest_of(s), s) for s in snapshots if s.kind == "agent_answer"}
    judge_labels = _initial(lb for lb in judge_file if lb.rater.kind == "model_judge")
    expert_labels = _initial(lb for lb in expert_file if lb.rater.kind == "expert")
    by_snapshot: dict[str, list[Label]] = defaultdict(list)
    for label in judge_labels + expert_labels:
        by_snapshot[label.snapshot].append(label)
    references: dict[str, Mapping[str, Any] | None] = {}

    episodes: list[EpisodeScore] = []
    traces: dict[str, tuple[TraceRow, ...]] = {}
    pairs: list[PairRow] = []
    candidates: list[CandidateRow] = []
    demo_pairs: list[DemoPairRow] = []
    unchanged = 0
    for record in records:
        first = record.steps[0] if record.steps else None
        outcome = first.attempt.outcome if first else None
        imported = (
            first is not None
            and isinstance(first.attempt.action, ImportRun)
            and isinstance(outcome, Completed)
            and isinstance(outcome.result, ImportedRunResult)
        )
        model = outcome.result.agent.model_id if imported else "-"
        if not imported:
            continue
        result = outcome.result
        if record.human_contributions:
            # A human's path past the answer; the agent's own episode is its own row.
            candidate = _candidate(record, model)
            if candidate is None:
                unchanged += 1
                continue
            demo = _demo_pair(record, candidate, snapshots, signoffs)
            if demo is not None:
                demo_pairs.append(demo[0])
            candidates.append(candidate.model_copy(update={"demo_pair": demo is not None}))
            continue
        episodes.append(
            _score(record, result, answer_of, by_snapshot, feedback, references, reference, traces)
        )

    pairs.extend(row for row, _ in _pair_items(records, manifest.dataset_version))
    episodes.sort(key=lambda e: (e.case_id, e.model, e.job))
    matrix = _matrix(episodes)
    failures = _failures(episodes)
    judges, judge_pairs = _judge_stats(judge_labels)
    owner_labels = sum(1 for lb in expert_file if lb.rater.kind == "owner")
    recon = _reconcile(
        manifest, episodes, records, snapshots, expert_file, judge_file, feedback, signoffs
    )
    caveats = (
        "No model was trained on this data or this reward. Reward v0 is a declared spec to inspect, not a validated training signal.",
        f"Model-judge labels are model judgments, not expert labels. Expert labels: {len([lb for lb in expert_file if lb.rater.kind == 'expert'])}."
        f" Owner labels ({owner_labels}) are revealed demo labels and never count.",
        "Judge items in the reward carry no expert calibration: "
        + "; ".join(
            f"{p.model_a} vs {p.model_b}: Cohen's kappa {'none' if p.kappa is None else f'{p.kappa:.2f}'} on {p.comparable} comparable pairs ({p.agree} agree)"
            for p in judge_pairs
        )
        + "; "
        + "; ".join(f"{j.model} answered yes on {j.yes} of {j.labels}" for j in judges)
        + ".",
        "Reference match is the published reference phase list at formula level, never 'correct': DEV006's reference omits"
        " Bi2CO5, which the DARA authors report. Owner only.",
        "Analyst coding is an agent's reading of the U6 trajectories against custody ("
        + FAILURE_DOC
        + "), not an expert label.",
    )
    return ScorecardView(
        generated_at_utc=now or datetime.now(UTC),
        dataset_version=manifest.dataset_version,
        scope=manifest.scope,
        steps=STEP_DEFINITIONS,
        failure_codes=FAILURE_CODES,
        episodes=tuple(episodes),
        matrix=matrix,
        failures=failures,
        reward_spec=REWARD_SPEC_V0,
        pairs=tuple(pairs),
        candidates=tuple(candidates),
        demo_pairs=tuple(demo_pairs),
        unchanged_paths=unchanged,
        judges=judges,
        judge_pairs=judge_pairs,
        expert_labels=sum(1 for lb in expert_file if lb.rater.kind == "expert"),
        owner_labels=owner_labels,
        reference_cases=tuple(sorted(c for c, r in references.items() if r is not None)),
        excluded_episodes=next(
            (c.count for c in manifest.counts if c.name == "episodes_excluded"), 0
        ),
        traces=traces,
        reconciliation=recon,
        caveats=caveats,
    )


def _score(
    record: EpisodeRecord,
    result: ImportedRunResult,
    answer_of: Mapping[str, tuple[str, AnalysisSnapshot]],
    by_snapshot: Mapping[str, list[Label]],
    feedback: list[StepFeedback],
    references: dict[str, Mapping[str, Any] | None],
    reference: Callable[[str], Mapping[str, Any] | None],
    traces: dict[str, tuple[TraceRow, ...]],
) -> EpisodeScore:
    first = record.steps[0]
    episode = first.digest
    case_id = record.root.case.case_id
    action = first.attempt.action
    assert isinstance(action, ImportRun)
    job = action.harbor.job_name
    rows = tuple(
        TraceRow(index=t.index, tool=t.tool, summary=t.summary, marks=_marks(t.tool, t.summary))
        for t in result.trajectory
    )
    traces[episode] = rows
    refine_calls = [r for r in rows if "refine" in r.marks]
    screens = [r for r in rows if "screen" in r.marks]
    answer_calls = [r for r in rows if "answer" in r.marks]
    submitted = isinstance(result.end, ImportedSubmitted)
    end = (
        "submitted"
        if submitted
        else getattr(result.end, "reason", None) or getattr(result.end, "cause", "invalid")
    )
    digest, snap = answer_of.get(episode, (None, None))
    answer_ids = (
        tuple(p.phase_id for p in snap.phases)
        if snap
        else tuple(result.end.answer.phases)
        if submitted
        else ()
    )
    cited = (
        next(
            (r for r in refine_calls if set(CANDIDATE.findall(r.summary)) == set(answer_ids)), None
        )
        if answer_ids
        else None
    )
    fit_call = cited or (refine_calls[-1] if refine_calls else None)
    anchors: dict[StepName, int | None] = {
        "phase_search": (screens or refine_calls or rows[:1] or [None])[0].index
        if (screens or refine_calls or rows)
        else None,
        "phase_selection": fit_call.index if fit_call else None,
        "refinement": fit_call.index if fit_call else None,
        "quantification": answer_calls[-1].index if answer_calls else None,
        "final_answer": answer_calls[-1].index
        if answer_calls
        else rows[-1].index
        if rows
        else None,
    }
    if case_id not in references:
        references[case_id] = reference(case_id)
    ref = references[case_id]
    labels = by_snapshot.get(digest, []) if digest else []
    judge = [lb for lb in labels if lb.rater.kind == "model_judge"]
    expert = [lb for lb in labels if lb.rater.kind == "expert"]
    by_model = lambda lb: lb.rater.judge.model if lb.rater.judge else lb.rater.principal_id  # noqa: E731
    by_rater = lambda lb: lb.rater.principal_id  # noqa: E731
    first_refine = refine_calls[0].index if refine_calls else len(rows)

    def ref_tried() -> Reading:
        if ref is None:
            return Reading(
                source="reference_match",
                check="reference.tried",
                outcome="unknown",
                detail="no published reference on this host",
            )
        tried = {c for r in refine_calls for c in CANDIDATE.findall(r.summary)}
        missed = [p["formula"] for p in ref["phases"] if not tried & set(p["candidates"])]
        if not refine_calls:
            return Reading(
                source="reference_match",
                check="reference.tried",
                outcome="fail",
                detail="no refinement call in the trace",
            )
        return Reading(
            source="reference_match",
            check="reference.tried",
            outcome="fail" if missed else "pass",
            detail=f"{len(refine_calls)} refinement calls; never tried: {', '.join(missed)}"
            if missed
            else f"every reference formula tried in {len(refine_calls)} refinement calls",
        )

    def ref_list() -> Reading:
        if ref is None:
            return Reading(
                source="reference_match",
                check="reference.phase_list",
                outcome="unknown",
                detail="no published reference on this host",
            )
        if not answer_ids:
            return Reading(
                source="reference_match",
                check="reference.phase_list",
                outcome="unknown",
                detail="no answer",
            )
        match = snapshot_module._match(answer_ids, ref, "answer")
        formulas = {c.candidate_id: c.formula_sum for c in record.root.case.candidates} | (
            {p.phase_id: p.formula for p in snap.phases} if snap else {}
        )
        extra = [formulas.get(c, c) for c in match["extra"]]
        detail = (
            "matches published reference"
            if match["phase_list_exact"]
            else (
                "does not match published reference: "
                + "; ".join(
                    x
                    for x in (
                        f"missing {', '.join(match['missed'])}" if match["missed"] else "",
                        f"extra {', '.join(extra)}" if extra else "",
                    )
                    if x
                )
            )
        )
        if any((case_id, f) in DISPUTED for f in extra):
            detail += " (disputed: the DARA authors report it; an expert call)"
        return Reading(
            source="reference_match",
            check="reference.phase_list",
            outcome="pass" if match["phase_list_exact"] else "fail",
            detail=detail,
        )

    def submitted_reading() -> Reading:
        ok = submitted and result.basis_fit is not None
        detail = (
            f"submitted, cites a {result.basis_fit.stage} fit at Rwp {result.basis_fit.rwp_percent:.2f}%"
            if ok
            # Missing evidence is named, never scored as the agent's failure (R11).
            else (
                "submitted, but the export holds no basis fit: "
                + ", ".join(getattr(result.completeness, "missing", ()) or ("not recorded",))
                if submitted
                else f"no answer: {end} after {len(rows)} calls"
            )
        )
        return Reading(
            source="deterministic_check",
            check="submitted",
            outcome="pass" if ok else "unknown" if submitted else "fail",
            detail=detail,
        )

    def k1() -> Reading:
        reading = _check(snap, "K1")
        if snap is not None and reading.outcome == "pass" and snap.stage != "rietveld":
            return reading.model_copy(
                update={
                    "outcome": "fail",
                    "detail": f"cited fit stage is {snap.stage}, not Rietveld",
                }
            )
        return reading

    def phase_items(
        items: tuple[str, ...],
        source: Source,
        pool: list[Label],
        who: Callable[[Label], str],
        check: str,
    ) -> Reading:
        return _aggregate(source, check, (lb for lb in pool if lb.item in items), who)

    no_answer = Reading(
        source="deterministic_check", check="-", outcome="unknown", detail="no answer"
    )
    readings: dict[StepName, list[Reading]] = {
        "phase_search": [_expert_marks(feedback, episode, set(range(first_refine))), ref_tried()],
        "phase_selection": [
            phase_items(("phase_supported",), "expert_label", expert, by_rater, "phase_supported"),
            ref_list(),
            _check(snap, "K2") if snap else no_answer.model_copy(update={"check": "K2"}),
            phase_items(
                ("phase_supported", "phase_plausible"),
                "model_judge",
                judge,
                by_model,
                "phase_supported, phase_plausible",
            ),
        ],
        "refinement": [
            _expert_marks(feedback, episode, {r.index for r in refine_calls}),
            k1()
            if snap
            else no_answer.model_copy(
                update={
                    "check": "K1",
                    "detail": f"no answer; {len(refine_calls)} refinement calls, no saved fit in the export",
                }
            ),
        ],
        "quantification": [
            phase_items(
                ("amounts_defensible",), "expert_label", expert, by_rater, "amounts_defensible"
            ),
            _check(snap, "K5") if snap else no_answer.model_copy(update={"check": "K5"}),
            phase_items(
                ("amounts_defensible",), "model_judge", judge, by_model, "amounts_defensible"
            ),
        ],
        "final_answer": [
            phase_items(
                ("fit_supports_conclusion",),
                "expert_label",
                expert,
                by_rater,
                "fit_supports_conclusion",
            ),
            submitted_reading(),
            phase_items(
                ("fit_supports_conclusion",),
                "model_judge",
                judge,
                by_model,
                "fit_supports_conclusion",
            ),
        ],
    }

    def code_for(step: StepName) -> Callable[[Reading], str]:
        def code(reading: Reading) -> str:
            if step == "phase_search":
                return "S4"
            if step == "phase_selection":
                return "S5"
            if step == "refinement":
                return "S8"
            if step == "quantification":
                return "S9"
            return "S11" if reading.check == "submitted" else "S10"

        return code

    steps = tuple(_decide(step, readings[step], anchors[step], code_for(step)) for step in STEPS)
    analyst = ANALYST_CODING.get(job)
    reward = _reward(steps, snap is not None, judge)
    return EpisodeScore(
        episode=episode,
        leaf=record.leaf,
        case_id=case_id,
        model=result.agent.model_id,
        harness=result.agent.harness,
        job=job,
        trial=action.harbor.trial_id,
        answered=submitted,
        end=str(end),
        snapshot=digest,
        rwp_percent=result.basis_fit.rwp_percent if result.basis_fit else None,
        tool_calls=len(rows),
        steps=steps,
        analyst=analyst,
        analyst_none=job in ANALYST_NONE,
        reward=reward,
    )


def _reward(steps: tuple[StepOutcome, ...], answered: bool, judge: list[Label]) -> RewardBreakdown:
    by_key = {(r.source, r.check): r for s in steps for r in s.readings}
    terms = []
    for c in REWARD_SPEC_V0.components:
        if not answered:
            terms.append(
                RewardTerm(
                    id=c.id, weight=c.weight, value=0.0, missing=False, detail="no answer snapshot"
                )
            )
        elif c.source == "model_judge":
            answers = Counter(lb.answer for lb in judge if lb.item == c.check)
            n = answers["yes"] + answers["no"]
            terms.append(
                RewardTerm(
                    id=c.id,
                    weight=c.weight,
                    value=round(answers["yes"] / n, 4) if n else 0.0,
                    missing=n == 0,
                    detail=f"{answers['yes']} yes of {n} yes or no answers ({answers['cannot_assess']} cannot assess)",
                )
            )
        else:
            reading = by_key.get((c.source, c.check))
            outcome = reading.outcome if reading else "unknown"
            terms.append(
                RewardTerm(
                    id=c.id,
                    weight=c.weight,
                    value=1.0 if outcome == "pass" else 0.0,
                    missing=outcome == "unknown",
                    detail=reading.detail[:200] if reading else "no reading",
                )
            )
    judge_ids = {c.id for c in REWARD_SPEC_V0.components if c.source == "model_judge"}
    total = sum(t.weight * t.value for t in terms)
    judged = sum(t.weight * t.value for t in terms if t.id in judge_ids)
    return RewardBreakdown(
        spec_version=REWARD_SPEC_V0.version,
        total=round(total, 4),
        verifiable=round(total - judged, 4),
        judge=round(judged, 4),
        terms=tuple(terms),
    )


def _preference_use(record: EpisodeRecord) -> str:
    return next((d.status for d in record.eligibility if d.use == "preference"), "absent")


def _model_of(record: EpisodeRecord) -> str:
    first = record.steps[0] if record.steps else None
    outcome = first.attempt.outcome if first else None
    imported = isinstance(outcome, Completed) and isinstance(outcome.result, ImportedRunResult)
    return outcome.result.agent.model_id if imported else "-"


def _path_terms(paths: list[EpisodeRecord]) -> list[dict[str, Any]]:
    return [
        {
            "leaf": r.leaf,
            "reference_exposed": r.reference_exposed,
            "eligibility": [d.model_dump(mode="json") for d in r.eligibility],
            "rights": [x.model_dump(mode="json") for x in r.rights],
        }
        for r in paths
    ]


def _pair_items(
    records: list[EpisodeRecord], dataset_version: str
) -> list[tuple[PairRow, dict[str, Any] | None]]:
    """Each explicit preference the export carries, once, with both endpoints resolved across
    every exported path through them (CR17). A path's exposure and eligibility count against
    the pair wherever it touches either endpoint. An endpoint missing from the export (its path
    excluded, or the root) withholds the pair: no line, never an empty stand-in."""
    through: dict[str, list[EpisodeRecord]] = defaultdict(list)
    state: dict[str, Hypothesis | None] = {}
    for record in records:
        for step in record.steps:
            through[step.digest].append(record)
            state[step.digest] = step.state_after.hypothesis
    seen: set[tuple] = set()
    items = []
    for record in records:
        for p in record.preference_pairs:
            key = (p.chosen, p.rejected, p.fork, p.reason, p.author.principal_id)
            if key in seen:
                continue  # a preference on a shared node rides on every path through it
            seen.add(key)
            chosen, rejected = through.get(p.chosen, []), through.get(p.rejected, [])
            resolved = bool(chosen) and bool(rejected)

            def use(paths: list[EpisodeRecord]) -> str | None:
                if not paths:
                    return None
                statuses = {_preference_use(r) for r in paths}
                return (
                    "eligible"
                    if statuses == {"eligible"}
                    else ", ".join(sorted(statuses - {"eligible"}))
                )

            c_use, r_use = use(chosen), use(rejected)
            c_exposed = any(r.reference_exposed for r in chosen)
            r_exposed = any(r.reference_exposed for r in rejected) if rejected else None
            withheld = "; ".join(
                x
                for x in (
                    "" if chosen else "chosen endpoint not in this export",
                    "" if rejected else "rejected endpoint not in this export (excluded or absent)",
                    "a reference-exposed path" if c_exposed or r_exposed else "",
                    "" if c_use in (None, "eligible") else f"chosen path preference use {c_use}",
                    "" if r_use in (None, "eligible") else f"rejected path preference use {r_use}",
                )
                if x
            )
            c_h, r_h = state.get(p.chosen), state.get(p.rejected)
            case = record.root.case
            model = _model_of(record)
            row = PairRow(
                case_id=case.case_id,
                model=model,
                rejected=p.rejected,
                chosen=p.chosen,
                fork=p.fork,
                reason=p.reason,
                author=p.author.principal_id,
                rejected_rwp=r_h.fit.rwp_percent if r_h and r_h.fit else None,
                chosen_rwp=c_h.fit.rwp_percent if c_h and c_h.fit else None,
                rejected_phases=tuple(r_h.phases) if r_h else (),
                chosen_phases=tuple(c_h.phases) if c_h else (),
                resolved=resolved,
                chosen_exposed=c_exposed,
                rejected_exposed=r_exposed,
                chosen_use=c_use or "absent",
                rejected_use=r_use,
                review_eligible=not withheld,
                withheld_because=withheld or None,
            )
            line = (
                {
                    "prompt": case.task,
                    "chosen": json.dumps(_phase_text(c_h), sort_keys=True),
                    "rejected": json.dumps(_phase_text(r_h), sort_keys=True),
                    "metadata": {
                        "dataset_version": dataset_version,
                        "source": "an explicit preference recorded in the workbench (review.Preference)",
                        "case_id": case.case_id,
                        "producer": model,
                        "rejected_attempt": p.rejected,
                        "chosen_attempt": p.chosen,
                        "fork": p.fork,
                        "reason": p.reason,
                        "author": p.author.principal_id,
                        "chosen_paths": _path_terms(chosen),
                        "rejected_paths": _path_terms(rejected),
                        "review_eligible": not withheld,
                        "withheld_because": withheld or None,
                        "training_permission": "not checked: artifact rights, contributor and model-output terms"
                        " are not resolved here",
                        "trained_on": False,
                    },
                }
                if resolved
                else None
            )
            items.append((row, line))
    return items


def _candidate(record: EpisodeRecord, model: str) -> CandidateRow | None:
    """None when the path left the answer unchanged (a refused or no-op edit)."""
    before, after = record.steps[0].state_after.hypothesis, record.steps[-1].state_after.hypothesis
    if _phase_text(before) == _phase_text(after):
        return None
    chosen = {p.chosen for p in record.preference_pairs}
    return CandidateRow(
        case_id=record.root.case.case_id,
        model=model,
        before=record.steps[0].digest,
        after=record.leaf,
        contributions=tuple(c.kind for c in record.human_contributions),
        before_rwp=before.fit.rwp_percent if before and before.fit else None,
        after_rwp=after.fit.rwp_percent if after and after.fit else None,
        before_phases=tuple(before.phases) if before else (),
        after_phases=tuple(after.phases) if after else (),
        after_fit=after is not None and after.fit is not None,
        reference_exposed=record.reference_exposed,
        preference_use=_preference_use(record),
        has_preference=bool(chosen & {s.digest for s in record.steps}),
        demo_pair=False,
    )


def _demo_pair(
    record: EpisodeRecord,
    candidate: CandidateRow,
    snapshots: list[AnalysisSnapshot],
    signoffs: list[SignOff],
) -> tuple[DemoPairRow, dict[str, Any]] | None:
    """The pair DEMO_PAIR_RULE defines, or None. Each condition is checked, none assumed."""
    authors = {c.principal_id for c in record.human_contributions}
    leaf = record.steps[-1]
    if len(authors) != 1 or not candidate.after_fit or leaf.digest != record.leaf:
        return None
    if not (
        isinstance(leaf.attempt.outcome, Completed)
        and getattr(leaf.attempt.action, "verb", None) == "refine"
    ):
        return None
    (author,) = authors
    answer = record.steps[0].digest
    answer_snapshots = {
        digest_of(s) for s in snapshots if s.origin == answer and s.kind == "agent_answer"
    }
    refit_at = leaf.attempt.created_at_utc
    signed = sorted(
        (
            so
            for so in signoffs
            if so.snapshot in answer_snapshots
            and so.rater.principal_id == author
            and so.rater.kind == "owner"
            and so.created_at_utc >= refit_at
        ),
        key=lambda so: so.created_at_utc,
    )
    if not signed:
        return None
    signoff = signed[0]
    before, after = record.steps[0].state_after.hypothesis, leaf.state_after.hypothesis
    row = DemoPairRow(
        case_id=candidate.case_id,
        model=candidate.model,
        rejected=answer,
        chosen=record.leaf,
        rejected_rwp=candidate.before_rwp,
        chosen_rwp=candidate.after_rwp,
        rejected_phases=candidate.before_phases,
        chosen_phases=candidate.after_phases,
        author=author,
        contributions=candidate.contributions,
        signoff_verdict=signoff.verdict,
        signed_at_utc=signoff.created_at_utc,
        refit_at_utc=refit_at,
    )
    line = {
        "prompt": record.root.case.task,
        "chosen": json.dumps(_phase_text(after), sort_keys=True),
        "rejected": json.dumps(_phase_text(before), sort_keys=True),
        "metadata": {
            "bucket": "demo_revealed",
            "rule": DEMO_PAIR_RULE,
            "case_id": candidate.case_id,
            "producer": candidate.model,
            "rejected_attempt": answer,
            "chosen_attempt": record.leaf,
            "author": author,
            "author_kind": "owner",
            "contributions": list(candidate.contributions),
            "signoff_verdict": signoff.verdict,
            "signoff_exposure": signoff.exposure,
            "reference_exposed": record.reference_exposed,
            "eligibility": [d.model_dump(mode="json") for d in record.eligibility],
            "review_eligible": False,
            "expert": False,
            "training_permission": "not checked; a demo pair is never for training",
            "trained_on": False,
        },
    }
    return row, line


def demo_pair_lines(dataset: Path) -> bytes:
    """The revealed demo bucket as JSONL in the same DPO shape, apart from the main file."""
    snapshots = [
        AnalysisSnapshot.model_validate_json(line) for line in _lines(dataset / "snapshots.jsonl")
    ]
    signoffs = [SignOff.model_validate_json(line) for line in _lines(dataset / "signoffs.jsonl")]
    out = []
    for line in _lines(dataset / "episodes.jsonl"):
        record = EpisodeRecord.model_validate_json(line)
        if not record.human_contributions or _model_of(record) == "-":
            continue
        candidate = _candidate(record, _model_of(record))
        demo = _demo_pair(record, candidate, snapshots, signoffs) if candidate else None
        if demo is not None:
            out.append(json.dumps(demo[1], sort_keys=True).encode())
    return b"".join(line + b"\n" for line in out)


def preference_lines(dataset: Path, eligible_only: bool = True) -> bytes:
    """Explicit preferences as JSONL in the common DPO shape: prompt, chosen, rejected, metadata.
    By default only pairs eligible for review; a pair with an endpoint missing from the export
    is never a line, and a correction without a recorded preference is never a line. Training
    permission is not checked here, and each line says so."""
    manifest = DatasetManifest.model_validate_json((dataset / "manifest.json").read_bytes())
    records = [
        EpisodeRecord.model_validate_json(line) for line in _lines(dataset / "episodes.jsonl")
    ]
    out = [
        json.dumps(item, sort_keys=True).encode()
        for row, item in _pair_items(records, manifest.dataset_version)
        if item is not None and (row.review_eligible or not eligible_only)
    ]
    return b"".join(line + b"\n" for line in out)


def _tally(outcomes: list[str], denominator: int) -> Count:
    c = Counter(outcomes)
    return Count(passed=c["pass"], failed=c["fail"], unknown=c["unknown"], denominator=denominator)


def _matrix(episodes: list[EpisodeScore]) -> tuple[MatrixRow, ...]:
    rows = []
    for model in sorted({e.model for e in episodes}):
        mine = [e for e in episodes if e.model == model]
        n = len(mine)
        rows.append(
            MatrixRow(
                model=model,
                source="decided",
                episodes=n,
                cells={
                    step: _tally([s.outcome for e in mine for s in e.steps if s.step == step], n)
                    for step in STEPS
                },
            )
        )
        for source in SOURCES:
            cells = {}
            for step in STEPS:
                outcomes = [
                    r.outcome
                    for e in mine
                    for s in e.steps
                    if s.step == step
                    for r in s.readings
                    if r.source == source
                ]
                if outcomes:
                    cells[step] = _tally(outcomes, n)
            rows.append(MatrixRow(model=model, source=source, episodes=n, cells=cells))
    return tuple(rows)


def _failures(episodes: list[EpisodeScore]) -> tuple[FailureRow, ...]:
    title = {c.code: c for c in FAILURE_CODES}
    grouped: dict[tuple[str, StepName, str, str], list[str]] = defaultdict(list)
    for e in episodes:
        for s in e.steps:
            if s.outcome == "fail" and s.failure_code and s.source:
                check = (s.check or "-") + (
                    ", disputed (an expert call)" if "disputed" in s.detail else ""
                )
                grouped[(s.failure_code, s.step, s.source, check)].append(e.episode)
        if e.analyst is not None:
            code = title[e.analyst.code]
            grouped[
                (
                    code.code,
                    code.step,
                    "analyst",
                    "disputed call" if e.analyst.disputed else "first divergence",
                )
            ].append(e.episode)
    order = {c.code: i for i, c in enumerate(FAILURE_CODES)}
    analyst_read = sum(e.analyst is not None or e.analyst_none for e in episodes)
    return tuple(
        FailureRow(
            code=code,
            title=title[code].title,
            step=step,
            source=source,
            check=check,
            episodes=tuple(eps),
            denominator=analyst_read if source == "analyst" else len(episodes),
        )
        for (code, step, source, check), eps in sorted(
            grouped.items(), key=lambda kv: (order[kv[0][0]], kv[0][2], kv[0][3])
        )
    )


def _judge_stats(labels: list[Label]) -> tuple[tuple[JudgeStat, ...], tuple[JudgePairStat, ...]]:
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    targets: dict[tuple, dict[str, str]] = defaultdict(dict)
    for label in labels:
        model = label.rater.judge.model if label.rater.judge else label.rater.principal_id
        by_model[model][label.answer] += 1
        targets[(label.snapshot, label.item, label.phase_id)][model] = label.answer
    stats = tuple(
        JudgeStat(
            model=m,
            labels=sum(c.values()),
            yes=c["yes"],
            no=c["no"],
            cannot_assess=c["cannot_assess"],
        )
        for m, c in sorted(by_model.items())
    )
    pairs = []
    for a, b in combinations(sorted(by_model), 2):
        both = [(t[a], t[b]) for t in targets.values() if a in t and b in t]
        comparable = [(x, y) for x, y in both if "cannot_assess" not in (x, y)]
        n = len(comparable)
        agree = sum(x == y for x, y in comparable)
        kappa = None
        if n:
            ca, cb = Counter(x for x, _ in comparable), Counter(y for _, y in comparable)
            expected = sum(ca[k] * cb[k] for k in ("yes", "no")) / (n * n)
            kappa = None if expected == 1 else round((agree / n - expected) / (1 - expected), 3)
        pairs.append(
            JudgePairStat(
                model_a=a,
                model_b=b,
                paired_targets=len(both),
                comparable=n,
                agree=agree,
                kappa=kappa,
            )
        )
    return stats, tuple(pairs)


def _reconcile(
    manifest: DatasetManifest,
    episodes: list[EpisodeScore],
    records: list[EpisodeRecord],
    snapshots: list[AnalysisSnapshot],
    expert_file: list[Label],
    judge_file: list[Label],
    feedback: list[StepFeedback],
    signoffs: list[SignOff],
) -> tuple[ReconLine, ...]:
    counts = {c.name: c.count for c in manifest.counts}

    def line(name: str, page: int, field: str, value: int) -> ReconLine:
        return ReconLine(
            name=name, page=page, manifest=value, manifest_field=field, ok=page == value
        )

    lines = [
        line(
            "episodes in episodes.jsonl",
            len(records),
            "counts.episodes_exported",
            counts.get("episodes_exported", -1),
        ),
        line(
            "agent episodes scored",
            len(episodes),
            "sum(agent_episodes.episodes)",
            sum(a.episodes for a in manifest.agent_episodes),
        ),
        line(
            "answered (submitted)",
            sum(e.answered for e in episodes),
            "sum(agent_episodes.submitted)",
            sum(a.submitted for a in manifest.agent_episodes),
        ),
        line(
            "unanswered (unsuccessful)",
            sum(not e.answered for e in episodes),
            "sum(agent_episodes.unsuccessful)",
            sum(a.unsuccessful for a in manifest.agent_episodes),
        ),
        line(
            "answer snapshots scored",
            sum(e.snapshot is not None for e in episodes),
            "counts.agent_snapshots",
            counts.get("agent_snapshots", -1),
        ),
        line("snapshots read", len(snapshots), "counts.snapshots", counts.get("snapshots", -1)),
        line(
            "model-judge labels read",
            len(judge_file),
            "counts.labels_model_judge",
            counts.get("labels_model_judge", -1),
        ),
        line(
            "snapshots with model-judge labels",
            len({lb.snapshot for lb in judge_file}),
            "counts.snapshots_with_model_judge_labels",
            counts.get("snapshots_with_model_judge_labels", -1),
        ),
        line(
            "labels in labels.expert.jsonl",
            len(expert_file),
            "counts.labels_expert",
            counts.get("labels_expert", -1),
        ),
        line("sign-offs read", len(signoffs), "counts.signoffs", counts.get("signoffs", -1)),
        line(
            "step feedback records",
            len(feedback),
            "counts.step_feedback",
            counts.get("step_feedback", -1),
        ),
        line(
            "explicit preferences attached to episodes (a shared node's preference counts per path)",
            sum(len(r.preference_pairs) for r in records),
            "episode_export.preference_pairs",
            manifest.episode_export.preference_pairs,
        ),
    ]
    page_rows = Counter((e.case_id, e.model, e.harness) for e in episodes)
    page_answered = Counter((e.case_id, e.model, e.harness) for e in episodes if e.answered)
    for a in sorted(manifest.agent_episodes, key=lambda a: (a.case_id, a.model)):
        key = (a.case_id, a.model, a.harness)
        lines.append(
            line(
                f"{a.case_id} {a.model} episodes",
                page_rows[key],
                "agent_episodes.episodes",
                a.episodes,
            )
        )
        lines.append(
            line(
                f"{a.case_id} {a.model} answered",
                page_answered[key],
                "agent_episodes.submitted",
                a.submitted,
            )
        )
    return tuple(lines)


def reconciliation_text(view: ScorecardView) -> str:
    width = max(len(r.name) for r in view.reconciliation)
    rows = [
        f"{r.name:<{width}}  page {r.page:>4}  manifest {r.manifest:>4}  {'ok' if r.ok else 'MISMATCH'}  ({r.manifest_field})"
        for r in view.reconciliation
    ]
    return "\n".join([f"dataset {view.dataset_version} ({view.scope})", *rows])


# --- rendering ---------------------------------------------------------------------------------

E = html.escape
SHORT: Mapping[str, str] = {
    "expert_label": "expert",
    "reference_match": "reference",
    "deterministic_check": "check",
    "model_judge": "judge",
}


def _table(
    caption: str, head: Iterable[str], rows: Iterable[Iterable[str]], empty: str, cls: str = ""
) -> str:
    """Cells are pre-escaped HTML."""
    head = list(head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    body = body or f'<tr><td class="empty" colspan="{len(head)}">{E(empty)}</td></tr>'
    heads = "".join(f"<th>{E(h)}</th>" for h in head)
    return f'<table class="{cls}"><caption>{E(caption)}</caption><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>'


def _count(c: Count | None) -> str:
    if c is None:
        return '<span class="na">not read</span>'
    decided = c.passed + c.failed
    rate = f"pass rate {c.passed / decided:.0%} of {decided} decided" if decided else "none decided"
    return (
        f'<span class="frac"><b>{c.passed} / {c.denominator}</b> pass</span>'
        f'<span class="sub">{c.failed} fail · {c.unknown} unknown · {rate}</span>'
    )


def render(view: ScorecardView) -> str:
    """The scorecard's tables as one HTML fragment; the page inserts it after an authenticated
    fetch. Every value is escaped."""
    title = {d.step: d.title for d in view.steps}
    short = lambda d: E(d[:12])  # noqa: E731
    parts = [
        f'<p class="meta">Dataset {E(view.dataset_version)} ({E(view.scope)}), generated {view.generated_at_utc:%Y-%m-%d %H:%M} UTC.'
        f" {len(view.episodes)} agent episodes. Published reference on this host for: {E(', '.join(view.reference_cases) or 'no case')}.</p>",
        '<details class="how"><summary>How to read this</summary><ul class="banners">'
        + "".join(f"<li>{E(c)}</li>" for c in view.caveats)
        + "</ul></details>",
        '<div id="lead"></div>',  # the page draws the heatmap and the run rows here from the view
        '<details class="fold" id="steps"><summary>What decides each step</summary>',
        _table(
            "One outcome per step: the first source in this order that reads pass or fail decides; the other readings stay on the cell",
            ("step", "question", "deciders, in precedence order", "trace opens at"),
            (
                (
                    f"<b>{E(d.title)}</b>",
                    E(d.question),
                    '<ol class="dec">'
                    + "".join(
                        f"<li><b>{E(SOURCE_TEXT[x.source])}</b> <code>{E(x.check)}</code>"
                        f"{' <i>(decides only on fail)</i>' if x.decides == 'fail_only' else ''}: {E(x.rule)}</li>"
                        for x in d.deciders
                    )
                    + "</ol>",
                    E(d.trace_anchor),
                )
                for d in view.steps
            ),
            "no steps",
            "steps",
        ),
        "</details>",
        '<details class="fold" id="matrix"><summary>Model by step, every source (table)</summary>',
        _table(
            "Pass rate per model and step. 'decided' is the cell outcome; each source row shows that source alone. Every count is over the model's episodes",
            ("model", "source", *(title[s] for s in STEPS)),
            (
                (
                    f'<b>{E(r.model)}</b><span class="sub">{r.episodes} episodes</span>',
                    "<b>decided</b>" if r.source == "decided" else E(SOURCE_TEXT[r.source]),
                    *(_count(r.cells.get(s)) for s in STEPS),
                )
                for r in view.matrix
            ),
            "no episodes",
            "matrix",
        ),
        "</details>",
        '<details class="fold" id="episodes"><summary>Every run, every column (table)</summary>',
        '<p class="hint">Click a step to open the trace at that step and the evidence for each source.</p>',
        _table(
            "One row per agent episode, one cell per step",
            (
                "case",
                "model",
                "run",
                "ending",
                "Rwp",
                *(title[s] for s in STEPS),
                "reward v0",
                "analyst coding",
            ),
            (
                (
                    E(e.case_id),
                    E(e.model),
                    f'<span class="mono">{E(e.job)}</span>',
                    E(e.end) + f'<span class="sub">{e.tool_calls} calls</span>',
                    "-" if e.rwp_percent is None else f"{e.rwp_percent:.2f}%",
                    *(
                        f'<button type="button" class="cell {s.outcome}" data-ep="{E(e.episode)}" data-step="{s.step}">{s.outcome}'
                        f"<small>{E(SHORT[s.source]) if s.source else 'no source'}{' · ' + E(s.failure_code) if s.failure_code else ''}</small></button>"
                        for s in e.steps
                    ),
                    f"<b>{e.reward.total:.2f}</b>",
                    (
                        f'<button type="button" class="cell analyst" data-ep="{E(e.episode)}" data-step="analyst">{E(e.analyst.code)}'
                        f"<small>{'disputed' if e.analyst.disputed else 'first divergence'}</small></button>"
                    )
                    if e.analyst
                    else ("none found" if e.analyst_none else '<span class="na">not coded</span>'),
                )
                for e in view.episodes
            ),
            "no episodes",
            "episodes",
        ),
        "</details>",
        '<h2 id="failures">Failure modes</h2>',
        _table(
            "Failed steps by code (seeded from the S1 to S11 legend of failures-trajectories.md), with the source that coded them",
            ("code", "failure", "step", "coded by", "check", "episodes", "runs"),
            (
                (
                    f"<b>{E(f.code)}</b>",
                    E(f.title),
                    E(title[f.step]),
                    "analyst coding" if f.source == "analyst" else E(SOURCE_TEXT[f.source]),
                    E(f.check),
                    f"{len(f.episodes)} / {f.denominator}"
                    + (
                        '<span class="sub">runs the analyst read</span>'
                        if f.source == "analyst"
                        else ""
                    ),
                    " ".join(
                        f'<a href="#ep-{E(x)}" class="mono" data-jump="{E(x)}">{E(_job(view, x))}</a>'
                        for x in f.episodes
                    ),
                )
                for f in view.failures
            ),
            "no failed steps",
        ),
        '<h2 id="rl">RL view: reward spec v0</h2>',
        '<p class="shortfall">No model was trained on this. The reward is declared so it can be inspected; it is not a validated training signal.</p>',
        _table(
            f"Reward spec {view.reward_spec.version}",
            ("component", "source", "check", "weight", "value"),
            (
                (
                    E(c.id),
                    E(SOURCE_TEXT[c.source]),
                    f"<code>{E(c.check)}</code>",
                    f"{c.weight:.2f}",
                    E(c.rule),
                )
                for c in view.reward_spec.components
            ),
            "no components",
        ),
        '<ul class="rules">'
        + "".join(f"<li>{E(r)}</li>" for r in view.reward_spec.rules)
        + "</ul>",
        _disputed_note(view),
        _table(
            "Model-judge items in the reward: agreement between judges and yes rates (no expert calibration exists)",
            ("judges", "paired targets", "comparable pairs", "agree", "Cohen's kappa"),
            (
                (
                    E(f"{p.model_a} vs {p.model_b}"),
                    str(p.paired_targets),
                    str(p.comparable),
                    f"{p.agree} / {p.comparable}",
                    "none" if p.kappa is None else f"{p.kappa:.2f}",
                )
                for p in view.judge_pairs
            ),
            "fewer than two judges",
        ),
        _table(
            "Judge answers",
            ("judge", "labels", "yes", "no", "cannot assess"),
            (
                (
                    E(j.model),
                    str(j.labels),
                    f"{j.yes} / {j.labels}",
                    str(j.no),
                    str(j.cannot_assess),
                )
                for j in view.judges
            ),
            "no model-judge labels",
        ),
        _reward_tables(view),
        '<h2 id="pairs">Preference pairs</h2>',
        '<p class="hint">Only an explicit preference (a reviewer naming chosen and rejected) is a pair. A correction is'
        " shown below as a before and after, unordered, until someone records a preference on it.</p>",
        _table(
            "Explicit preferences in the export, with the exposure and eligibility that decide training use",
            (
                "case",
                "producer",
                "rejected",
                "chosen",
                "Rwp rejected to chosen",
                "phases rejected",
                "phases chosen",
                "by",
                "reason",
                "eligible for review",
            ),
            (
                (
                    E(p.case_id),
                    E(p.model),
                    short(p.rejected),
                    short(p.chosen),
                    f"{'-' if p.rejected_rwp is None else f'{p.rejected_rwp:.2f}'} to {'-' if p.chosen_rwp is None else f'{p.chosen_rwp:.2f}'}",
                    E(", ".join(p.rejected_phases)),
                    E(", ".join(p.chosen_phases)),
                    E(p.author),
                    E(p.reason[:200]),
                    "yes" if p.review_eligible else f"no: {E(p.withheld_because or '')}",
                )
                for p in view.pairs
            ),
            "No explicit preference is in this export, so there are 0 pairs.",
        ),
        (
            f'<p><button type="button" id="pairs-download" {"disabled" if not any(p.review_eligible for p in view.pairs) else ""}>'
            f"Download {sum(p.review_eligible for p in view.pairs)} review-eligible pairs as JSONL (prompt, chosen, rejected,"
            f" metadata)</button> {sum(not p.review_eligible for p in view.pairs)} withheld pairs stay out of the file."
            " Training permission (artifact rights, contributor and model-output terms) is not checked here, so no pair is"
            " marked trainable.</p>"
        ),
        _table(
            "Correction candidates: a human path past an agent's answer that changed it, before and after, unordered",
            (
                "case",
                "producer",
                "before",
                "after",
                "edits",
                "Rwp before to after",
                "phases before",
                "phases after",
                "refit after the edit",
                "reference exposed",
                "preference use",
                "preference recorded",
                "demo pair",
            ),
            (
                (
                    E(c.case_id),
                    E(c.model),
                    short(c.before),
                    short(c.after),
                    E(", ".join(c.contributions)),
                    f"{'-' if c.before_rwp is None else f'{c.before_rwp:.2f}'} to {'-' if c.after_rwp is None else f'{c.after_rwp:.2f}'}",
                    E(", ".join(c.before_phases)),
                    E(", ".join(c.after_phases)),
                    "yes" if c.after_fit else "no",
                    "yes" if c.reference_exposed else "no",
                    E(c.preference_use),
                    "yes" if c.has_preference else "no",
                    "yes" if c.demo_pair else "no",
                )
                for c in view.candidates
            ),
            f"No human path in this export changed an agent's answer ({view.excluded_episodes} episodes excluded by the export policy).",
        ),
        f'<p class="meta">{view.unchanged_paths} human paths left the answer unchanged (refused or no-op edits) and are not listed.</p>',
        '<h3 id="demo">Demo pairs (revealed, owner account)</h3>',
        f'<p class="shortfall">Demo bucket: owner-account records made with the reference revealed. Never expert labels,'
        f" never eligible for review or training, never in the file above. {E(DEMO_PAIR_RULE)}</p>",
        _table(
            "The on-camera loop: an agent answer, the demo rater's refit, then the rater's sign-off",
            (
                "case",
                "producer",
                "rejected (agent answer)",
                "chosen (demo refit)",
                "Rwp rejected to chosen",
                "phases rejected",
                "phases chosen",
                "edits",
                "refit at",
                "signed off at",
                "sign-off verdict",
                "by",
            ),
            (
                (
                    E(d.case_id),
                    E(d.model),
                    short(d.rejected),
                    short(d.chosen),
                    f"{'-' if d.rejected_rwp is None else f'{d.rejected_rwp:.2f}'} to {'-' if d.chosen_rwp is None else f'{d.chosen_rwp:.2f}'}",
                    E(", ".join(d.rejected_phases)),
                    E(", ".join(d.chosen_phases)),
                    E(", ".join(d.contributions)),
                    f"{d.refit_at_utc:%H:%M:%S} UTC",
                    f"{d.signed_at_utc:%H:%M:%S} UTC",
                    E(d.signoff_verdict),
                    E(d.author),
                )
                for d in view.demo_pairs
            ),
            "No demo pair in this export.",
        ),
        f'<p><button type="button" id="demo-download" {"disabled" if not view.demo_pairs else ""}>Download {len(view.demo_pairs)}'
        " demo pairs as JSONL (revealed bucket; not for training or expert counts)</button></p>",
        '<h2 id="reconcile">Reconciliation with the dataset manifest</h2>',
        _table(
            "Every count on this page against the manifest field it must equal",
            ("count", "page", "manifest", "field", "status"),
            (
                (
                    E(r.name),
                    str(r.page),
                    str(r.manifest),
                    f"<code>{E(r.manifest_field)}</code>",
                    "ok" if r.ok else "<b>MISMATCH</b>",
                )
                for r in view.reconciliation
            ),
            "nothing to reconcile",
        ),
    ]
    return "\n".join(parts)


def _disputed_note(view: ScorecardView) -> str:
    """Episodes that lose the reference term only on a disputed phase: what a reference-match
    reward would teach a policy to stop reporting."""
    hit = [
        e
        for e in view.episodes
        if e.snapshot is not None  # scored under the reward; no snapshot scores 0 everywhere
        for s in e.steps
        if s.step == "phase_selection"
        for r in s.readings
        if r.source == "reference_match" and r.outcome == "fail" and "disputed" in r.detail
    ]
    if not hit:
        return ""
    cases = ", ".join(sorted({e.case_id for e in hit}))
    return (
        f'<p class="shortfall">{len(hit)} of {len(view.episodes)} episodes ({cases}) lose the reference term only on a'
        " disputed phase the source authors report. Trained on reward v0 as declared, a policy would learn to drop that"
        " call. An expert label on these answers should decide before any reference term is used as reward.</p>"
    )


def _job(view: ScorecardView, episode: str) -> str:
    return next((e.job for e in view.episodes if e.episode == episode), episode[:12])


def _reward_tables(view: ScorecardView) -> str:
    ids = [c.id for c in view.reward_spec.components]
    per_episode = _table(
        "Reward per episode under v0 (value x weight; a dash marks a missing reading scored 0)",
        ("case", "model", "run", *ids, "verifiable", "judge", "total"),
        (
            (
                E(e.case_id),
                E(e.model),
                f'<span class="mono">{E(e.job)}</span>',
                *(
                    f'<span title="{E(t.detail)}">{"-" if t.missing else f"{t.value * t.weight:.2f}"}</span>'
                    for t in e.reward.terms
                ),
                f"{e.reward.verifiable:.2f}",
                f"{e.reward.judge:.2f}",
                f"<b>{e.reward.total:.2f}</b>",
            )
            for e in view.episodes
        ),
        "no episodes",
        "reward",
    )
    by_model: dict[str, list[EpisodeScore]] = defaultdict(list)
    for e in view.episodes:
        by_model[e.model].append(e)
    per_model = _table(
        "Mean reward per model and case (n episodes each)",
        ("model", "case", "episodes", "mean verifiable", "mean judge", "mean total"),
        (
            (
                E(m),
                E(c),
                str(len(g)),
                f"{sum(e.reward.verifiable for e in g) / len(g):.2f}",
                f"{sum(e.reward.judge for e in g) / len(g):.2f}",
                f"<b>{sum(e.reward.total for e in g) / len(g):.2f}</b>",
            )
            for m in sorted(by_model)
            for c in sorted({e.case_id for e in by_model[m]})
            for g in ([e for e in by_model[m] if e.case_id == c],)
        ),
        "no episodes",
    )
    return per_model + per_episode


# --- routes ------------------------------------------------------------------------------------

router = APIRouter(tags=["workbench-scorecard"])


def _admin_owner(principal: Annotated[Principal, Depends(authenticate_principal)]) -> Principal:
    if not is_admin_owner(principal):
        raise HTTPException(status_code=403, detail="the scorecard is for the admin owner")
    return principal


def _dataset() -> Path:
    dataset = Path(os.environ.get("AXA_WORKBENCH_DATASET", str(DEFAULT_DATASET)))
    if not (dataset / "manifest.json").is_file():
        raise HTTPException(status_code=503, detail="no eval dataset is built")
    return dataset


@router.get("/scorecard", include_in_schema=False)
def get_scorecard_page() -> FileResponse:
    return FileResponse(ASSETS / "scorecard.html", media_type="text/html")


@router.get("/api/attempts/scorecard")
def get_scorecard(principal: Annotated[Principal, Depends(_admin_owner)]) -> dict[str, Any]:
    """The view and its rendered tables for the dataset at $AXA_WORKBENCH_DATASET."""
    view = scorecard_view(_dataset())
    return {"view": view.model_dump(mode="json"), "html": render(view)}


@router.get("/api/attempts/scorecard/pairs.jsonl")
def get_pairs(principal: Annotated[Principal, Depends(_admin_owner)]) -> Response:
    return Response(
        preference_lines(_dataset()),
        media_type="application/jsonl",
        headers={"Content-Disposition": 'attachment; filename="preference-pairs.jsonl"'},
    )


@router.get("/api/attempts/scorecard/demo-pairs.jsonl")
def get_demo_pairs(principal: Annotated[Principal, Depends(_admin_owner)]) -> Response:
    """The revealed demo bucket, apart from the review-eligible file."""
    return Response(
        demo_pair_lines(_dataset()),
        media_type="application/jsonl",
        headers={"Content-Disposition": 'attachment; filename="demo-pairs-revealed.jsonl"'},
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the scorecard's reconciliation with the dataset manifest."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("AXA_WORKBENCH_DATASET", str(DEFAULT_DATASET))),
    )
    parser.add_argument(
        "--pairs-out",
        type=Path,
        default=None,
        help="also write the review-eligible preference pairs JSONL here",
    )
    args = parser.parse_args()
    view = scorecard_view(args.dataset)
    print(reconciliation_text(view))
    if args.pairs_out:
        args.pairs_out.write_bytes(preference_lines(args.dataset))
        print(
            f"pairs: {sum(p.review_eligible for p in view.pairs)} review-eligible of {len(view.pairs)} written to {args.pairs_out}"
        )
    raise SystemExit(0 if all(r.ok for r in view.reconciliation) else 1)


if __name__ == "__main__":
    main()
