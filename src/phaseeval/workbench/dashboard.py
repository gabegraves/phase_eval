"""Evaluation dashboard: a pure summary of one eval dataset directory, the routing ledger and the
qualification reports (U8; origin R20, R34; final architect contract row U8).

Target path: src/phaseeval/workbench/dashboard.py (public). W6 export. GET /dashboard redirects to
/scorecard, which carries these tables (owner directive 2026-10-22). GET /api/attempts/dashboard returns the view and its
rendered tables to the admin owner only: the view names models and harnesses, so label-mode and
rater credentials are refused (Build D11). It stores nothing; every number is recomputed from
the files exports.write_dataset wrote, so it cannot disagree with the export it reads.

Rules it renders and never relaxes:
- Every count shows its denominator.
- Agreement is computed within one rater kind, per rubric item, only over matching snapshot,
  item, phase, rubric and guidance versions, from one initial blind response per distinct rater
  (the earliest). Rows show targets, one-sided targets, comparable and cannot-assess pairs, so
  an agreement rate is read as conditional. Revealed records are revisions, counted apart.
  Judge rows also show the frozen expected targets and those no judge answered; human rows mark
  that scope unavailable. Model judges are compared per frozen prompt, using the earliest observed
  run per (model, prompt), inferred from label times rather than read from a declared run:
  a repeat run, a second principal running the same model, or another prompt is never counted
  as another rater. Judge agreement is shown with and without labels on the judge's own model's
  episodes.
- No judge-versus-expert number is computed here. It appears only from a qualification report
  for a judge subject, which Qualifier.run refuses without a frozen split (origin R34), shown
  as the split's test partition with the source exposure of its roots; without one the judge
  table says no eligible report is available, and split status is read only from frozen sets.
- A human correction is not an expert correction: an attempt records no rater kind.

Routing decisions come from the triage ledger (triage.Triage.decisions, one read per bound
policy version); qualification reports from the private export's reports.jsonl
(qualification.QualificationReport). The routing table and the expert queue join each decision to
the store as it is now: its frozen snapshot (case, origin), the imported run under that origin
(model, harness) and the blind sign-offs on it, so a sign-off made after the export shows.
"""

from __future__ import annotations

import html
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from phaseeval.api.auth import Principal
from phaseeval.api.routes.attempts import authenticate_principal

from .exports import DatasetManifest, EpisodeRecord, JudgeManifest, stored_snapshots
from .labels import Label, LabelStore, SignOff, is_admin_owner
from .model import ChildAttempt, Completed, ImportedRunResult, ImportRun, Record, digest_of
from .qualification import (
    Interval,
    JudgeSummary,
    ProducerSummary,
    QualificationReport,
    QualificationSet,
    Qualifier,
)
from .seams import NoReference
from .snapshot import AnalysisSnapshot
from .store import WorkbenchStore
from .triage import (
    CHECK_ORDER,
    CheckOutcome,
    CheckPolicy,
    RouteDecision,
    Triage,
    audit_pick,
    expert_queue,
)

DEFAULT_DATASET = Path.home() / ".local/state/axa-workbench/dataset-v0"
RUBRIC_ITEMS = ("phase_supported", "phase_plausible", "amounts_defensible", "fit_supports_conclusion")
CHECK_CODES = ("K1", "K2", "K3", "K5", "K6")
CHECK_LEGEND = (
    ("K1", "Rietveld did not degrade after a good Pawley fit."),
    ("K2", "No unexplained residual peak at or above 5 percent of the strongest net peak; weaker ones down to 2 percent"
     " are listed for review, never flagged."),
    ("K3", "Largest lattice shift from the CIF cell. Advisory: it is shown and never routes."),
    ("K5", "The reported amounts equal the cited fit's amounts within the policy's wt%."),
    ("K6", "Sample displacement against its limit, how far the amounts moved between a refit with texture off"
     " and the final fit, and what the fit assumed. Advisory: it is shown and never routes."),
    ("No K4", "The old sums-to-100 check was dropped; K5 (the answer equals the cited fit) replaced it."),
)


def _yes(value: bool) -> str:
    return "yes" if value else "no"


class Ratio(Record):
    count: int
    denominator: int


class OutcomeRow(Record):
    case_id: str
    model: str
    harness: str
    episodes: int  # the denominator of every ratio in the row
    submitted: Ratio
    unsuccessful: Ratio
    invalid: Ratio
    contaminated: Ratio
    incomplete: Ratio
    human_corrected: Ratio  # a human's path past the answer; an attempt records no rater kind
    snapshots: Ratio  # frozen answer snapshots / submitted episodes


class CheckCount(Record):
    code: str
    passed: Ratio
    flagged: Ratio
    not_computable: Ratio


class SnapshotRow(Record):
    snapshot: str
    case_id: str
    model: str | None  # None for a human correction
    answered: bool
    judged: bool
    checks: dict[str, str]
    residual_peaks: int
    chemistry: bool
    labels_by_kind: dict[str, int]
    signoffs: int


class ExclusionRow(Record):
    kind: str
    reason: str
    count: int
    refs: tuple[str, ...]


class RaterLabelRow(Record):
    snapshot: str
    case_id: str
    rater: str  # pseudonym
    kind: str
    exposure: str
    yes: int
    no: int
    cannot_assess: int


AgreementKind = Literal["expert", "owner", "model_judge", "model_judge_without_own_model"]


class ItemAgreementRow(Record):
    kind: AgreementKind
    item: str
    prompt_sha256: str | None  # model judges: one row per frozen prompt; never pooled across prompts
    raters: tuple[str, ...]  # the selected raters: pseudonyms, or "<model>@<initial run>" for judges
    targets: int  # observed: snapshot-item-phase targets at least one selected rater answered
    one_sided_targets: int  # targets only one selected rater answered: no pair, counted apart
    # The frozen expected targets (judges: every phase item of every judged snapshot plus its
    # analysis items) and those no selected rater answered. None: the scope is not in the export.
    expected_targets: int | None
    wholly_missing_targets: int | None
    comparable_pairs: int
    agreeing_pairs: int
    agreement: float | None  # agreeing / comparable; None when no comparable pair
    cannot_assess_pairs: int
    revealed_revisions: int


class JudgeCoverageRow(Record):
    """One model-judge run over the frozen judge targets, counted per target from the run's
    earliest blind label on it. answered + cannot_assess + missing == expected_targets."""

    prompt_sha256: str
    judge: str  # "<model>@<run>"
    initial: bool  # the run the agreement table selects for this model and prompt
    expected_targets: int  # every phase item of every judged snapshot plus its analysis items
    answered: int  # yes or no
    cannot_assess: int
    missing: int  # no label from this run: a malformed, refused or never-made call lands here
    judged_snapshots: int
    snapshots_with_labels: int


class RouteCountRow(Record):
    policy_version: str
    provisional: bool
    route: str
    count: Ratio  # decisions with this route / decisions under this policy version
    unmeasured_flags: int  # flagged only by a check whose basis is "unmeasured" (TR3)


class RoutingRow(Record):
    """One routing decision as the admin owner reads it: it names the producer."""

    snapshot: str
    case_id: str | None  # None: the snapshot is not in the store
    model: str | None  # None: no imported run at the snapshot's origin
    harness: str | None
    policy_version: str
    provisional: bool
    route: str
    why: str
    audit: Literal["picked", "not picked", "not drawn: flagged"]  # the seeded draw over unflagged answers
    checks: tuple[CheckOutcome, ...]  # every routing check and every frozen advisory one
    blind_signoffs: int  # distinct raters whose sign-off on this snapshot is stamped blind
    revealed_signoffs: int  # distinct raters with only a revealed sign-off (reveal open or declared exposed)


class ExpertQueueView(Record):
    """triage.expert_queue under one policy version; every count is over its decisions."""

    policy_version: str
    provisional: bool
    routed: Ratio
    flagged: Ratio
    audit: Ratio
    cleared: Ratio  # not routed; still stored, exportable and reviewable
    not_computable: Ratio  # a routing check not computed (always routed: fail closed)
    unmeasured_flags: Ratio  # flagged only by a check whose basis is "unmeasured"
    order: tuple[str, ...]  # the routed snapshots in queue order: flagged, then audit


class Rate(Record):
    k: int
    n: int
    rate: float | None


class ProducerRow(Record):
    model_id: str
    harness: str
    harness_version: str
    set_digest: str
    roots: int
    by_verdict: dict[str, int]
    accepted: Rate  # accepted side over assessed roots; abstentions fail
    accepted_applicable: int  # roots with an accepted analysis: the rate's context, never its n
    rejected: Rate  # rejected side over assessed roots; abstentions pass
    rejected_applicable: int


class JudgeAgreementRow(Record):
    model: str
    prompt_sha256: str
    set_digest: str
    item: str
    eligible: int
    coverage: Rate  # eligible targets the judge answered yes or no
    abstained: int  # judge cannot_assess where an expert committed, counted per run
    # Unavailable (None): the report counts answered targets uniquely but abstentions per run, so
    # eligible - answered - abstained mixes units and is not a missing count.
    missing: int | None
    exposure: str  # the source exposure of the report's roots
    n: int
    judge_expert: Rate
    expert_expert: Rate | None
    tpr: Rate
    tnr: Rate
    judge_abstained: int


class DashboardView(Record):
    generated_at_utc: datetime
    dataset_version: str
    scope: str
    banners: tuple[str, ...]
    outcomes: tuple[OutcomeRow, ...]
    checks: tuple[CheckCount, ...]
    snapshots: tuple[SnapshotRow, ...]
    exclusions: tuple[ExclusionRow, ...]
    labels: tuple[RaterLabelRow, ...]
    agreement: tuple[ItemAgreementRow, ...]
    judge_coverage: tuple[JudgeCoverageRow, ...]
    judge_shortfall: tuple[str, ...]  # why no judge-versus-expert number is shown; empty when shown
    split_status: tuple[str, ...]  # read from frozen qualification sets, never inferred
    judge_agreement: tuple[JudgeAgreementRow, ...]
    routes: tuple[RouteCountRow, ...]
    routing: tuple[RoutingRow, ...] = ()
    expert_queues: tuple[ExpertQueueView, ...] = ()
    producers: tuple[ProducerRow, ...]


def _rate(interval: Interval) -> Rate:
    return Rate(k=interval.k, n=interval.n, rate=interval.rate)


# --- the view ----------------------------------------------------------------------------------


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines() if path.is_file() else []


def _initial_labels(labels: Iterable[Label]) -> dict[tuple, dict[str, Label]]:
    """(snapshot, item, phase, rubric, guidance) -> rater -> that rater's earliest blind label."""
    found: dict[tuple, dict[str, Label]] = defaultdict(dict)
    for label in labels:
        if label.exposure != "blind":
            continue
        key = (label.snapshot, label.item, label.phase_id, label.rubric_version, label.guidance_version)
        held = found[key].get(label.rater.principal_id)
        if held is None or label.created_at_utc < held.created_at_utc:
            found[key][label.rater.principal_id] = label
    return found


def _initial_judge_labels(labels: Iterable[Label]) -> dict[str, dict[tuple, dict[str, Label]]]:
    """prompt -> target -> "<model>@<run>" -> label. Per (model, prompt) the selected run is the
    earliest observed one (inferred from label times, not read from a declared run); every other
    run of that model and prompt is a repeat, and another principal running the same model is the
    same rater."""
    by_stratum: dict[tuple[str, str], dict[str, list[Label]]] = defaultdict(lambda: defaultdict(list))
    for label in labels:
        if label.exposure == "blind" and label.rater.judge is not None:
            judge = label.rater.judge
            by_stratum[(judge.prompt_sha256, judge.model)][judge.run_id].append(label)
    found: dict[str, dict[tuple, dict[str, Label]]] = defaultdict(lambda: defaultdict(dict))
    for (prompt, model), runs in by_stratum.items():
        run_id = min(runs, key=lambda run: (min(lb.created_at_utc for lb in runs[run]), run))
        for label in runs[run_id]:
            key = (label.snapshot, label.item, label.phase_id, label.rubric_version, label.guidance_version)
            held = found[prompt][key].get(f"{model}@{run_id}")
            if held is None or label.created_at_utc < held.created_at_utc:
                found[prompt][key][f"{model}@{run_id}"] = label
    return found


def _judge_coverage(labels: Iterable[Label], expected: frozenset[tuple]) -> tuple[JudgeCoverageRow, ...]:
    runs: dict[tuple[str, str, str], dict[tuple, Label]] = defaultdict(dict)
    first_seen: dict[tuple[str, str, str], datetime] = {}
    for label in labels:
        judge = label.rater.judge
        if label.exposure != "blind" or judge is None:
            continue
        run = (judge.prompt_sha256, judge.model, judge.run_id)
        first_seen[run] = min(first_seen.get(run, label.created_at_utc), label.created_at_utc)
        target = (label.snapshot, label.item, label.phase_id)
        held = runs[run].get(target)
        if target in expected and (held is None or label.created_at_utc < held.created_at_utc):
            runs[run][target] = label
    initial = {}
    for prompt, model, run_id in sorted(first_seen, key=lambda r: (first_seen[r], r[2])):
        initial.setdefault((prompt, model), run_id)
    judged = {t[0] for t in expected}
    rows = []
    for prompt, model, run_id in sorted(first_seen):
        found = runs[(prompt, model, run_id)]
        answers = Counter(lb.answer for lb in found.values())
        rows.append(
            JudgeCoverageRow(
                prompt_sha256=prompt,
                judge=f"{model}@{run_id}",
                initial=initial[(prompt, model)] == run_id,
                expected_targets=len(expected),
                answered=answers["yes"] + answers["no"],
                cannot_assess=answers["cannot_assess"],
                missing=len(expected) - len(found),
                judged_snapshots=len(judged),
                snapshots_with_labels=len({t[0] for t in found}),
            )
        )
    return tuple(rows)


def _agreement_rows(
    kind: AgreementKind,
    item: str,
    prompt: str | None,
    initial: Mapping[tuple, Mapping[str, Label]],
    revisions: int,
    expected: frozenset[tuple] | None,
) -> ItemAgreementRow:
    comparable = agreeing = cannot = one_sided = targets = 0
    raters: set[str] = set()
    observed: set[tuple] = set()
    for key, by_rater in initial.items():
        if key[1] != item:
            continue
        targets += 1
        observed.add(key[:3])
        raters |= set(by_rater)
        one_sided += len(by_rater) == 1
        for a, b in combinations(sorted(by_rater), 2):
            left, right = by_rater[a].answer, by_rater[b].answer
            if "cannot_assess" in (left, right):
                cannot += 1
            else:
                comparable += 1
                agreeing += left == right
    return ItemAgreementRow(
        kind=kind,
        item=item,
        prompt_sha256=prompt,
        raters=tuple(sorted(raters)),
        targets=targets,
        one_sided_targets=one_sided,
        comparable_pairs=comparable,
        agreeing_pairs=agreeing,
        agreement=agreeing / comparable if comparable else None,
        cannot_assess_pairs=cannot,
        revealed_revisions=revisions,
        expected_targets=None if expected is None else sum(1 for t in expected if t[1] == item),
        wholly_missing_targets=None if expected is None else sum(1 for t in expected if t[1] == item and t not in observed),
    )


def _agreement(
    kind: AgreementKind, labels: list[Label], expected: frozenset[tuple] | None = None
) -> tuple[ItemAgreementRow, ...]:
    def revisions(item: str, prompt: str | None) -> int:
        return sum(
            1
            for lb in labels
            if lb.item == item
            and lb.exposure == "revealed"
            and (prompt is None or (lb.rater.judge is not None and lb.rater.judge.prompt_sha256 == prompt))
        )

    if not kind.startswith("model_judge"):
        initial = _initial_labels(labels)
        return tuple(_agreement_rows(kind, item, None, initial, revisions(item, None), None) for item in RUBRIC_ITEMS)
    strata = _initial_judge_labels(labels)
    if not strata:
        return tuple(_agreement_rows(kind, item, None, {}, 0, expected) for item in RUBRIC_ITEMS)
    return tuple(
        _agreement_rows(kind, item, prompt, strata[prompt], revisions(item, prompt), expected)
        for prompt in sorted(strata)
        for item in RUBRIC_ITEMS
    )


def dashboard_view(
    dataset: Path,
    routes: Iterable[RouteDecision] = (),
    reports: Iterable[QualificationReport] = (),
    qualification_sets: Iterable[QualificationSet] = (),
    now: datetime | None = None,
    routing: Iterable[RoutingRow] = (),
    expert_queues: Iterable[ExpertQueueView] = (),
) -> DashboardView:
    manifest = DatasetManifest.model_validate_json((dataset / "manifest.json").read_bytes())
    episodes = [EpisodeRecord.model_validate_json(line) for line in _lines(dataset / "episodes.jsonl")]
    snapshots = [AnalysisSnapshot.model_validate_json(line) for line in _lines(dataset / "snapshots.jsonl")]
    expert_file = [Label.model_validate_json(line) for line in _lines(dataset / "labels.expert.jsonl")]
    judge_file = [Label.model_validate_json(line) for line in _lines(dataset / "labels.model_judge.jsonl")]
    signoffs = [SignOff.model_validate_json(line) for line in _lines(dataset / "signoffs.jsonl")]
    judge_path = dataset / "judge_manifest.json"
    judged = (
        {e.snapshot_digest for e in JudgeManifest.model_validate_json(judge_path.read_bytes()).entries if e.judged}
        if judge_path.is_file()
        else {e.digest for e in manifest.snapshot_index if e.judged}
    )

    # Outcomes per case, model and harness, from each episode's imported runs.
    run_of: dict[str, tuple[str, str, str]] = {}
    tally: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    scans: Counter[str] = Counter()
    for episode in episodes:
        keys = []
        for step in episode.steps:
            outcome = step.attempt.outcome
            if not isinstance(outcome, Completed):
                continue
            scans[outcome.custody_scan] += 1
            if isinstance(step.attempt.action, ImportRun) and isinstance(outcome.result, ImportedRunResult):
                agent = outcome.result.agent
                keys.append(run_of.setdefault(step.digest, (episode.root.case.case_id, agent.model_id, agent.harness)))
        if not keys:
            continue
        counts = tally[keys[0]]
        if episode.human_contributions:
            counts["human_corrected"] += 1  # a human's path past the answer; the agent episode is its own row
            continue
        counts["episodes"] += 1
        counts[episode.status.status] += 1
        counts["incomplete"] += any(not run.complete for run in episode.runs)
    snapshots_by_run = Counter(run_of[s.origin] for s in snapshots if s.kind == "agent_answer" and s.origin in run_of)
    outcomes = []
    for key in sorted(tally):
        counts, total = tally[key], tally[key]["episodes"]
        outcomes.append(
            OutcomeRow(
                case_id=key[0],
                model=key[1],
                harness=key[2],
                episodes=total,
                **{
                    name: Ratio(count=counts[name], denominator=total)
                    for name in ("submitted", "unsuccessful", "invalid", "contaminated", "incomplete", "human_corrected")
                },
                snapshots=Ratio(count=snapshots_by_run[key], denominator=counts["submitted"]),
            )
        )

    n_snap = len(snapshots)
    status_of = [{c.code: c.status for c in s.checks} for s in snapshots]
    checks = tuple(
        CheckCount(
            code=code,
            passed=Ratio(count=sum(1 for s in status_of if s.get(code) == "pass"), denominator=n_snap),
            flagged=Ratio(count=sum(1 for s in status_of if s.get(code) == "flag"), denominator=n_snap),
            not_computable=Ratio(count=sum(1 for s in status_of if s.get(code, "not_computable") == "not_computable"), denominator=n_snap),
        )
        for code in CHECK_CODES
    )
    labels_all = expert_file + judge_file
    snapshot_rows = tuple(
        SnapshotRow(
            snapshot=digest,
            case_id=snap.case_id,
            model=run_of.get(snap.origin, (None, None, None))[1],
            answered=snap.answered,
            judged=digest in judged,
            checks={c.code: c.status for c in snap.checks},
            residual_peaks=len(snap.residual_peaks),
            chemistry=snap.chemistry is not None,
            labels_by_kind=dict(Counter(lb.rater.kind for lb in labels_all if lb.snapshot == digest)),
            signoffs=sum(1 for so in signoffs if so.snapshot == digest),
        )
        for digest, snap in sorted((_digest(s), s) for s in snapshots)
    )
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for exclusion in manifest.exclusions:
        for reason in exclusion.reasons:
            grouped[(exclusion.kind, reason[:100])].append(exclusion.ref)
    exclusions = tuple(
        ExclusionRow(kind=kind, reason=reason, count=len(refs), refs=tuple(refs)) for (kind, reason), refs in sorted(grouped.items())
    )
    case_of = {_digest(s): s.case_id for s in snapshots}
    per_rater: dict[tuple, Counter[str]] = defaultdict(Counter)
    for label in labels_all:
        who = label.rater.judge.model if label.rater.judge else label.rater.principal_id  # a judge is its model
        per_rater[(label.snapshot, who, label.rater.kind, label.exposure)][label.answer] += 1
    label_rows = tuple(
        RaterLabelRow(
            snapshot=snap, case_id=case_of.get(snap, "?"), rater=rater, kind=kind, exposure=exposure,
            yes=c["yes"], no=c["no"], cannot_assess=c["cannot_assess"],
        )
        for (snap, rater, kind, exposure), c in sorted(per_rater.items())
    )
    # Frozen judge targets: every phase item of every judged snapshot, plus its analysis items.
    judge_targets = frozenset(
        target
        for snap in snapshots
        if (digest := _digest(snap)) in judged
        for target in (
            *((digest, item, p.phase_id) for item in ("phase_supported", "phase_plausible") for p in snap.phases),
            *((digest, item, None) for item in ("amounts_defensible", "fit_supports_conclusion")),
        )
    )
    agreement = (
        _agreement("expert", [lb for lb in expert_file if lb.rater.kind == "expert"])
        + _agreement("owner", [lb for lb in expert_file if lb.rater.kind == "owner"])
        + _agreement("model_judge", judge_file, judge_targets)
        + _agreement(
            "model_judge_without_own_model",
            [lb for lb in judge_file if not (lb.rater.judge and lb.rater.judge.judge_is_actor_model)],
            judge_targets,
        )
    )

    judge_coverage = _judge_coverage(judge_file, judge_targets)

    route_rows = _route_rows(routes)
    reports = tuple(reports)
    producers = tuple(
        ProducerRow(
            model_id=r.subject.model_id,
            harness=r.subject.harness,
            harness_version=r.subject.harness_version,
            set_digest=r.set_digest,
            roots=r.summary.roots,
            by_verdict=dict(sorted(r.summary.by_verdict.items())),
            accepted=_rate(r.summary.accepted_pass_rate),
            accepted_applicable=r.summary.accepted_applicable,
            rejected=_rate(r.summary.rejected_pass_rate),
            rejected_applicable=r.summary.rejected_applicable,
        )
        for r in reports
        if isinstance(r.summary, ProducerSummary)
    )
    judge_reports = [r for r in reports if isinstance(r.summary, JudgeSummary)]
    exposure_of = {e.root_digest: e.root.case.exposure for e in episodes}

    def exposure(report: QualificationReport) -> str:
        kinds = sorted({exposure_of.get(c.root, "unknown") for c in report.cases})
        caveat = " (not held-out scientific validation)" if "development_exposed" in kinds else ""
        return ", ".join(kinds) + caveat

    judge_agreement = tuple(
        JudgeAgreementRow(
            model=r.subject.model,
            prompt_sha256=r.subject.prompt_sha256,
            set_digest=r.set_digest,
            item=item.item,
            eligible=item.eligible,
            coverage=_rate(item.coverage),
            abstained=item.judge_abstained,
            missing=None,  # unavailable until the report counts abstained targets uniquely
            exposure=exposure(r),
            n=item.n,
            judge_expert=_rate(item.judge_expert_pairwise),
            expert_expert=_rate(item.expert_expert_pairwise) if item.expert_expert_pairwise else None,
            tpr=_rate(item.tpr),
            tnr=_rate(item.tnr),
            judge_abstained=item.judge_abstained,
        )
        for r in judge_reports
        for item in r.summary.items
    )
    n_expert = sum(1 for lb in expert_file if lb.rater.kind == "expert")
    n_owner = sum(1 for lb in expert_file if lb.rater.kind == "owner")
    split_status = tuple(
        f"Qualification set {digest_of(q)[:12]} ({q.lab}/{q.name}): "
        + (
            f"split frozen, {sum(p == 'held_out' for p in q.split.values())} of {len(q.roots)} roots in the test partition"
            if q.split
            else "frozen without a split"
        )
        for q in qualification_sets
    ) or ("No qualification set is frozen.",)
    shortfall = []
    if not judge_reports:
        shortfall.append("No eligible judge qualification report available, so no judge-versus-expert number is shown.")
    if n_expert == 0:
        shortfall.append(f"No expert-kind labels: 0 expert labels ({n_owner} owner labels, {len(judge_file)} model-judge labels).")

    banners = ["Model-judge labels are model judgments, not expert labels. Agreement is computed within one rater kind."]
    banners += ["Training hold in force: no learning gains are claimed."]
    if n_expert == 0:
        banners.append("No expert-kind labels yet.")
    if not_scanned := scans.get("not_configured", 0):
        banners.append(f"Custody scan not configured on {not_scanned} of {sum(scans.values())} completed attempts.")
    if later := sum(1 for e in manifest.snapshot_index if not e.judged):
        banners.append(f"{later} of {len(manifest.snapshot_index)} snapshots were frozen after the judge manifest (judged false).")
    if any(r.provisional for r in route_rows):
        banners.append("Routing thresholds are provisional.")
    return DashboardView(
        generated_at_utc=now or datetime.now(UTC),
        dataset_version=manifest.dataset_version,
        scope=manifest.scope,
        banners=tuple(banners),
        outcomes=tuple(outcomes),
        checks=checks,
        snapshots=snapshot_rows,
        exclusions=exclusions,
        labels=label_rows,
        agreement=agreement,
        judge_coverage=judge_coverage,
        judge_shortfall=tuple(shortfall),
        split_status=split_status,
        judge_agreement=judge_agreement,
        routes=route_rows,
        routing=tuple(routing),
        expert_queues=tuple(expert_queues),
        producers=producers,
    )


def _digest(snapshot: AnalysisSnapshot) -> str:
    return digest_of(snapshot)


def _route_rows(decisions: Iterable[RouteDecision]) -> tuple[RouteCountRow, ...]:
    by_version: dict[str, list[RouteDecision]] = defaultdict(list)
    for decision in decisions:
        by_version[decision.policy_version].append(decision)
    rows = []
    for version, group in sorted(by_version.items()):
        provisional = any(d.provisional for d in group)
        unmeasured = Counter(d.route for d in group if d.unmeasured_flag)
        for route, count in sorted(Counter(d.route for d in group).items()):
            rows.append(
                RouteCountRow(
                    policy_version=version,
                    provisional=provisional,
                    route=route,
                    count=Ratio(count=count, denominator=len(group)),
                    unmeasured_flags=unmeasured[route],
                )
            )
    return tuple(rows)


def routing_view(
    ledger: Iterable[tuple[CheckPolicy, Iterable[RouteDecision]]],
    snapshots: Mapping[str, AnalysisSnapshot],
    producers: Mapping[str, tuple[str, str]],
    signoffs: Iterable[SignOff],
) -> tuple[tuple[RoutingRow, ...], tuple[ExpertQueueView, ...]]:
    """Pure: one row per decision and one expert queue per policy version. `producers` maps an
    imported run's attempt digest to its (model, harness)."""
    signed: dict[str, dict[str, set[str]]] = {"blind": defaultdict(set), "revealed": defaultdict(set)}
    for signoff in signoffs:
        signed[signoff.exposure][signoff.snapshot].add(signoff.rater.principal_id)
    rows, queues = [], []
    for policy, group in ledger:
        group = tuple(group)
        queue = expert_queue(policy, group)
        n = len(group)
        queues.append(
            ExpertQueueView(
                policy_version=policy.version,
                provisional=policy.provisional,
                routed=Ratio(count=len(queue.routed), denominator=n),
                flagged=Ratio(count=queue.counts["expert_flagged"], denominator=n),
                audit=Ratio(count=queue.counts["expert_audit"], denominator=n),
                cleared=Ratio(count=queue.counts["cleared"], denominator=n),
                not_computable=Ratio(count=queue.not_computable, denominator=n),
                unmeasured_flags=Ratio(count=queue.unmeasured_flags, denominator=n),
                order=tuple(d.snapshot for d in queue.routed),
            )
        )
        for decision in group:
            snap = snapshots.get(decision.snapshot)
            model, harness = producers.get(snap.origin, (None, None)) if snap else (None, None)
            causes = [f"{c.code} {c.status.replace('_', ' ')}" for c in decision.checks if c.routes and c.status != "pass"]
            flagged = decision.route == "expert_flagged"
            rows.append(
                RoutingRow(
                    snapshot=decision.snapshot,
                    case_id=snap.case_id if snap else None,
                    model=model,
                    harness=harness,
                    policy_version=decision.policy_version,
                    provisional=decision.provisional,
                    route=decision.route,
                    why=(
                        "routing check " + ", ".join(causes)
                        if flagged
                        else "audit sample of unflagged answers"
                        if decision.route == "expert_audit"
                        else "no routing check fired; not drawn for audit"
                    ),
                    audit="not drawn: flagged" if flagged else "picked" if audit_pick(decision.snapshot, policy) else "not picked",
                    checks=decision.checks,
                    blind_signoffs=len(signed["blind"][decision.snapshot]),
                    revealed_signoffs=len(signed["revealed"][decision.snapshot] - signed["blind"][decision.snapshot]),
                )
            )
    rows.sort(key=lambda r: (r.policy_version, r.case_id or "", r.snapshot))
    return tuple(rows), tuple(queues)


# --- rendering ---------------------------------------------------------------------------------


def _ratio(r: Ratio) -> str:
    return f"{r.count} / {r.denominator}"


def _pct(rate: Rate | None) -> str:
    if rate is None:
        return "none"
    return f"{rate.k} / {rate.n}" + ("" if rate.rate is None else f" ({rate.rate:.0%})")


def _table(caption: str, head: Iterable[str], rows: Iterable[Iterable[object]], empty: str) -> str:
    rows = list(rows)
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    body = cells or f'<tr><td class="empty" colspan="{len(list(head))}">{html.escape(empty)}</td></tr>'
    heads = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    return f"<table><caption>{html.escape(caption)}</caption><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>"


# The unit and basis of each check's value and threshold (triage.CheckPolicy field comments).
CHECK_UNITS = {
    "K1": "Rwp percentage points of rise after a good Pawley fit",
    "K2": "% of the strongest net observed peak",
    "K3": "largest absolute a/b/c shift from the CIF (% of that CIF axis; phases at least 2 wt%)",
    "K5": "wt% points, answer minus cited fit",
    "K6": "EPS2 against its bound; largest wt% change from the refit with texture off to the final fit",
}


def _check_cell(outcome: CheckOutcome | None) -> str:
    if outcome is None:
        return "-"
    cell = outcome.status.replace("_", " ")
    numbers = [f"{outcome.value:g}"] if outcome.value is not None else []
    numbers += [f"threshold {outcome.threshold:g}"] if outcome.threshold is not None else []
    if numbers:
        unit = CHECK_UNITS.get(outcome.code)
        cell += " " + ", ".join(numbers) + (f" ({unit})" if unit else "")
    cell += "; routes" if outcome.routes else "; advisory"
    return cell + ("; unmeasured basis" if outcome.unmeasured else "")


def _signed(row: RoutingRow) -> str:
    cell = f"yes ({row.blind_signoffs} rater{'s' * (row.blind_signoffs != 1)})" if row.blind_signoffs else "no"
    return cell + (f"; {row.revealed_signoffs} revealed only" if row.revealed_signoffs else "")


def _producer(row: RoutingRow) -> str:
    return f"{row.model} ({row.harness})" if row.model else "no imported run in the store"


def _routing_tables(view: DashboardView) -> list[str]:
    short = lambda digest: digest[:12]  # noqa: E731
    codes = [code for code in CHECK_ORDER if any(c.code == code for r in view.routing for c in r.checks)]
    parts = [
        _table(
            "Triage: route per snapshot (admin owner only; never served to a rater before that rater's sign-off;"
            " sign-off cells show presence, not agreement or accuracy)",
            ("case", "snapshot", "producer", "policy version", "route", "why", *codes, "audit draw", "blind sign-off"),
            (
                (r.case_id or "not in the store", short(r.snapshot), _producer(r),
                 r.policy_version + (" (provisional)" if r.provisional else ""), r.route, r.why,
                 *(_check_cell(next((c for c in r.checks if c.code == code), None)) for code in codes), r.audit, _signed(r))
                for r in view.routing
            ),
            "no routing decisions yet: run scripts/workbench/triage_sweep.py",
        )
    ]
    by_snapshot = {(r.policy_version, r.snapshot): r for r in view.routing}
    for q in view.expert_queues:
        status = "provisional" if q.provisional else "final"
        queued = [by_snapshot[(q.policy_version, d)] for d in q.order if (q.policy_version, d) in by_snapshot]
        cleared = [r for r in view.routing if r.policy_version == q.policy_version and r.route == "cleared"]
        parts.append(
            f"<h2>Expert queue under {html.escape(q.policy_version)} ({status})</h2>"
            f"<p class=\"meta\">Routed to an expert: {_ratio(q.routed)} decisions (flagged {_ratio(q.flagged)}, "
            f"audit sample {_ratio(q.audit)}). Cleared: {_ratio(q.cleared)}. A routing check not computable: "
            f"{_ratio(q.not_computable)}. Flagged only by checks with an unmeasured basis: {_ratio(q.unmeasured_flags)}.</p>"
        )
        parts.append(
            _table(
                f"Routed to an expert, in queue order ({_ratio(q.routed)} decisions)",
                ("position", "case", "snapshot", "producer", "route", "why", "blind sign-off"),
                ((f"{i} / {len(queued)}", r.case_id or "not in the store", short(r.snapshot), _producer(r), r.route, r.why, _signed(r))
                 for i, r in enumerate(queued, 1)),
                "nothing routed to an expert",
            )
        )
        parts.append(
            _table(
                f"Cleared ({_ratio(q.cleared)} decisions): not routed, still stored, exportable and reviewable",
                ("case", "snapshot", "producer", "why", "blind sign-off"),
                ((r.case_id or "not in the store", short(r.snapshot), _producer(r), r.why, _signed(r)) for r in cleared),
                "nothing cleared",
            )
        )
    return parts


def render_tables(view: DashboardView) -> str:
    """The dashboard's tables as one HTML fragment; the page inserts it after an authenticated
    fetch. Every value is escaped."""
    short = lambda digest: digest[:12]  # noqa: E731
    parts = [
        f"<p class=\"meta\">Dataset {html.escape(view.dataset_version)} ({html.escape(view.scope)}), "
        f"generated {view.generated_at_utc:%Y-%m-%d %H:%M} UTC</p>",
        "<ul class=\"banners\">" + "".join(f"<li>{html.escape(b)}</li>" for b in view.banners) + "</ul>",
        _table(
            "Outcomes per case, model and harness",
            ("case", "model", "harness", "episodes", "submitted", "unsuccessful", "invalid", "contaminated",
             "incomplete runs", "human corrected", "answer snapshots / submitted"),
            ((o.case_id, o.model, o.harness, o.episodes, _ratio(o.submitted), _ratio(o.unsuccessful), _ratio(o.invalid),
              _ratio(o.contaminated), _ratio(o.incomplete), _ratio(o.human_corrected), _ratio(o.snapshots))
             for o in view.outcomes),
            "no episodes",
        ),
        _table(
            "Checks over frozen snapshots",
            ("check", "pass", "flag", "not computable"),
            ((c.code, _ratio(c.passed), _ratio(c.flagged), _ratio(c.not_computable)) for c in view.checks),
            "no snapshots",
        ),
        "<dl class=\"codes\">"
        + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in CHECK_LEGEND)
        + "</dl>",
        _table(
            "Snapshots",
            ("snapshot", "case", "model", "answered", "judged", *CHECK_CODES, "residual peaks", "chemistry", "labels", "sign-offs"),
            ((short(s.snapshot), s.case_id, s.model or "human correction", _yes(s.answered), _yes(s.judged),
              *(s.checks.get(code, "-") for code in CHECK_CODES), s.residual_peaks, _yes(s.chemistry),
              ", ".join(f"{k} {n}" for k, n in sorted(s.labels_by_kind.items())) or "none", s.signoffs)
             for s in view.snapshots),
            "no snapshots",
        ),
        _table(
            "Exclusions with reasons",
            ("kind", "reason", "count", "refs"),
            ((e.kind, e.reason, e.count, ", ".join(r[:40] for r in e.refs)) for e in view.exclusions),
            "no exclusions",
        ),
        _table(
            "Labels per snapshot and rater",
            ("snapshot", "case", "rater", "kind", "exposure", "yes", "no", "cannot assess"),
            ((short(lb.snapshot), lb.case_id, lb.rater, lb.kind, lb.exposure, lb.yes, lb.no, lb.cannot_assess) for lb in view.labels),
            "no labels yet",
        ),
        _table(
            "Coverage per model-judge run over the frozen judge targets (earliest blind label per target; a malformed,"
            " refused or never-made call counts as missing; a run with no ingested label has no row)",
            ("judge", "prompt", "initial run", "expected targets", "answered yes/no", "cannot assess", "missing",
             "snapshots with labels / judged"),
            ((c.judge, short(c.prompt_sha256), "yes" if c.initial else "no (repeat)", c.expected_targets,
              f"{c.answered} / {c.expected_targets}", c.cannot_assess, c.missing,
              f"{c.snapshots_with_labels} / {c.judged_snapshots}")
             for c in view.judge_coverage),
            "no model-judge labels",
        ),
        _table(
            "Agreement within one rater kind (initial blind responses; model judges per prompt, the earliest observed"
            " run per model and prompt, inferred from label times, not a declared run; agreement is conditional on"
            " comparable pairs)",
            ("kind", "item", "prompt", "raters", "expected targets", "observed targets", "wholly missing",
             "one-sided targets", "comparable pairs", "agreeing", "agreement", "cannot-assess pairs", "revealed revisions"),
            ((a.kind, a.item, short(a.prompt_sha256) if a.prompt_sha256 else "-",
              f"{len(a.raters)}: " + ", ".join(r[:40] for r in a.raters) if a.raters else "0",
              "unavailable" if a.expected_targets is None else a.expected_targets, a.targets,
              "unavailable" if a.wholly_missing_targets is None else a.wholly_missing_targets,
              a.one_sided_targets, a.comparable_pairs, a.agreeing_pairs,
              "none" if a.agreement is None else f"{a.agreeing_pairs} / {a.comparable_pairs} ({a.agreement:.0%}, conditional)",
              a.cannot_assess_pairs, a.revealed_revisions)
             for a in view.agreement),
            "no labels",
        ),
        "<h2>Judge versus expert</h2>"
        + ("<p class=\"shortfall\"><b>Not gradable: 0 expert labels.</b></p>" if not _has_expert(view) else "")
        + "".join(f"<p class=\"shortfall\">{html.escape(s)}</p>" for s in view.judge_shortfall)
        + "".join(f"<p class=\"meta\">{html.escape(s)}</p>" for s in view.split_status)
        + (
            _table(
                "Judge versus expert on the test partition of a frozen calibration split (from qualification reports;"
                " rates are conditional on the judge answering)",
                ("judge", "prompt", "set", "item", "source exposure", "eligible", "answered", "abstained (per run)", "missing", "n",
                 "judge-expert pairwise", "expert-expert pairwise", "TPR", "TNR"),
                ((j.model, short(j.prompt_sha256), short(j.set_digest), j.item, j.exposure, j.eligible, _pct(j.coverage),
                  j.abstained, "unavailable" if j.missing is None else j.missing, j.n, _pct(j.judge_expert), _pct(j.expert_expert), _pct(j.tpr), _pct(j.tnr))
                 for j in view.judge_agreement),
                "none",
            )
            if view.judge_agreement
            else ""
        ),
        _table(
            "Triage: routes per policy version",
            ("policy version", "status", "route", "count / decisions", "flagged only by unmeasured checks"),
            ((r.policy_version, "provisional" if r.provisional else "final", r.route, _ratio(r.count), r.unmeasured_flags)
             for r in view.routes),
            "no routing decisions yet",
        ),
        *_routing_tables(view),
        _table(
            "Qualification: verdicts per producer (accepted and rejected sides, never pooled)",
            ("producer", "harness", "set", "roots", "verdicts", "accepted side", "rejected side"),
            ((p.model_id, f"{p.harness} {p.harness_version}", short(p.set_digest), p.roots,
              ", ".join(f"{k} {n}" for k, n in p.by_verdict.items()),
              f"{_pct(p.accepted)} pass of assessed; {p.accepted.n} assessed of {p.accepted_applicable} applicable",
              f"{_pct(p.rejected)} pass of assessed; {p.rejected.n} assessed of {p.rejected_applicable} applicable")
             for p in view.producers),
            "Not gradable: 0 expert labels. No qualification reports yet."
            if not _has_expert(view)
            else "no qualification reports yet",
        ),
    ]
    return "\n".join(parts)


def _has_expert(view: DashboardView) -> bool:
    return not any(b == "No expert-kind labels yet." for b in view.banners)


# --- routes ------------------------------------------------------------------------------------

router = APIRouter(tags=["workbench-dashboard"])


def _admin_owner(principal: Annotated[Principal, Depends(authenticate_principal)]) -> Principal:
    if not is_admin_owner(principal):
        raise HTTPException(status_code=403, detail="the dashboard is for the admin owner")
    return principal


def routing_ledger(
    store: WorkbenchStore, principal: Principal
) -> tuple[tuple[CheckPolicy, tuple[RouteDecision, ...]], ...]:
    """Each policy version bound under routing-policies/ with every decision under it that this
    principal may export, read through the triage ledger."""
    policies = store.root / "routing-policies"
    if not policies.is_dir():
        return ()
    labels = LabelStore(store)
    found = []
    for path in sorted(policies.glob("*.json")):
        policy = CheckPolicy.model_validate_json(path.read_bytes())
        found.append((policy, Triage(store, labels, policy).decisions(principal, policy.version)))
    return tuple(found)


def routing_decisions(store: WorkbenchStore, principal: Principal) -> tuple[RouteDecision, ...]:
    """Every decision this principal may export, for each bound policy version."""
    return tuple(decision for _, group in routing_ledger(store, principal) for decision in group)


def imported_producers(store: WorkbenchStore, roots: Iterable[str]) -> dict[str, tuple[str, str]]:
    """Attempt digest -> (model, harness) of every imported run under `roots`. Trusted caller:
    the roots come from decisions the admin owner may export."""
    found = {}
    for root in sorted(set(roots)):
        for digest, attempt in store.walk(root):
            if not isinstance(attempt, ChildAttempt) or not isinstance(attempt.action, ImportRun):
                continue
            if isinstance(attempt.outcome, Completed) and isinstance(attempt.outcome.result, ImportedRunResult):
                found[digest] = (attempt.outcome.result.agent.model_id, attempt.outcome.result.agent.harness)
    return found


def qualification_sets(store: WorkbenchStore, principal: Principal) -> tuple[QualificationSet, ...]:
    """Every frozen qualification set, read and re-hashed through the Qualifier (owner only)."""
    directory = store.root / "qualification-sets"
    if not directory.is_dir():
        return ()
    qualifier = Qualifier(store, LabelStore(store), NoReference())
    return tuple(qualifier.qualification_set(principal, path.stem) for path in sorted(directory.glob("*.json")))


@router.get("/dashboard", include_in_schema=False)
def get_dashboard_page() -> RedirectResponse:
    """The dashboard page is retired: /scorecard shows its tables (owner directive 2026-10-22)."""
    return RedirectResponse(url="/scorecard", status_code=307)


@router.get("/api/attempts/dashboard")
def get_dashboard(request: Request, principal: Annotated[Principal, Depends(_admin_owner)]) -> dict[str, Any]:
    """The view and its rendered tables for the dataset at $AXA_WORKBENCH_DATASET."""
    dataset = Path(os.environ.get("AXA_WORKBENCH_DATASET", str(DEFAULT_DATASET)))
    if not (dataset / "manifest.json").is_file():
        raise HTTPException(status_code=503, detail="no eval dataset is built")
    reports = [QualificationReport.model_validate_json(line) for line in _lines(dataset / "reports.jsonl")]
    store = getattr(request.app.state, "workbench_store", None)
    has_store = isinstance(store, WorkbenchStore)
    ledger = routing_ledger(store, principal) if has_store else ()
    routes = tuple(decision for _, group in ledger for decision in group)
    rows, queues = (
        routing_view(
            ledger,
            stored_snapshots(store.root),
            imported_producers(store, (d.root for d in routes)),
            LabelStore(store).export_records("internal_evaluation")[2],
        )
        if routes
        else ((), ())
    )
    sets = qualification_sets(store, principal) if has_store else ()
    view = dashboard_view(dataset, routes, reports, sets, routing=rows, expert_queues=queues)
    return {"view": view.model_dump(mode="json"), "html": render_tables(view)}
