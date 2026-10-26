"""U16b qualification (AXA-43zwk.26): lab-expectation-v2 table tests over hand-built labels and
snapshots, then seam tests where the real LabelStore writes the records and the Qualifier reads
them.

Every case, candidate and judgment below is fictional (case SYN001, candidates CAND....). None is
a real case's label, reference or verdict, and the reference is a fake reader's."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.support.store import BLOBS, CASE, OWNER, FakeRefine, _clock

from phaseeval.api.auth import CaseOperation, Principal
from phaseeval.workbench.labels import (
    StageProjection,
    JudgeIdentity,
    Label,
    LabelRequest,
    LabelStore,
    Rater,
    SignOff,
    SignOffRequest,
)
from phaseeval.workbench.qualification import (
    CaseExpectation,
    CaseResult,
    CrossBuildBounds,
    JudgeVersion,
    NoFrozenSplit,
    NotGradable,
    QualificationPolicy,
    Qualifier,
    Refiner,
    RootLedger,
    Triager,
    TunedOn,
    TunedOnHeldOut,
    derive_expectation,
    grade_answer,
    grade_judge,
    partition_of,
    phase_key,
    pin_of,
    snapshot_digest,
    summarize_producer,
)
from phaseeval.workbench.seams import NoReference, SealedReference, SealedReferencePhase, public_seams
from phaseeval.workbench.snapshot import AnalysisSnapshot, SnapshotPhase
from phaseeval.workbench.store import ActionRequest, WorkbenchStore
from phaseeval.workbench.triage import PROVISIONAL_2026_10_21, Triage

ROOT = "a" * 64
A1 = ("CAND0101", "MgO", "Fm-3m (225)")
A2 = ("CAND0102", "MgO", "Fm-3m (225)")  # library duplicates of A1
A3 = ("CAND0103", "MgO", "Fm-3m (225)")
B = ("CAND0201", "ZnO", "P6_3mc (186)")
C = ("CAND0301", "Al2O3", "R-3c (167)")
X = ("CAND0401", "MgAl2O4", "Fd-3m (227)")
_TICKS = itertools.count()


def _at() -> datetime:
    return datetime(2026, 10, 22, 6, tzinfo=UTC) + timedelta(seconds=next(_TICKS))


def snap(
    origin: str,
    phases: tuple[tuple[str, str, str], ...],
    kind: str = "agent_answer",
    quantification: str = "restricted_profile",
    weights: tuple[float, ...] | None = None,
    answered: bool = True,
    curve: str | None = None,
) -> AnalysisSnapshot:
    return AnalysisSnapshot(
        kind=kind,
        origin=origin * 64,
        root=ROOT,
        case_id="SYN001",
        pattern_sha256="b" * 64,
        fit_curve_sha256=curve,
        stage="rietveld",
        rwp_percent=8.0,
        gof=1.5,
        quantification=quantification,
        phases=tuple(
            SnapshotPhase(
                phase_id=pid,
                formula=formula,
                space_group=sg,
                weight_percent=(weights or (25.0,) * len(phases))[i],
                weight_percent_error=None,
            )
            for i, (pid, formula, sg) in enumerate(phases)
        ),
        residual_peaks=(),
        checks=(),
        answered=answered,
    )


def label(s, item, answer, rater="rater-a", phase=None, kind="expert", exposure="blind") -> Label:
    return Label(
        snapshot=snapshot_digest(s),
        item=item,
        phase_id=phase[0] if phase else None,
        answer=answer,
        reason="peaks overlap" if answer == "cannot_assess" else "",
        rater=Rater(principal_id=rater, kind=kind),
        exposure=exposure,
        created_at_utc=_at(),
    )


def supported(s: AnalysisSnapshot, rater: str, *, kind: str = "expert", but: dict | None = None) -> list[Label]:
    """phase_supported and phase_plausible yes on every phase, except the answers in `but`."""
    out = []
    for phase in s.phases:
        row = (phase.phase_id, phase.formula, phase.space_group)
        out.append(label(s, "phase_supported", (but or {}).get(phase.phase_id, "yes"), rater, row, kind))
        out.append(label(s, "phase_plausible", "yes", rater, row, kind))
    return out


def sign(s, verdict, rater="rater-a", kind="expert", exposure="blind") -> SignOff:
    return SignOff(
        snapshot=snapshot_digest(s),
        root=s.root,
        verdict=verdict,
        rater=Rater(principal_id=rater, kind=kind),
        exposure=exposure,
        created_at_utc=_at(),
    )


def policy(**changes: object) -> QualificationPolicy:
    fields = {
        "version": "lab-a-2026q4",
        "justified_by": "hand-built test policy",
        "rater_kinds": ("expert", "owner"),
        "require_blind": False,
        "min_raters": 1,
        "phase_identity": "formula_and_space_group",
        "amount_abs_wt_pct": None,
        "expectation_rule": "lab-expectation-v2",
    }
    return QualificationPolicy(**{**fields, **changes})


def reference(*phases: tuple[str, str, str]) -> SealedReference:
    return SealedReference(
        case_id="SYN001",
        source="fake reader",
        phases=tuple(SealedReferencePhase(formula=f, space_group=None, candidates=(pid,)) for pid, f, _ in phases),
    )


def derive(snaps, labels, signoffs, rule=None, ref=None, exposed=()):
    ledger = RootLedger(
        root=ROOT, snapshots=tuple(snaps), labels=tuple(labels), signoffs=tuple(signoffs), exposed=tuple(exposed)
    )
    return derive_expectation(ledger, "SYN001", rule or policy(), ref)


def stances(expectation: CaseExpectation) -> dict[str, tuple[str, str, str]]:
    return {p.candidate_id: (p.expect, p.because, p.reference) for p in expectation.phases}


WITH_X = snap("1", (A1, B, C, X))  # an agent answer that adds X
CORRECTION = snap("2", (A1, B, C), kind="expert_correction")
ANSWER = snap("3", (A2, B, C))
RERUN_X = snap("4", (A3, B, C, X))
EMPTY = snap("8", ())  # answered, no phase


def rejected_with_x() -> tuple[list[Label], list[SignOff]]:
    """The rater refuses WITH_X (X unsupported) and signs CORRECTION, which drops X."""
    labels = [*supported(WITH_X, "rater-a", but={"CAND0401": "no"}), *supported(CORRECTION, "rater-a")]
    return labels, [sign(WITH_X, "would_not_sign"), sign(CORRECTION, "would_sign")]


# --- derive_expectation ------------------------------------------------------------------------


def test_a_refused_fits_negative_stays_scoped_to_that_fit() -> None:
    labels, signoffs = rejected_with_x()
    expectation = derive([WITH_X, CORRECTION], labels, signoffs, ref=reference(A1, B, C, X))
    assert expectation.accepted == snapshot_digest(CORRECTION)
    assert stances(expectation) == {
        "CAND0101": ("required", "supported_in_accepted", "agrees"),
        "CAND0201": ("required", "supported_in_accepted", "agrees"),
        "CAND0301": ("required", "supported_in_accepted", "agrees"),
        # Refused with its fit and dropped by the signed correction: still not a root-wide
        # exclusion. The reference lists it and the accepted analysis does not: disputes.
        "CAND0401": ("free", "negative_in_one_fit", "disputes"),
    }
    (rejected,) = expectation.rejected
    assert (rejected.snapshot, rejected.keys) == (
        snapshot_digest(WITH_X),
        ("Al2O3|167", "MgAl2O4|227", "MgO|225", "ZnO|186"),
    )
    assert expectation.amounts_graded is False  # restricted_profile never grades amounts
    assert expectation.provenance == "lab_expert_blind"
    assert {p.reference for p in derive([WITH_X, CORRECTION], labels, signoffs).phases} == {"absent"}


def test_a_negative_on_a_fit_nobody_refused_does_not_forbid() -> None:
    labels = [*supported(WITH_X, "rater-a", but={"CAND0401": "no"}), *supported(CORRECTION, "rater-a")]
    expectation = derive([WITH_X, CORRECTION], labels, [sign(CORRECTION, "would_sign")])
    assert stances(expectation)["CAND0401"][:2] == ("free", "negative_in_one_fit")
    graded = grade_answer(expectation, RERUN_X, policy())
    assert (graded.verdict, graded.reasons) == ("needs_review", ("negative_in_one_fit:CAND0401",))
    # Without an accepted analysis a refused fit's negative stays scoped to that fit too.
    rejected_only = derive([WITH_X], supported(WITH_X, "rater-a", but={"CAND0401": "no"}), [sign(WITH_X, "would_not_sign")])
    assert stances(rejected_only)["CAND0401"][:2] == ("free", "negative_in_one_fit")


def test_library_duplicates_share_one_phase_key() -> None:
    def phase(row: tuple[str, str, str]) -> SnapshotPhase:
        return SnapshotPhase(phase_id=row[0], formula=row[1], space_group=row[2], weight_percent=None, weight_percent_error=None)

    assert {phase_key(phase(row), "formula_and_space_group") for row in (A1, A2, A3)} == {"MgO|225"}
    assert phase_key(phase(A1), "candidate_id") == "id:CAND0101"


def test_a_two_rater_split_is_free_disputed() -> None:
    labels = [*supported(CORRECTION, "rater-a"), *supported(CORRECTION, "rater-b", but={"CAND0201": "no"})]
    signoffs = [sign(CORRECTION, "would_sign", "rater-a"), sign(CORRECTION, "would_sign", "rater-b")]
    assert stances(derive([CORRECTION], labels, signoffs))["CAND0201"][:2] == ("free", "disputed")


def test_an_accepted_analysis_without_phase_judgments_is_not_gradable() -> None:
    expectation = derive([ANSWER], [], [sign(ANSWER, "would_sign")])
    assert expectation == NotGradable(root=ROOT, reason="no_phase_judgments")
    assert grade_answer(expectation, EMPTY, policy()).verdict == "not_gradable"


def test_cannot_assess_only_constrains_nothing_and_passes_nothing() -> None:
    labels = [label(CORRECTION, "phase_supported", "cannot_assess", "rater-a", row) for row in (A1, B, C)]
    expectation = derive([CORRECTION], labels, [sign(CORRECTION, "would_sign")])
    assert {v[:2] for v in stances(expectation).values()} == {("free", "unassessable")}
    for answer in (EMPTY, ANSWER):
        graded = grade_answer(expectation, answer, policy())
        assert (graded.verdict, graded.accepted_side) == ("needs_review", "needs_review")
        assert "no_required_phase" in graded.reasons


def test_an_empty_answer_fails_an_accepted_case() -> None:
    expectation = derive([CORRECTION], supported(CORRECTION, "rater-a"), [sign(CORRECTION, "would_sign")])
    graded = grade_answer(expectation, EMPTY, policy())
    assert (graded.verdict, graded.accepted_side) == ("fail", "fail")


@pytest.mark.parametrize(
    ("labels", "signoffs", "reason"),
    [
        ([], [], "no_counted_judgment"),
        (supported(ANSWER, "rater-a"), [], "no_resolved_analysis"),
        (supported(ANSWER, "rater-a"), [sign(ANSWER, "unresolved")], "no_resolved_analysis"),
        ([], [sign(ANSWER, "would_sign", "rater-a"), sign(ANSWER, "would_not_sign", "rater-b")], "signoff_disputed"),
    ],
)
def test_no_resolved_analysis_is_not_gradable(labels, signoffs, reason) -> None:
    expectation = derive([ANSWER], labels, signoffs)
    assert expectation == NotGradable(root=ROOT, reason=reason)
    graded = grade_answer(expectation, ANSWER, policy())
    assert (graded.verdict, graded.reasons, graded.accepted_side, graded.rejected_side) == (
        "not_gradable",
        (reason,),
        None,
        None,
    )


def test_require_blind_fails_closed_on_any_revealed_record_or_exposed_correction() -> None:
    blind_rule = policy(require_blind=True)
    # The signed analysis is blind; a revealed "no" sits on another, unsigned fit.
    labels = [*supported(ANSWER, "rater-a"), label(WITH_X, "phase_supported", "no", "rater-b", X, exposure="revealed")]
    signoffs = [sign(ANSWER, "would_sign")]
    assert derive([ANSWER, WITH_X], labels, signoffs, blind_rule) == NotGradable(root=ROOT, reason="not_blind")
    assert derive([ANSWER, WITH_X], labels, signoffs).provenance == "lab_expert_revealed"
    # Every record is blind, but the correction's author had seen the reference.
    clean = supported(CORRECTION, "rater-a")
    exposed = (snapshot_digest(CORRECTION),)
    corrected = [sign(CORRECTION, "would_sign")]
    assert derive([CORRECTION], clean, corrected, blind_rule, exposed=exposed) == NotGradable(root=ROOT, reason="not_blind")
    assert derive([CORRECTION], clean, corrected, exposed=exposed).provenance == "lab_expert_revealed"
    assert derive([CORRECTION], clean, corrected, blind_rule).provenance == "lab_expert_blind"


def test_owner_judgments_are_never_expert_provenance() -> None:
    owner = supported(CORRECTION, "owner-rater", kind="owner")
    assert derive([CORRECTION], owner, [sign(CORRECTION, "would_sign", "owner-rater", "owner")]).provenance == (
        "lab_owner_blind"
    )
    mixed = derive([CORRECTION], [*owner, *supported(CORRECTION, "rater-a")], [sign(CORRECTION, "would_sign")])
    assert mixed.provenance == "lab_mixed_blind"


def test_a_model_judge_record_never_enters_an_expectation() -> None:
    judge = Label(
        snapshot=snapshot_digest(ANSWER),
        item="phase_supported",
        phase_id="CAND0102",
        answer="yes",
        rater=Rater(
            principal_id="judge-opus",
            kind="model_judge",
            judge=JudgeIdentity(
                model="claude-opus-5-5",
                prompt_sha256="c" * 64,
                input_sha256="d" * 64,
                run_id="r1",
                judge_is_actor_model=False,
            ),
        ),
        exposure="blind",
        created_at_utc=_at(),
    )
    with pytest.raises(ValidationError, match="model-judge"):
        RootLedger(root=ROOT, snapshots=(ANSWER,), labels=(judge,), signoffs=())
    with pytest.raises(ValidationError):
        policy(rater_kinds=("model_judge",))


def test_a_withdrawn_raters_records_change_the_derivation_and_the_pin() -> None:
    labels = [*supported(CORRECTION, "rater-a"), *supported(CORRECTION, "rater-b", but={"CAND0201": "no"})]
    signoffs = [sign(CORRECTION, "would_sign", "rater-a"), sign(CORRECTION, "would_sign", "rater-b")]
    both = RootLedger(root=ROOT, snapshots=(CORRECTION,), labels=tuple(labels), signoffs=tuple(signoffs))
    kept = RootLedger(
        root=ROOT,
        snapshots=(CORRECTION,),
        labels=tuple(x for x in labels if x.rater.principal_id == "rater-a"),
        signoffs=tuple(signoffs[:1]),
    )
    alone = derive_expectation(kept, "SYN001", policy(), None)
    assert stances(alone)["CAND0201"][:2] == ("required", "supported_in_accepted")
    assert pin_of(both) != pin_of(kept)
    reordered = RootLedger(root=ROOT, snapshots=(CORRECTION,), labels=tuple(reversed(labels)), signoffs=tuple(signoffs))
    assert pin_of(both) == pin_of(reordered)


# --- grade_answer and the producer summary ------------------------------------------------------


def test_the_rejected_analysis_itself_fails_and_new_evidence_with_its_phases_needs_review() -> None:
    labels, signoffs = rejected_with_x()
    expectation = derive([WITH_X, CORRECTION], labels, signoffs)
    itself = grade_answer(expectation, WITH_X, policy())
    assert (itself.verdict, itself.accepted_side, itself.rejected_side) == ("fail", "needs_review", "fail")
    assert itself.reasons == (f"repeats_rejected_analysis:{snapshot_digest(WITH_X)}", "negative_in_one_fit:CAND0401")
    rerun = grade_answer(expectation, RERUN_X, policy())  # new evidence, the rejected phase set
    assert (rerun.verdict, rerun.rejected_side) == ("needs_review", "needs_review")
    assert rerun.reasons == ("negative_in_one_fit:CAND0401", f"same_phases_as_rejected:{snapshot_digest(WITH_X)}")
    good = grade_answer(expectation, ANSWER, policy())
    assert (good.verdict, good.accepted_side, good.rejected_side, good.reasons) == ("pass", "pass", "pass", ())


def test_a_rejection_that_shares_the_accepted_phases_is_about_that_analysis() -> None:
    rejected = snap("5", (A2, B, C), curve="e" * 64)
    labels = [*supported(CORRECTION, "rater-a"), *supported(rejected, "rater-a")]
    expectation = derive([rejected, CORRECTION], labels, [sign(rejected, "would_not_sign"), sign(CORRECTION, "would_sign")])
    exact = grade_answer(expectation, rejected, policy())
    assert (exact.verdict, exact.accepted_side, exact.rejected_side) == ("fail", "pass", "fail")
    assert exact.reasons == (f"repeats_rejected_analysis:{snapshot_digest(rejected)}",)
    same_curve = grade_answer(expectation, snap("6", (A3, B, C), curve="e" * 64), policy())
    assert same_curve.verdict == "needs_review"  # a curve does not carry what the rejection was about
    assert grade_answer(expectation, CORRECTION, policy()).verdict == "pass"  # the signed analysis itself
    new_fit = grade_answer(expectation, snap("7", (A3, B, C), curve="f" * 64), policy())
    assert (new_fit.verdict, new_fit.accepted_side, new_fit.rejected_side) == ("needs_review", "pass", "needs_review")
    assert new_fit.reasons == (f"same_phases_as_rejected:{snapshot_digest(rejected)}",)


def test_a_signed_correction_sharing_the_rejected_curve_passes() -> None:
    rule = policy(amount_abs_wt_pct=1.0)
    rejected = snap("5", (A1, B, C), quantification="quantitative", weights=(70.0, 20.0, 10.0), curve="e" * 64)
    corrected = snap("6", (A1, B, C), "expert_correction", "quantitative", (50.0, 30.0, 20.0), curve="e" * 64)
    labels = [*supported(rejected, "rater-a"), *supported(corrected, "rater-a"), label(corrected, "amounts_defensible", "yes")]
    expectation = derive([rejected, corrected], labels, [sign(rejected, "would_not_sign"), sign(corrected, "would_sign")], rule)
    graded = grade_answer(expectation, corrected, rule)
    assert (graded.verdict, graded.accepted_side, graded.rejected_side, graded.reasons) == ("pass", "pass", "pass", ())
    assert grade_answer(expectation, rejected, rule).reasons[-1] == f"repeats_rejected_analysis:{snapshot_digest(rejected)}"


def test_a_rejected_only_root_is_a_negative_case() -> None:
    expectation = derive([WITH_X], [], [sign(WITH_X, "would_not_sign")])
    assert expectation.accepted is None and expectation.phases == ()
    itself = grade_answer(expectation, WITH_X, policy())
    assert (itself.verdict, itself.accepted_side, itself.rejected_side) == ("fail", None, "fail")
    same_phases = grade_answer(expectation, RERUN_X, policy())
    assert (same_phases.verdict, same_phases.rejected_side) == ("needs_review", "needs_review")
    other = grade_answer(expectation, ANSWER, policy())
    assert (other.verdict, other.rejected_side) == ("needs_review", "needs_review")  # nobody judged its phases


def test_an_unjudged_extra_phase_needs_review_and_a_disputed_phase_is_skipped() -> None:
    labels = [*supported(ANSWER, "rater-a"), *supported(ANSWER, "rater-b")]
    labels += [*supported(WITH_X, "rater-a"), *supported(WITH_X, "rater-b", but={"CAND0401": "no"})]
    signoffs = [sign(ANSWER, "would_sign", "rater-a"), sign(ANSWER, "would_sign", "rater-b")]
    disputed = derive([ANSWER, WITH_X], labels, signoffs)
    assert stances(disputed)["CAND0401"][:2] == ("free", "disputed")
    skipped = grade_answer(disputed, RERUN_X, policy())
    assert (skipped.verdict, skipped.reasons) == ("pass", ("disputed_skipped:CAND0401",))
    only = derive([ANSWER], supported(ANSWER, "rater-a"), [sign(ANSWER, "would_sign")])
    extra = grade_answer(only, RERUN_X, policy())
    assert (extra.verdict, extra.accepted_side, extra.reasons) == ("needs_review", "needs_review", ("unjudged_extra:CAND0401",))


def test_an_abstention_fails_the_accepted_side_and_passes_the_rejected_side() -> None:
    labels, signoffs = rejected_with_x()
    expectation = derive([WITH_X, CORRECTION], labels, signoffs)
    silent = snap("9", (), answered=False)
    graded = grade_answer(expectation, silent, policy())
    assert (graded.verdict, graded.accepted_side, graded.rejected_side) == ("no_answer", "fail", "pass")

    mixed = [
        expectation,  # accepted and rejected
        derive([CORRECTION], supported(CORRECTION, "rater-a"), [sign(CORRECTION, "would_sign")]),  # accepted only
        derive([WITH_X], [], [sign(WITH_X, "would_not_sign")]),  # rejected only
        derive([ANSWER], [], []),  # not gradable
    ]
    summary = summarize_producer(grade_answer(e, silent, policy()) for e in mixed)
    assert (summary.accepted_pass_rate.k, summary.accepted_pass_rate.n) == (0, 2)
    assert (summary.rejected_pass_rate.k, summary.rejected_pass_rate.n) == (2, 2)
    assert (summary.roots, summary.accepted_applicable, summary.rejected_applicable) == (4, 2, 2)
    assert summary.by_verdict == {"no_answer": 3, "not_gradable": 1}


def test_conditional_rates_carry_their_applicable_roots() -> None:
    def case(verdict: str) -> CaseResult:
        return CaseResult(
            root=ROOT,
            partition=None,
            expectation_sha256=None,
            observed=None,
            verdict=verdict,
            accepted_side=verdict,
            applicable=("accepted",),
            reasons=(),
        )

    summary = summarize_producer([case("pass"), *(case("needs_review") for _ in range(9))])
    assert (summary.accepted_pass_rate.k, summary.accepted_pass_rate.n) == (1, 1)  # conditional on assessed
    assert (summary.roots, summary.accepted_applicable, summary.by_verdict["needs_review"]) == (10, 10, 9)


def test_a_root_not_run_stays_applicable_and_unassessed() -> None:
    expectation = derive([CORRECTION], supported(CORRECTION, "rater-a"), [sign(CORRECTION, "would_sign")])
    missing = grade_answer(expectation, None, policy())
    assert (missing.verdict, missing.accepted_side, missing.applicable) == ("not_run", None, ("accepted",))
    summary = summarize_producer([missing])
    assert (summary.roots, summary.accepted_applicable, summary.accepted_pass_rate.n) == (1, 1, 0)


def test_amounts_are_graded_only_on_a_quantitative_defensible_fit() -> None:
    fit = snap("5", (A1, B, C), "expert_correction", "quantitative", (50.0, 30.0, 20.0))
    close = snap("6", (A1, B, C), quantification="quantitative", weights=(50.5, 29.5, 20.0))
    far = snap("7", (A1, B, C), quantification="quantitative", weights=(55.0, 25.0, 20.0))
    labels = [*supported(fit, "rater-a"), label(fit, "amounts_defensible", "yes")]
    rule = policy(amount_abs_wt_pct=1.0)
    expectation = derive([fit], labels, [sign(fit, "would_sign")], rule)
    assert expectation.amounts_graded is True
    assert grade_answer(expectation, close, rule).verdict == "pass"
    assert grade_answer(expectation, far, rule).reasons == (
        "amount_outside:CAND0101:+5.00",
        "amount_outside:CAND0201:-5.00",
    )
    assert grade_answer(expectation, ANSWER, rule).verdict == "fail"  # restricted profile: not comparable
    assert derive([fit], labels, [sign(fit, "would_sign")]).amounts_graded is False  # no tolerance


# --- grade_judge ---------------------------------------------------------------------------------


def _judge_label(s, item, answer, phase=None, *, prompt="c", actor_model=False) -> Label:
    return Label(
        snapshot=snapshot_digest(s),
        item=item,
        phase_id=phase[0] if phase else None,
        answer=answer,
        reason="unclear" if answer == "cannot_assess" else "",
        rater=Rater(
            principal_id="judge-opus",
            kind="model_judge",
            judge=JudgeIdentity(
                model="claude-opus-5-5",
                prompt_sha256=prompt * 64,
                input_sha256="d" * 64,
                run_id="r2" if actor_model else "r1",
                judge_is_actor_model=actor_model,
            ),
        ),
        exposure="blind",
        created_at_utc=_at(),
    )


JUDGE = JudgeVersion(model="claude-opus-5-5", prompt_sha256="c" * 64, tuned_on=())


def test_grade_judge_reports_tpr_tnr_and_coverage_beside_expert_expert() -> None:
    labels = [
        *supported(ANSWER, "rater-a"),
        *supported(ANSWER, "rater-b", but={"CAND0201": "no"}),
        label(ANSWER, "amounts_defensible", "no", "rater-a"),
        label(ANSWER, "amounts_defensible", "no", "rater-b"),
    ]
    signoffs = [sign(ANSWER, "would_sign", "rater-a"), sign(ANSWER, "would_sign", "rater-b")]
    expectation = derive([ANSWER], labels, signoffs)
    judge_labels = [
        _judge_label(ANSWER, "phase_supported", "yes", A2),  # agrees
        _judge_label(ANSWER, "phase_supported", "yes", B),  # experts split: no TPR/TNR
        _judge_label(ANSWER, "phase_supported", "no", C),  # disagrees
        _judge_label(ANSWER, "amounts_defensible", "no"),  # agrees
        _judge_label(ANSWER, "phase_plausible", "cannot_assess", A2),
        _judge_label(ANSWER, "phase_plausible", "no", A2, prompt="e"),  # another prompt: ignored
        _judge_label(ANSWER, "phase_plausible", "no", B, actor_model=True),
    ]
    cases, summary = grade_judge(JUDGE, judge_labels, labels, {ROOT: expectation}, frozenset({ROOT}))
    assert [(c.verdict, c.reasons) for c in cases] == [
        ("fail", (f"disagrees:phase_supported:CAND0301:{snapshot_digest(ANSWER)[:12]}",))
    ]
    items = {i.item: i for i in summary.items}
    sup = items["phase_supported"]
    assert sup.n == 3 and (sup.tpr.k, sup.tpr.n, sup.tnr.n) == (1, 2, 0)
    assert (sup.coverage.k, sup.coverage.n, sup.eligible) == (3, 3, 3)
    assert (sup.judge_expert_pairwise.k, sup.judge_expert_pairwise.n) == (3, 6)
    assert (sup.expert_expert_pairwise.k, sup.expert_expert_pairwise.n) == (2, 3)
    assert (items["amounts_defensible"].tnr.k, items["amounts_defensible"].tnr.n) == (1, 1)
    plausible = items["phase_plausible"]
    assert (plausible.judge_abstained, plausible.n, plausible.coverage.k, plausible.coverage.n) == (1, 0, 0, 3)
    assert summary.judge_is_actor_model_excluded == 1


def test_an_always_abstaining_judge_shows_zero_coverage() -> None:
    labels = supported(ANSWER, "rater-a")
    expectation = derive([ANSWER], labels, [sign(ANSWER, "would_sign")])
    silent = [_judge_label(ANSWER, "phase_supported", "cannot_assess", row) for row in (A2, B, C)]
    cases, summary = grade_judge(JUDGE, silent, labels, {ROOT: expectation}, frozenset({ROOT}))
    sup = {i.item: i for i in summary.items}["phase_supported"]
    assert (sup.coverage.k, sup.coverage.n, sup.judge_abstained) == (0, 3, 3)
    assert cases[0].verdict == "not_gradable" and cases[0].reasons == ("no_decided_target",)


def test_grade_judge_drops_tune_roots_and_has_no_expert_expert_with_one_expert() -> None:
    labels = supported(ANSWER, "rater-a")
    expectation = derive([ANSWER], labels, [sign(ANSWER, "would_sign")])
    judged = [_judge_label(ANSWER, "phase_supported", "yes", A2)]
    tune, _ = grade_judge(JUDGE, judged, labels, {ROOT: expectation}, frozenset())
    assert [(c.verdict, c.reasons) for c in tune] == [("not_gradable", ("tune_partition",))]
    held, summary = grade_judge(JUDGE, judged, labels, {ROOT: expectation}, frozenset({ROOT}))
    assert held[0].verdict == "pass"
    assert {i.item: i.expert_expert_pairwise for i in summary.items}["phase_supported"] is None


# --- seam: the real LabelStore writes, the Qualifier reads --------------------------------------

from tests.support.qualification import (
    ALLOW,
    ASTRA_V2,
    Bench,
    JUDGE_PRINCIPAL,
    QWEN_V2,
    RATER_A,
    RATER_B,
    RIGHT,
    RULE,
    WITHDRAW,
    WRONG,
)
from tests.support.qualification import _import, _stored

def test_seam_label_store_records_become_expectations_and_reports(tmp_path) -> None:
    bench = Bench(tmp_path)
    identity = JudgeIdentity(
        model="claude-opus-5-5", prompt_sha256="c" * 64, input_sha256="d" * 64, run_id="r1", judge_is_actor_model=False
    )
    bench.labels.judge_label(
        JUDGE_PRINCIPAL,
        Label(
            snapshot=bench.qwen,
            item="phase_supported",
            phase_id="CIF000001",
            answer="no",
            rater=Rater(principal_id="judge-opus", kind="model_judge", judge=identity),
            exposure="blind",
            created_at_utc=datetime(2026, 10, 22, 6, tzinfo=UTC),
        ),
    )
    digest = bench.freeze()
    assert bench.freeze() == digest  # the same request over the same ledger
    expectations = bench.qualifier.expectations(OWNER, digest)
    first = expectations[bench.root]
    assert {p.key: p.expect for p in first.phases} == {
        "id:CIF000001": "required",  # the judge's "no" never counts
        "id:CIF000002": "required",
        "id:CIF000003": "free",  # a refused fit's negative is scoped to that fit
    }
    assert first.accepted == bench.qwen and [r.snapshot for r in first.rejected] == [bench.astra]
    assert first.provenance == "lab_expert_blind"
    assert expectations[bench.other] == NotGradable(root=bench.other, reason="no_resolved_analysis")  # 1 rater < 2

    good = bench.qualifier.run(OWNER, digest, QWEN_V2)
    assert good.claim == "agreement_with_lab_judgments"
    assert [c.verdict for c in good.cases] == ["pass", "not_gradable"]
    assert (good.summary.accepted_pass_rate.k, good.summary.accepted_pass_rate.n) == (1, 1)
    assert (good.summary.roots, good.summary.accepted_applicable) == (2, 1)
    bad = bench.qualifier.run(OWNER, digest, ASTRA_V2)
    assert bad.cases[0].verdict == "fail"
    assert bad.cases[0].reasons == (f"repeats_rejected_analysis:{bench.astra}", "negative_in_one_fit:CIF000003")
    assert bad.cases[1].verdict == "not_gradable" and len(bad.episodes) == 1
    assert bench.qualifier.plan(OWNER, digest, ASTRA_V2) == (bench.other,)
    assert bench.qualifier.run(OWNER, digest, QWEN_V2) == good
    assert len(list((bench.store.root / "qualification-reports").iterdir())) == 2  # published once each
    bounds = CrossBuildBounds(version="x", justified_by="x", rwp_abs_points=1, weight_abs_wt_pct=1, cell_length_rel=1)
    with pytest.raises(NotImplementedError, match="CrossBuildBounds"):
        bench.qualifier.run(OWNER, digest, Refiner(bounds=bounds))
    unrouted = bench.qualifier.run(OWNER, digest, Triager(policy=PROVISIONAL_2026_10_21))
    assert unrouted.cases[0].verdict == "not_run"  # nobody swept these answers under the policy
    assert unrouted.cases[0].reasons == tuple(sorted(f"unrouted:{d}" for d in (bench.qwen, bench.astra)))
    with pytest.raises(NoFrozenSplit):
        bench.qualifier.run(OWNER, digest, JUDGE)
    with pytest.raises(PermissionError):
        bench.qualifier.expectations(RATER_A, digest)

    # A label after the cut changes nothing in this set.
    late = LabelRequest(snapshot=bench.qwen, item="phase_supported", phase_id="CIF000002", answer="no")
    bench.labels.label(RATER_A, late)
    assert bench.qualifier.expectations(OWNER, digest)[bench.root] == first

    # Rater B withdraws: only the root B judged reads records_changed; both roots stay reported.
    bench.labels.record_permission(RATER_B, WITHDRAW)
    after = bench.qualifier.expectations(OWNER, digest)
    assert after[bench.root] == NotGradable(root=bench.root, reason="records_changed")
    assert after[bench.other] == expectations[bench.other]
    report = bench.qualifier.run(OWNER, digest, QWEN_V2)
    assert [(c.verdict, c.reasons) for c in report.cases] == [
        ("not_gradable", ("records_changed",)),
        ("not_gradable", ("no_resolved_analysis",)),
    ]


def test_seam_a_correction_by_an_exposed_rater_is_marked_exposed(tmp_path) -> None:
    bench = Bench(tmp_path)
    bench.labels.declare_exposure(OWNER, RATER_A.principal_id, bench.root, "saw the reference before review")
    request = ActionRequest(idempotency_key="fix-1", action={"verb": "add_phases", "candidates": ["CIF000003"]})
    (step,) = bench.store.gesture(RATER_A, bench.root, (request,))
    correction = bench.labels.put_snapshot(_stored(bench.root, CASE, step.digest, WRONG, kind="expert_correction"))
    ledgers, _ = bench.qualifier._ledgers(RULE, None, "internal_evaluation")
    assert ledgers[bench.root].exposed == (correction,)


def test_seam_judge_split_and_tuned_on_refusal(tmp_path) -> None:
    bench = Bench(tmp_path)
    roots = (bench.root, bench.other)
    seed = next(f"s{i}" for i in range(64) if [partition_of(f"s{i}", r) for r in roots] == ["held_out", "tune"])
    digest = bench.freeze("judged", seed)
    tuned_here = JUDGE.model_copy(update={"tuned_on": (TunedOn(set_digest=digest, partition="held_out"),)})
    with pytest.raises(TunedOnHeldOut):
        bench.qualifier.run(OWNER, digest, tuned_here)
    tuned_on_tune = JUDGE.model_copy(update={"tuned_on": (TunedOn(set_digest=digest, partition="tune"),)})
    report = bench.qualifier.run(OWNER, digest, tuned_on_tune)
    assert [c.partition for c in report.cases] == ["held_out", "tune"]
    assert [c.verdict for c in report.cases] == ["not_run", "not_gradable"]  # no judge labels yet


def _swept(tmp_path) -> tuple[LabelStore, str, dict[str, str]]:
    """One root, a qwen and an astra fixture answer, both swept under the provisional policy.
    Returns the label store, the root and each model's swept snapshot digest."""
    store = WorkbenchStore(
        tmp_path / "store",
        FakeRefine(),
        public_seams(),
        _clock(datetime(2026, 10, 22, 6, tzinfo=UTC)),
        import_root=tmp_path / "runs",
        exposure=StageProjection(tmp_path / "store"),
    )
    labels = LabelStore(store)
    root = store.load_case(OWNER, CASE, BLOBS)
    for rater in (RATER_A, RATER_B):
        store.grant(OWNER, root, rater.principal_id, (CaseOperation.INSPECT,))
        labels.record_permission(rater, ALLOW)
    origins = {
        "qwen": _import(store, root, CASE, "dev900__qwen", "qwen36-27b", RIGHT),
        "astra": _import(store, root, CASE, "dev900__astra", "gpt-6-astra", WRONG),
    }
    triage = Triage(store, labels, PROVISIONAL_2026_10_21)
    triage.sweep(OWNER, (root,))
    by_origin = {labels._snapshot(d.snapshot).origin: d for d in triage.decisions(OWNER, PROVISIONAL_2026_10_21.version)}
    # Both fixture answers lack K readings, so the policy routes both (fail closed).
    assert {d.route for d in by_origin.values()} == {"expert_flagged"}
    swept = {name: by_origin[origin].snapshot for name, origin in origins.items()}
    # A rater reads only its cohort before the reveal (CR-07).
    labels.assign(OWNER, root, (RATER_A.principal_id, RATER_B.principal_id), (swept["qwen"], swept["astra"]))
    return labels, root, swept


def _route_report(labels: LabelStore, root: str, rule: QualificationPolicy):
    qualifier = Qualifier(labels._store, labels, NoReference())
    digest = qualifier.freeze(OWNER, lab="lab-a", name="routes", roots=(root,), policy=rule, split_seed=None)
    return qualifier.run(OWNER, digest, Triager(policy=PROVISIONAL_2026_10_21))


def _sign_with_labels(labels: LabelStore, rater: Principal, snapshot: str, verdict: str) -> None:
    for phase in labels._snapshot(snapshot).phases:
        answer = "no" if phase.phase_id == "CIF000003" else "yes"
        labels.label(rater, LabelRequest(snapshot=snapshot, item="phase_supported", phase_id=phase.phase_id, answer=answer))
    labels.sign_off(rater, SignOffRequest(snapshot=snapshot, verdict=verdict))


def test_seam_triager_grades_the_decisions_the_sweep_published(tmp_path) -> None:
    labels, root, swept = _swept(tmp_path)
    for rater in (RATER_A, RATER_B):
        _sign_with_labels(labels, rater, swept["qwen"], "would_sign")
        _sign_with_labels(labels, rater, swept["astra"], "would_not_sign")
    report = _route_report(labels, root, RULE)
    flagged = report.summary.grade.by_route["expert_flagged"]
    # The signed qwen answer is a false alarm and the refused astra answer is caught.
    assert (flagged.decisions, flagged.wrong, flagged.right) == (2, 1, 1)
    assert [(c.verdict, c.reasons) for c in report.cases] == [("fail", ("false_alarms:1",))]
    assert report.summary.grade.recall_estimate is None and report.summary.grade.recall_withheld


def test_seam_triager_counts_a_sign_off_without_phase_labels(tmp_path) -> None:
    labels, root, swept = _swept(tmp_path)
    labels.sign_off(RATER_A, SignOffRequest(snapshot=swept["qwen"], verdict="would_sign"))  # no phase label
    report = _route_report(labels, root, policy(rater_kinds=("expert",), min_raters=1, phase_identity="candidate_id"))
    flagged = report.summary.grade.by_route["expert_flagged"]
    assert (flagged.wrong, flagged.right, flagged.unreviewed) == (0, 1, 1)
    assert [(c.verdict, c.reasons) for c in report.cases] == [("fail", ("false_alarms:1",))]


def test_seam_triager_keeps_a_sign_off_below_quorum_unreviewed(tmp_path) -> None:
    labels, root, swept = _swept(tmp_path)
    for rater in (RATER_A, RATER_B):
        _sign_with_labels(labels, rater, swept["qwen"], "would_sign")
    _sign_with_labels(labels, RATER_A, swept["astra"], "would_not_sign")  # one refusal, quorum is two
    report = _route_report(labels, root, RULE)
    flagged = report.summary.grade.by_route["expert_flagged"]
    assert (flagged.wrong, flagged.right, flagged.unreviewed) == (0, 1, 1)  # astra: unreviewed, not caught
    assert [(c.verdict, c.reasons) for c in report.cases] == [("fail", ("false_alarms:1",))]


def test_seam_triager_quorum_counts_blind_sign_offs_only(tmp_path) -> None:
    labels, root, swept = _swept(tmp_path)
    for rater in (RATER_A, RATER_B):
        _sign_with_labels(labels, rater, swept["qwen"], "would_sign")
    _sign_with_labels(labels, RATER_A, swept["astra"], "would_not_sign")
    labels.declare_exposure(OWNER, RATER_B.principal_id, root, "saw the reference after the qwen review")
    _sign_with_labels(labels, RATER_B, swept["astra"], "would_not_sign")  # stamped revealed
    report = _route_report(labels, root, RULE)  # require_blind False, min_raters 2
    grade = report.summary.grade
    flagged = grade.by_route["expert_flagged"]
    assert (flagged.wrong, flagged.right, flagged.unreviewed, len(grade.misses)) == (0, 1, 1, 0)
    assert [(c.verdict, c.reasons) for c in report.cases] == [("fail", ("false_alarms:1",))]
