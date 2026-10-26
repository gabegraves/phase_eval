"""Episode scorecard and RL view: every exported agent episode scored per workflow step with a
named source, counts that reconcile with the manifest, model-judge labels never read as expert
labels, reward v0, preference pairs from corrections in the DPO shape, and admin-only serving
(AXA-43zwk.11). Every test drives the real export (exports.build_dataset, write_dataset) into
the scorecard; the last drives a real Harbor import through the real snapshot and blind view."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_export import (
    CASE,
    FORMULAS,
    INPUTS,
    NOW,
    OWNER,
    OWNER_RATER,
    PERMIT,
    RATER,
    WORKTREE,
    Labeled,
    Tree,
)
from test_tree import REFINE

from phaseeval.api.app import app
from phaseeval.api.auth import CaseOperation, Principal, PrincipalAuthenticator, PrincipalType
from phaseeval.workbench import exports, importer, policy, scorecard, snapshot
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.exports import RaterPseudonyms, build_dataset, write_dataset
from phaseeval.workbench.labels import (
    ExposureLedger,
    JudgeIdentity,
    Label,
    LabelStore,
    Rater,
    SignOffRequest,
)
from phaseeval.workbench.model import Refused
from phaseeval.workbench.review import PreferenceRequest
from phaseeval.workbench.scorecard import (
    REWARD_SPEC_V0,
    STEP_DEFINITIONS,
    STEPS,
    demo_pair_lines,
    preference_lines,
    reconciliation_text,
    render,
    scorecard_view,
)
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import ActionRequest, WorkbenchStore


def no_reference(case_id: str) -> None:
    return None


@pytest.fixture
def labeled(tmp_path: Path) -> Labeled:
    return Labeled(Tree(tmp_path / "store"), tmp_path)


def episode(view, digest: str):  # noqa: ANN001, ANN201
    (found,) = [e for e in view.episodes if e.episode == digest]
    return found


def step(score, name: str):  # noqa: ANN001, ANN201
    (found,) = [s for s in score.steps if s.step == name]
    return found


def test_every_exported_agent_episode_gets_one_outcome_per_step_and_reconciles(
    labeled: Labeled,
) -> None:
    dataset = labeled.build()
    view = scorecard_view(labeled.out, reference=no_reference)
    tree = labeled.tree

    # The agent's own answer stays a row after the owner corrects it; the corrected path is a pair.
    assert {e.episode for e in view.episodes} == {tree.answered, tree.incomplete, tree.unsuccessful}
    assert len(view.episodes) == sum(a.episodes for a in dataset.manifest.agent_episodes)
    assert all(r.ok for r in view.reconciliation), reconciliation_text(view)
    assert all(tuple(s.step for s in e.steps) == STEPS for e in view.episodes)
    assert [d.step for d in STEP_DEFINITIONS] == list(STEPS)

    unsuccessful = step(episode(view, tree.unsuccessful), "final_answer")
    # A run that ended without an answer is still a fail in every count, shown apart (PI 19:50).
    assert (unsuccessful.no_answer, unsuccessful.disputed) == (True, False)
    (decided_row,) = [r for r in view.matrix if r.source == "decided"]
    assert (
        decided_row.cells["final_answer"].failed,
        decided_row.cells["final_answer"].no_answer,
    ) == (1, 1)
    assert (
        unsuccessful.outcome,
        unsuccessful.source,
        unsuccessful.check,
        unsuccessful.failure_code,
    ) == (
        "fail",
        "deterministic_check",
        "submitted",
        "S11",
    )
    answered = episode(view, tree.answered)
    assert (
        answered.answered and answered.snapshot is not None and answered.job == "u6-dev900-qwen-006"
    )
    # A submission is a fact, not support: with no support label the final answer is unknown.
    final = step(answered, "final_answer")
    assert (final.outcome, final.source) == ("unknown", None)
    assert [(r.check, r.outcome) for r in final.readings if r.source == "deterministic_check"] == [
        ("submitted", "pass")
    ]
    incomplete = step(
        episode(view, tree.incomplete), "final_answer"
    )  # missing evidence, not a failure
    assert (incomplete.outcome, incomplete.failure_code) == ("unknown", None)
    assert "no basis fit: basis fit: output/run1/run1.lst" in incomplete.readings[1].detail
    # The fixture snapshot carries no K checks: unknown, never a pass.
    k2 = next(r for r in step(answered, "phase_selection").readings if r.check == "K2")
    assert (k2.outcome, k2.detail) == ("unknown", "K2 not on the snapshot")
    # The trace opens at the recorded refinement call.
    assert step(answered, "refinement").call_index == 0
    assert [t.marks for t in view.traces[tree.answered]] == [("refine",)]

    # Every matrix cell shows its denominator: the model's episodes.
    (decided,) = [r for r in view.matrix if r.source == "decided"]
    assert decided.episodes == 3 and all(c.denominator == 3 for c in decided.cells.values())
    assert (
        decided.cells["final_answer"].passed
        + decided.cells["final_answer"].failed
        + decided.cells["final_answer"].unknown
        == 3
    )


def test_model_judge_labels_read_as_model_judge_and_owner_labels_never_count(
    labeled: Labeled,
) -> None:
    first = labeled.build()
    digest = next(j.snapshot_digest for j in first.judge_inputs)
    labeled.label_both_kinds(digest)  # owner-rater yes, expert rater yes, model judge no
    labeled.build()

    view = scorecard_view(labeled.out, reference=no_reference)
    (scored,) = [e for e in view.episodes if e.snapshot == digest]
    final = step(scored, "final_answer")
    readings = {r.source: r for r in final.readings}

    assert (
        readings["model_judge"].outcome == "fail"
        and "claude-opus-5-5: 0 yes, 1 no" in readings["model_judge"].detail
    )
    # One expert-kind label; the owner's label is a revealed demo label and is not read.
    assert (
        readings["expert_label"].outcome == "pass"
        and readings["expert_label"].detail.count("1 yes") == 1
    )
    assert ";" not in readings["expert_label"].detail
    assert (final.outcome, final.source) == ("pass", "expert_label")  # expert precedes the judge
    assert (view.expert_labels, view.owner_labels) == (1, 1)
    fragment = render(view)
    assert (
        "Model-judge labels are model judgments, not expert labels. Expert labels: 1." in fragment
    )
    assert all(r.ok for r in view.reconciliation), reconciliation_text(view)


def test_reference_match_reads_the_published_list_and_never_says_correct(labeled: Labeled) -> None:
    labeled.build()

    def reference(case_id: str) -> dict:
        return {
            "source": "test",
            "phases": [{"formula": FORMULAS["CIF000001"], "candidates": ["CIF000001"]}],
        }

    view = scorecard_view(labeled.out, reference=reference)
    selection = step(episode(view, labeled.tree.answered), "phase_selection")

    assert (selection.outcome, selection.source, selection.failure_code) == (
        "fail",
        "reference_match",
        "S5",
    )
    assert selection.detail == f"does not match published reference: extra {FORMULAS['CIF000002']}"
    search = step(episode(view, labeled.tree.answered), "phase_search")
    assert (search.outcome, search.source) == ("pass", "reference_match")
    words = " ".join(r.detail for e in view.episodes for s in e.steps for r in s.readings).lower()
    assert "correct" not in words
    fragment = render(view)
    assert "<b>published reference</b>" in fragment and "never &#x27;correct&#x27;" in fragment


def test_a_reference_miss_on_a_disputed_phase_is_named_where_the_reward_uses_it(
    labeled: Labeled, monkeypatch: pytest.MonkeyPatch
) -> None:
    labeled.build()
    monkeypatch.setattr(scorecard, "DISPUTED", frozenset({(CASE.case_id, FORMULAS["CIF000002"])}))

    def reference(case_id: str) -> dict:
        return {
            "source": "test",
            "phases": [{"formula": FORMULAS["CIF000001"], "candidates": ["CIF000001"]}],
        }

    view = scorecard_view(labeled.out, reference=reference)
    selection = step(episode(view, labeled.tree.answered), "phase_selection")

    assert selection.outcome == "fail" and selection.detail.endswith(
        "(disputed: the DARA authors report it; an expert call)"
    )
    # The only disagreement is the disputed phase: still a fail, shown apart as disputed.
    assert selection.disputed and not selection.no_answer
    (decided_row,) = [r for r in view.matrix if r.source == "decided"]
    cell = decided_row.cells["phase_selection"]
    assert (cell.failed, cell.disputed) == (2, 2)  # the answered and the fit-less submitted answer
    assert {t.id: t.value for t in episode(view, labeled.tree.answered).reward.terms}[
        "reference_phase_list"
    ] == 0.0
    assert "1 of 3 episodes (DEV900) lose the reference term only on a disputed phase" in render(
        view
    )
    disputed = [f for f in view.failures if f.code == "S5" and "disputed" in f.check]
    assert [(f.check, set(f.episodes)) for f in disputed] == [
        (
            "reference.phase_list, disputed (an expert call)",
            {labeled.tree.answered, labeled.tree.incomplete},
        )
    ]


def test_reward_v0_is_declared_once_and_an_episode_with_no_answer_scores_zero(
    labeled: Labeled,
) -> None:
    labeled.build()
    view = scorecard_view(labeled.out, reference=no_reference)

    assert round(sum(c.weight for c in REWARD_SPEC_V0.components), 6) == 1.0
    assert (
        round(sum(c.weight for c in REWARD_SPEC_V0.components if c.source == "model_judge"), 6)
        == 0.2
    )
    assert episode(view, labeled.tree.unsuccessful).reward.total == 0.0
    answered = episode(view, labeled.tree.answered).reward
    assert {t.id: t.value for t in answered.terms}["answered"] == 1.0
    assert answered.total == round(sum(t.weight * t.value for t in answered.terms), 4)
    assert "No model was trained on this" in render(view)


def test_analyst_rows_count_over_the_runs_the_analyst_read_and_rule_rows_over_every_episode(
    labeled: Labeled,
) -> None:
    labeled.build()
    view = scorecard_view(labeled.out, reference=no_reference)

    (row,) = [f for f in view.failures if f.code == "S11"]
    assert (row.source, row.check, row.episodes, row.denominator) == (
        "deterministic_check",
        "submitted",
        (labeled.tree.unsuccessful,),
        len(view.episodes),
    )
    assert "none decided" in render(view) and "of 0 decided" not in render(view)


def judge(labeled: Labeled, digest: str, model: str, answer: str) -> None:
    """A model-judge label written through the real LabelStore.judge_label."""
    principal = Principal(f"judge-{model}", PrincipalType.MODEL_JUDGE, frozenset())
    identity = JudgeIdentity(
        model=model,
        prompt_sha256="7" * 64,
        input_sha256="8" * 64,
        run_id=f"run-{model}",
        judge_is_actor_model=False,
    )
    labeled.labels.judge_label(
        principal,
        Label(
            snapshot=digest,
            item="fit_supports_conclusion",
            phase_id=None,
            answer=answer,
            rater=Rater(principal_id=principal.principal_id, kind="model_judge", judge=identity),
            exposure="blind",
            created_at_utc=NOW,
        ),
    )


def test_a_submission_never_masks_judges_who_say_the_fit_does_not_support_it(
    labeled: Labeled,
) -> None:
    """CR15: submitted=yes and support=no stay distinguishable in the cell, and judges who
    disagree decide nothing."""
    first = labeled.build()
    digest = next(j.snapshot_digest for j in first.judge_inputs)
    judge(labeled, digest, "claude-opus-5-5", "no")
    labeled.build()

    final = step(
        next(
            e
            for e in scorecard_view(labeled.out, reference=no_reference).episodes
            if e.snapshot == digest
        ),
        "final_answer",
    )
    assert (final.outcome, final.source, final.check, final.failure_code) == (
        "fail",
        "model_judge",
        "fit_supports_conclusion",
        "S10",
    )
    assert {(r.check, r.outcome) for r in final.readings if r.source == "deterministic_check"} == {
        ("submitted", "pass")
    }

    judge(labeled, digest, "gpt-6-astra", "yes")
    labeled.build()
    final = step(
        next(
            e
            for e in scorecard_view(labeled.out, reference=no_reference).episodes
            if e.snapshot == digest
        ),
        "final_answer",
    )
    judges = next(r for r in final.readings if r.source == "model_judge")
    assert (final.outcome, judges.outcome) == ("unknown", "unknown") and judges.detail.startswith(
        "raters disagree; "
    )
    fail_only = {
        (x.source, x.check)
        for d in STEP_DEFINITIONS
        for x in d.deciders
        if x.decides == "fail_only"
    }
    assert fail_only == {
        ("deterministic_check", "K2"),
        ("deterministic_check", "K5"),
        ("deterministic_check", "submitted"),
        ("model_judge", "phase_supported, phase_plausible"),
        ("model_judge", "amounts_defensible"),
        ("model_judge", "fit_supports_conclusion"),
    }


@pytest.mark.parametrize(
    ("answer", "outcome", "source"),
    [("yes", "unknown", None), ("no", "fail", "model_judge")],
)
def test_an_uncalibrated_judge_alone_can_fail_a_step_and_never_pass_it(
    labeled: Labeled, answer: str, outcome: str, source: str | None
) -> None:
    """With no expert label, a judge yes leaves the step unknown and a judge no fails it."""
    first = labeled.build()
    digest = next(j.snapshot_digest for j in first.judge_inputs)
    judge(labeled, digest, "claude-opus-5-5", answer)
    labeled.build()
    final = step(
        next(
            e
            for e in scorecard_view(labeled.out, reference=no_reference).episodes
            if e.snapshot == digest
        ),
        "final_answer",
    )
    judges = next(r for r in final.readings if r.source == "model_judge")
    assert judges.outcome == ("pass" if answer == "yes" else "fail")
    assert (final.outcome, final.source) == (outcome, source)


def test_a_correction_is_an_unordered_candidate_never_a_pair(labeled: Labeled) -> None:
    """CR16: a human path past an answer is shown before and after; nothing is chosen."""
    labeled.build()
    view = scorecard_view(labeled.out, reference=no_reference)
    tree = labeled.tree

    assert view.pairs == () and preference_lines(labeled.out, eligible_only=False) == b""
    (candidate,) = view.candidates
    assert (candidate.before, candidate.after, candidate.model) == (
        tree.answered,
        tree.corrected,
        "qwen36-27b",
    )
    assert candidate.contributions == ("phase_edit", "refine") and candidate.after_fit
    assert (
        candidate.before_phases == ("CIF000001", "CIF000002")
        and "CIF000003" in candidate.after_phases
    )
    assert (
        candidate.reference_exposed and not candidate.has_preference
    )  # the owner declared exposure


def test_a_refused_edit_is_counted_and_never_listed(labeled: Labeled) -> None:
    refused = labeled.tree.world.act(OWNER, labeled.tree.unsuccessful, "refine-nothing", REFINE)
    assert isinstance(refused.attempt.outcome, Refused)
    labeled.build()

    view = scorecard_view(labeled.out, reference=no_reference)
    assert [c.after for c in view.candidates] == [labeled.tree.corrected]
    assert view.unchanged_paths == 1 and view.pairs == ()
    assert "1 human paths left the answer unchanged (refused or no-op edits)" in render(view)


def prefer(tree: Tree, chosen: str, rejected: str, reason: str) -> None:
    """An explicit preference through the real WorkbenchStore.record (review.Preference)."""
    tree.world.store.record(
        OWNER, chosen, PreferenceRequest(chosen=chosen, rejected=rejected, reason=reason)
    )


def test_an_explicit_preference_is_a_pair_held_back_by_exposure_and_never_called_trainable(
    labeled: Labeled,
) -> None:
    tree = labeled.tree
    prefer(tree, tree.corrected, tree.answered, "CIF000003 explains the residual")
    dataset = labeled.build()
    view = scorecard_view(labeled.out, reference=no_reference)

    (pair,) = view.pairs
    assert (pair.rejected, pair.chosen, pair.model, pair.reason, pair.resolved) == (
        tree.answered,
        tree.corrected,
        "qwen36-27b",
        "[withheld]",
        True,
    )
    assert pair.rejected_phases == ("CIF000001", "CIF000002") and "CIF000003" in pair.chosen_phases
    assert not pair.review_eligible and pair.withheld_because.startswith("a reference-exposed path")
    assert pair.training_permission == "not checked"  # CR18: no gate here proves training rights
    assert view.candidates[0].has_preference
    assert preference_lines(labeled.out) == b""  # withheld pairs stay out of the served file
    assert next(
        r for r in view.reconciliation if r.name.startswith("explicit preferences attached")
    ).ok
    assert dataset.manifest.episode_export.preference_pairs == 1

    (line,) = preference_lines(labeled.out, eligible_only=False).splitlines()
    record = json.loads(line)
    assert (
        set(record) == {"prompt", "chosen", "rejected", "metadata"}
        and record["prompt"] == CASE.task
    )
    assert [p["candidate_id"] for p in json.loads(record["rejected"])["phases"]] == [
        "CIF000001",
        "CIF000002",
    ]
    assert "CIF000003" in [p["candidate_id"] for p in json.loads(record["chosen"])["phases"]]
    meta = record["metadata"]
    assert (
        meta["chosen_attempt"],
        meta["rejected_attempt"],
        meta["review_eligible"],
        meta["trained_on"],
    ) == (
        tree.corrected,
        tree.answered,
        False,
        False,
    )
    assert meta["training_permission"].startswith("not checked") and "trainable" not in meta
    # The rejected answer rides on the agent's own episode and on the corrected path: both carried.
    assert {x["leaf"] for x in meta["rejected_paths"]} == {tree.answered, tree.corrected}
    assert all(
        set(x) == {"leaf", "reference_exposed", "eligibility", "rights"}
        for x in meta["chosen_paths"]
    )


def test_a_sibling_preference_resolves_the_rejected_branch_and_its_restrictions(
    labeled: Labeled,
) -> None:
    """CR17: the rejected attempt sits on another branch, not on the chosen episode's path."""
    tree, world = labeled.tree, labeled.tree.world
    removed = world.act(
        OWNER, tree.answered, "remove-2", {"verb": "remove_phases", "candidates": ["CIF000002"]}
    )
    sibling = world.act(OWNER, removed.digest, "refit-1", REFINE).digest
    prefer(tree, tree.corrected, sibling, "keeping CIF000002 fits better")
    labeled.build()

    (pair,) = scorecard_view(labeled.out, reference=no_reference).pairs
    assert (pair.chosen, pair.rejected, pair.resolved) == (tree.corrected, sibling, True)
    assert pair.rejected_phases == (
        "CIF000001",
    )  # the sibling's own answer, never an empty stand-in
    assert pair.rejected_exposed is True and pair.rejected_use is not None
    (line,) = preference_lines(labeled.out, eligible_only=False).splitlines()
    meta = json.loads(line)["metadata"]
    assert [x["leaf"] for x in meta["rejected_paths"]] == [sibling]
    assert [p["candidate_id"] for p in json.loads(json.loads(line)["rejected"])["phases"]] == [
        "CIF000001"
    ]


def test_a_preference_whose_rejected_path_is_not_exported_is_withheld_and_never_a_line(
    labeled: Labeled,
) -> None:
    """CR17: the host agent session's path is excluded from the export (development_session)."""
    tree = labeled.tree
    prefer(tree, tree.corrected, tree.host_session, "the corrected fit beats the session")
    labeled.build()

    (pair,) = scorecard_view(labeled.out, reference=no_reference).pairs
    assert (pair.resolved, pair.rejected_exposed, pair.rejected_use, pair.rejected_phases) == (
        False,
        None,
        None,
        (),
    )
    assert (
        "rejected endpoint not in this export" in pair.withheld_because and not pair.review_eligible
    )
    assert preference_lines(labeled.out, eligible_only=False) == b""


def test_a_demo_raters_refit_then_sign_off_is_one_pair_in_the_revealed_demo_bucket(
    labeled: Labeled,
) -> None:
    """PI D-sub-6 (b): the on-camera loop. The owner account's refit of an agent answer, then its
    sign-off on that answer, is one pair in the demo bucket: never a preference pair, never
    eligible, never an expert label, never in the main file."""
    tree = labeled.tree
    first = labeled.build()
    (digest,) = [exports.digest_of(s) for s in first.snapshots if s.origin == tree.answered]
    labeled.store.grant(
        OWNER,
        tree.world.root,
        OWNER_RATER.principal_id,
        (CaseOperation.INSPECT, CaseOperation.ACTION),
    )
    labeled.labels.assign(OWNER, tree.world.root, (OWNER_RATER.principal_id,), (digest,))
    labeled.labels.record_permission(OWNER_RATER, PERMIT)
    steps = labeled.store.gesture(
        OWNER_RATER,
        tree.answered,
        (
            ActionRequest(
                idempotency_key="demo-add",
                action={"verb": "add_phases", "candidates": ["CIF000003"]},
            ),
            ActionRequest(idempotency_key="demo-refit", action=REFINE),
        ),
    )
    assert [s.status for s in steps] == ["recorded", "completed"]
    refit = steps[-1].digest
    labeled.labels.sign_off(OWNER_RATER, SignOffRequest(snapshot=digest, verdict="would_not_sign"))
    labeled.build()

    view = scorecard_view(labeled.out, reference=no_reference)
    (demo,) = view.demo_pairs
    assert (demo.rejected, demo.chosen, demo.bucket, demo.signoff_verdict) == (
        tree.answered,
        refit,
        "demo_revealed",
        "would_not_sign",
    )
    assert demo.rejected_phases == ("CIF000001", "CIF000002") and "CIF000003" in demo.chosen_phases
    assert demo.refit_at_utc <= demo.signed_at_utc
    # The admin owner's own correction is signed by nobody: it stays a candidate, not a pair.
    assert {c.after: c.demo_pair for c in view.candidates} == {tree.corrected: False, refit: True}
    assert view.pairs == () and view.expert_labels == 0
    assert (
        preference_lines(labeled.out) == b""
        and preference_lines(labeled.out, eligible_only=False) == b""
    )
    (line,) = demo_pair_lines(labeled.out).splitlines()
    record = json.loads(line)
    assert (
        set(record) == {"prompt", "chosen", "rejected", "metadata"}
        and record["prompt"] == CASE.task
    )
    meta = record["metadata"]
    assert (
        meta["bucket"],
        meta["author_kind"],
        meta["review_eligible"],
        meta["expert"],
        meta["trained_on"],
    ) == (
        "demo_revealed",
        "owner",
        False,
        False,
        False,
    )
    assert (meta["rejected_attempt"], meta["chosen_attempt"]) == (tree.answered, refit)
    assert all(r.ok for r in view.reconciliation), reconciliation_text(view)
    assert "Demo pairs (revealed, owner account)" in render(view)


def test_dogfood_wording_names_the_untried_reference_structure_and_the_expert_file(
    labeled: Labeled,
) -> None:
    """PI dogfood A1, F6: a search miss names the reference structure it looked for, not the
    formula; the expert-file count says how many are owner demo labels."""
    first = labeled.build()
    labeled.label_both_kinds(next(j.snapshot_digest for j in first.judge_inputs))
    labeled.build()

    def reference(case_id: str) -> dict:
        return {
            "source": "test",
            "phases": [{"formula": "GeO2", "space_group": 152, "candidates": ["CIF999999"]}],
        }

    view = scorecard_view(labeled.out, reference=reference)
    search = step(episode(view, labeled.tree.answered), "phase_search")
    assert (
        search.outcome == "fail"
        and "did not try the published reference structure for GeO2 (SG 152, CIF999999)"
        in search.detail
    )
    assert "never tried" not in search.detail
    line = next(
        r for r in view.reconciliation if r.name.startswith("labels in labels.expert.jsonl")
    )
    assert line.ok and line.name.endswith("1 owner (revealed demo), 1 expert")


def test_dogfood_dashboard_tables_read_plainly(labeled: Labeled) -> None:
    """PI dogfood F8, F10: a K-check legend that explains the missing K4, judges named by model,
    yes and no for booleans, and "Not gradable" without expert labels."""
    first = labeled.build()
    labeled.label_both_kinds(next(j.snapshot_digest for j in first.judge_inputs))
    labeled.build()
    from phaseeval.workbench.dashboard import dashboard_view, render_tables

    view = dashboard_view(labeled.out)
    assert any(lb.rater == "claude-opus-5-5" and lb.kind == "model_judge" for lb in view.labels)
    fragment = render_tables(view)
    assert "<dt>No K4</dt>" in fragment and "<dt>K2</dt>" in fragment
    assert "<td>True</td>" not in fragment and "<td>False</td>" not in fragment
    assert "Not gradable" not in fragment  # one expert-kind label exists here
    no_expert = view.model_copy(update={"banners": (*view.banners, "No expert-kind labels yet.")})
    assert render_tables(no_expert).count("Not gradable: 0 expert labels.") == 2


# --- serving ------------------------------------------------------------------------------------

TOKENS = {
    OWNER.principal_id: ("admin-token", "user", ["owner"]),
    "owner-rater": ("label-token", "user", ["owner_rater"]),
    RATER.principal_id: ("rater-token", "user", ["rater"]),
    "agent-1": ("agent-token", "agent", []),
    "judge-opus": ("judge-token", "model_judge", []),
}


@pytest.fixture
def client(
    labeled: Labeled, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    labeled.build()
    principals = tmp_path / "principals.json"
    principals.write_text(
        json.dumps(
            {
                "schema_version": "axa.api_principals.v1",
                "principals": [
                    {
                        "principal_id": pid,
                        "principal_type": kind,
                        "credential_sha256": hashlib.sha256(token.encode()).hexdigest(),
                        "roles": roles,
                    }
                    for pid, (token, kind, roles) in TOKENS.items()
                ],
            }
        )
    )
    monkeypatch.setenv("AXA_WORKBENCH_DATASET", str(labeled.out))
    saved = dict(app.state._state)
    app.state.workbench_store = labeled.store
    app.state.attempt_authenticator = PrincipalAuthenticator.from_file(principals)
    with TestClient(app) as test_client:
        yield test_client
    app.state._state.clear()
    app.state._state.update(saved)


def test_the_scorecard_is_served_to_the_admin_owner_only(client: TestClient) -> None:
    page = client.get("/scorecard")
    assert page.status_code == 200 and "/api/attempts/scorecard" in page.text
    assert (
        "u6-dev900" not in page.text and "qwen36-27b" not in page.text
    )  # the shell carries no data
    for path in (
        "/api/attempts/scorecard",
        "/api/attempts/scorecard/pairs.jsonl",
        "/api/attempts/scorecard/demo-pairs.jsonl",
    ):
        assert client.get(path).status_code == 401
        for token in ("label-token", "rater-token", "agent-token", "judge-token"):
            response = client.get(path, headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 403, (path, token)
            assert "u6-dev900" not in response.text and "qwen36-27b" not in response.text
        assert client.get(path, headers={"Authorization": "Bearer admin-token"}).status_code == 200
    headers = {"Authorization": "Bearer admin-token"}
    assert (
        client.get("/api/attempts/scorecard?model=qwen36-27b", headers=headers).status_code == 200
    )
    assert (
        client.get("/api/attempts/scorecard?model=no-such-model", headers=headers).status_code
        == 404
    )
    assert (
        client.get(
            "/api/attempts/scorecard?model=qwen36-27b",
            headers={"Authorization": "Bearer rater-token"},
        ).status_code
        == 403
    )
    admin = client.get(
        "/api/attempts/scorecard", headers={"Authorization": "Bearer admin-token"}
    ).json()
    assert (
        "Model by step" in admin["html"] and admin["view"]["episodes"][0]["model"] == "qwen36-27b"
    )
    dashboard_page = client.get("/dashboard", follow_redirects=False)
    assert (dashboard_page.status_code, dashboard_page.headers["location"]) == (307, "/scorecard")
    # The former dashboard's tables ride in the unscoped page, from the dashboard's own route.
    assert (
        "(formerly the dashboard)" in admin["html"]
        and "Outcomes per case, model and harness" in admin["html"]
    )
    scoped = client.get("/api/attempts/scorecard?model=qwen36-27b", headers=headers).json()
    assert "(formerly the dashboard)" not in scoped["html"]


def test_local_mode_serves_the_owner_with_no_token_and_never_by_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner directive 20:28 UTC: on the launcher's single-user machine the scorecard loads with
    no token paste; the switch is the explicit flag the launcher sets, and the shared server
    keeps bearer auth."""
    local = TestClient(app, base_url="http://127.0.0.1:18793", client=("127.0.0.1", 50000))
    remote = TestClient(app, base_url="http://127.0.0.1:18793", client=("10.0.0.5", 50000))
    path = "/api/attempts/scorecard"

    monkeypatch.delenv(scorecard.LOCAL_OWNER_ENV, raising=False)
    assert local.get(path).status_code == 401  # flag unset: loopback still needs the token
    monkeypatch.setenv(scorecard.LOCAL_OWNER_ENV, "1")
    response = local.get(path)
    assert response.status_code == 200 and response.json()["local_mode"] is True
    assert local.get(path + "/pairs.jsonl").status_code == 200
    assert remote.get(path).status_code == 401  # local mode answers loopback clients only
    assert local.get(path, headers={"Authorization": "Bearer rater-token"}).status_code == 403
    # CR23: the grant needs this server's own loopback authority and a same-origin request.
    assert (
        local.get(
            path, headers={"Origin": "http://127.0.0.1:18793", "Sec-Fetch-Site": "same-origin"}
        ).status_code
        == 200
    )
    assert local.get(path, headers={"host": "localhost:18793"}).status_code == 200
    for refused in (
        {"host": "evil.example:18793"},  # DNS rebinding: a foreign name on the loopback socket
        {"host": "127.0.0.1:3000"},  # another port's authority
        {"Origin": "http://localhost:3000"},  # another local app's page
        {"Origin": "http://localhost:5173", "Sec-Fetch-Site": "same-site"},
        {"Sec-Fetch-Site": "cross-site"},
    ):
        assert local.get(path, headers=refused).status_code == 401, refused
    assert (
        local.get(path, headers={"Authorization": "Bearer admin-token"}).json()["local_mode"]
        is False
    )


# --- the real producers: importer, snapshot, blind view, custody reference -----------------------

CUSTODY = Path(os.environ.get("AXA_WORKBENCH_CUSTODY", WORKTREE / "data/custody/workbench"))
QWEN_V2 = CUSTODY / "runs/jobs/u6-dev006-qwen-006/task-dev006__5SKiLMp"
ASTRA_V2 = CUSTODY / "runs/jobs/u6-dev006-astra-006/task-dev006__eTpQJSY"  # reports Bi2CO5


@pytest.mark.skipif(
    not (CUSTODY / "DEV006/custody.json").is_file()
    or not QWEN_V2.is_dir()
    or not ASTRA_V2.is_dir(),
    reason=f"custody records or U6 runs absent under {CUSTODY}; set AXA_WORKBENCH_CUSTODY",
)
def test_a_real_import_scores_through_the_real_export_and_the_custody_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AXA_WORKBENCH_CUSTODY", str(CUSTODY))
    owner = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
    importer_service = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
    store = WorkbenchStore(
        tmp_path / "store",
        default_registry().run,
        public_seams(),
        lambda: NOW,
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(tmp_path / "store"),
    )
    case, blobs = importer.read_case_package(
        WORKTREE / "data/workbench/cases/DEV006",
        WORKTREE / "data/workbench/case_labels.json",
        custody=CUSTODY / "DEV006/custody.json",
    )
    root = store.load_case(owner, case, blobs)
    run, result, run_blobs = importer.read_harbor_run(QWEN_V2, case)
    origin = store.import_run(importer_service, root, run, result, run_blobs)
    astra_run, astra_result, astra_blobs = importer.read_harbor_run(ASTRA_V2, case)
    astra = store.import_run(importer_service, root, astra_run, astra_result, astra_blobs)
    labels = LabelStore(store)

    def export(out: Path) -> None:
        dataset = build_dataset(
            store,
            labels,
            owner,
            [root],
            INPUTS.model_copy(update={"rights": policy.rights_registry_v0((case,))}),
            "internal_evaluation",
            RaterPseudonyms(store.root / "private"),
            build_snapshot=snapshot.build_snapshot,
            review_payload=exports.blind_view,
        )
        write_dataset(dataset, out)

    export(tmp_path / "dataset")

    view = scorecard_view(tmp_path / "dataset")
    by_episode = {e.episode: e for e in view.episodes}
    assert set(by_episode) == {origin, astra}
    scored = by_episode[origin]

    assert (scored.episode, scored.case_id, scored.model, scored.job) == (
        origin,
        "DEV006",
        "qwen36-27b",
        "u6-dev006-qwen-006",
    )
    assert view.reference_cases == ("DEV006",)
    assert all(r.ok for r in view.reconciliation), reconciliation_text(view)
    selection = step(scored, "phase_selection")
    by_source = {r.source: r for r in selection.readings}
    assert (
        by_source["reference_match"].outcome == "pass"
        and by_source["reference_match"].detail == "matches published reference"
    )
    assert by_source["deterministic_check"].outcome in (
        "pass",
        "fail",
    )  # the real snapshot's K2, never absent
    assert step(scored, "phase_search").outcome == "pass"
    calls = {t.index for t in view.traces[origin]}
    assert len(calls) == len(result.trajectory) and all(
        s.call_index in calls for s in scored.steps if s.call_index is not None
    )
    assert scored.reward.total > 0 and view.pairs == ()

    # Astra reports Bi2CO5: no published-reference match, named as disputed, and the reward
    # section says that term would train the call away.
    reference = next(
        r
        for r in step(by_episode[astra], "phase_selection").readings
        if r.source == "reference_match"
    )
    assert reference.outcome == "fail" and reference.detail == (
        "does not match published reference: extra Bi2CO5 (disputed: the DARA authors report it; an expert call)"
    )
    assert {t.id: t.value for t in by_episode[astra].reward.terms}["reference_phase_list"] == 0.0
    assert "1 of 2 episodes (DEV006) lose the reference term only on a disputed phase" in render(
        view
    )
    assert step(by_episode[astra], "phase_selection").disputed

    # Q6: the qwen-only page holds no Astra run anywhere, and still reconciles its rows.
    scoped = scorecard_view(tmp_path / "dataset", model="qwen36-27b")
    assert [e.episode for e in scoped.episodes] == [origin] and scoped.scope_model == "qwen36-27b"
    assert all(r.ok for r in scoped.reconciliation), reconciliation_text(scoped)
    assert "gpt-6-astra" not in render(scoped) and "gpt-6-astra" not in scoped.model_dump_json()
    assert "of qwen36-27b only" in render(scoped)
    assert "Published reference answers are loaded for DEV006." in render(view)

    # CR15 on the snapshot the video shows (49b3d49f, u6-dev006-qwen-006): K5 says the amounts
    # are the fit's, which is not defensibility, and a submission is not support. Both frozen
    # judges read that fit as not supporting the conclusion; the cell must say so.
    quantification = step(scored, "quantification")
    assert quantification.outcome == "unknown"
    assert next(r for r in quantification.readings if r.check == "K5").outcome == "pass"
    for model in ("claude-opus-5-5", "gpt-6-astra"):
        principal = Principal(f"judge-{model}", PrincipalType.MODEL_JUDGE, frozenset())
        identity = JudgeIdentity(
            model=model,
            prompt_sha256="7" * 64,
            input_sha256="8" * 64,
            run_id=f"run-{model}",
            judge_is_actor_model=False,
        )
        labels.judge_label(
            principal,
            Label(
                snapshot=scored.snapshot,
                item="fit_supports_conclusion",
                phase_id=None,
                answer="no",
                rater=Rater(
                    principal_id=principal.principal_id, kind="model_judge", judge=identity
                ),
                exposure="blind",
                created_at_utc=NOW,
            ),
        )
    export(tmp_path / "judged")
    final = step(
        next(e for e in scorecard_view(tmp_path / "judged").episodes if e.episode == origin),
        "final_answer",
    )
    assert (final.outcome, final.source, final.check) == (
        "fail",
        "model_judge",
        "fit_supports_conclusion",
    )
    assert next(r for r in final.readings if r.check == "submitted").outcome == "pass"
