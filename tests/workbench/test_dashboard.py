"""U8 evaluation dashboard: counts that match the export, agreement within one rater kind from
initial blind responses, no judge-versus-expert number without a frozen split, the triage and
qualification tables, and admin-only serving (origin R20, R34; final architect contract U8)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_export import NOW, OWNER, OWNER_RATER, PERMIT, RATER, WORKTREE, Labeled, Tree, lines

from phaseeval.api.app import app
from phaseeval.api.auth import Principal, PrincipalAuthenticator, PrincipalType
from phaseeval.workbench import exports, importer
from phaseeval.workbench.dashboard import _route_rows, dashboard_view, render_tables, routing_decisions
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.labels import (
    ExposureLedger,
    JudgeIdentity,
    Label,
    LabelRequest,
    LabelStore,
    Rater,
)
from phaseeval.workbench.qualification import (
    CaseResult,
    ItemAgreement,
    JudgeSummary,
    JudgeVersion,
    Producer,
    QualificationReport,
    interval,
    summarize_producer,
)
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import WorkbenchStore
from phaseeval.workbench.triage import PROVISIONAL_2026_10_21 as POLICY
from phaseeval.workbench.triage import CheckReading, Triage, route


def decision(n: int, *statuses: tuple[str, str]):  # noqa: ANN201
    """A routing decision made by the real triage.route under the provisional policy."""
    readings = tuple(CheckReading(code=c, status=s, value=None, computed_under=POLICY.version) for c, s in statuses)
    return route(f"{n:064x}", "9" * 64, readings, POLICY)


ROUTES = [
    decision(1, ("K2", "flag"), ("K5", "pass")),
    decision(2, ("K2", "pass"), ("K5", "flag")),  # flagged only by the unmeasured K5
    decision(3, ("K2", "pass"), ("K5", "pass")),
    decision(4, ("K2", "pass"), ("K5", "pass")),
]


def case(n: int, verdict: str, accepted: str | None, rejected: str | None) -> CaseResult:
    applicable = tuple(side for side, value in (("accepted", accepted), ("rejected", rejected)) if value is not None)
    return CaseResult(
        root=f"{n:064x}", partition=None, expectation_sha256=None, observed=None, verdict=verdict,
        accepted_side=accepted, rejected_side=rejected, applicable=applicable, reasons=(),
    )


def producer_report(model: str, cases: tuple[CaseResult, ...]) -> QualificationReport:
    """A real qualification report for a Producer, summarized by the real summarize_producer."""
    return QualificationReport(
        set_digest="5" * 64,
        subject=Producer(model_id=model, harness="codex", harness_version="0.162.1"),
        expectations_sha256="0" * 64,
        cases=cases,
        summary=summarize_producer(cases),
    )


REPORTS = [
    # pooled pass / (pass + fail) is 1 / 3; the accepted and rejected sides are each 2 / 3
    producer_report("producer-model-a", (case(1, "pass", "pass", "pass"), case(2, "fail", "fail", "pass"), case(3, "fail", "pass", "fail"))),
    producer_report("producer-model-b", (case(4, "pass", "pass", None), case(5, "no_answer", "fail", "pass"))),
]
JUDGE_REPORT = QualificationReport(
    set_digest="6" * 64,
    subject=JudgeVersion(model="claude-opus-5-5", prompt_sha256="7" * 64, tuned_on=()),
    expectations_sha256="0" * 64,
    cases=(),
    summary=JudgeSummary(
        items=(
            ItemAgreement(
                item="phase_supported", eligible=4, coverage=interval(3, 4), n=3,
                judge_expert_pairwise=interval(3, 4), expert_expert_pairwise=None,
                tpr=interval(2, 2), tnr=interval(1, 2), judge_abstained=1,
            ),
        ),
        judge_is_actor_model_excluded=0,
    ),
)


@pytest.fixture
def labeled(tmp_path: Path) -> Labeled:
    return Labeled(Tree(tmp_path / "store"), tmp_path)


def row_of(fragment: str, text: str) -> str:
    (row,) = [r for r in re.findall(r"<tr>.*?</tr>", fragment) if f"<td>{text}</td>" in r]
    return row


def test_episode_counts_by_model_and_harness_match_the_export(labeled: Labeled) -> None:
    dataset = labeled.build()
    view = dashboard_view(labeled.out)
    episodes = lines(labeled.out / "episodes.jsonl")

    rows = {(o.case_id, o.model, o.harness): o for o in view.outcomes}
    exported = {(a.case_id, a.model, a.harness): a for a in dataset.manifest.agent_episodes}
    assert rows.keys() == exported.keys() == {("DEV900", "qwen36-27b", "harbor-local-mcp")}
    for key, row in rows.items():
        assert (row.episodes, row.submitted.count, row.unsuccessful.count, row.incomplete.count) == (
            exported[key].episodes, exported[key].submitted, exported[key].unsuccessful, exported[key].incomplete,
        )
    (row,) = rows.values()
    assert (row.episodes, row.submitted.count, row.submitted.denominator, row.unsuccessful.count) == (3, 2, 3, 1)
    assert row.human_corrected.count == 1 and len(episodes) == 4  # 3 agent episodes plus the corrected path
    assert row.snapshots.count == 1 and row.snapshots.denominator == 2  # the incomplete answer has no fit
    assert [(e.kind, e.reason, e.count) for e in view.exclusions] == [
        ("episode", "development_session", 1),
        ("episode", "open", 1),
        ("snapshot", "LookupError: no fit to freeze", 1),
    ]


def test_k6_is_listed_as_advisory_and_a_snapshot_without_it_counts_as_not_computable(labeled: Labeled) -> None:
    labeled.build()
    view = dashboard_view(labeled.out)
    (k6,) = [c for c in view.checks if c.code == "K6"]
    assert k6.passed.count == k6.flagged.count == 0 and k6.not_computable.count == k6.not_computable.denominator
    legend = render_tables(view)
    assert "<dt>K6</dt>" in legend and "it is shown and never routes" in legend


def test_no_judge_versus_expert_metric_renders_without_a_frozen_split(labeled: Labeled) -> None:
    first = labeled.build()
    labeled.label_both_kinds(first.judge_inputs[0].snapshot_digest)
    labeled.build()

    view = dashboard_view(labeled.out)
    fragment = render_tables(view)

    assert view.judge_agreement == ()
    assert view.judge_shortfall[0] == "No eligible judge qualification report available, so no judge-versus-expert number is shown."
    assert view.split_status == ("No qualification set is frozen.",)  # read from sets, never inferred
    assert "judge-expert pairwise" not in fragment and "TPR" not in fragment

    report = JUDGE_REPORT.model_copy(update={"cases": (case(9, "pass", None, None).model_copy(update={"root": labeled.tree.world.root}),)})
    with_report = dashboard_view(labeled.out, reports=[report])
    (row,) = with_report.judge_agreement
    assert (row.item, row.judge_expert.k, row.judge_expert.n, row.prompt_sha256) == ("phase_supported", 3, 4, "7" * 64)
    # Abstentions are counted per run and answers per target, so a missing count would mix units.
    assert (row.eligible, row.coverage.k, row.abstained, row.missing) == (4, 3, 1, None)
    assert row.exposure == "development_exposed (not held-out scientific validation)"
    rendered = render_tables(with_report)
    assert "test partition of a frozen calibration split" in rendered and "held-out roots" not in rendered


def test_the_triage_and_qualification_tables_show_provisional_and_both_rates(labeled: Labeled) -> None:
    first = labeled.build()
    digest = first.judge_inputs[0].snapshot_digest
    # A rater reads only its cohort before the reveal (CR-07).
    labeled.labels.assign(OWNER, labeled.tree.world.root, (OWNER_RATER.principal_id,), (digest,))
    labeled.labels.record_permission(OWNER_RATER, PERMIT)
    labeled.labels.label(OWNER_RATER, LabelRequest(snapshot=digest, item="fit_supports_conclusion", answer="yes"))
    labeled.build()

    view = dashboard_view(labeled.out, routes=ROUTES, reports=REPORTS)
    fragment = render_tables(view)

    routes = {r.route: r for r in view.routes}
    assert {r.policy_version for r in view.routes} == {POLICY.version} and all(r.provisional for r in view.routes)
    assert (routes["expert_flagged"].count.count, routes["expert_flagged"].count.denominator) == (2, 4)
    assert routes["expert_flagged"].unmeasured_flags == 1
    assert sum(r.count.count for r in view.routes) == 4
    assert "provisional" in row_of(fragment, "expert_flagged")
    row_a, row_b = row_of(fragment, "producer-model-a"), row_of(fragment, "producer-model-b")
    assert row_a.count("2 / 3") == 2 and "1 / 3" not in row_a  # both sides shown, the pooled rate never
    assert "1 / 2" in row_b and "1 / 1" in row_b  # accepted 1 / 2 (abstention fails), rejected 1 / 1
    assert "2 assessed of 2 applicable" in row_b and "1 assessed of 1 applicable" in row_b
    assert "No expert-kind labels: 0 expert labels (1 owner labels, 0 model-judge labels)." in fragment


def test_agreement_counts_one_initial_blind_response_per_distinct_rater_within_a_kind(labeled: Labeled) -> None:
    dataset = labeled.build()
    digest = dataset.judge_inputs[0].snapshot_digest

    def judge_label(model: str, principal: str, answer: str, minutes: int, own: bool = False) -> Label:
        identity = JudgeIdentity(
            model=model, prompt_sha256="7" * 64, input_sha256="8" * 64, run_id=f"run-{principal}-{minutes}",
            judge_is_actor_model=own,
        )
        return Label(
            snapshot=digest, item="fit_supports_conclusion", phase_id=None, answer=answer,
            rater=Rater(principal_id=principal, kind="model_judge", judge=identity), exposure="blind",
            created_at_utc=NOW + timedelta(minutes=minutes),
        )

    def expert_label(rater: str, answer: str, minutes: int, exposure: str = "blind") -> Label:
        return Label(
            snapshot=digest, item="fit_supports_conclusion", phase_id=None, answer=answer,
            reason="" if answer != "cannot_assess" else "the residual is outside the plotted range",
            rater=Rater(principal_id=rater, kind="expert"), exposure=exposure, created_at_utc=NOW + timedelta(minutes=minutes),
        )

    experts = [
        expert_label("r-aaaaaaaaaaaa", "yes", 1),
        expert_label("r-aaaaaaaaaaaa", "no", 9, exposure="revealed"),  # a revision, never the initial response
        expert_label("r-bbbbbbbbbbbb", "yes", 2),
        expert_label("r-cccccccccccc", "cannot_assess", 3),
    ]
    judges = [
        judge_label("claude-opus-5-5", "judge-opus-a", "no", 1),
        judge_label("claude-opus-5-5", "judge-opus-b", "yes", 2),  # same model, a later run: not its initial run
        judge_label("gpt-6-astra", "judge-astra", "no", 3, own=True),
        judge_label("claude-opus-5-5", "judge-opus-a", "yes", 0).model_copy(  # earliest, under another prompt
            update={"rater": Rater(principal_id="judge-opus-a", kind="model_judge", judge=JudgeIdentity(
                model="claude-opus-5-5", prompt_sha256="a" * 64, input_sha256="8" * 64, run_id="run-prompt-a",
                judge_is_actor_model=False))}
        ),
    ]
    (labeled.out / "labels.expert.jsonl").write_text("".join(lb.model_dump_json() + "\n" for lb in experts))
    (labeled.out / "labels.model_judge.jsonl").write_text("".join(lb.model_dump_json() + "\n" for lb in judges))

    rows = {(a.kind, a.item, a.prompt_sha256): a for a in dashboard_view(labeled.out).agreement}
    item = "fit_supports_conclusion"
    expert = rows[("expert", item, None)]
    judge = rows[("model_judge", item, "7" * 64)]
    other_prompt = rows[("model_judge", item, "a" * 64)]
    without_own = rows[("model_judge_without_own_model", item, "7" * 64)]

    assert (len(expert.raters), expert.comparable_pairs, expert.agreeing_pairs, expert.cannot_assess_pairs) == (3, 1, 1, 2)
    assert (expert.targets, expert.one_sided_targets, expert.revealed_revisions, expert.agreement) == (1, 0, 1, 1.0)
    # One initial run per model and prompt: opus's earliest run says "no", the same as astra.
    assert judge.raters == ("claude-opus-5-5@run-judge-opus-a-1", "gpt-6-astra@run-judge-astra-3")
    assert (judge.comparable_pairs, judge.agreeing_pairs) == (1, 1)
    # Another prompt is its own stratum: never paired with the first, never a second model.
    assert (other_prompt.raters, other_prompt.one_sided_targets, other_prompt.comparable_pairs) == (
        ("claude-opus-5-5@run-prompt-a",), 1, 0,
    )
    assert (len(without_own.raters), without_own.comparable_pairs) == (1, 0) and without_own.agreement is None
    assert rows[("owner", item, None)].raters == ()


def test_a_frozen_target_no_judge_answered_is_shown_missing_not_dropped(labeled: Labeled) -> None:
    """Science counterexample (.11 09:09): both judges answer phase_supported for phase 1 only of a
    judged two-phase snapshot; the second target must show as wholly missing, not vanish."""
    dataset = labeled.build()
    digest = dataset.judge_inputs[0].snapshot_digest
    labels = [
        Label(
            snapshot=digest, item="phase_supported", phase_id="CIF000001", answer="yes",
            rater=Rater(principal_id=principal, kind="model_judge", judge=JudgeIdentity(
                model=model, prompt_sha256="7" * 64, input_sha256="8" * 64, run_id=f"run-{model}",
                judge_is_actor_model=False)),
            exposure="blind", created_at_utc=NOW,
        )
        for model, principal in (("claude-opus-5-5", "judge-opus"), ("gpt-6-astra", "judge-astra"))
    ]
    (labeled.out / "labels.model_judge.jsonl").write_text("".join(lb.model_dump_json() + "\n" for lb in labels))

    view = dashboard_view(labeled.out)
    rows = {(a.kind, a.item, a.prompt_sha256): a for a in view.agreement}
    supported = rows[("model_judge", "phase_supported", "7" * 64)]
    human = rows[("expert", "phase_supported", None)]

    assert (supported.expected_targets, supported.targets, supported.wholly_missing_targets) == (2, 1, 1)
    assert (supported.comparable_pairs, supported.agreeing_pairs) == (1, 1)
    assert (human.expected_targets, human.wholly_missing_targets) == (None, None)  # scope not in the export
    fragment = render_tables(view)
    assert "earliest observed run per model and prompt, inferred from label times, not a declared run" in fragment



def test_each_judge_run_shows_its_own_coverage_of_the_frozen_targets(labeled: Labeled) -> None:
    """Reviewer R10 (.21 16:31): a judge whose calls were malformed has no labels on those
    snapshots, and the dashboard must show that per judge, not only pooled with the other judge."""
    dataset = labeled.build()
    digest = dataset.judge_inputs[0].snapshot_digest
    (snap,) = [s for s in dataset.snapshots if exports.digest_of(s) == digest]
    targets = [*((item, p.phase_id) for item in ("phase_supported", "phase_plausible") for p in snap.phases),
               ("amounts_defensible", None), ("fit_supports_conclusion", None)]

    def label(model: str, run: str, item: str, phase: str | None, answer: str, minutes: int) -> Label:
        return Label(
            snapshot=digest, item=item, phase_id=phase, answer=answer,
            reason="the residual is outside the plotted range" if answer == "cannot_assess" else "",
            rater=Rater(principal_id=f"judge-{model}", kind="model_judge", judge=JudgeIdentity(
                model=model, prompt_sha256="7" * 64, input_sha256="8" * 64, run_id=run, judge_is_actor_model=False)),
            exposure="blind", created_at_utc=NOW + timedelta(minutes=minutes),
        )

    opus = [label("claude-opus-5-5", "opus-initial-v0", item, phase, "cannot_assess" if i == 0 else "yes", 1)
            for i, (item, phase) in enumerate(targets)]
    astra = [label("gpt-6-astra", "astra-initial-v0", *targets[0], "no", 2)]  # the rest of its output was malformed
    repeat = [label("gpt-6-astra", "astra-retry-v0", item, phase, "yes", 30) for item, phase in targets[:2]]
    (labeled.out / "labels.model_judge.jsonl").write_text("".join(lb.model_dump_json() + "\n" for lb in opus + astra + repeat))

    view = dashboard_view(labeled.out)
    rows = {c.judge: c for c in view.judge_coverage}
    expected = len(targets)
    assert expected == 6  # a judged two-phase snapshot: 2 x 2 phase items + 2 analysis items
    assert [(c.initial, c.expected_targets, c.answered, c.cannot_assess, c.missing, c.snapshots_with_labels)
            for c in (rows["claude-opus-5-5@opus-initial-v0"], rows["gpt-6-astra@astra-initial-v0"],
                      rows["gpt-6-astra@astra-retry-v0"])] == [
        (True, 6, 5, 1, 0, 1),
        (True, 6, 1, 0, 5, 1),
        (False, 6, 2, 0, 4, 1),
    ]
    fragment = render_tables(view)
    assert "Coverage per model-judge run over the frozen judge targets" in fragment
    assert row_of(fragment, "gpt-6-astra@astra-initial-v0").count("<td>") == 8
    assert "<td>1 / 6</td>" in row_of(fragment, "gpt-6-astra@astra-initial-v0")
    assert "no (repeat)" in row_of(fragment, "gpt-6-astra@astra-retry-v0")


# --- serving ------------------------------------------------------------------------------------

TOKENS = {OWNER.principal_id: ("admin-token", ["owner"]), OWNER_RATER.principal_id: ("label-token", ["owner_rater"])}


@pytest.fixture
def client(labeled: Labeled, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    labeled.build()
    principals = tmp_path / "principals.json"
    principals.write_text(
        json.dumps(
            {
                "schema_version": "axa.api_principals.v1",
                "principals": [
                    {
                        "principal_id": pid,
                        "principal_type": "user",
                        "credential_sha256": hashlib.sha256(token.encode()).hexdigest(),
                        "roles": roles,
                    }
                    for pid, (token, roles) in TOKENS.items()
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


def test_the_dashboard_data_is_served_to_the_admin_owner_only(client: TestClient) -> None:
    page = client.get("/dashboard", follow_redirects=False)
    anonymous = client.get("/api/attempts/dashboard")
    label_mode = client.get("/api/attempts/dashboard", headers={"Authorization": "Bearer label-token"})
    admin = client.get("/api/attempts/dashboard", headers={"Authorization": "Bearer admin-token"})

    # The page is retired into /scorecard (owner directive 2026-10-22); the data route stays.
    assert (page.status_code, page.headers["location"]) == (307, "/scorecard")
    assert anonymous.status_code == 401
    assert label_mode.status_code == 403
    assert admin.status_code == 200
    assert "Outcomes per case, model and harness" in admin.json()["html"]
    assert admin.json()["view"]["outcomes"][0]["model"] == "qwen36-27b"
    assert RATER.principal_id not in admin.text


# --- the real triage ledger --------------------------------------------------------------------

CUSTODY = Path(os.environ.get("AXA_WORKBENCH_CUSTODY", WORKTREE / "data/custody/workbench"))
QWEN_V2 = CUSTODY / "runs/jobs/u6-dev006-qwen-006/task-dev006__5SKiLMp"


@pytest.mark.skipif(
    not (CUSTODY / "DEV006/custody.json").is_file() or not QWEN_V2.is_dir(),
    reason=f"custody records or U6 runs absent under {CUSTODY}; set AXA_WORKBENCH_CUSTODY",
)
def test_the_triage_table_reads_decisions_the_real_sweep_published(tmp_path: Path) -> None:
    owner = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
    importer_service = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
    root_dir = tmp_path / "store"
    store = WorkbenchStore(
        root_dir, default_registry().run, public_seams(), lambda: NOW, import_root=CUSTODY / "runs", exposure=ExposureLedger(root_dir)
    )
    case, blobs = importer.read_case_package(
        WORKTREE / "data/workbench/cases/DEV006", WORKTREE / "data/workbench/case_labels.json", custody=CUSTODY / "DEV006/custody.json"
    )
    root = store.load_case(owner, case, blobs)
    run, result, run_blobs = importer.read_harbor_run(QWEN_V2, case)
    store.import_run(importer_service, root, run, result, run_blobs)
    queue = Triage(store, LabelStore(store), POLICY).sweep(owner, (root,))

    decisions = routing_decisions(store, owner)
    rows = _route_rows(decisions)

    assert len(decisions) == 1 and decisions[0].policy_version == POLICY.version
    assert [(r.route, r.count.count, r.count.denominator, r.provisional) for r in rows] == [
        (decisions[0].route, 1, 1, True)
    ]
    assert sum(r.unmeasured_flags for r in rows) == queue.unmeasured_flags
