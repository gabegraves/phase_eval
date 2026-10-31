"""Triage (C1a): CheckPolicy, pure routing, the audit sample, publish-once routing records and
routing accuracy against sign-offs.

The pure tests run on hand-built readings and sign-offs. The store tests need a real
WorkbenchStore. The sweep tests import the 12 harness v2 trials (DEV006 and DEV102, series 006
to 008) into a temporary store and need the view lane's snapshots (SnapshotCheck.policy_version)
and the labels lane's LabelStore; they skip until both are on this branch. The U6 runs are
git-excluded and exist only on the rig: from another worktree, point AXA_WORKBENCH_CUSTODY at a
data/custody/workbench directory.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from phaseeval.api.auth import CaseOperation, Principal, PrincipalType
from phaseeval.workbench import importer
from phaseeval.workbench import snapshot as snapshots
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.labels import ExposureLedger, JudgeIdentity, LabelStore, Rater, SignOff
from phaseeval.workbench.model import canonical_bytes
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import AttemptAccessError, WorkbenchStore
from phaseeval.workbench.triage import (
    PROVISIONAL_2026_10_21 as P1,
)
from phaseeval.workbench.triage import (
    CheckOutcome,
    CheckPolicy,
    CheckReading,
    PolicyMismatch,
    PolicyVersionConflict,
    RouteDecision,
    Triage,
    audit_pick,
    grade_routing,
    readings_v1,
    route,
)

WORKTREE = Path(__file__).resolve().parents[2]
CASES = WORKTREE / "data/workbench/cases"
LABELS = WORKTREE / "data/workbench/case_labels.json"
CUSTODY = Path(os.environ.get("AXA_WORKBENCH_CUSTODY", WORKTREE / "data/custody/workbench"))
V2_TRIALS = sorted((CUSTODY / "runs/jobs").glob("u6-dev*-*-00[678]/task-*"))

OWNER = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
RATER = Principal("rater-1", PrincipalType.USER, frozenset({"rater"}))
STRANGER = Principal("stranger", PrincipalType.USER, frozenset())
IMPORTER = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))

SNAP = "a" * 64
ROOT = "b" * 64
NO_AUDIT = P1.model_copy(update={"audit_rate": 0.0})
ALL_AUDIT = P1.model_copy(update={"audit_rate": 1.0})
T0 = datetime(2026, 10, 22, 6, 0, tzinfo=UTC)


def reading(code: str, status: str = "pass", value: float | None = None) -> CheckReading:
    return CheckReading(code=code, status=status, value=value, computed_under=P1.version)


def readings(**changed: tuple[str, float | None]) -> tuple[CheckReading, ...]:
    """The four v1 checks, passing unless named: readings(K2=("flag", 7.04))."""
    statuses = {
        "K1": ("pass", None),
        "K2": ("pass", None),
        "K3": ("pass", None),
        "K5": ("pass", None),
    }
    return tuple(reading(code, *value) for code, value in (statuses | changed).items())


def outcome(decision: RouteDecision, code: str) -> CheckOutcome:
    (found,) = [c for c in decision.checks if c.code == code]
    return found


# --- the policy ---------------------------------------------------------------------------------


def test_p1_copies_the_checkpoint_checks_rows() -> None:
    assert P1.version == "kchecks-2026-10-21-p1" and P1.provisional
    assert (
        "checkpoint-checks.md" in P1.justified_by
        and "research/checkpoint-checks" in P1.justified_by
    )
    assert P1.routes_on == ("K2", "K5") and P1.routing_checks() == {"K2", "K5"}
    assert (P1.k2_flag_height_percent, P1.k2_list_height_percent) == (5.0, 2.0)
    assert (P1.k3_flag_shift_percent, P1.k3_min_weight_percent) == (1.0, 2.0)
    assert (P1.k1_pawley_good_max_rwp_percent, P1.k1_rise_flag_points) == (15.0, 10.0)
    k2 = P1.basis["K2"]
    assert (k2.scans, k2.instruments) == (7, 1)
    assert (k2.matched_fits, k2.matched_flagged, k2.near_miss_fits, k2.near_miss_flagged) == (
        33,
        1,
        25,
        21,
    )
    assert (P1.basis["K3"].matched_flagged, P1.basis["K3"].near_miss_flagged) == (3, 9)
    assert P1.basis["K5"] == "unmeasured"
    assert P1.audit_rate == 0.10


def test_no_threshold_has_a_default() -> None:
    assert all(field.is_required() for field in CheckPolicy.model_fields.values())
    data = P1.model_dump()
    del data["k2_flag_height_percent"]
    with pytest.raises(ValidationError):
        CheckPolicy.model_validate(data)


def test_a_routing_check_names_its_basis() -> None:
    data = P1.model_dump()
    del data["basis"]["K5"]
    with pytest.raises(ValidationError, match="without a basis"):
        CheckPolicy.model_validate(data)


# --- route --------------------------------------------------------------------------------------


def test_routing_is_deterministic() -> None:
    first = route(SNAP, ROOT, readings(K2=("pass", 3.18)), P1)
    again = route(SNAP, ROOT, tuple(reversed(readings(K2=("pass", 3.18)))), P1)
    assert canonical_bytes(first) == canonical_bytes(again)


def test_k2_routes_at_five_percent_and_the_dev006_weak_peak_does_not() -> None:
    flagged = route(SNAP, ROOT, readings(K2=("flag", 7.04)), NO_AUDIT)
    assert flagged.route == "expert_flagged"
    assert outcome(flagged, "K2").threshold == 5.0 and outcome(flagged, "K2").value == 7.04
    weak = route(SNAP, ROOT, readings(K2=("pass", 3.18)), NO_AUDIT)  # qwen's DEV006 residual
    assert weak.route == "cleared"


def test_a_routing_check_that_cannot_be_computed_routes_to_an_expert() -> None:
    decision = route(SNAP, ROOT, readings(K2=("not_computable", None)), NO_AUDIT)
    assert decision.route == "expert_flagged"
    assert outcome(decision, "K2").threshold is None


def test_a_missing_routing_reading_routes_to_an_expert() -> None:
    decision = route(SNAP, ROOT, [r for r in readings() if r.code != "K5"], NO_AUDIT)
    assert decision.route == "expert_flagged"
    assert outcome(decision, "K5").status == "not_computable"


@pytest.mark.parametrize("status", ["flag", "not_computable"])
def test_k1_and_k3_never_route(status: str) -> None:
    decision = route(SNAP, ROOT, readings(K1=(status, None), K3=(status, None)), NO_AUDIT)
    assert decision.route == "cleared"
    for code in ("K1", "K3"):
        assert not outcome(decision, code).routes and outcome(decision, code).status == status
        assert outcome(decision, code).threshold is None


@pytest.mark.parametrize("status", ["pass", "flag", "not_computable"])
def test_k6_is_shown_and_never_routes(status: str) -> None:
    decision = route(SNAP, ROOT, (*readings(), reading("K6", status)), NO_AUDIT)
    assert decision.route == "cleared"
    assert not outcome(decision, "K6").routes and outcome(decision, "K6").status == status
    assert outcome(decision, "K6").threshold is None


def test_a_snapshot_frozen_without_k6_gets_no_k6_outcome() -> None:
    assert all(c.code != "K6" for c in route(SNAP, ROOT, readings(), NO_AUDIT).checks)


def test_k5_routes_and_says_it_is_unmeasured() -> None:
    decision = route(SNAP, ROOT, readings(K5=("flag", None)), NO_AUDIT)
    assert decision.route == "expert_flagged"
    assert outcome(decision, "K5").unmeasured and not outcome(decision, "K2").unmeasured
    assert decision.unmeasured_flag
    both = route(SNAP, ROOT, readings(K2=("flag", 6.0), K5=("flag", None)), NO_AUDIT)
    assert not both.unmeasured_flag


def test_a_reading_computed_under_another_policy_is_refused() -> None:
    stale = CheckReading(code="K2", status="pass", value=None, computed_under="kchecks-old")
    with pytest.raises(PolicyMismatch):
        route(SNAP, ROOT, [*readings()[:1], stale], P1)


def test_a_decision_is_flagged_exactly_when_a_routing_check_did_not_pass() -> None:
    passing = route(SNAP, ROOT, readings(), NO_AUDIT)
    with pytest.raises(ValidationError):
        passing.model_validate(passing.model_dump() | {"route": "expert_flagged"})
    flagged = route(SNAP, ROOT, readings(K2=("flag", 9.0)), NO_AUDIT)
    with pytest.raises(ValidationError):
        flagged.model_validate(flagged.model_dump() | {"route": "cleared"})


def digests(n: int) -> list[str]:
    return [hashlib.sha256(str(i).encode()).hexdigest() for i in range(n)]


def test_the_audit_sample_is_seeded_and_near_its_rate() -> None:
    picked = {d for d in digests(4000) if audit_pick(d, P1)}
    assert 0.08 < len(picked) / 4000 < 0.12
    assert picked == {d for d in digests(4000) if audit_pick(d, P1)}
    reseeded = P1.model_copy(update={"audit_seed": "another-seed"})
    assert picked != {d for d in digests(4000) if audit_pick(d, reseeded)}
    assert not any(audit_pick(d, NO_AUDIT) for d in digests(200))
    assert all(audit_pick(d, ALL_AUDIT) for d in digests(200))


def test_a_policy_change_under_one_version_changes_the_decision_bytes() -> None:
    """CR-05: a threshold or audit seed edited under the same version can leave the route as it
    was; the decision still names the exact policy bytes."""
    edited = P1.model_copy(update={"k2_flag_height_percent": 10.0, "audit_seed": "another-seed"})
    assert edited.version == P1.version and edited.sha256 != P1.sha256
    same_route = readings(K2=("not_computable", None))
    a, b = route(SNAP, ROOT, same_route, P1), route(SNAP, ROOT, same_route, edited)
    assert a.route == b.route == "expert_flagged"
    assert canonical_bytes(a) != canonical_bytes(b) and a.policy_sha256 == P1.sha256


def test_only_an_unflagged_analysis_is_audited() -> None:
    assert route(SNAP, ROOT, readings(), ALL_AUDIT).route == "expert_audit"
    assert route(SNAP, ROOT, readings(K2=("flag", 6.0)), ALL_AUDIT).route == "expert_flagged"


# --- publish once -------------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> WorkbenchStore:
    return WorkbenchStore(
        tmp_path / "store",
        default_registry().run,
        public_seams(),
        lambda: T0,
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(tmp_path / "store"),
    )


def routing_file(store: WorkbenchStore, snapshot: str, version: str) -> Path:
    return store.root / "routing" / snapshot / f"{version}.json"


def test_publishing_the_same_decision_again_is_a_no_op(store: WorkbenchStore) -> None:
    triage = Triage(store, None, P1)  # type: ignore[arg-type]  # publishing reads no label
    decision = route(SNAP, ROOT, readings(), P1)
    triage._publish(decision)
    path = routing_file(store, SNAP, P1.version)
    before = (path.read_bytes(), path.stat().st_ino)
    triage._publish(decision)
    assert (path.read_bytes(), path.stat().st_ino) == before
    assert before[0] == canonical_bytes(decision)


def test_changed_bytes_under_one_policy_version_raise(store: WorkbenchStore) -> None:
    triage = Triage(store, None, P1)  # type: ignore[arg-type]
    triage._publish(route(SNAP, ROOT, readings(), P1))
    path = routing_file(store, SNAP, P1.version)
    before = path.read_bytes()
    with pytest.raises(PolicyVersionConflict):
        triage._publish(route(SNAP, ROOT, readings(K2=("flag", 6.0)), P1))
    assert path.read_bytes() == before
    bumped = P1.model_copy(update={"version": "kchecks-test-p2"})
    unpinned = [r.model_copy(update={"computed_under": None}) for r in readings(K2=("flag", 6.0))]
    Triage(store, None, bumped)._publish(route(SNAP, ROOT, unpinned, bumped))  # type: ignore[arg-type]
    assert routing_file(store, SNAP, "kchecks-test-p2").is_file() and path.read_bytes() == before


def test_a_version_stays_bound_to_its_policy_bytes(store: WorkbenchStore) -> None:
    """CR-05: the edited policy would route this snapshot exactly as p1 does, and is still refused
    before its decision is written."""
    Triage(store, None, P1)._publish(route(SNAP, ROOT, readings(), P1))  # type: ignore[arg-type]
    bound = store.root / "routing-policies" / f"{P1.version}.json"
    assert bound.read_bytes() == canonical_bytes(P1)
    edited = P1.model_copy(update={"k2_flag_height_percent": 10.0, "audit_seed": "another-seed"})
    other = "e" * 64
    with pytest.raises(PolicyVersionConflict):
        Triage(store, None, edited)._publish(  # type: ignore[arg-type]
            route(other, ROOT, readings(K2=("not_computable", None)), edited)
        )
    assert not routing_file(store, other, P1.version).exists()
    assert bound.read_bytes() == canonical_bytes(P1)
    with pytest.raises(PolicyMismatch):
        Triage(store, None, P1)._publish(route(other, ROOT, readings(), edited))  # type: ignore[arg-type]


needs_custody = pytest.mark.skipif(
    not (CUSTODY / "DEV006/custody.json").is_file(),
    reason=f"custody records absent under {CUSTODY}; set AXA_WORKBENCH_CUSTODY",
)


@needs_custody
def test_only_a_principal_who_may_export_the_root_reads_decisions(store: WorkbenchStore) -> None:
    case, blobs = importer.read_case_package(
        CASES / "DEV006", LABELS, custody=CUSTODY / "DEV006/custody.json"
    )
    root = store.load_case(OWNER, case, blobs)
    store.grant(OWNER, root, RATER.principal_id, (CaseOperation.INSPECT,))
    store.grant(OWNER, root, STRANGER.principal_id, (CaseOperation.INSPECT,))
    triage = Triage(store, None, P1)  # type: ignore[arg-type]
    decision = route(SNAP, root, readings(), P1)
    triage._publish(decision)
    assert triage.decisions(OWNER, P1.version) == (decision,)
    assert triage.decisions(STRANGER, P1.version) == ()
    with pytest.raises(AttemptAccessError):
        triage.decisions(RATER, P1.version)
    with pytest.raises(ValueError):
        triage.decisions(OWNER, "../escape")


# --- grade_routing ------------------------------------------------------------------------------


AUDITED = [d for d in digests(400) if audit_pick(d, P1)]
UNAUDITED = [d for d in digests(400) if not audit_pick(d, P1)]


def decision(i: int, kind: str) -> RouteDecision:
    """A p1 decision on a distinct snapshot whose seeded audit draw gives the wanted route."""
    if kind == "expert_flagged":
        return route(UNAUDITED[100 + i], ROOT, readings(K2=("flag", 8.0)), P1)
    snapshot = AUDITED[i] if kind == "expert_audit" else UNAUDITED[i]
    decided = route(snapshot, ROOT, readings(), P1)
    assert decided.route == kind
    return decided


def sign(
    d: RouteDecision,
    verdict: str,
    rater: str = "owner-rater",
    rater_obj: Rater | None = None,
    **extra: object,
) -> SignOff:
    fields = {
        "snapshot": d.snapshot,
        "root": d.root,
        "verdict": verdict,
        "rater": rater_obj or Rater(principal_id=rater, kind="owner"),
        "exposure": "blind",
        "created_at_utc": T0,
    }
    return SignOff(**(fields | extra))


def batch() -> tuple[list[RouteDecision], list[SignOff]]:
    flagged = [decision(i, "expert_flagged") for i in range(4)]
    audited = [decision(i, "expert_audit") for i in range(4, 7)]
    cleared = [decision(i, "cleared") for i in range(7, 10)]
    signoffs = [sign(d, "would_not_sign") for d in flagged[:3]] + [sign(flagged[3], "would_sign")]
    signoffs += [sign(audited[0], "would_not_sign")] + [sign(d, "would_sign") for d in audited[1:]]
    signoffs += [sign(cleared[0], "would_not_sign")]  # reviewed on request, outside the sample
    return flagged + audited + cleared, signoffs


def test_routing_is_graded_against_sign_offs_with_misses_listed() -> None:
    decisions, signoffs = batch()
    grade = grade_routing(decisions, signoffs, P1, min_audit_reviews=3)
    assert (grade.caught, grade.false_alarms, grade.missed) == (3, 1, 2)
    assert [(m.snapshot, m.route) for m in grade.misses] == sorted(
        [(decisions[4].snapshot, "expert_audit"), (decisions[7].snapshot, "cleared")]
    )
    cleared = grade.by_route["cleared"]
    assert (cleared.decisions, cleared.wrong, cleared.right, cleared.unreviewed) == (3, 1, 0, 2)
    # flagged wrong 3/4 x 4 = 3; unflagged wrong 1/3 audited x 6 unflagged = 2; recall 3/5
    assert grade.recall_estimate == pytest.approx(0.6)
    flagged = grade.by_route["expert_flagged"]
    assert flagged.reviewed_roots == 1  # four fits of one pattern count as one root
    assert grade.rater_kinds == ("owner",) and grade.provisional


def test_recall_waits_for_enough_reviewed_audits() -> None:
    decisions, signoffs = batch()
    grade = grade_routing(decisions, signoffs, P1, min_audit_reviews=30)
    assert grade.recall_estimate is None and "3 reviewed audit" in grade.recall_withheld
    assert grade.missed == 2  # counts and misses are still reported
    with pytest.raises(ValueError):
        grade_routing(decisions, signoffs, P1, min_audit_reviews=0)


def test_only_each_raters_latest_blind_verdict_counts() -> None:
    d = decision(0, "expert_audit")
    later = T0 + timedelta(minutes=5)
    signoffs = [
        sign(d, "would_sign"),
        sign(d, "would_not_sign", created_at_utc=later),
        sign(d, "would_sign", created_at_utc=later + timedelta(minutes=5), exposure="revealed"),
        sign(d, "unresolved", rater="expert-2"),
    ]
    grade = grade_routing([d], signoffs, P1, min_audit_reviews=1)
    assert grade.by_route["expert_audit"].wrong == 1 and grade.missed == 1
    assert grade.rater_kinds == ("owner",)
    unresolved_only = grade_routing([d], signoffs[3:], P1, min_audit_reviews=1)
    assert (
        unresolved_only.rater_kinds == ()
        and unresolved_only.by_route["expert_audit"].unreviewed == 1
    )


def test_raters_who_disagree_are_disputed_not_counted() -> None:
    d = decision(0, "expert_flagged")
    grade = grade_routing(
        [d],
        [sign(d, "would_sign"), sign(d, "would_not_sign", rater="expert-2")],
        P1,
        min_audit_reviews=1,
    )
    flagged = grade.by_route["expert_flagged"]
    assert (flagged.disputed, flagged.wrong, flagged.right) == (1, 0, 0)


def test_a_model_judge_never_grades_routing() -> None:
    d = decision(0, "cleared")
    identity = JudgeIdentity(
        model="m",
        prompt_sha256="c" * 64,
        input_sha256="d" * 64,
        run_id="r1",
        judge_is_actor_model=False,
    )
    judge = Rater(principal_id="judge", kind="model_judge", judge=identity)
    with pytest.raises(ValueError, match="model judge"):
        grade_routing(
            [d],
            [sign(d, "would_sign", rater=judge.principal_id, rater_obj=judge)],
            P1,
            min_audit_reviews=1,
        )


def test_decisions_under_another_policy_are_refused() -> None:
    other = P1.model_copy(update={"version": "kchecks-test-p2"})
    with pytest.raises(PolicyMismatch):
        grade_routing([decision(0, "cleared")], [], other, min_audit_reviews=1)


# --- sweep over the 12 harness v2 answers -------------------------------------------------------

LANES_LANDED = "policy_version" in snapshots.SnapshotCheck.model_fields
# A harness v1 DEV102 trial that ran out of budget without an answer (unsuccessful).
UNANSWERED = sorted((CUSTODY / "runs/jobs").glob("u6-dev102-qwen-001/task-*"))
sweep_world = pytest.mark.skipif(
    not LANES_LANDED
    or len(V2_TRIALS) != 12
    or len(UNANSWERED) != 1
    or not (CUSTODY / "DEV006/custody.json").is_file(),
    reason=(
        "needs the view lane (SnapshotCheck.policy_version, build_snapshot, blind_payload) and the "
        f"labels lane on this branch, and the 12 harness v2 trials under {CUSTODY}/runs/jobs"
    ),
)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    store_root = tmp_path_factory.mktemp("triage") / "store"
    store = WorkbenchStore(
        store_root,
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=CUSTODY / "runs",
        exposure=ExposureLedger(store_root),
    )
    roots, jobs = {}, {}
    for case_id in ("DEV006", "DEV102"):
        case, blobs = importer.read_case_package(
            CASES / case_id, LABELS, custody=CUSTODY / f"{case_id}/custody.json"
        )
        roots[case_id] = store.load_case(OWNER, case, blobs)
        for trial in [*V2_TRIALS, *UNANSWERED]:
            if f"task-{case_id.lower()}__" in trial.name:
                run, result, run_blobs = importer.read_harbor_run(trial, case)
                jobs[store.import_run(IMPORTER, roots[case_id], run, result, run_blobs)] = (
                    trial.parent.name
                )
    labels = LabelStore(store)
    triage = Triage(store, labels, P1)
    yield {
        "store": store,
        "labels": labels,
        "triage": triage,
        "roots": roots,
        "jobs": jobs,
        "queue": triage.sweep(OWNER),
    }


@sweep_world
def test_the_sweep_routes_every_answered_v2_episode_once(world: dict) -> None:
    queue, triage = world["queue"], world["triage"]
    assert sum(queue.counts.values()) == 12 and queue.policy_version == P1.version
    decisions = triage.decisions(OWNER, P1.version)
    assert len(decisions) == 12 and queue.provisional
    snaps = {d.snapshot: world["labels"]._snapshot(d.snapshot) for d in decisions}
    routed_jobs = {world["jobs"][s.origin] for s in snaps.values()}
    assert "u6-dev102-qwen-001" in world["jobs"].values()  # imported, never routed
    assert routed_jobs == {trial.parent.name for trial in V2_TRIALS}
    assert {d.snapshot for d in queue.routed} <= {d.snapshot for d in decisions}
    # No v2 answer is flagged under p1. The qwen DEV006 answers keep the weak 3.18% peak at
    # 23.95 deg (checkpoint-checks.md K2); Astra's Bi2CO5 removes it.
    assert queue.counts["expert_flagged"] == 0
    k2 = {
        world["jobs"][s.origin]: outcome(d, "K2").value
        for d in decisions
        for s in [snaps[d.snapshot]]
    }
    assert {job: v for job, v in k2.items() if job.startswith("u6-dev006-qwen")} == {
        f"u6-dev006-qwen-00{n}": 3.18 for n in (6, 7, 8)
    }


@sweep_world
def test_a_rerun_of_the_sweep_is_a_no_op(world: dict) -> None:
    store = world["store"]
    files = sorted((store.root / "routing").glob("*/*.json"))
    before = [(f.read_bytes(), f.stat().st_ino) for f in files]
    again = world["triage"].sweep(OWNER)
    assert canonical_bytes(again) == canonical_bytes(world["queue"])
    assert sorted((store.root / "routing").glob("*/*.json")) == files
    assert [(f.read_bytes(), f.stat().st_ino) for f in files] == before


@sweep_world
def test_a_rater_can_neither_sweep_nor_read_decisions(world: dict) -> None:
    with pytest.raises(PermissionError):
        world["triage"].sweep(RATER)
    with pytest.raises(PermissionError):
        world["triage"].decisions(RATER, P1.version)


@sweep_world
def test_a_sweep_under_an_edited_policy_is_refused_before_it_freezes(world: dict) -> None:
    edited = P1.model_copy(update={"audit_seed": "another-seed"})
    before = sorted((world["store"].root / "routing").glob("*/*.json"))
    with pytest.raises(PolicyVersionConflict):
        Triage(world["store"], world["labels"], edited).sweep(OWNER)
    assert sorted((world["store"].root / "routing").glob("*/*.json")) == before


@sweep_world
def test_another_policy_version_cannot_route_v1_statuses(world: dict) -> None:
    bumped = P1.model_copy(update={"version": "kchecks-test-p2"})
    with pytest.raises(PolicyMismatch):
        Triage(world["store"], world["labels"], bumped).sweep(OWNER)
    assert not list((world["store"].root / "routing").glob("*/kchecks-test-p2.json"))


@sweep_world
def test_readings_v1_cite_the_policy_their_statuses_were_computed_under(world: dict) -> None:
    for d in world["triage"].decisions(OWNER, P1.version):
        frozen = world["labels"]._snapshot(d.snapshot)
        assert {r.computed_under for r in readings_v1(frozen)} == {P1.version}
        assert route(d.snapshot, d.root, readings_v1(frozen), P1) == d


def route_fields(value: object, path: str = "$") -> Iterator[str]:
    names = {"expert_flagged", "expert_audit", "cleared", "route.v1", "route-v1"}
    if isinstance(value, dict):
        for key, item in value.items():
            if "route" in key.lower() or "routing" in key.lower():
                yield f"{path}.{key}"
            yield from route_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from route_fields(item, f"{path}[{index}]")
    elif isinstance(value, str) and value in names:
        yield f"{path}: {value}"


@sweep_world
def test_a_blind_rater_payload_carries_no_route_field(world: dict) -> None:
    store, labels = world["store"], world["labels"]
    decisions = world["triage"].decisions(OWNER, P1.version)
    assert decisions
    for root in world["roots"].values():
        store.grant(OWNER, root, RATER.principal_id, (CaseOperation.INSPECT,))
        cohort = tuple(sorted(d.snapshot for d in decisions if d.root == root))
        labels.assign(OWNER, root, (RATER.principal_id,), cohort)  # the review cohort (Build D12)
    for d in decisions:
        payload = snapshots.blind_payload(store, labels, RATER, d.snapshot)
        assert list(route_fields(payload)) == []
