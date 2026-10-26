"""U3 labels: rubric labels, peak calls, sign-offs, reveal, step feedback and permissions
(R29, R32, R33, N8). A real WorkbenchStore in tmp_path; every HTTP case goes through TestClient."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from phaseeval.kernel.errors import Denied
from tests.support.store import write_judge_label
from tests.support.store import OWNER

from tests.support.store import app
from phaseeval.api.auth import CaseOperation, Principal, PrincipalType
from phaseeval.workbench import snapshot as snapshots
from phaseeval.workbench.labels import (
    StageProjection,
    JudgeIdentity,
    Label,
    LabelStore,
    LabelRequest,
    PermissionRequest,
    Rater,
    RevealOpened,
    SignOffRequest,
    rater_kind_for,
)
from phaseeval.workbench.model import digest_of
from phaseeval.workbench.review import projection_for
from phaseeval.workbench.store import ActionRequest

from tests.support.store import (
    AGENT,
    ALLOW,
    BANNED,
    Bench,
    IMPORTER,
    JUDGE,
    MIXED,
    OWNER_RATER,
    RATER_A,
    RATER_B,
)
from tests.support.store import _headers, _import, _snapshot


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_legacy_reveals_have_no_writer_at_wiring(world):
    LabelStore(world.store)
    assert not any(c.directory.startswith('reveals/') and c.mode != 'legacy' for c in world.store.ledger.layout())


def test_reopened_store_derives_exposure_without_publishing_reveals(demo_store, monkeypatch):
    from phaseeval.kernel.ledger import Collection
    from phaseeval.workbench import cli

    root = demo_store.roots['DEV006']
    cohort = demo_store.labels.cohort(OWNER_RATER, root)
    assert cohort
    demo_store.labels.record_permission(OWNER_RATER, PermissionRequest(**ALLOW))
    for snapshot in cohort:
        demo_store.labels.sign_off(OWNER_RATER, SignOffRequest(snapshot=snapshot, verdict='would_sign'))
    marker = demo_store.store.root / 'reveals' / root / f'{OWNER_RATER.principal_id}.json'
    assert not marker.exists()
    reopened = cli.open_store(demo_store.store.root, demo_store.directory)
    published = []
    put = Collection.put

    def observe(collection, key, record):
        result = put(collection, key, record)
        if isinstance(record, RevealOpened):
            published.append(collection._ledger)
        return result

    monkeypatch.setattr(Collection, 'put', observe)
    acted = reopened.act(OWNER_RATER, root, ActionRequest(
        idempotency_key='after-reopen', action={'verb': 'add_phases', 'candidates': ['cod_1010093']},
    ))
    attempt = dict(reopened.walk(root))[acted.digest]
    assert attempt.actor.reference_exposure == 'exposed'
    assert not marker.exists()
    assert published == []


@pytest.mark.parametrize('operation', ['sign_off', 'reveal_state', 'lineage'])
def test_stage_derivation_never_publishes_reveal_markers(bench, monkeypatch, operation):
    from phaseeval.kernel.ledger import Collection

    bench.labels.assign(OWNER, bench.root, (RATER_A.principal_id,), bench.snaps)
    bench.labels.record_permission(RATER_A, PermissionRequest(**ALLOW))
    for snapshot in bench.snaps:
        bench.labels.sign_off(RATER_A, SignOffRequest(snapshot=snapshot, verdict='would_sign'))
    published = []
    put = Collection.put

    def observe(collection, key, record):
        result = put(collection, key, record)
        if isinstance(record, RevealOpened):
            published.append(collection._ledger)
        return result

    monkeypatch.setattr(Collection, 'put', observe)
    if operation == 'sign_off':
        bench.labels.sign_off(RATER_A, SignOffRequest(snapshot=bench.snaps[0], verdict='would_sign'))
    elif operation == 'lineage':
        bench.store.lineage(RATER_A, bench.runs[0])
    else:
        assert bench.labels.reveal_state(RATER_A, bench.root).revealed
    assert published == []
    assert StageProjection(bench.store.root).exposure(RATER_A.principal_id, bench.root) == 'exposed'


def test_unauthenticated_calls_are_refused(bench: Bench) -> None:
    snap, root, run = bench.snaps[0], bench.root, bench.runs[0]
    calls = [
        ("post", "/snapshots", {"origin": run, "kind": "agent_answer"}),
        ("get", f"/snapshots/{snap}", None),
        ("post", f"/snapshots/{snap}/labels", {"item": "amounts_defensible", "answer": "yes"}),
        ("get", f"/snapshots/{snap}/labels", None),
        ("post", f"/snapshots/{snap}/peak-calls", {"peak_id": "p23.95", "cause": "missing_phase"}),
        ("post", f"/snapshots/{snap}/signoff", {"verdict": "would_sign"}),
        ("get", f"/roots/{root}/reveal-state", None),
        ("post", f"/roots/{root}/assignment", {"raters": ["rater-a"], "snapshots": [snap]}),
        ("post", "/step-feedback", {"step": {"attempt": run, "step_kind": "attempt"}, "mark": "good_move"}),
        ("get", f"/step-feedback?attempt={run}", None),
        ("post", "/permissions", ALLOW),
    ]
    for method, path, body in calls:
        for headers in ({}, {"Authorization": "Bearer wrong-secret"}):
            response = bench.client.request(method, f"/api/attempts{path}", json=body, headers=headers)
            assert response.status_code == 401, (method, path, response.text)
    assert bench.labels.export_records("internal_evaluation") == ((), (), (), ())


def test_a_blind_rater_sees_no_reference_and_no_other_raters_label(assigned: Bench) -> None:
    bench, snap = assigned, assigned.snaps[0]
    assert bench.label(RATER_A, snap, answer="no").status_code == 201

    seen = bench.get(RATER_B, f"/snapshots/{snap}")
    labels = bench.get(RATER_B, f"/snapshots/{snap}/labels")
    own = bench.label(RATER_B, snap)
    state = bench.get(RATER_B, f"/roots/{bench.root}/reveal-state")

    assert seen.status_code == 200 and seen.json()["kind"] == "agent_answer"
    assert labels.status_code == 200 and labels.json() == []
    assert own.json()["exposure"] == "blind" and own.json()["rater"] == {"principal_id": "rater-b", "kind": "expert", "judge": None}
    assert [label["rater"]["principal_id"] for label in bench.get(RATER_B, f"/snapshots/{snap}/labels").json()] == ["rater-b"]
    assert state.json()["revealed"] is False and state.json()["pending_raters"] == ["rater-a", "rater-b"]
    for response in (seen, labels, own, state):
        assert not _keys(response.json()) & BANNED, response.json()
    assert bench.get(RATER_B, f"/snapshots/{bench.reference}").status_code == 403


def test_the_reveal_waits_for_every_assigned_rater_and_a_later_label_is_a_new_revealed_record(
    assigned: Bench,
) -> None:
    bench, snap = assigned, assigned.snaps[0]
    blind = bench.label(RATER_A, snap, answer="no").json()
    bench.label(RATER_B, snap, answer="yes")
    bench.sign_all(RATER_A)

    waiting = bench.get(RATER_A, f"/roots/{bench.root}/reveal-state").json()
    assert waiting["signed_off"] is True and waiting["revealed"] is False and waiting["pending_raters"] == ["rater-b"]
    assert bench.get(RATER_A, f"/snapshots/{bench.reference}").status_code == 403
    assert {label["rater"]["principal_id"] for label in bench.get(RATER_A, f"/snapshots/{snap}/labels").json()} == {"rater-a"}

    bench.sign_all(RATER_B)
    opened = bench.get(RATER_A, f"/roots/{bench.root}/reveal-state").json()
    later = bench.label(RATER_A, snap, answer="yes").json()
    visible = bench.get(RATER_A, f"/snapshots/{snap}/labels").json()

    assert opened["revealed"] is True and opened["pending_raters"] == []
    assert bench.get(RATER_A, f"/snapshots/{bench.reference}").status_code == 200
    assert blind["exposure"] == "blind" and later["exposure"] == "revealed"
    assert [(label["rater"]["principal_id"], label["answer"], label["exposure"]) for label in visible] == [
        ("rater-a", "no", "blind"),
        ("rater-b", "yes", "blind"),
        ("rater-a", "yes", "revealed"),
    ]
    # A later assignment does not close a reveal that opened: exposure stays monotone.
    late = _import(bench.store, bench.root, "task-dev900__late", ("CIF000003",))
    third = bench.labels.put_snapshot(_snapshot(bench.root, late, ("CIF000003",)))
    assert bench.assign(["rater-a", "rater-b"], snapshots=[third]).status_code == 201
    assert bench.get(RATER_A, f"/roots/{bench.root}/reveal-state").json()["revealed"] is True
    assert bench.label(RATER_A, third, phase_id="CIF000003").json()["exposure"] == "revealed"


def test_a_snapshot_frozen_after_the_sign_offs_never_joins_the_cohort(assigned: Bench) -> None:
    """Build D12: export, refit or judge snapshots of the same root cannot block the reveal."""
    bench = assigned
    bench.sign_all(RATER_A)
    bench.sign_all(RATER_B)
    late = _import(bench.store, bench.root, "task-dev900__late", ("CIF000003",))
    bench.labels.put_snapshot(_snapshot(bench.root, late, ("CIF000003",)))

    for rater in (RATER_A, RATER_B):
        state = bench.get(rater, f"/roots/{bench.root}/reveal-state").json()
        assert state["signed_off"] is True and state["revealed"] is True and state["pending_raters"] == []


def test_a_cohort_holds_only_agent_answers_of_its_root(bench: Bench) -> None:
    assert bench.assign(["rater-a"], snapshots=[bench.reference]).status_code == 422
    assert bench.assign(["rater-a"], snapshots=[bench.snaps[0], bench.snaps[0]]).status_code == 422
    assert bench.assign(["rater-a"], snapshots=[]).status_code == 422
    assert bench.assign(["rater-a"], snapshots=["f" * 64]).status_code == 403
    assert bench.post(OWNER, f"/roots/{bench.root}/assignment", {"raters": ["rater-a"]}).status_code == 422


def test_an_unassigned_rater_is_never_revealed(bench: Bench) -> None:
    bench.assign(["rater-a"])
    bench.allow(RATER_A)
    bench.allow(RATER_B)
    bench.sign_all(RATER_A)
    # An unassigned rater has no cohort, so it reads no answer to sign (CR-07).
    assert bench.post(RATER_B, f"/snapshots/{bench.snaps[0]}/signoff", {"verdict": "would_sign"}).status_code == 403

    assert bench.get(RATER_A, f"/roots/{bench.root}/reveal-state").json()["revealed"] is True
    assert bench.get(RATER_B, f"/roots/{bench.root}/reveal-state").json()["revealed"] is False
    assert bench.assign(["rater-a"], principal=RATER_A).status_code == 403


def test_cannot_assess_needs_a_reason_and_the_export_keeps_it(assigned: Bench) -> None:
    bench, snap = assigned, assigned.snaps[0]
    refused = bench.label(RATER_A, snap, item="amounts_defensible", phase_id=None, answer="cannot_assess")
    stored = bench.label(
        RATER_A, snap, item="amounts_defensible", phase_id=None, answer="cannot_assess", reason="no esd reported"
    )
    wrong_phase = bench.label(RATER_A, snap, phase_id="CIF000003")

    assert refused.status_code == 422
    assert wrong_phase.status_code == 422
    assert stored.status_code == 201
    labels = bench.labels.export_records("internal_evaluation")[0]
    assert [(label.answer, label.reason) for label in labels] == [("cannot_assess", "no esd reported")]


def test_agents_and_model_judges_cannot_write_as_humans(assigned: Bench) -> None:
    bench, snap, run = assigned, assigned.snaps[0], assigned.runs[0]
    for principal in (AGENT, JUDGE, IMPORTER):
        assert bench.label(principal, snap).status_code == 403
        assert bench.post(principal, f"/snapshots/{snap}/signoff", {"verdict": "would_sign"}).status_code == 403
        step = {"step": {"attempt": run, "call_index": 0, "step_kind": "refine"}, "mark": "good_move"}
        assert bench.post(principal, "/step-feedback", step).status_code == 403
        assert bench.post(principal, "/permissions", ALLOW).status_code == 403
    assert bench.get(AGENT, f"/snapshots/{snap}").status_code == 403
    assert projection_for(JUDGE) == "actor"
    assert bench.labels.export_records("internal_evaluation") == ((), (), (), ())


def test_label_mode_has_its_own_credentials() -> None:
    """Build D11: the admin owner never labels, and no principal is both admin and rater."""
    assert rater_kind_for(OWNER_RATER) == "owner"
    assert rater_kind_for(RATER_A) == "expert"
    assert rater_kind_for(JUDGE) == "model_judge"
    for refused in (
        OWNER,
        MIXED,
        Principal("rater-mixed", PrincipalType.USER, frozenset({"owner", "rater"})),
        Principal("no-role", PrincipalType.USER, frozenset()),
        AGENT,
        IMPORTER,
    ):
        with pytest.raises((PermissionError, Denied)):
            rater_kind_for(refused)


def test_the_admin_owner_is_refused_every_label_mode_call(bench: Bench) -> None:
    snap, run = bench.snaps[0], bench.runs[0]
    for admin in (OWNER, MIXED):
        calls = [
            ("post", "/permissions", ALLOW),
            ("post", f"/snapshots/{snap}/labels", {"item": "amounts_defensible", "answer": "yes"}),
            ("post", f"/snapshots/{snap}/peak-calls", {"peak_id": "p23.95", "cause": "missing_phase"}),
            ("post", f"/snapshots/{snap}/signoff", {"verdict": "would_sign"}),
            ("post", "/step-feedback", {"step": {"attempt": run, "step_kind": "attempt"}, "mark": "good_move"}),
            ("post", "/snapshots", {"origin": run, "kind": "agent_answer"}),
            ("get", f"/snapshots/{snap}", None),
            ("get", f"/snapshots/{snap}/labels", None),
            ("get", f"/roots/{bench.root}/reveal-state", None),
            ("get", f"/step-feedback?attempt={run}", None),
        ]
        for method, path, body in calls:
            response = bench.client.request(method, f"/api/attempts{path}", json=body, headers=_headers(admin))
            assert response.status_code == 403, (admin.principal_id, method, path, response.text)
    assert bench.assign(["owner-rater"], principal=MIXED).status_code == 403
    assert bench.labels.export_records("internal_evaluation") == ((), (), (), ())


def test_the_owner_rater_labels_as_kind_owner(bench: Bench) -> None:
    assert bench.assign(["owner-rater"]).status_code == 201
    bench.allow(OWNER_RATER)

    stored = bench.label(OWNER_RATER, bench.snaps[0])
    bench.sign_all(OWNER_RATER)

    assert stored.json()["rater"] == {"principal_id": "owner-rater", "kind": "owner", "judge": None}
    assert stored.json()["exposure"] == "blind"
    assert bench.get(OWNER_RATER, f"/roots/{bench.root}/reveal-state").json()["revealed"] is True


def test_a_model_judge_label_is_stored_as_model_judge_and_blind(assigned: Bench) -> None:
    bench, snap = assigned, assigned.snaps[0]
    judge = JudgeIdentity(
        model="claude-opus-5-5",
        prompt_sha256="8" * 64,
        input_sha256="9" * 64,
        run_id="judge-run-1",
        judge_is_actor_model=False,
    )
    claimed = Label(
        snapshot=snap,
        item="fit_supports_conclusion",
        phase_id=None,
        answer="no",
        rater=Rater(principal_id="rater-a", kind="model_judge", judge=judge),
        exposure="revealed",
        created_at_utc=datetime(2020, 1, 1, tzinfo=UTC),
    )

    stored = write_judge_label(bench.labels, JUDGE, claimed)

    assert stored.rater.principal_id == 'judge-opus'
    assert stored.rater.kind == 'model_judge'
    assert stored.rater.judge.model == judge.model
    assert stored.rater.judge.prompt_sha256 == judge.prompt_sha256
    assert stored.exposure == "blind" and stored.created_at_utc.year == 2026
    assert bench.labels.snapshot(JUDGE, snap).kind == "agent_answer"
    with pytest.raises((PermissionError, Denied)):
        bench.labels.snapshot(JUDGE, bench.reference)
    with pytest.raises((PermissionError, Denied)):
        write_judge_label(bench.labels, RATER_A, claimed)
    with pytest.raises((PermissionError, Denied)):
        bench.labels.label(JUDGE, LabelRequest(snapshot=snap, item="fit_supports_conclusion", answer="no"))
    # A judge's labels never gate the reveal and reach the export with no permission record.
    bench.sign_all(RATER_A)
    bench.sign_all(RATER_B)
    assert bench.get(RATER_A, f"/roots/{bench.root}/reveal-state").json()["revealed"] is True
    assert [label.rater.kind for label in bench.labels.export_records("training")[0]] == ["model_judge"]


@pytest.mark.parametrize(
    ("path", "extra"),
    [
        ("labels", {"rater_kind": "owner"}),
        ("labels", {"rater": {"principal_id": "rater-b", "kind": "expert"}}),
        ("labels", {"exposure": "blind"}),
        ("signoff", {"rater_kind": "expert"}),
        ("peak-calls", {"exposure": "blind"}),
    ],
)
def test_a_body_that_names_the_rater_or_exposure_is_rejected(assigned: Bench, path: str, extra: dict) -> None:
    bodies = {
        "labels": {"item": "amounts_defensible", "answer": "yes"},
        "signoff": {"verdict": "would_sign"},
        "peak-calls": {"peak_id": "p23.95", "cause": "missing_phase"},
    }
    response = assigned.post(RATER_A, f"/snapshots/{assigned.snaps[0]}/{path}", bodies[path] | extra)

    assert response.status_code == 422, response.text
    assert assigned.labels.export_records("internal_evaluation") == ((), (), (), ())


def test_a_step_feedback_or_permission_body_naming_a_rater_is_rejected(assigned: Bench) -> None:
    step = {"step": {"attempt": assigned.runs[0], "step_kind": "attempt"}, "mark": "good_move"}
    assert assigned.post(RATER_A, "/step-feedback", step | {"rater_kind": "owner"}).status_code == 422
    assert assigned.post(RATER_A, "/permissions", ALLOW | {"principal_id": "rater-b"}).status_code == 422


def test_nothing_is_stored_without_participation_and_a_withdrawal_leaves_the_export(bench: Bench) -> None:
    snap, run = bench.snaps[0], bench.runs[0]
    assert bench.assign(["rater-a", "rater-b"]).status_code == 201  # a rater reads only its cohort (CR-07)
    assert bench.label(RATER_A, snap).status_code == 403
    bench.allow(RATER_A, participation_and_storage=False)
    assert bench.label(RATER_A, snap).status_code == 403
    assert bench.labels.export_records("internal_evaluation") == ((), (), (), ())

    bench.allow(RATER_A)
    bench.allow(RATER_B)
    for rater in (RATER_A, RATER_B):
        assert bench.label(rater, snap).status_code == 201
        assert bench.post(rater, f"/snapshots/{snap}/peak-calls", {"peak_id": "p23.95", "cause": "missing_phase"}).status_code == 201
        assert bench.post(rater, f"/snapshots/{snap}/signoff", {"verdict": "unresolved"}).status_code == 201
        step = {"step": {"attempt": run, "call_index": 3, "step_kind": "refine"}, "mark": "wrong_move"}
        assert bench.post(rater, "/step-feedback", step).status_code == 201

    def raters(use: str) -> list[set[str]]:
        return [{r.rater.principal_id for r in records} for records in bench.labels.export_records(use)]

    assert raters("internal_evaluation") == [{"rater-a", "rater-b"}] * 4
    assert raters("public_release") == [set()] * 4
    bench.allow(RATER_A, participation_and_storage=False, internal_evaluation_and_calibration=False)
    assert raters("internal_evaluation") == [{"rater-b"}] * 4
    assert bench.label(RATER_A, snap).status_code == 403


def test_step_feedback_on_an_imported_step_is_its_own_record(assigned: Bench) -> None:
    bench, run, snap = assigned, assigned.runs[0], assigned.snaps[0]
    step = {"attempt": run, "call_index": 3, "step_kind": "refine", "label": "refine CIF000001"}

    stored = bench.post(RATER_A, "/step-feedback", {"step": step, "mark": "went_wrong_here", "note": "dropped CIF000002"})
    unknown_call = bench.post(RATER_A, "/step-feedback", {"step": step | {"call_index": 1}, "mark": "good_move"})
    empty_note = bench.post(RATER_A, "/step-feedback", {"step": step, "mark": "note"})

    assert stored.status_code == 201 and stored.json()["exposure"] == "blind"
    assert unknown_call.status_code == 422 and empty_note.status_code == 422
    assert [f["mark"] for f in bench.get(RATER_A, f"/step-feedback?attempt={run}").json()] == ["went_wrong_here"]
    assert bench.get(RATER_B, f"/step-feedback?attempt={run}").json() == []
    assert bench.get(RATER_A, f"/snapshots/{snap}/labels").json() == []
    labels, _, _, feedback = bench.labels.export_records("internal_evaluation")
    assert labels == () and [(f.attempt, f.call_index, f.mark) for f in feedback] == [(run, 3, "went_wrong_here")]
    bench.sign_all(RATER_A)
    bench.sign_all(RATER_B)
    assert [f["rater"]["principal_id"] for f in bench.get(RATER_B, f"/step-feedback?attempt={run}").json()] == ["rater-a"]


def test_post_snapshots_freezes_the_built_snapshot(assigned: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = assigned
    bench.store.grant(OWNER, bench.root, "rater-a", (CaseOperation.ACTION,))
    request = ActionRequest(idempotency_key="a-add", action={"verb": "add_phases", "candidates": ["CIF000003"]})
    run = bench.store.act(RATER_A, bench.runs[0], request).digest  # rater-a's own correction
    off_cohort = _import(bench.store, bench.root, "task-dev900__new", ("CIF000003",))
    run = bench.store.act(RATER_A, run, ActionRequest(idempotency_key="a-refine", action={"verb": "refine"})).digest
    built = snapshots.build_snapshot(bench.store, RATER_A, run, "expert_correction")

    created = bench.post(RATER_A, "/snapshots", {"origin": run, "kind": "expert_correction"})
    reference = bench.post(OWNER, "/snapshots", {"origin": run, "kind": "reference"})
    agent = bench.post(AGENT, "/snapshots", {"origin": run, "kind": "agent_answer"})
    outside = bench.post(RATER_A, "/snapshots", {"origin": off_cohort, "kind": "agent_answer"})

    assert created.status_code == 201
    assert created.json()["digest"] == digest_of(built)
    assert created.json()["snapshot"] == built.model_dump(mode="json")
    assert bench.get(RATER_B, f"/snapshots/{created.json()['digest']}").status_code == 403  # until the reveal
    assert reference.status_code == 403 and agent.status_code == 403
    assert outside.status_code == 403  # an answer outside the rater's cohort (CR-07)
    with pytest.raises(ValueError, match="under its root"):
        bench.labels.put_snapshot(_snapshot(bench.root, bench.root, ("CIF000003",)))


def test_an_attempt_is_frozen_once_per_kind(bench: Bench) -> None:
    again = bench.labels.put_snapshot(snapshots.build_snapshot(bench.store, OWNER, bench.runs[0], "agent_answer"))

    assert again == bench.snaps[0]
    assert bench.labels.put_snapshot(_snapshot(bench.root, bench.runs[0], ("CIF000002",))) == again


def test_another_raters_correction_stays_hidden_until_the_reveal(assigned: Bench) -> None:
    bench = assigned
    bench.store.grant(OWNER, bench.root, "rater-a", (CaseOperation.ACTION,))
    request = ActionRequest(idempotency_key="a-add", action={"verb": "add_phases", "candidates": ["CIF000001"]})
    child = bench.store.act(RATER_A, bench.root, request)
    correction = bench.labels.put_snapshot(_snapshot(bench.root, child.digest, ("CIF000001",), kind="expert_correction"))

    assert bench.get(RATER_A, f"/snapshots/{correction}").status_code == 200
    assert bench.get(RATER_B, f"/snapshots/{correction}").status_code == 403
    bench.sign_all(RATER_A)
    bench.sign_all(RATER_B)
    assert bench.get(RATER_B, f"/snapshots/{correction}").status_code == 200


def test_permission_records_are_the_principals_own(bench: Bench) -> None:
    record = bench.labels.record_permission(RATER_A, PermissionRequest(**ALLOW))
    assert record.principal_id == "rater-a"
    assert bench.post(RATER_B, "/permissions", ALLOW).json()["principal_id"] == "rater-b"


def test_exposure_reads_the_reveal_marker_and_owner_declarations(assigned: Bench) -> None:
    """Architect D2 (LB1): one reader answers "has this rater seen the reference"."""
    bench, snap = assigned, assigned.snaps[0]
    ledger = StageProjection(bench.store.root)
    assert ledger.exposure("rater-a", bench.root) == "unexposed"

    declared = bench.labels.declare_exposure(OWNER, "rater-b", bench.root, "saw the reference in the dataset paper")
    label = bench.label(RATER_B, snap).json()
    state = bench.get(RATER_B, f"/roots/{bench.root}/reveal-state").json()
    assert bench.label(RATER_A, snap, answer="no").status_code == 201

    assert declared.declared_by == "owner-1" and ledger.exposure("rater-b", bench.root) == "exposed"
    assert label["exposure"] == "revealed"
    assert state["revealed"] is True
    assert {label["rater"]["principal_id"] for label in bench.get(RATER_B, f"/snapshots/{snap}/labels").json()} == {"rater-a", "rater-b"}
    assert ledger.exposure("rater-a", bench.root) == "unexposed"

    bench.sign_all(RATER_A)
    bench.sign_all(RATER_B)
    assert bench.get(RATER_A, f"/roots/{bench.root}/reveal-state").json()["revealed"] is True
    assert ledger.exposure("rater-a", bench.root) == "exposed"


def test_only_the_admin_owner_declares_exposure(bench: Bench) -> None:
    for principal in (OWNER_RATER, MIXED, RATER_A, JUDGE, AGENT, IMPORTER):
        with pytest.raises((PermissionError, Denied)):
            bench.labels.declare_exposure(principal, "rater-a", bench.root, "reason")
    with pytest.raises(ValueError):
        bench.labels.declare_exposure(OWNER, "rater-a", bench.root, "")
    assert StageProjection(bench.store.root).exposure("rater-a", bench.root) == "unexposed"


def test_an_attempt_records_the_actors_exposure_from_the_ledger(bench: Bench) -> None:
    """Architect D3 (LB2): HumanActor.reference_exposure reads the same ledger as the labels."""
    bench.store.grant(OWNER, bench.root, "rater-a", (CaseOperation.ACTION,))
    add = {"verb": "add_phases", "candidates": ["CIF000001"]}
    blind = bench.store.act(RATER_A, bench.root, ActionRequest(idempotency_key="blind", action=add))
    bench.labels.declare_exposure(OWNER, "owner-1", bench.root, "the owner has seen every demo reference")
    declared = bench.store.act(OWNER, bench.root, ActionRequest(idempotency_key="declared", action=add))

    assert blind.attempt.actor.reference_exposure == "unexposed"
    assert declared.attempt.actor.reference_exposure == "exposed"


def test_a_model_judge_reads_and_labels_only_an_imported_agent_answer(assigned: Bench) -> None:
    """CR-08: a judge never gets an expert correction, even one by a reference-exposed owner, nor an
    answer frozen from a human attempt; an imported agent answer stays open to it."""
    bench = assigned
    bench.store.grant(OWNER, bench.root, JUDGE.principal_id, (CaseOperation.INSPECT,))
    bench.labels.declare_exposure(OWNER, "owner-1", bench.root, "the owner has seen every demo reference")
    edit = bench.store.act(
        OWNER, bench.root, ActionRequest(idempotency_key="owner-edit", action={"verb": "add_phases", "candidates": ["CIF000002"]})
    )
    assert edit.attempt.actor.reference_exposure == "exposed"
    correction = bench.labels.put_snapshot(_snapshot(bench.root, edit.digest, ("CIF000002",), kind="expert_correction"))
    mislabeled = bench.labels.put_snapshot(_snapshot(bench.root, edit.digest, ("CIF000002",), kind="agent_answer"))
    judge = JudgeIdentity(model="claude-opus-5-5", prompt_sha256="8" * 64, input_sha256="9" * 64, run_id="judge-cr08", judge_is_actor_model=False)

    def judged(snap: str) -> Label:
        return Label(
            snapshot=snap,
            item="fit_supports_conclusion",
            phase_id=None,
            answer="no",
            rater=Rater(principal_id="judge-opus", kind="model_judge", judge=judge),
            exposure="blind",
            created_at_utc=datetime(2026, 10, 22, tzinfo=UTC),
        )

    for snap in (correction, mislabeled, bench.reference):
        assert bench.get(JUDGE, f"/snapshots/{snap}").status_code == 403
        assert bench.get(JUDGE, f"/snapshots/{snap}/labels").status_code == 403
        with pytest.raises((PermissionError, Denied)):
            write_judge_label(bench.labels, JUDGE, judged(snap))
    assert bench.get(JUDGE, f"/snapshots/{bench.snaps[0]}").status_code == 200
    assert write_judge_label(bench.labels, JUDGE, judged(bench.snaps[0])).exposure == "blind"
    assert [label.snapshot for label in bench.labels.export_records("internal_evaluation")[0]] == [bench.snaps[0]]
