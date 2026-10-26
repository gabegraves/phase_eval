"""Every rater read and gesture route uses the same lineage projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.support.store import AGENT, JUDGE, OWNER, RATER_A, RATER_B, Bench, _import, _snapshot

from tests.support.store import app
from phaseeval.api.auth import CaseOperation, Principal, PrincipalType
from phaseeval.workbench.store import ActionRequest

ADD = {"verb": "add_phases", "candidates": ["CIF000003"]}
NOTE = {"mark": "note", "note": "looks wrong"}


def _edit(key: str, action: dict[str, Any] = ADD) -> dict[str, Any]:
    return {"steps": [{"idempotency_key": key, "action": action}]}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    """One root, a two-answer cohort assigned to rater-a and rater-b, and the work a rater must
    not see before the reveal: the owner's reference-exposed correction, rater-a's correction,
    rater-b's own edit on top of the owner's correction, and an answer outside the cohort."""
    bench = Bench(tmp_path)
    bench.client = TestClient(app, raise_server_exceptions=False)
    monkeypatch.setattr(app.state, "workbench_store", bench.store, raising=False)
    monkeypatch.setattr(app.state, "attempt_authenticator", bench.authenticator, raising=False)
    store, labels, root = bench.store, bench.labels, bench.root
    for rater in ("rater-a", "rater-b"):
        store.grant(OWNER, root, rater, (CaseOperation.ACTION,))
    store.grant(OWNER, root, AGENT.principal_id, (CaseOperation.INSPECT,))
    store.grant(OWNER, root, JUDGE.principal_id, (CaseOperation.INSPECT,))
    assert bench.assign(["rater-a", "rater-b"]).status_code == 201
    bench.allow(RATER_A)
    bench.allow(RATER_B)
    labels.declare_exposure(OWNER, OWNER.principal_id, root, "the owner has seen every demo reference")

    refit = (ActionRequest(idempotency_key="o-add", action=ADD), ActionRequest(idempotency_key="o-refine", action={"verb": "refine"}))  # fmt: skip
    owner_edit, owner_refit = (step.digest for step in store.gesture(OWNER, bench.runs[0], refit))
    assert store._load(owner_refit).actor.reference_exposure == "exposed"
    a_steps = bench.post(RATER_A, f"/{bench.runs[0]}/actions", {"steps": [r.model_dump(mode="json") for r in refit]})
    assert a_steps.status_code == 201, a_steps.text
    a_refit = a_steps.json()["attempts"][-1]["digest"]
    # rater-b's id under a credential without a label-mode role: an attempt of the rater's own
    # that descends from the owner's correction, as one made before D11 would sit in the store.
    b_plain = Principal("rater-b", PrincipalType.USER, frozenset())
    remove = ActionRequest(idempotency_key="b-remove", action={"verb": "remove_phases", "candidates": ["CIF000003"]})
    descendant = store.act(b_plain, owner_refit, remove).digest
    late = _import(store, root, "task-dev900__late", ("CIF000002",))

    bench.hidden = {"owner_edit": owner_edit, "owner_refit": owner_refit, "a_refit": a_refit, "descendant": descendant}
    bench.owner_snap = labels.put_snapshot(_snapshot(root, owner_refit, ("CIF000001", "CIF000003"), kind="expert_correction"))
    bench.late_snap = labels.put_snapshot(_snapshot(root, late, ("CIF000002",)))
    bench.a_refit = a_refit
    return bench


def test_review_of_a_hidden_attempt_or_its_descendant_is_refused(world: Bench) -> None:
    for name, digest in world.hidden.items():
        response = world.get(RATER_B, f"/{digest}/review")
        assert response.status_code == 403, (name, response.status_code, response.text[:200])


def test_a_gesture_never_builds_on_a_hidden_attempt(world: Bench) -> None:
    for name, digest in world.hidden.items():
        response = world.post(RATER_B, f"/{digest}/actions", _edit(f"b-{name}", {"verb": "refine"}))
        assert response.status_code == 403, (name, response.status_code, response.text[:200])
    assert world.post(RATER_B, f"/{world.runs[0]}/actions", _edit("b-own")).status_code == 201


def test_freezing_a_hidden_attempt_is_refused_and_stores_nothing(world: Bench, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = sorted((world.store.root / "snapshots").iterdir())

    response = world.post(RATER_B, "/snapshots", {"origin": world.a_refit, "kind": "expert_correction"})

    assert response.status_code == 403, response.text
    assert sorted((world.store.root / "snapshots").iterdir()) == frozen
    own = world.post(RATER_A, "/snapshots", {"origin": world.a_refit, "kind": "expert_correction"})
    assert own.status_code == 201, own.text


def test_snapshot_reads_show_only_the_cohort_and_own_corrections(world: Bench) -> None:
    for snap in (world.owner_snap, world.late_snap, world.reference):
        for path in (f"/snapshots/{snap}", f"/snapshots/{snap}/labels", f"/snapshots/{snap}/review"):
            response = world.get(RATER_B, path)
            assert response.status_code == 403, (path, response.status_code, response.text[:200])
    for snap in world.snaps:
        assert world.get(RATER_B, f"/snapshots/{snap}").status_code == 200


def test_step_feedback_reads_and_writes_only_visible_attempts(world: Bench) -> None:
    for name, digest in world.hidden.items():
        step = {"attempt": digest, "step_kind": "attempt"}
        assert world.get(RATER_B, f"/step-feedback?attempt={digest}").status_code == 403, name
        assert world.post(RATER_B, "/step-feedback", {"step": step} | NOTE).status_code == 403, name
    own = {"attempt": world.runs[0], "call_index": 0, "step_kind": "tool_call"}
    assert world.post(RATER_B, "/step-feedback", {"step": own} | NOTE).status_code == 201


def test_raw_notes_and_preferences_refuse_raters(world: Bench) -> None:
    for name, digest in world.hidden.items():
        note = world.post(RATER_B, f"/{digest}/records", {"kind": "note", "text": "seen"})
        assert note.status_code == 403, (name, note.status_code)
    own = world.post(RATER_B, f"/{world.runs[0]}/actions", _edit("b-own")).json()["attempts"][0]["digest"]
    preference = {"kind": "preference", "chosen": own, "rejected": world.a_refit, "reason": "better"}
    assert world.post(RATER_B, f"/{own}/records", preference).status_code == 403
    assert world.post(RATER_B, f"/{own}/records", {"kind": "note", "text": "mine"}).status_code == 403


def test_the_raw_routes_refuse_a_rater(world: Bench) -> None:
    run, root = world.runs[0], world.root
    sha = json.dumps({"trial": "task-dev900__qwen"}).encode()
    for path in (
        f"/{run}",
        f"/{run}/path",
        f"/roots/{root}/tree",
        f"/compare?a={run}&b={world.runs[1]}",
        f"/{run}/artifacts/{hashlib.sha256(sha).hexdigest()}",
    ):
        assert world.get(RATER_B, path).status_code == 403, path


def test_agents_and_model_judges_read_no_review_lineage(world: Bench) -> None:
    for principal in (AGENT, JUDGE):
        for digest in (world.hidden["owner_refit"], world.runs[0]):
            response = world.get(principal, f"/{digest}/review")
            assert response.status_code == 403, (principal.principal_id, response.status_code)


def test_a_rater_still_reads_and_builds_on_their_own_work(world: Bench) -> None:
    a_refit = world.a_refit
    assert world.post(RATER_A, f"/{a_refit}/actions", _edit("a-again", {"verb": "remove_phases", "candidates": ["CIF000003"]})).status_code == 201  # fmt: skip
    assert world.get(RATER_A, f"/step-feedback?attempt={a_refit}").status_code == 200
    assert world.post(RATER_A, "/step-feedback", {"snapshot": world.snaps[0], "step": {"attempt": a_refit, "step_kind": "attempt"}} | NOTE).status_code == 201  # fmt: skip


def test_the_reveal_opens_other_raters_work_but_never_the_raw_routes(world: Bench) -> None:
    world.sign_all(RATER_A)
    assert world.get(RATER_B, f"/snapshots/{world.owner_snap}").status_code == 403  # rater-b unsigned
    world.sign_all(RATER_B)
    assert world.get(RATER_B, f"/snapshots/{world.owner_snap}").status_code == 200
    assert world.get(RATER_B, f"/step-feedback?attempt={world.a_refit}").status_code == 200
    assert world.get(RATER_B, f"/{world.a_refit}").status_code == 403
