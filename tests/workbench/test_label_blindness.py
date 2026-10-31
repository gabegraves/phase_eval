"""U10: the custody and blindness boundary of the label store, checked over generated histories.

Property: before a rater's reveal can open, no response to that rater carries a reference
snapshot, another rater's label, step feedback or correction, an agent answer outside its
cohort, or a run-identity or reference key; a model judge or an agent never receives one; and
every record is stamped with the exposure its author had when it was written (R32, R33, KTD8,
KTD12). The reveal waits on the latest assignment's review cohort (Build D12); an owner's
exposure declaration changes the stamp, never the reveal (architect D2). Triage (R2c): every
agent answer of an imported run is routed, and no response before the reveal carries a route
field; the revealed payload carries a snapshot's route only once the rater signed off that
snapshot and its cohort (triage.Triage.revealed). CR-07: before its
reveal, a rater reads, refits, freezes or annotates only its own lineage, its own attempts and
its cohort's origins, on every route its page calls: never another rater's or the owner's
refit, anything built on one, or an agent answer outside its cohort.

Hypothesis drives generated sequences of every route a rater's page calls (D10: queue, snapshot
and attempt review, labels, peak calls, sign-off, step feedback, the gesture result, plus
snapshots, reveal state and permissions) and LabelStore.judge_label, which has no route,
against one real WorkbenchStore per example, and checks each response against an oracle of the
reveal rule written from the labels.py contract, not from its code. Skill and coverage record: .claude/skills/label-blindness/SKILL.md.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, Phase, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule
from test_labels import (
    AGENT,
    ALLOW,
    BANNED,
    IMPORTER,
    JUDGE,
    OWNER_RATER,
    RATER_A,
    RATER_B,
    Bench,
    _headers,
    _import,
    _snapshot,
)
from test_tree import OWNER
from test_triage import route_fields

from phaseeval.api.app import app
from phaseeval.api.auth import AuthorizationError, CaseOperation, Principal, PrincipalType
from phaseeval.workbench.labels import JudgeIdentity, Label, Rater
from phaseeval.workbench.page_routes import ReviewQueue, queue_path
from phaseeval.workbench.store import ActionRequest
from phaseeval.workbench.triage import PROVISIONAL_2026_10_21, Triage, readings_v1, route

HUMANS = (OWNER_RATER, RATER_A, RATER_B)  # label-mode credentials; OWNER is the admin (D11)
AUTHORS = (RATER_A, RATER_B)  # each freezes one expert correction
STRANGERS = (JUDGE, IMPORTER, AGENT)
OWNER_ONLY = ("/scorecard", "/scorecard/pairs.jsonl", "/scorecard/demo-pairs.jsonl")  # the admin owner's scorecard: models, runs, the reference
REFITTERS = (OWNER, *HUMANS)  # the admin owner's refit is reference-informed: foreign to every rater
JUDGE_ID = JudgeIdentity(
    model="claude-opus-5-5", prompt_sha256="8" * 64, input_sha256="9" * 64, run_id="judge-run", judge_is_actor_model=False
)
EXAMPLES = int(os.environ.get("AXA_BLINDNESS_EXAMPLES", "150"))
SHRINK = os.environ.get("AXA_BLINDNESS_SHRINK", "1") != "0"
# Every record write fsyncs; a tmpfs store keeps one run near a minute instead of several.
SCRATCH = os.environ.get("AXA_BLINDNESS_TMP", "/dev/shm" if os.path.isdir("/dev/shm") else None)


def test_launcher_restore_returns_only_the_label_mode_credential(tmp_path):
    from fastapi import FastAPI
    from phaseeval.launcher import Nonces, Session, launcher_router

    launcher_app = FastAPI()
    nonces = Nonces()
    session = Session(tmp_path / "principals.json", {"owner": "private-owner", "owner-rater": "blind-rater"}, ())
    launcher_app.include_router(launcher_router(session, nonces, "test", 18828, lambda: "/workbench"))
    client = TestClient(launcher_app, base_url="http://127.0.0.1:18828", client=("127.0.0.1", 12345))
    assert client.post("/api/launcher/exchange", json={"nonce": nonces.mint()}).status_code == 200
    restored = client.post("/api/launcher/review-session", headers={"Origin": "http://127.0.0.1:18828"})
    assert restored.status_code == 200
    assert restored.json() == {"token": "blind-rater"}


def test_k6_carries_no_run_identity_for_any_status() -> None:
    from phaseeval.workbench.snapshot import FitRecord, k6

    run = "task-dev900__qwen"
    corrections = {"eps1": None, "eps2": {"value": 0.0099, "esd": 0.00001}, "eps3": None}
    texture = {"rwp_control": 9.0, "rwp_final": 6.0, "largest_change_wt_points": 8.0, "largest_change_phase": "CIF000001", "exceeds_threshold": True}
    for fields in ({}, {"angle_corrections": corrections, "eps2_bound": "0_-0.01^0.01", "texture_dependence": texture, "texture_order": 4}):
        fit = FitRecord(label=run, candidates=("CIF000001",), failed=False, stages={}, final=True, rwp=3.5, gof=1.8, phases=(), **fields)
        check = k6(fit, (10.0, 80.0))
        assert not _payload_keys(check) & BANNED
        assert "qwen" not in json.dumps(check) and "dev900" not in json.dumps(check)


RUN_NAMES = ("RUNIDENTITY", "qwen", "dev900")


def test_a_recorded_bound_outside_the_grammar_is_not_recorded_and_never_printed() -> None:
    from phaseeval.workbench.snapshot import _recorded_fit, k6

    stage = {"rwp": 3.5, "gof": 1.8, "converged": True, "phases": [], "angle_corrections": {"eps2": {"value": 0.0008, "esd": 0.002}}}
    result = {"refinement_failed": False, "rietveld": stage}
    request = {"candidates": [], "config": {"eps2": "RUNIDENTITY-qwen-dev900_-0.0001^0.0001", "rietveld": {"gewicht": "SPHAR4"}}}
    check = k6(_recorded_fit("r1", result, request), (10.0, 80.0))
    assert check["status"] == "not_computable" and "limit is not recorded" in check["detail"]
    assert not any(name in json.dumps(check) for name in RUN_NAMES)


def test_a_bound_in_the_grammar_reaches_a_rater_only_as_parsed_limits() -> None:
    from phaseeval.workbench.snapshot import FitRecord, k6

    corrections = {"eps2": {"value": 0.0099, "esd": 0.00001}}
    fit = FitRecord(label="r1", candidates=(), failed=False, stages={}, final=True, rwp=3.5, gof=1.8, phases=(),
                    angle_corrections=corrections, eps2_bound="1234.5e0_-0.01^0.01")
    check = k6(fit, (10.0, 80.0))
    assert "at its bound (-0.01 to 0.01)" in check["detail"] and "1234.5" not in check["detail"]


def test_a_recorded_texture_phase_outside_the_requested_candidates_is_not_printed() -> None:
    from phaseeval.workbench.snapshot import _recorded_fit, k6

    texture = {"rwp_control": 9.0, "rwp_final": 6.0, "largest_change_wt_points": 8.0, "largest_change_phase": "RUNIDENTITY", "exceeds_threshold": True}
    stage = {"rwp": 3.5, "gof": 1.8, "converged": True, "phases": [{"phase_id": "RUNIDENTITY"}]}
    result = {"refinement_failed": False, "rietveld": stage, "texture_dependence": texture}
    request = {"candidates": [{"candidate_id": "cod_1010093"}], "config": {"rietveld": {"gewicht": "SPHAR4"}}}
    assert "RUNIDENTITY" not in k6(_recorded_fit("r1", result, request), (10.0, 80.0))["detail"]


def test_a_recorded_gewicht_cannot_stop_the_review_or_reach_a_rater() -> None:
    from phaseeval.workbench.snapshot import _recorded_fit, k6

    for gewicht in ("SPHAR" + "9" * 5000, "SPHAR4-qwen", "qwen"):
        result = {"refinement_failed": False, "rietveld": {"rwp": 3.5, "gof": 1.8, "converged": True, "phases": []}}
        fit = _recorded_fit("r1", result, {"candidates": [], "config": {"rietveld": {"gewicht": gewicht}}})
        assert fit.texture_order is None and "qwen" not in k6(fit, (10.0, 80.0))["detail"]


def test_the_k6_a_rater_reads_before_sign_off_names_no_source_or_identity(tmp_path: Path) -> None:
    from test_import_external import BUNDLE, _case, _client, _queue, _restore, run_cli

    case, store = _case(tmp_path), tmp_path / "state/store"
    run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(BUNDLE), "--case", "DEV006", "--store", str(store))
    (digest,) = _queue(store)
    client, saved = _client(tmp_path, store)
    try:
        with client:
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
    finally:
        _restore(saved)
    (check,) = [c for c in review.json()["checks"] if c["code"] == "K6"]
    assert not _payload_keys(check) & BANNED
    text = json.dumps(check)
    assert "bgmn_saved_output" not in text and "reference" not in text and "EPS2" in check["detail"]


def test_the_preparation_record_in_the_brief_a_rater_reads_adds_no_identity(tmp_path: Path) -> None:
    from test_import_external import BUNDLE, _case, _client, _queue, _restore, run_cli

    case, store = _case(tmp_path), tmp_path / "state/store"
    manifest = json.loads((case / "manifest.json").read_text())
    manifest["cases"][0]["material_context"]["specimen_preparation"] = {"mount": "back-loaded"}
    (case / "manifest.json").write_text(json.dumps(manifest))
    run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(BUNDLE), "--case", "DEV006", "--store", str(store))
    (digest,) = _queue(store)
    client, saved = _client(tmp_path, store)
    try:
        with client:
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
    finally:
        _restore(saved)
    preparation = review.json()["brief"]["specimen_preparation"]
    assert preparation["mount"] == "back-loaded" and not _payload_keys(preparation) & BANNED


def _payload_keys(value: object) -> set[str]:
    """Every key at any depth, minus each record's own `rater`: that names the record's author
    (for a judge, its own model), never the episode's run identity."""
    if isinstance(value, dict):
        return set(value) | {k for key, v in value.items() if key != "rater" for k in _payload_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _payload_keys(v)}
    return set()


RATER_IDS = [h.principal_id for h in HUMANS]


class Blindness(RuleBasedStateMachine):
    """One root with two imported agent answers, a reference snapshot, two late agent answers
    and one correction per rater that can be frozen mid-history, three human raters, the admin
    owner, a model judge and an agent."""

    @initialize(
        assigned=st.sets(st.sampled_from(RATER_IDS), min_size=1),
        first_cohort=st.sets(st.sampled_from((0, 1)), min_size=1),
        permitted=st.sets(st.sampled_from(HUMANS)),
    )
    def open_bench(self, assigned: set[str], first_cohort: set[int], permitted: set[Principal]) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="label-blindness-", dir=SCRATCH))
        self.bench = bench = Bench(self.tmp)
        self.saved = {name: getattr(app.state, name, None) for name in ("workbench_store", "attempt_authenticator")}
        app.state.workbench_store, app.state.attempt_authenticator = bench.store, bench.authenticator
        # Routed before the late answers are imported: the sweep would freeze them early.
        self.triage = Triage(bench.store, bench.labels, PROVISIONAL_2026_10_21)
        self.routed: set[str] = {d.snapshot for d in self.triage.sweep(OWNER).routed}
        assert self.routed == set(bench.snaps), "every agent answer of an imported run is routed"
        self.late = {
            "any_time": _import(bench.store, bench.root, "task-dev900__late", ("CIF000003",)),
            "after_a_reveal": _import(bench.store, bench.root, "task-dev900__later", ("CIF000002",)),
        }
        self.loose = TestClient(app, raise_server_exceptions=False)  # a crash after admission is no refusal
        # Lineage of every attempt: the human authors and the imports on its path from the root.
        self.humans: dict[str, frozenset[str]] = {}
        self.imports: dict[str, frozenset[str]] = {}
        for run in (*bench.runs, *self.late.values()):
            self.humans[run], self.imports[run] = frozenset(), frozenset({run})
        self.edits: dict[str, str] = {}
        for rater in HUMANS:
            bench.store.grant(OWNER, bench.root, rater.principal_id, (CaseOperation.ACTION,))
        for author in AUTHORS:
            request = ActionRequest(idempotency_key=f"{author.principal_id}-add", action={"verb": "add_phases", "candidates": ["CIF000002"]})
            edit = self.edits[author.principal_id] = bench.store.act(author, bench.root, request).digest
            self.humans[edit], self.imports[edit] = frozenset({author.principal_id}), frozenset()
        self.attempts: list[str] = [*bench.runs, *self.late.values(), *self.edits.values()]
        self.keys = 0
        # An agent_answer frozen from a human edit: the judge must refuse it by its origin (CR-08).
        mislabeled = bench.labels.put_snapshot(
            _snapshot(bench.root, self.edits["rater-a"], ("CIF000002",), kind="agent_answer")
        )
        self.pool: list[str] = [*bench.snaps, bench.reference, mislabeled]
        self.origin_of: dict[str, str] = {
            bench.snaps[0]: bench.runs[0],
            bench.snaps[1]: bench.runs[1],
            bench.reference: bench.runs[0],
            mislabeled: self.edits["rater-a"],
        }
        self.kind_of: dict[str, str] = dict.fromkeys(bench.snaps, "agent_answer")
        self.kind_of |= {bench.reference: "reference", mislabeled: "agent_answer"}
        self.import_runs: frozenset[str] = frozenset((*bench.runs, *self.late.values()))
        self.agent_snaps: set[str] = set(bench.snaps)
        self.assigned: tuple[str, ...] = ()
        self.cohort: frozenset[str] = frozenset()
        self.declared: dict[str, bool] = defaultdict(bool)
        self.participates: dict[str, bool] = defaultdict(bool)
        self.signed: dict[str, set[str]] = defaultdict(set)
        self.ever: dict[str, bool] = defaultdict(bool)  # the reveal condition has held at some point
        self.seen: dict[str, bool] = defaultdict(bool)  # the store has shown this rater gated content
        self.post_assignment(assigned, {bench.snaps[i] for i in first_cohort})
        for rater in permitted:
            self.permit(rater, True)

    def teardown(self) -> None:
        if hasattr(self, "tmp"):
            for name, value in self.saved.items():
                setattr(app.state, name, value)
            shutil.rmtree(self.tmp, ignore_errors=True)

    # ----- the oracle: the reveal rule as the labels.py contract states it ------------------

    def done(self, rater: str) -> bool:
        return bool(self.cohort) and self.cohort <= self.signed[rater]

    def condition(self, rater: str) -> bool:
        return rater in self.assigned and all(self.done(r) for r in self.assigned)

    def foreign(self, attempt: str, rater: str) -> bool:
        """Another human's attempt is on its lineage: another rater's or the owner's refit, or
        anything built on one (CR-07)."""
        return bool(self.humans[attempt] - {rater})

    def admissible(self, attempt: str, rater: str) -> bool:
        """Every attempt after the root is the rater's own or an origin of its cohort; a rater has
        a cohort only while the latest assignment names it (D12)."""
        cohort = self.cohort if rater in self.assigned else frozenset()
        return self.humans[attempt] <= {rater} and self.imports[attempt] <= {self.origin_of[s] for s in cohort}

    def hidden(self, snap: str, rater: str) -> bool:
        """A reference snapshot, one frozen from a foreign attempt, or an agent answer outside this
        rater's latest cohort (CR-07: before the reveal a rater reads its cohort and its own work
        only)."""
        return not self.visible(snap, rater)

    def visible(self, snap: str, rater: str) -> bool:
        """What a rater must be able to read before its reveal: a snapshot of its own lineage."""
        return snap != self.bench.reference and self.admissible(self.origin_of[snap], rater)

    def judge_reads(self, snap: str) -> bool:
        """CR-08: a model judge reads and labels only an agent_answer snapshot of an imported run."""
        return self.kind_of[snap] == "agent_answer" and self.origin_of[snap] in self.import_runs

    def describe(self, attempt: str) -> str:
        return f"attempt {attempt[:8]} (humans {sorted(self.humans[attempt])})"

    def saw_outside(self, rater: str, what: str, response: Any) -> None:
        self.ever[rater] = self.ever[rater] or self.condition(rater)
        assert self.ever[rater], (
            f"{rater} was admitted to {what} before its reveal could open: {response.status_code} {response.text[:300]}"
        )
        self.seen[rater] = True

    def call(self, principal: Principal, method: str, path: str, body: Any = None) -> Any:
        return self.loose.request(method, f"/api/attempts{path}", json=body, headers=_headers(principal))

    def open_to(self, rater: str) -> bool:
        """When the store must already answer as revealed: the condition holds now, or it has
        shown this rater gated content before (the reveal never closes)."""
        return self.condition(rater) or self.seen[rater]

    def saw_gated(self, rater: str, what: str) -> None:
        self.ever[rater] = self.ever[rater] or self.condition(rater)  # a read inside a rule sees this state
        assert self.ever[rater], f"{rater} received {what} before its reveal could open"
        self.seen[rater] = True

    def clean(self, viewer: Principal, body: Any) -> None:
        if viewer.principal_type is not PrincipalType.USER or not self.seen[viewer.principal_id]:
            assert not _payload_keys(body) & BANNED, f"{viewer.principal_id} received {_payload_keys(body) & BANNED}"
            routes = list(route_fields(body))
            assert not routes, f"{viewer.principal_id} received a route before its reveal: {routes[:3]}"

    def stamped(self, rater: str, record: dict[str, Any], open_before: bool) -> None:
        if open_before or self.declared[rater]:
            assert record["exposure"] == "revealed", f"{rater} wrote blind after its reveal opened or was declared"
        if record["exposure"] == "revealed" and not self.declared[rater]:
            self.saw_gated(rater, "a revealed exposure stamp")

    @invariant()
    def track_condition(self) -> None:
        for rater in self.assigned:
            self.ever[rater] = self.ever[rater] or self.condition(rater)

    # ----- writes --------------------------------------------------------------------------

    def post_assignment(self, raters: set[str], cohort: set[str]) -> None:
        body = {"raters": sorted(raters), "snapshots": sorted(cohort)}
        response = self.bench.post(OWNER, f"/roots/{self.bench.root}/assignment", body)
        assert response.status_code == 201, response.text
        self.assigned, self.cohort = tuple(sorted(raters)), frozenset(cohort)
        for rater in HUMANS:  # as build_queue.py does: each assigned rater's queue is its cohort
            path = queue_path(self.bench.store, rater.principal_id)
            path.parent.mkdir(mode=0o700, exist_ok=True)
            if rater.principal_id in raters:
                path.write_text(ReviewQueue(principal_id=rater.principal_id, snapshots=tuple(sorted(cohort))).model_dump_json())
            else:
                path.unlink(missing_ok=True)

    @rule(raters=st.sets(st.sampled_from(RATER_IDS), min_size=1), data=st.data())
    def assign(self, raters: set[str], data: st.DataObject) -> None:
        """Every assignment change is followed by each rater's reveal state, so a reassignment
        that closes an opened reveal is seen at once."""
        self.post_assignment(raters, data.draw(st.sets(st.sampled_from(sorted(self.agent_snaps)), min_size=1)))
        self.track_condition()
        self.read_reveal_states()

    @precondition(lambda self: any(self.seen.values()))
    @rule(raters=st.sets(st.sampled_from(RATER_IDS), min_size=1), data=st.data())
    def reassign_after_a_reveal(self, raters: set[str], data: st.DataObject) -> None:
        """A new cohort or rater set must not close a reveal that already opened."""
        self.assign(raters, data)

    @precondition(lambda self: any(self.seen.values()))
    @rule(data=st.data())
    def reassign_a_revealed_rater_to_unsigned_work(self, data: st.DataObject) -> None:
        """The case a reveal marker exists for: an assignment the revealed rater cannot have
        completed (a snapshot it has not signed, or a second rater who has not signed the cohort)
        makes the condition false again; the reveal must stay open."""
        rater = data.draw(st.sampled_from(sorted(r for r, seen in self.seen.items() if seen)))
        unsigned = sorted(self.agent_snaps - self.signed[rater])
        if unsigned:
            raters, cohort = {rater}, {data.draw(st.sampled_from(unsigned))}
        else:
            cohort = set(self.agent_snaps)
            behind = sorted(h.principal_id for h in HUMANS if not cohort <= self.signed[h.principal_id])
            if not behind:
                return
            raters = {rater, data.draw(st.sampled_from(behind))}
        self.post_assignment(raters, cohort)
        self.track_condition()
        self.read_reveal_states()

    @rule(rater=st.sampled_from(HUMANS))
    def declare(self, rater: Principal) -> None:
        """The admin owner declares a rater exposed: the stamp changes, the reveal does not."""
        self.bench.labels.declare_exposure(OWNER, rater.principal_id, self.bench.root, "seen outside the workbench")
        self.declared[rater.principal_id] = True
        self.read_labels(rater)  # a declaration changes the stamp only: the rater sees no new label

    @rule(rater=st.sampled_from(HUMANS), participation=st.booleans())
    def permit(self, rater: Principal, participation: bool) -> None:
        response = self.bench.post(rater, "/permissions", ALLOW | {"participation_and_storage": participation})
        assert response.status_code == 201, response.text
        self.participates[rater.principal_id] = participation

    def write(self, rater: Principal, snap: str, path: str, body: dict[str, Any]) -> dict[str, Any] | None:
        rid, open_before = rater.principal_id, self.open_to(rater.principal_id)
        response = self.bench.post(rater, f"/snapshots/{snap}/{path}", body)
        if self.participates[rid] and (self.visible(snap, rid) or open_before):
            assert response.status_code == 201, response.text
        if response.status_code != 201:
            assert response.status_code == 403, response.text
            return None
        assert self.participates[rid], f"{rid} stored a record without participation (R33)"
        if self.hidden(snap, rid):
            self.saw_gated(rid, "a gated snapshot")
        self.clean(rater, response.json())
        self.stamped(rid, response.json(), open_before)
        return response.json()

    @rule(rater=st.sampled_from(HUMANS), data=st.data())
    def label(self, rater: Principal, data: st.DataObject) -> None:
        snap = data.draw(st.sampled_from(self.pool))
        if self.write(rater, snap, "labels", {"item": "amounts_defensible", "answer": "yes"}) is not None:
            self.read_labels_of(JUDGE, snap)  # a human label now exists: the judge must not get it

    @rule(rater=st.sampled_from(HUMANS), data=st.data())
    def method_review(self, rater: Principal, data: st.DataObject) -> None:
        snap = data.draw(st.sampled_from(self.pool))
        response = self.bench.get(rater, f"/snapshots/{snap}/trajectory-review")
        if response.status_code == 200:
            self.clean(rater, response.json())
            assert all(r["rater"]["principal_id"] == rater.principal_id for r in response.json()["reviews"])
            if self.hidden(snap, rater.principal_id):
                self.saw_gated(rater.principal_id, "method review of a gated snapshot")
        else:
            assert response.status_code == 403, response.text
        history = response.json()["reviews"] if response.status_code == 200 else []
        self.keys += 1
        self.write(rater, snap, "trajectory-review", {
            "submission_id": f"method-{self.keys}",
            "supersedes": history[-1]["submission_id"] if history else None,
            "verdict": "defer", "note": "Synthetic test: no recorded conversation.",
        })
        # No method verdict updates the oracle's signed set or unlocks a reveal.
        self.read_reveal_state(rater)

    @rule(rater=st.sampled_from(HUMANS), data=st.data())
    def sign_off(self, rater: Principal, data: st.DataObject) -> None:
        snap = data.draw(st.sampled_from(self.pool))
        if self.write(rater, snap, "signoff", {"verdict": "would_sign"}) is not None:
            self.signed[rater.principal_id].add(snap)
            self.read_every_revealed(rater)

    @rule(rater=st.sampled_from(HUMANS))
    def sign_off_every_answer(self, rater: Principal) -> None:
        """What a rater does at the end of a review: sign off each snapshot of the cohort."""
        for snap in sorted(self.cohort):
            if self.write(rater, snap, "signoff", {"verdict": "would_sign"}) is not None:
                self.signed[rater.principal_id].add(snap)
            self.track_condition()
        for path in OWNER_ONLY:  # no reveal opens the owner's scorecard to a rater
            assert self.call(rater, "GET", path).status_code == 403, (rater.principal_id, path)
        self.read_every_revealed(rater)

    def read_every_revealed(self, rater: Principal) -> None:
        """The page loads the reveal right after a sign-off: a route must not come with it for a
        snapshot this rater has not signed off, even once its cohort is signed."""
        for snap in list(self.pool):
            self.read_revealed_of(rater, snap)

    @rule(rater=st.sampled_from(HUMANS))
    def step_feedback(self, rater: Principal) -> None:
        """A step of runs[0], the origin of the first agent answer: gated as that answer is."""
        rid, open_before = rater.principal_id, self.open_to(rater.principal_id)
        step = {"attempt": self.bench.runs[0], "call_index": 0, "step_kind": "refine"}
        response = self.bench.post(rater, "/step-feedback", {"step": step, "mark": "good_move"})
        admissible = self.admissible(self.bench.runs[0], rid)
        if self.participates[rid] and (admissible or open_before):
            assert response.status_code == 201, response.text
        if response.status_code != 201:
            assert response.status_code == 403, response.text
            return
        assert self.participates[rid], f"{rid} stored step feedback without participation (R33)"
        if not admissible:
            self.saw_gated(rid, "a step of an attempt outside its lineage")
        self.clean(rater, response.json())
        self.stamped(rid, response.json(), open_before)

    @rule(data=st.data())
    def judge_label(self, data: st.DataObject) -> None:
        snap = data.draw(st.sampled_from(self.pool))
        label = Label(
            snapshot=snap,
            item="fit_supports_conclusion",
            phase_id=None,
            answer="no",
            rater=Rater(principal_id="judge-opus", kind="model_judge", judge=JUDGE_ID),
            exposure="revealed",
            created_at_utc=datetime(2026, 10, 22, tzinfo=UTC),
        )
        if not self.judge_reads(snap):
            with pytest.raises(AuthorizationError):
                self.bench.labels.judge_label(JUDGE, label)
            return
        assert self.bench.labels.judge_label(JUDGE, label).exposure == "blind"

    def freeze_late(self, which: str, phase: str) -> None:
        origin = self.late.pop(which)
        digest = self.bench.labels.put_snapshot(_snapshot(self.bench.root, origin, (phase,)))
        self.pool.append(digest)
        self.agent_snaps.add(digest)
        self.origin_of[digest], self.kind_of[digest] = origin, "agent_answer"
        snap = self.bench.labels._snapshot(digest)
        self.triage._publish(route(digest, self.bench.root, readings_v1(snap), PROVISIONAL_2026_10_21))
        self.routed.add(digest)

    @precondition(lambda self: "any_time" in self.late)
    @rule()
    def freeze_late_answer(self) -> None:
        """A new agent answer mid-history (an export or refit) never joins the current cohort."""
        self.freeze_late("any_time", "CIF000003")

    @precondition(lambda self: "after_a_reveal" in self.late and any(self.seen.values()))
    @rule()
    def freeze_answer_after_a_reveal(self) -> None:
        """A new agent answer after a reveal, which a later assignment may name."""
        self.freeze_late("after_a_reveal", "CIF000002")

    @rule(author=st.sampled_from(AUTHORS))
    def freeze_correction(self, author: Principal) -> None:
        edit = self.edits[author.principal_id]
        digest = self.bench.labels.put_snapshot(_snapshot(self.bench.root, edit, ("CIF000002",), kind="expert_correction"))
        if digest not in self.origin_of:
            self.pool.append(digest)
            self.origin_of[digest], self.kind_of[digest] = edit, "expert_correction"

    # ----- reads ---------------------------------------------------------------------------

    @rule(viewer=st.sampled_from((*HUMANS, JUDGE)), data=st.data())
    def read_snapshot(self, viewer: Principal, data: st.DataObject) -> None:
        snap = data.draw(st.sampled_from(self.pool))
        response = self.bench.get(viewer, f"/snapshots/{snap}")
        if viewer is JUDGE:
            assert response.status_code == (200 if self.judge_reads(snap) else 403), response.text
            if response.status_code == 200:
                assert response.json()["kind"] == "agent_answer"
                self.clean(viewer, response.json())
            return
        vid = viewer.principal_id
        if self.visible(snap, vid) or self.open_to(vid):
            assert response.status_code == 200, response.text
        if response.status_code == 200:
            if self.hidden(snap, vid):
                self.saw_gated(vid, f"gated snapshot {response.json()['kind']}")
            self.clean(viewer, response.json())

    @rule(viewer=st.sampled_from((*HUMANS, JUDGE)))
    def read_labels(self, viewer: Principal) -> None:
        """The viewer reads the labels of every snapshot, as the page and the judge runner do."""
        for snap in list(self.pool):
            self.read_labels_of(viewer, snap)

    def read_labels_of(self, viewer: Principal, snap: str) -> None:
        response = self.bench.get(viewer, f"/snapshots/{snap}/labels")
        vid = viewer.principal_id
        if viewer is JUDGE:
            assert response.status_code == (200 if self.judge_reads(snap) else 403), response.text
        elif self.visible(snap, vid) or self.open_to(vid):
            assert response.status_code == 200, response.text
        if response.status_code != 200:
            assert response.status_code == 403, response.text
            return
        others = {label["rater"]["principal_id"] for label in response.json()} - {vid}
        if viewer is JUDGE:
            assert not others, f"the model judge received labels by {others}"
            self.clean(viewer, response.json())
            return
        if others:
            self.saw_gated(vid, f"labels by {sorted(others)}")
        if self.hidden(snap, vid):
            self.saw_gated(vid, "the labels of a gated snapshot")
        self.clean(viewer, response.json())

    @rule(viewer=st.sampled_from(HUMANS))
    def read_step_feedback(self, viewer: Principal) -> None:
        vid = viewer.principal_id
        response = self.bench.get(viewer, f"/step-feedback?attempt={self.bench.runs[0]}")
        if self.admissible(self.bench.runs[0], vid) or self.open_to(vid):
            assert response.status_code == 200, response.text
        if response.status_code != 200:
            assert response.status_code == 403, response.text
            return
        if not self.admissible(self.bench.runs[0], vid):
            self.saw_gated(vid, "the steps of an attempt outside its lineage")
        others = {f["rater"]["principal_id"] for f in response.json()} - {viewer.principal_id}
        if others:
            self.saw_gated(viewer.principal_id, f"step feedback by {sorted(others)}")
        self.clean(viewer, response.json())

    @rule()
    def read_reveal_states(self) -> None:
        for viewer in HUMANS:
            self.read_reveal_state(viewer)

    def read_reveal_state(self, viewer: Principal) -> None:
        vid = viewer.principal_id
        open_before = self.open_to(vid)
        state = self.bench.get(viewer, f"/roots/{self.bench.root}/reveal-state").json()
        assert state["pending_raters"] == [r for r in self.assigned if not self.done(r)], state
        assert state["signed_off"] == self.done(vid), state
        if open_before:
            assert state["revealed"] is True, f"{vid}'s reveal closed or never opened: {state}"
        if state["revealed"]:
            self.saw_gated(vid, "revealed=true")

    # ----- attempts: the gesture result, attempt review, freeze, annotate (CR-07) -------------

    def admits(self, viewer: Principal, attempt: str, response: Any, what: str) -> bool:
        """Check one attempt-route response; True when the route admitted the call (not 403)."""
        vid = viewer.principal_id
        admitted = response.status_code != 403
        if self.admissible(attempt, vid) or self.open_to(vid):
            assert admitted, f"{vid} was refused {what} of its own lineage: {response.text[:300]}"
        if admitted and not self.admissible(attempt, vid):
            self.saw_outside(vid, f"{what} of {self.describe(attempt)}", response)
        return admitted

    @rule(author=st.sampled_from(REFITTERS), data=st.data())
    def refit(self, author: Principal, data: st.DataObject) -> None:
        """A refine gesture on any attempt of the root, as the page and MCP post it (D10)."""
        parent = data.draw(st.sampled_from(self.attempts))
        self.keys += 1
        body = {"steps": [{"idempotency_key": f"refit-{self.keys}", "action": {"verb": "refine"}}]}
        response = self.call(author, "POST", f"/{parent}/actions", body)
        if author is OWNER:
            assert response.status_code == 201, response.text
        else:
            self.admits(author, parent, response, "a gesture")
        if response.status_code == 201:
            child = response.json()["attempts"][-1]["digest"]
            self.humans[child] = self.humans[parent] | {author.principal_id}
            self.imports[child] = self.imports[parent]
            self.attempts.append(child)
            self.clean(author, response.json())

    @rule(viewer=st.sampled_from(HUMANS), data=st.data())
    def review_attempt(self, viewer: Principal, data: st.DataObject) -> None:
        """GET /{digest}/review: the page's blind view of a refit beside its parent (D10)."""
        target = data.draw(st.sampled_from(self.attempts))
        response = self.call(viewer, "GET", f"/{target}/review")
        if self.admits(viewer, target, response, "the review") and response.status_code == 200:
            self.clean(viewer, response.json())

    @rule(viewer=st.sampled_from(HUMANS), data=st.data())
    def freeze_attempt(self, viewer: Principal, data: st.DataObject) -> None:
        """POST /snapshots: "label this correction" on any attempt."""
        target = data.draw(st.sampled_from(self.attempts))
        response = self.call(viewer, "POST", "/snapshots", {"origin": target, "kind": "expert_correction"})
        if self.admits(viewer, target, response, "freezing") and response.status_code == 201:
            digest = response.json()["digest"]
            if digest not in self.origin_of:
                self.pool.append(digest)
                self.origin_of[digest], self.kind_of[digest] = target, "expert_correction"

    @rule(viewer=st.sampled_from(HUMANS), data=st.data())
    def annotate_attempt(self, viewer: Principal, data: st.DataObject) -> None:
        """Step feedback on any attempt, the page's mark on a refinement-strip entry (N8)."""
        target, vid = data.draw(st.sampled_from(self.attempts)), viewer.principal_id
        open_before = self.open_to(vid)
        body = {"step": {"attempt": target, "step_kind": "attempt"}, "mark": "good_move"}
        response = self.call(viewer, "POST", "/step-feedback", body)
        assert response.status_code in (201, 403), response.text
        if self.participates[vid] and (self.admissible(target, vid) or open_before):
            assert response.status_code == 201, response.text
        if response.status_code == 201:
            if not self.admissible(target, vid):
                self.saw_outside(vid, f"step feedback on {self.describe(target)}", response)
            self.clean(viewer, response.json())
            self.stamped(vid, response.json(), open_before)

    @rule(viewer=st.sampled_from(HUMANS))
    def read_queue(self, viewer: Principal) -> None:
        response = self.call(viewer, "GET", "/queue")
        vid = viewer.principal_id
        if vid in self.assigned or self.open_to(vid):
            assert response.status_code == 200, response.text
        if response.status_code == 200:
            for item in response.json()["items"]:
                if self.hidden(item["digest"], vid):
                    self.saw_gated(vid, f"a queued gated snapshot {item['digest'][:8]}")
            self.clean(viewer, response.json())

    @rule(viewer=st.sampled_from(HUMANS), data=st.data())
    def review_snapshot(self, viewer: Principal, data: st.DataObject) -> None:
        """GET /snapshots/{digest}/review, the page's blind payload of a snapshot."""
        snap, vid = data.draw(st.sampled_from(self.pool)), viewer.principal_id
        response = self.call(viewer, "GET", f"/snapshots/{snap}/review")
        admitted = response.status_code != 403
        if self.visible(snap, vid) or self.open_to(vid):
            assert admitted, response.text
        if admitted and self.hidden(snap, vid):
            self.saw_gated(vid, f"the review of gated snapshot {snap[:8]}: {response.status_code}")
        if response.status_code == 200:
            self.clean(viewer, response.json())

    @rule(viewer=st.sampled_from(HUMANS), data=st.data())
    def trace_call(self, viewer: Principal, data: st.DataObject) -> None:
        """GET /snapshots/{digest}/trace/{call}: one call of the blind trace in full, behind the
        review's gate (snapshot.trace_call)."""
        snap, vid = data.draw(st.sampled_from(self.pool)), viewer.principal_id
        response = self.call(viewer, "GET", f"/snapshots/{snap}/trace/0")
        admitted = response.status_code != 403
        if self.visible(snap, vid) or self.open_to(vid):
            assert admitted, response.text
        if admitted and self.hidden(snap, vid):
            self.saw_gated(vid, f"a trace call of gated snapshot {snap[:8]}: {response.status_code}")
        if response.status_code == 200:
            self.clean(viewer, response.json())

    @rule(viewer=st.sampled_from(HUMANS), data=st.data())
    def read_revealed(self, viewer: Principal, data: st.DataObject) -> None:
        """GET /snapshots/{digest}/revealed: run identity after the rater's cohort sign-off, and the
        snapshot's route only once the rater signed off that snapshot too (Triage.revealed)."""
        self.read_revealed_of(viewer, data.draw(st.sampled_from(self.pool)))

    def read_revealed_of(self, viewer: Principal, snap: str) -> None:
        vid = viewer.principal_id
        response = self.call(viewer, "GET", f"/snapshots/{snap}/revealed")
        if response.status_code != 200:
            return
        route, signed = response.json().get("route"), snap in self.signed[vid] and self.done(vid)
        if route is not None:
            assert signed, f"{vid} received the route of {snap[:8]} without its own sign-off on it and its cohort"
            assert route["snapshot"] == snap, route
        else:
            assert not (signed and snap in self.routed), f"{vid} signed off routed {snap[:8]} and its cohort but got no route"

    @rule(stranger=st.sampled_from(STRANGERS), data=st.data())
    def stranger_writes(self, stranger: Principal, data: st.DataObject) -> None:
        snap = data.draw(st.sampled_from(self.pool))
        for path, body in (("labels", {"item": "amounts_defensible", "answer": "yes"}), ("signoff", {"verdict": "would_sign"})):
            response = self.bench.post(stranger, f"/snapshots/{snap}/{path}", body)
            assert response.status_code == 403, (stranger.principal_id, path, response.text)
        assert self.bench.get(stranger, f"/snapshots/{snap}/trace/0").status_code == 403
        assert self.bench.get(stranger, f"/snapshots/{snap}/trajectory-review").status_code == 403
        assert self.bench.post(stranger, f"/snapshots/{snap}/trajectory-review", {
            "submission_id": "stranger", "verdict": "defer", "note": "synthetic",
        }).status_code == 403
        if stranger is not JUDGE:
            assert self.bench.get(stranger, f"/snapshots/{snap}").status_code == 403
            assert self.bench.get(stranger, f"/snapshots/{snap}/labels").status_code == 403
        target = data.draw(st.sampled_from(self.attempts))
        assert self.call(stranger, "GET", f"/{target}/review").status_code == 403
        for path in OWNER_ONLY:
            assert self.call(stranger, "GET", path).status_code == 403, (stranger.principal_id, path)


Blindness.TestCase.settings = settings(
    max_examples=EXAMPLES,
    stateful_step_count=30,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    phases=[p for p in Phase if SHRINK or p is not Phase.shrink],
)
test_blindness = Blindness.TestCase
