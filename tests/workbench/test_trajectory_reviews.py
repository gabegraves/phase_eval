from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_labels import AGENT, ALLOW, JUDGE, OWNER, RATER_A, RATER_B
from test_trace_route import _snap, bench

from phaseeval.workbench.labels import TrajectoryReviewRequest
from phaseeval.workbench.store import CustodyError


def recorded_trace():
    def action(ids, message, outputs):
        return {"source": "agent", "message": message, "timestamp": "2026-10-21T10:00:05Z",
                "tool_calls": [{"tool_call_id": i, "function_name": "screen", "arguments": {}} for i in ids],
                "observation": {"results": [{"source_call_id": i, "content": out} for i, out in zip(ids, outputs)]}}
    return json.dumps({"schema_version": "ATIF-v1.2", "session_id": "synthetic-method", "steps": [
        {"source": "user", "timestamp": "2026-10-21T10:00:00Z", "message": "Analyze the supplied scan."},
        action(["a"], "Read the pattern", ["early evidence " * 1000]),
        action(["b", "c"], "Compare two hypotheses", ["sibling one", "sibling two"]),
        {"source": "user", "message": "Late instruction, not previously known."},
        {"source": "agent", "message": "Final unsupported interpretation without a tool call."},
        {"source": "agent", "message": "I retract the attribution; the residual remains unexplained."},
    ]}).encode()


def setup(bench):
    snap = _snap(bench, [recorded_trace()], "method-test")
    assert bench.post(RATER_A, "/permissions", ALLOW).status_code == 201
    return snap, f"/snapshots/{snap}/trajectory-review"


def judgment(**updates):
    return {"submission_id": "review-1", "verdict": "fail", "note": "Earlier observations do not support the final claim.",
            "evidence": [{"turn_index": 1, "call_index": 0}],
            "earliest_error": {"turn_index": 4}, "repair_status": "repaired",
            "observed_repair": {"turn_index": 5}, **updates}


def test_historical_trace_retains_full_shown_text_late_context_and_final_prose(bench):
    _, url = setup(bench)
    response = bench.get(RATER_A, url)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reviews"] == []
    assert len(body["trace"][1]["calls"][0]["output"]) > 6000
    assert not body["trace"][1]["calls"][0]["clipped"]
    assert "Late instruction" in body["trace"][3]["message"]
    assert body["trace"][4]["calls"] == []
    assert "Final unsupported" in body["trace"][4]["message"]
    assert not {"fit", "checks", "answer", "reference", "executor"} & body.keys()


def test_method_review_is_durable_idempotent_and_does_not_sign_off(bench):
    snap, url = setup(bench)
    body = judgment()
    first = bench.post(RATER_A, url, body)
    assert first.status_code == 201, first.text
    assert first.json()["rater"]["principal_id"] == RATER_A.principal_id
    assert first.json()["exposure"] == "blind"
    assert bench.post(RATER_A, url, body).json() == first.json()
    assert len(bench.get(RATER_A, url).json()["reviews"]) == 1
    assert not bench.labels.has_signed(RATER_A, snap)
    assert not bench.labels.reveal_state(RATER_A, bench.root).revealed
    assert bench.post(RATER_A, url, body | {"note": "different"}).status_code == 422
    correction = body | {"submission_id": "review-2", "supersedes": "review-1", "note": "Correction kept in history."}
    assert bench.post(RATER_A, url, correction).status_code == 201
    assert bench.post(RATER_A, url, body | {"submission_id": "stale"}).status_code == 422
    assert [r["submission_id"] for r in bench.get(RATER_A, url).json()["reviews"]] == ["review-1", "review-2"]


@pytest.mark.parametrize("patch", [
    {"evidence": [{"turn_index": 4}]},  # the later claim is not evidence available beforehand
    {"earliest_error": {"turn_index": 2, "call_index": 2}, "evidence": [{"turn_index": 2, "call_index": 1}]},
    {"evidence": [{"turn_index": 1, "call_index": 2}]},
    {"observed_repair": {"turn_index": 3}},
    {"observed_repair": {"turn_index": 4}},
    {"earliest_error": {"turn_index": 3}},
    {"note": "   "}, {"protocol_version": "invented"}, {"rater": {}},
    {"verdict": "pass", "repair_status": "unrepaired", "observed_repair": None},
])
def test_invalid_or_hindsight_judgments_are_rejected(bench, patch):
    _, url = setup(bench)
    assert bench.post(RATER_A, url, judgment(**patch)).status_code == 422
    assert bench.get(RATER_A, url).json()["reviews"] == []


def test_own_only_cohort_roles_and_participation_boundary(bench):
    snap, url = setup(bench)
    assert bench.post(RATER_A, url, judgment()).status_code == 201
    for principal in (RATER_B, AGENT, JUDGE, OWNER):
        assert bench.get(principal, url).status_code == 403
        assert bench.post(principal, url, judgment()).status_code == 403
    bench.assign([RATER_A.principal_id, RATER_B.principal_id], snapshots=[snap])
    assert bench.get(RATER_B, url).json()["reviews"] == []
    assert len(bench.labels.export_trajectory_reviews("internal_evaluation")) == 1
    assert not bench.labels.export_trajectory_reviews("training")
    bench.post(RATER_A, "/permissions", ALLOW | {"participation_and_storage": False})
    assert not bench.labels.export_trajectory_reviews("internal_evaluation")
    assert bench.post(RATER_A, url, judgment(submission_id="later", supersedes="review-1")).status_code == 403


def test_concurrent_same_submission_publishes_one_revision(bench):
    snap, _ = setup(bench)
    request = TrajectoryReviewRequest.model_validate(judgment() | {"snapshot": snap})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: bench.labels.review_trajectory(RATER_A, request), range(2)))
    assert results[0] == results[1]
    assert len(bench.labels.own_trajectory_reviews(RATER_A, snap)) == 1


def test_record_content_hash_is_checked_on_read(bench):
    snap, url = setup(bench)
    assert bench.post(RATER_A, url, judgment()).status_code == 201
    path = next((bench.store.root / "trajectory-review-records").glob("*.json"))
    path.write_text(path.read_text().replace("Earlier observations", "Altered observations"))
    with pytest.raises(CustodyError):
        bench.labels.own_trajectory_reviews(RATER_A, snap)
