import json

import pytest
from pydantic import ValidationError

from phaseeval.workbench.assisted_review import Assessment, AssistedReviews, PROMPT
from phaseeval.workbench.model import canonical_bytes
from phaseeval.workbench.triage import ASSISTED_POLICY, assisted_disposition


SNAPSHOT = "a" * 64
ROOT = "b" * 64
TRACE = [
    {"source": "user", "message": "pattern supplied", "calls": []},
    {"source": "agent", "message": "identified phase", "calls": []},
    {"source": "agent", "message": "reconsidered phase", "calls": []},
]


def response(**changes):
    value = {
        "verdict": "fail", "note": "The first phase call was unsupported.",
        "evidence": [{"turn_index": 0}], "earliest_error": {"turn_index": 1},
        "observed_repair": None, "repair_status": "unrepaired",
        "proposed_repair": "Re-check the candidate.", "issue": "science_question",
        "question": "Which phase is supported?",
    }
    value.update(changes)
    return json.dumps(value).encode()


def test_complete_local_teacher_flow_and_immutable_retry(tmp_path):
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    local = reviews.submit(prepared["packet"], response(), model="local", run_id="one", context="fresh_packet")
    assert local["disposition"]["kind"] == "teacher_needed"
    teacher_packet = local["teacher"]["packet"]
    teacher_response = response(verdict="pass", earliest_error=None, repair_status="not_applicable", issue="none", question=None)
    teacher = reviews.submit(teacher_packet, teacher_response, model="teacher", run_id="two", context="inherited_context")
    assert teacher["disposition"]["kind"] == "expert_question"
    assert reviews.submit(teacher_packet, teacher_response, model="teacher", run_id="two", context="inherited_context") == teacher
    with pytest.raises(ValueError, match="immutable artifact conflicts"):
        reviews.submit(teacher_packet, teacher_response, model="other", run_id="three", context="unknown")
    assert reviews.status(prepared["packet"])["stage"] == "complete"
    report = reviews.report()
    assert report["total"] == 1
    question = report["expert_questions"]["none"][0]
    assert question["source_packet"] == teacher_packet
    assert question["assessment"]["evidence"] == [{"turn_index": 0, "call_index": None}]
    assert set(report["blind_audit_invitations"][0]) == {"packet"} if report["blind_audit_invitations"] else True
    assert reviews.submit(prepared["packet"], response(), model="local", run_id="one", context="fresh_packet")["disposition"] == teacher["disposition"]


def test_rejects_tamper_and_invented_reference(tmp_path):
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    packet_dir = tmp_path / "packets" / prepared["packet"]
    packet_dir.joinpath("prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="binding was changed"):
        reviews.submit(prepared["packet"], response(), model="local", run_id="one", context="fresh_packet")
    packet_dir.joinpath("prompt.txt").write_text(__import__("phaseeval.workbench.assisted_review", fromlist=["PROMPT"]).PROMPT)
    with pytest.raises(ValueError, match="no recorded turn"):
        reviews.submit(prepared["packet"], response(evidence=[{"turn_index": 99}]), model="local", run_id="one", context="fresh_packet")


def test_mechanical_is_terminal_but_preserves_audit_selection(tmp_path):
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    result = reviews.submit(prepared["packet"], response(issue="mechanical", question=None), model="local", run_id="one", context="unknown")
    assert result["disposition"]["kind"] == "ops_needed"
    assert result["disposition"]["audit_invited"] == reviews._load_packet(prepared["packet"]).audit_invited


def test_report_counts_prepared_packet_before_submission(tmp_path):
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    report = reviews.report()
    assert report["total"] == report["pending"] == 1
    assert report["items"][0]["packet"] == prepared["packet"]
    assert report["items"][0]["stage"] == "pending_local"


def test_pending_audit_invitation_is_selected_from_frozen_packet(tmp_path, monkeypatch):
    monkeypatch.setattr("phaseeval.workbench.assisted_review.audit_pick", lambda *_: True)
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    status = reviews.status(prepared["packet"])
    assert status["audit_invited"] is True
    assert reviews.report()["blind_audit_invitations"] == [{"packet": prepared["packet"]}]


@pytest.mark.parametrize("artifact", ["proposal", "response", "slot"])
def test_status_rejects_tampered_chain_artifacts(tmp_path, artifact):
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    result = reviews.submit(prepared["packet"], response(issue="mechanical", question=None), model="local", run_id="one", context="unknown")
    if artifact == "proposal":
        path = tmp_path / "proposals" / f"{result['proposal']}.json"
    elif artifact == "response":
        path = tmp_path / "responses" / f"{result['proposal']}.json"
    else:
        path = tmp_path / "chains" / prepared["packet"] / "local.json"
    path.write_bytes(path.read_bytes().replace(b"mechanical", b"mechanicaX") if artifact == "proposal" else b"{}")
    with pytest.raises((ValueError, ValidationError, KeyError)):
        reviews.status(prepared["packet"])


def test_rejects_changed_or_missing_frozen_policy(tmp_path):
    reviews = AssistedReviews(None, tmp_path)
    prepared = reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
    policy = tmp_path / "packets" / prepared["packet"] / "policy.json"
    policy.write_text("{}\n")
    with pytest.raises(ValueError, match="binding was changed"):
        reviews.status(prepared["packet"])


def test_rejects_blank_note_and_unsafe_public_ids(tmp_path):
    reviews = AssistedReviews(None, tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        reviews.status("../labels")
    with pytest.raises(PermissionError, match="owner"):
        reviews.prepare(SNAPSHOT)
    with pytest.raises(ValidationError, match="non-blank note"):
        reviews._packet(stage="local", snapshot=SNAPSHOT, root=ROOT, trace=TRACE)
        __import__("phaseeval.workbench.assisted_review", fromlist=["Assessment"]).Assessment.model_validate(json.loads(response(note="   ")))


def test_prompt_schema_explain_chronology_and_specific_unresolved_questions():
    schema = Assessment.model_json_schema()["properties"]
    assert "strictly before" in PROMPT and "strictly before" in schema["evidence"]["description"]
    assert "not merely endorse" in PROMPT
    assert "specific unresolved" in schema["question"]["description"]
    assert "issue=none" in PROMPT


def test_repair_disagreement_routes_to_expert_question():
    local = Assessment.model_validate(json.loads(response(issue="none", question=None)))
    teacher = Assessment.model_validate(json.loads(response(issue="none", question=None, repair_status="repaired", observed_repair={"turn_index": 2})))
    disposition = assisted_disposition(local=local, teacher=teacher, audit_invited=False, policy_sha256=ASSISTED_POLICY.sha256)
    assert disposition.kind == "expert_question"
    assert b"repair_disagreement" in canonical_bytes(ASSISTED_POLICY)
