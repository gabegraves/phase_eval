import json

import pytest

from phaseeval.workbench.assisted_review import AssistedReviews


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
    assert reviews.report()["total"] == 1


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
