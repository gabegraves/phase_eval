"""Canonical legacy bytes generated once from pinned 43189bc; no current producer imports."""
from pathlib import Path

import pytest

from phaseeval.kernel.records import canonical_bytes
from phaseeval.workbench import labels

FIXTURES = Path(__file__).with_name("fixtures")

READERS = {
    "step_feedback_v1": labels.StepFeedbackV1,
    "step_feedback_v1_note": labels.StepFeedbackV1,
    "reveal_opened": labels.RevealOpened,
    "permission_v1": labels.PermissionRecord,
    "assignment_v1": labels.AssignmentV1,
    "label_v1": labels.Label,
    "label_v1_judge": labels.Label,
    "peak_call_v1": labels.PeakCall,
    "signoff_v1": labels.SignOff,
    "exposure_declaration_v1": labels.ExposureDeclaration,
}


@pytest.mark.parametrize("name", sorted(READERS))
def test_reader_reproduces_committed_43189bc_bytes(name):
    raw = (FIXTURES / (name + ".json")).read_bytes()
    assert canonical_bytes(READERS[name].model_validate_json(raw)) == raw


def test_step_feedback_v1_has_step_and_v2_has_snapshot_and_attempt() -> None:
    assert "step" in labels.StepFeedbackV1.model_fields and "snapshot" not in labels.StepFeedbackV1.model_fields
    assert {"snapshot", "attempt", "call_index"} <= set(labels.StepFeedback.model_fields)
    assert labels.StepFeedback.model_fields["schema_version"].default == "phaseeval.step-feedback.v2"


def test_reveal_opened_keeps_opened_at_utc() -> None:
    assert "opened_at_utc" in labels.RevealOpened.model_fields
