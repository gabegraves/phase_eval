from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from phaseeval.workbench.admission import DOMAIN_EXPERT_ENV, CaseAdmission, SingleExpertCaseAdmission, admission_state

PACKET_TEXT = b"# DEV006 admission packet\n"
PACKET = hashlib.sha256(PACKET_TEXT).hexdigest()


def review(reviewer: str, verdict: str = "admit", packet: str = PACKET) -> dict:
    return {"reviewer_id": reviewer, "verdict": verdict, "packet_sha256": packet, "reviewed_at_utc": "2026-11-10T15:00:00Z"}


def write(directory: Path, *reviews: dict, case_id: str = "DEV006") -> Path:
    directory.mkdir(exist_ok=True)
    record = {"schema_version": "case-admission.v1", "case_id": case_id, "packet_sha256": PACKET, "reviews": list(reviews)}
    (directory / f"{case_id}.json").write_text(json.dumps(record))
    (directory / f"{case_id}.packet.md").write_bytes(PACKET_TEXT)
    return directory


def test_no_admission_directory_admits_nothing(tmp_path: Path) -> None:
    assert admission_state("DEV006", None) == (False, "case DEV006 not admitted: no admission records are configured")
    assert admission_state("DEV006", tmp_path)[0] is False


def test_two_distinct_experts_admitting_the_same_packet_admit_the_case(tmp_path: Path) -> None:
    directory = write(tmp_path / "admissions", review("expert-a"), review("expert-b"))
    assert admission_state("DEV006", directory) == (True, "case DEV006 admitted by 2 expert reviews of its packet")


@pytest.mark.parametrize(
    ("reviews", "why"),
    [
        ((review("expert-a"),), "1 of 2 required expert admissions"),
        ((review("expert-a"), review("expert-a")), "1 of 2 required expert admissions"),
        ((review("expert-a"), review("expert-b"), review("expert-c", "exclude")), "an expert excluded it"),
    ],
)
def test_one_expert_twice_or_any_exclusion_does_not_admit(tmp_path: Path, reviews: tuple, why: str) -> None:
    admitted, reason = admission_state("DEV006", write(tmp_path / "admissions", *reviews))
    assert not admitted and why in reason


def test_a_review_of_another_packet_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="names another packet"):
        CaseAdmission.model_validate(
            {"case_id": "DEV006", "packet_sha256": PACKET, "reviews": [review("expert-a"), review("expert-b", packet="b" * 64)]}
        )


def test_a_record_filed_under_another_case_is_refused(tmp_path: Path) -> None:
    directory = write(tmp_path / "admissions", review("expert-a"), review("expert-b"), case_id="DEV102")
    (directory / "DEV102.json").rename(directory / "DEV006.json")
    admitted, reason = admission_state("DEV006", directory)
    assert not admitted and "records case DEV102" in reason


def test_an_admission_does_not_survive_a_change_to_its_packet(tmp_path: Path) -> None:
    directory = write(tmp_path / "admissions", review("expert-a"), review("expert-b"))
    (directory / "DEV006.packet.md").write_bytes(PACKET_TEXT + b"revised reference solution\n")
    admitted, reason = admission_state("DEV006", directory)
    assert not admitted and "packet is not the one its experts reviewed" in reason
    (directory / "DEV006.packet.md").unlink()
    assert admission_state("DEV006", directory) == (False, "case DEV006 not admitted: its packet is missing")


@pytest.fixture
def single_expert_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(DOMAIN_EXPERT_ENV, "gabe")
    write(tmp_path, review("gabe"))
    path = tmp_path / "DEV006.json"
    record = json.loads(path.read_text())
    record.update(schema_version="case-admission.v2", policy_version="single-domain-expert-v1", domain_expert_id="gabe")
    path.write_text(json.dumps(record))
    return path


def test_single_configured_expert_admits_v2(single_expert_record: Path) -> None:
    assert admission_state("DEV006", single_expert_record.parent) == (
        True, "case DEV006 admitted by gabe under single-domain-expert-v1"
    )


@pytest.mark.parametrize("timestamp", ["2026-10-28T00:00:00", 1793145600])
def test_v2_refuses_timestamps_without_an_explicit_offset(single_expert_record: Path, timestamp: object) -> None:
    record = json.loads(single_expert_record.read_text())
    record["reviews"][0]["reviewed_at_utc"] = timestamp
    single_expert_record.write_text(json.dumps(record))
    assert admission_state("DEV006", single_expert_record.parent)[0] is False


def test_v2_valid_record_satisfies_schema_formats(single_expert_record: Path) -> None:
    from jsonschema import Draft202012Validator, FormatChecker

    validator = Draft202012Validator(SingleExpertCaseAdmission.model_json_schema(), format_checker=FormatChecker())
    validator.validate(json.loads(single_expert_record.read_text()))


@pytest.mark.parametrize("expert", [None, "someone-else"])
def test_v2_requires_the_configured_expert(
    single_expert_record: Path, monkeypatch: pytest.MonkeyPatch, expert: str | None
) -> None:
    if expert is None:
        monkeypatch.delenv(DOMAIN_EXPERT_ENV)
    else:
        monkeypatch.setenv(DOMAIN_EXPERT_ENV, expert)
    assert admission_state("DEV006", single_expert_record.parent)[0] is False


@pytest.mark.parametrize(
    "reviews",
    [[], [review("someone-else")], [review("gabe"), review("gabe")], [review("gabe"), review("other")],
     [review("gabe", packet="b" * 64)], [review("gabe", "exclude")]],
)
def test_v2_refuses_missing_extra_mismatched_or_excluding_reviews(single_expert_record: Path, reviews: list) -> None:
    record = json.loads(single_expert_record.read_text())
    record["reviews"] = reviews
    single_expert_record.write_text(json.dumps(record))
    assert admission_state("DEV006", single_expert_record.parent)[0] is False


@pytest.mark.parametrize("field,value", [("schema_version", "case-admission.v3"), ("policy_version", "unknown")])
def test_v2_refuses_unknown_versions(single_expert_record: Path, field: str, value: str) -> None:
    record = json.loads(single_expert_record.read_text())
    record[field] = value
    single_expert_record.write_text(json.dumps(record))
    assert admission_state("DEV006", single_expert_record.parent)[0] is False


@pytest.mark.parametrize("field", ["schema_version", "policy_version", "domain_expert_id"])
def test_v2_requires_explicit_policy_and_identity(single_expert_record: Path, field: str) -> None:
    record = json.loads(single_expert_record.read_text())
    del record[field]
    single_expert_record.write_text(json.dumps(record))
    assert admission_state("DEV006", single_expert_record.parent)[0] is False


def test_v2_preserves_packet_and_case_checks(single_expert_record: Path) -> None:
    directory = single_expert_record.parent
    packet = directory / "DEV006.packet.md"
    packet.write_bytes(PACKET_TEXT + b"changed\n")
    assert "packet is not the one" in admission_state("DEV006", directory)[1]
    packet.unlink()
    assert "packet is missing" in admission_state("DEV006", directory)[1]
    record = json.loads(single_expert_record.read_text())
    record["case_id"] = "DEV102"
    single_expert_record.write_text(json.dumps(record))
    assert "records case DEV102" in admission_state("DEV006", directory)[1]


def test_single_expert_configuration_does_not_reinterpret_v1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DOMAIN_EXPERT_ENV, "gabe")
    write(tmp_path, review("gabe"))
    assert "1 of 2 required expert admissions" in admission_state("DEV006", tmp_path)[1]
    write(tmp_path, review("expert-a"), review("expert-b"))
    assert admission_state("DEV006", tmp_path)[0] is True
    path = tmp_path / "DEV006.json"
    record = json.loads(path.read_text())
    del record["schema_version"]
    path.write_text(json.dumps(record))
    assert admission_state("DEV006", tmp_path)[0] is True


def test_published_v2_schema_matches_record() -> None:
    path = Path(__file__).resolve().parents[2] / "schemas" / "case-admission.v2.json"
    assert json.loads(path.read_text()) == SingleExpertCaseAdmission.model_json_schema()
