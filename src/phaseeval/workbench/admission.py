"""Case admission under a versioned expert policy.

V1 requires two distinct reviewer IDs. V2 requires exactly one review by the domain expert
named in both the record and $PHASEEVAL_DOMAIN_EXPERT_ID. Both require the same packet sha256,
no exclusion, and the packet beside the record still hashing to it. The code cannot check that
an ID is a human domain expert, independent of the case's author, or who wrote the record.
Those rest on a trusted operator, who attests each review from the expert's signed form and keeps
the records directory outside any agent's or trainer's write access.

The packet holds what the agent can observe, an independent reference solution worked from those
observations only, its alternatives and what stays unidentifiable; its sha256 binds each review to
the bytes the expert read. Records are expert judgments, so they live outside the repository
(AGENTS.md rule 3), in the directory $PHASEEVAL_ADMISSIONS names: per case, the record
`<case_id>.json` and the packet the experts reviewed, `<case_id>.packet.md`. A record admits only
while that packet still hashes to the sha256 its reviews name. No directory admits nothing.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError, model_validator

from .model import Digest, Record, SafeID

ADMISSIONS_ENV = "PHASEEVAL_ADMISSIONS"
DOMAIN_EXPERT_ENV = "PHASEEVAL_DOMAIN_EXPERT_ID"
REQUIRED_EXPERTS = 2


class ExpertReview(Record):
    reviewer_id: SafeID  # names the domain expert; the operator attests it, the code cannot
    verdict: Literal["admit", "exclude"]
    packet_sha256: Digest
    reviewed_at_utc: datetime
    note: str = Field(default="", max_length=4000)


class CaseAdmission(Record):
    schema_version: Literal["case-admission.v1"] = "case-admission.v1"
    case_id: SafeID
    packet_sha256: Digest
    reviews: tuple[ExpertReview, ...]

    @model_validator(mode="after")
    def reviews_name_the_packet(self) -> Self:
        if any(r.packet_sha256 != self.packet_sha256 for r in self.reviews):
            raise ValueError("a review names another packet than the record's")
        return self


class SingleExpertReview(ExpertReview):
    reviewed_at_utc: AwareDatetime = Field(strict=True)


class SingleExpertCaseAdmission(CaseAdmission):
    schema_version: Literal["case-admission.v2"]
    policy_version: Literal["single-domain-expert-v1"]
    domain_expert_id: SafeID
    reviews: tuple[SingleExpertReview]

    @model_validator(mode="after")
    def review_names_the_domain_expert(self) -> Self:
        if self.reviews[0].reviewer_id != self.domain_expert_id:
            raise ValueError("the review does not name the record's domain expert")
        return self


_ADMISSION_RECORD = TypeAdapter(CaseAdmission | SingleExpertCaseAdmission)


def admissions_dir() -> Path | None:
    value = os.environ.get(ADMISSIONS_ENV)
    return Path(value) if value else None


def admission_state(case_id: str, directory: Path | None) -> tuple[bool, str]:
    """Whether `case_id` is admitted, and why in words."""
    if directory is None:
        return False, f"case {case_id} not admitted: no admission records are configured"
    path = directory / f"{case_id}.json"
    if not path.is_file():
        return False, f"case {case_id} not admitted: no admission record"
    try:
        record = _ADMISSION_RECORD.validate_json(path.read_bytes())
    except ValidationError as error:
        return False, f"case {case_id} not admitted: its record is invalid ({error.errors()[0]['msg']})"
    if record.case_id != case_id:
        return False, f"case {case_id} not admitted: its file records case {record.case_id}"
    packet = directory / f"{case_id}.packet.md"
    if not packet.is_file():
        return False, f"case {case_id} not admitted: its packet is missing"
    if hashlib.sha256(packet.read_bytes()).hexdigest() != record.packet_sha256:
        return False, f"case {case_id} not admitted: its packet is not the one its experts reviewed"
    if any(r.verdict == "exclude" for r in record.reviews):
        return False, f"case {case_id} not admitted: an expert excluded it"
    if isinstance(record, SingleExpertCaseAdmission):
        expert = os.environ.get(DOMAIN_EXPERT_ENV)
        if not expert:
            return False, f"case {case_id} not admitted: no domain expert is configured"
        if record.domain_expert_id != expert:
            return False, f"case {case_id} not admitted: its domain expert is not the configured expert"
        return True, f"case {case_id} admitted by {expert} under {record.policy_version}"
    admitting = {r.reviewer_id for r in record.reviews if r.verdict == "admit"}
    if len(admitting) < REQUIRED_EXPERTS:
        return False, f"case {case_id} not admitted: {len(admitting)} of {REQUIRED_EXPERTS} required expert admissions"
    return True, f"case {case_id} admitted by {len(admitting)} expert reviews of its packet"
