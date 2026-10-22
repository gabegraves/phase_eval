"""Shared governance records for pipeline gates and terminal routes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

TERMINAL_ROUTE_VERDICTS = {"ACCEPT", "SET", "UNKNOWN", "REQUEST_MORE_DATA"}
_LEGACY_ROUTE_VERDICTS = {
    "set": "SET",
    "unknown": "UNKNOWN",
    "request_more_data": "REQUEST_MORE_DATA",
}

# Closed disambiguation vocabulary for SET / REQUEST_MORE_DATA routes.
# Each value names a concrete follow-up measurement that would resolve a
# same-element-set degeneracy a single 1D PXRD scan cannot. This is a closed
# enum on purpose: the loop may only emit one of these named next actions, it
# may not invent free-text measurement requests.
REQUESTED_DATA_TYPES = frozenset(
    {
        "variable_temperature",
        "complementary_technique",
        "higher_resolution_scan",
        "larger_2theta_range",
    }
)


def validate_requested_data_type(value: Any) -> str:
    """Return ``value`` if it is a known requested_data_type, else raise.

    Used on the SET / REQUEST_MORE_DATA route to keep the disambiguation
    vocabulary closed. ``None``/empty is the caller's concern (the field is
    optional); this validates only non-empty values.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"requested_data_type must be a non-empty string, got {value!r}")
    if value not in REQUESTED_DATA_TYPES:
        raise ValueError(
            f"unknown requested_data_type {value!r}; expected one of {sorted(REQUESTED_DATA_TYPES)}"
        )
    return value


@dataclass
class GateRecord:
    """Structured record of a single governance gate decision."""

    sample_id: str
    gate_name: str
    status: str
    decision: str
    reason_codes: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] | None = None
    input_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class TerminalRoute:
    """Contract-aligned terminal route emitted by governance gates."""

    sample_id: str
    verdict: str
    gate_name: str
    status: str
    reason_codes: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    reported_hypothesis: dict[str, Any] | None = None
    ranked_hypotheses: list[dict[str, Any]] | None = None
    set_payload: dict[str, Any] | None = None
    phase_fractions: dict[str, float] | None = None
    metrology_status: str | None = None
    chemistry_prior_summary: str | None = None
    refinement_summary: str | None = None
    request_more_data_fields: list[str] | None = None
    requested_data_type: str | None = None
    disambiguation_reason: str | None = None
    recommendation_details: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.requested_data_type is not None:
            validate_requested_data_type(self.requested_data_type)
        self.recommendation_details = _validated_recommendation_details(
            self.recommendation_details,
            recommended_actions=self.recommended_actions,
            route_requested_data_type=self.requested_data_type,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def normalize_terminal_route(
    value: Any,
    *,
    sample_id: str | None = None,
    source: str | None = None,
    evidence_items: list[dict[str, Any]] | None = None,
) -> TerminalRoute:
    """Normalize legacy route-like values into the canonical terminal route contract."""
    payload = _route_payload(value)
    route_sample_id = str(payload.get("sample_id") or sample_id or "")
    reason_codes = _str_list(payload.get("reason_codes"))
    recommended_actions = _str_list(
        payload.get("recommended_actions", payload.get("recommended_action"))
    )
    details_allowed = bool(recommended_actions)
    request_fields = _str_list(payload.get("request_more_data_fields"))
    raw_verdict = str(payload.get("verdict", payload.get("decision", "")) or "")
    verdict = _normalize_verdict(
        raw_verdict,
        payload=payload,
        reason_codes=reason_codes,
    )

    if verdict == "REQUEST_MORE_DATA" and not recommended_actions:
        recommended_actions = ["Provide missing PXRD measurement metadata"]
    elif verdict == "UNKNOWN" and not recommended_actions:
        recommended_actions = ["Review gate history and PXRD evidence artifacts"]
    elif verdict == "ACCEPT" and not recommended_actions:
        recommended_actions = ["Verify result against known chemistry and custody gates"]
    elif verdict == "SET" and not recommended_actions:
        recommended_actions = ["Review ranked hypotheses before selecting a final phase set"]

    requested_data_type = payload.get("requested_data_type")
    disambiguation_reason = payload.get("disambiguation_reason")
    ranked_hypotheses = payload.get("ranked_hypotheses")
    if verdict == "REQUEST_MORE_DATA":
        prerequisite_form = bool(request_fields) and not any(
            (requested_data_type, disambiguation_reason, ranked_hypotheses)
        )
        post_analysis_form = not request_fields and all(
            (requested_data_type, disambiguation_reason, ranked_hypotheses)
        )
        if post_analysis_form:
            try:
                validate_requested_data_type(requested_data_type)
            except ValueError:
                post_analysis_form = False
        if prerequisite_form == post_analysis_form:
            verdict = "UNKNOWN"
            reason_codes.append("malformed_request_more_data_contract")
            recommended_actions = ["Review the malformed request-more-data route"]
            request_fields = []
            requested_data_type = None
            disambiguation_reason = None
            ranked_hypotheses = None
            details_allowed = False

    evidence_summary = str(payload.get("evidence_summary", "") or "")
    if not evidence_summary:
        evidence_summary = (
            f"Terminal route normalized from {source}."
            if source
            else "Terminal route normalized without an evidence summary."
        )

    raw_recommendation_details = (
        payload.get("recommendation_details") if details_allowed else None
    )
    route = TerminalRoute(
        sample_id=route_sample_id,
        verdict=verdict,
        gate_name=str(payload.get("gate_name", "") or ""),
        status=str(payload.get("status", "") or ""),
        reason_codes=reason_codes,
        recommended_actions=recommended_actions,
        evidence_summary=evidence_summary,
        reported_hypothesis=payload.get("reported_hypothesis"),
        ranked_hypotheses=ranked_hypotheses,
        set_payload=payload.get("set_payload"),
        phase_fractions=payload.get("phase_fractions"),
        metrology_status=payload.get("metrology_status"),
        chemistry_prior_summary=payload.get("chemistry_prior_summary"),
        refinement_summary=payload.get("refinement_summary"),
        request_more_data_fields=request_fields or None,
        requested_data_type=requested_data_type,
        disambiguation_reason=disambiguation_reason,
        recommendation_details=(
            None if evidence_items is not None else raw_recommendation_details
        ),
    )
    if evidence_items is not None:
        route.recommendation_details = _validated_recommendation_details(
            raw_recommendation_details,
            recommended_actions=route.recommended_actions,
            route_requested_data_type=route.requested_data_type,
            evidence_items=evidence_items,
        )
    return route


def _route_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot normalize terminal route of type {type(value).__name__}")


def _normalize_verdict(
    raw_verdict: str,
    *,
    payload: dict[str, Any],
    reason_codes: list[str],
) -> str:
    if raw_verdict in TERMINAL_ROUTE_VERDICTS:
        return raw_verdict
    if raw_verdict == "ok":
        if _has_accept_route_context(payload):
            return "ACCEPT"
        if "legacy_ok_missing_route_context" not in reason_codes:
            reason_codes.append("legacy_ok_missing_route_context")
        return "UNKNOWN"
    legacy = _LEGACY_ROUTE_VERDICTS.get(raw_verdict.lower())
    if legacy:
        return legacy
    if "invalid_terminal_route_verdict" not in reason_codes:
        reason_codes.append("invalid_terminal_route_verdict")
    return "UNKNOWN"


def _has_accept_route_context(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("gate_name")
        and payload.get("status")
        and payload.get("reported_hypothesis")
        and payload.get("evidence_summary")
    )


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _validated_recommendation_details(
    value: Any,
    *,
    recommended_actions: list[str],
    route_requested_data_type: str | None,
    evidence_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]] | None:
    """Keep only independently validated details with resolvable live evidence."""
    if not isinstance(value, list) or not value:
        return None

    valid: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        action = item.get("recommended_action")
        diagnosis_id = item.get("diagnosis_id")
        requested_data_type = item.get("requested_data_type")
        if (
            not isinstance(action, str)
            or action not in recommended_actions
            or item.get("rendered_fix") != action
            or not isinstance(diagnosis_id, str)
            or not diagnosis_id
            or item.get("validation_status") != "validated"
            or item.get("validated_diagnosis_id") != diagnosis_id
            or item.get("validated_requested_data_type") != requested_data_type
            or not _nonempty_string(item.get("rationale"))
            or not _nonempty_string(item.get("evidence_statement"))
            or not _nonempty_string_list(item.get("source_reason_codes"))
            or not _is_sha256(item.get("evaluator_manifest_sha256"))
        ):
            continue
        try:
            validate_requested_data_type(requested_data_type)
        except ValueError:
            continue
        if (
            route_requested_data_type is not None
            and requested_data_type != route_requested_data_type
        ):
            continue
        references = item.get("evidence_references")
        if not _references_resolve(references, evidence_items=evidence_items):
            continue
        valid.append(dict(item))
    return valid or None


def _references_resolve(
    references: Any,
    *,
    evidence_items: list[dict[str, Any]] | None,
) -> bool:
    if not isinstance(references, list) or not references:
        return False
    custody = None
    if evidence_items is not None:
        custody = {
            (str(item.get("role", "")), str(item.get("path", "")), str(item.get("sha256", "")))
            for item in evidence_items
            if isinstance(item, dict)
        }
    for reference in references:
        if not isinstance(reference, dict):
            return False
        role = reference.get("role")
        path = reference.get("path")
        digest = reference.get("sha256")
        if (
            not isinstance(role, str)
            or not role.strip()
            or not isinstance(path, str)
            or not path.strip()
            or not isinstance(digest, str)
            or not _is_sha256(digest)
        ):
            return False
        if custody is not None and (role, path, digest) not in custody:
            return False
        if not _file_matches_sha256(Path(path), digest):
            return False
    return True


def _file_matches_sha256(path: Path, expected_digest: str) -> bool:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected_digest


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_nonempty_string(item) for item in value)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
