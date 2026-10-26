from __future__ import annotations

import enum
import hashlib
import json
import math
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


class _SinceMarker:
    pass


SINCE = _SinceMarker()


class Since:
    def __class_getitem__(cls, tp: Any) -> Any:
        return Annotated[tp | None, SINCE]


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _check_finite(getattr(value, name))
    elif isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_finite(item)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def _finite_numbers(self) -> Self:
        _check_finite(self)
        return self

    @model_serializer(mode="wrap")
    def _omit_unset_since_fields(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        for name, info in type(self).model_fields.items():
            if any(m is SINCE for m in info.metadata) and data.get(name) is None:
                data.pop(name, None)
        return data


def canonical_bytes(record: BaseModel) -> bytes:
    _check_finite(record)
    return (
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def digest_of(record: BaseModel) -> str:
    return hashlib.sha256(canonical_bytes(record)).hexdigest()


LEGACY_SCHEMA_IDS: frozenset[str] = frozenset(
    {
        "analysis-snapshot.v1",
        "assignment.v1",
        "axa.attempt.v3",
        "axa.workbench.case.v1",
        "axa.workbench.dataset.v1",
        "axa.workbench.episode.v1",
        "case-expectation.v2",
        "chemistry-evidence.v1",
        "composition-measurement.v1",
        "exposure-declaration.v1",
        "gpx-analysis.v3",
        "implied-composition.v1",
        "judge-input.v1",
        "judge-manifest.v1",
        "label.v1",
        "peak-call.v1",
        "permission.v1",
        "qualification-report.v2",
        "qualification-set.v2",
        "review-queue.v1",
        "route.v1",
        "signoff.v1",
        "step-feedback.v1",
    }
)


Clock = Callable[[], datetime]


class Audience(enum.StrEnum):
    PUBLIC = "public"
    AGENT = "agent"
    AGENT_OWN = "agent_own"
    RATER_BLIND = "rater_blind"
    IDENTITY_OPEN = "identity_open"
    REVEALED = "revealed"
    EXPERT = "expert"
    SERVICE = "service"


class AudienceRecord(Record):
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        missing = [
            name
            for name, info in cls.model_fields.items()
            if not any(isinstance(m, Audience) for m in info.metadata)
        ]
        if missing:
            raise TypeError(f"{cls.__name__}: fields without an Audience annotation: {missing}")


def _audiences_of(info: Any) -> frozenset[Audience]:
    return frozenset(m for m in info.metadata if isinstance(m, Audience))


def check_audiences(root: BaseModel, permitted: frozenset[Audience]) -> None:
    from phaseeval.kernel.errors import AudienceViolation

    def walk(value: Any, path: str) -> None:
        if isinstance(value, AudienceRecord):
            for name, info in type(value).model_fields.items():
                audiences = _audiences_of(info)
                if not audiences & permitted:
                    raise AudienceViolation(f"{path}.{name}")
                walk(getattr(value, name), f"{path}.{name}")
        elif isinstance(value, BaseModel):
            raise AudienceViolation(f"{path}: {type(value).__name__} is not an AudienceRecord")
        elif isinstance(value, (list, tuple, set, frozenset)):
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")
        elif isinstance(value, dict):
            for k, item in value.items():
                walk(item, f"{path}[{k!r}]")

    walk(root, type(root).__name__)


def field_has_set(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if annotation in (set, frozenset) or origin in (set, frozenset):
        return True
    return any(field_has_set(arg) for arg in get_args(annotation))
