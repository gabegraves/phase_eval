from __future__ import annotations

import importlib
import pkgutil
from typing import Literal, get_args, get_origin

import phaseeval
from phaseeval.kernel.layers import ACTIVE_MODULES, LEGACY
import pytest

if "phaseeval.kernel.records" not in ACTIVE_MODULES:
    pytest.skip("record/audience checks activate at P11, before importing their dependencies", allow_module_level=True)

from phaseeval.kernel.records import LEGACY_SCHEMA_IDS, Record, field_has_set

BASE_SERIALIZERS = set(Record.__pydantic_decorators__.model_serializers)


def _import_everything() -> None:
    for info in pkgutil.walk_packages(phaseeval.__path__, "phaseeval."):
        if info.name.rsplit(".", 1)[-1] != "__main__":
            if not LEGACY or info.name in ACTIVE_MODULES:
                importlib.import_module(info.name)


def _subclasses(base: type) -> list[type]:
    out = []
    for sub in base.__subclasses__():
        out.append(sub)
        out.extend(_subclasses(sub))
    return out


def _literals(annotation: object) -> list[str]:
    if get_origin(annotation) is Literal:
        return [a for a in get_args(annotation) if isinstance(a, str)]
    return [lit for arg in get_args(annotation) for lit in _literals(arg)]


def test_no_record_defines_its_own_serializer() -> None:
    _import_everything()
    bad = [
        f"{cls.__module__}.{cls.__name__}: {sorted(extra)}"
        for cls in _subclasses(Record)
        if (extra := set(cls.__pydantic_decorators__.model_serializers) - BASE_SERIALIZERS)
    ]
    assert bad == []


def test_every_schema_id_is_legacy_or_phaseeval() -> None:
    _import_everything()
    bad = []
    for cls in _subclasses(Record):
        info = cls.model_fields.get("schema_version")
        if info is None:
            continue
        literals = _literals(info.annotation)
        assert literals, f"{cls.__name__}.schema_version is not a Literal"
        bad.extend(
            f"{cls.__name__}: {lit}" for lit in literals if lit not in LEGACY_SCHEMA_IDS and not lit.startswith("phaseeval.")
        )
    assert bad == []


def test_no_record_field_is_a_set() -> None:
    _import_everything()
    bad = [
        f"{cls.__module__}.{cls.__name__}.{name}"
        for cls in _subclasses(Record)
        if cls.__module__.startswith("phaseeval.")
        for name, info in cls.model_fields.items()
        if field_has_set(info.annotation)
    ]
    assert bad == []
