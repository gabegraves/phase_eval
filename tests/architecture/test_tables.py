from __future__ import annotations

import importlib
from typing import get_args

import pytest

from phaseeval.kernel.layers import ACTIVE_MODULES, ACTIVE_THROUGH, LEGACY, MODULE_LANDS_IN


LEGACY_CHECKS = {
    "phaseeval.web.errors": (11, "phaseeval.api.errors"),
    "phaseeval.tree.access": (14, "phaseeval.api.auth"),
    "phaseeval.review.checks": (20, "phaseeval.workbench.snapshot"),
    "phaseeval.review.policy": (21, "phaseeval.workbench.triage"),
}


def _module(name: str):
    if LEGACY and name not in ACTIVE_MODULES:
        pr, legacy = LEGACY_CHECKS.get(name, (MODULE_LANDS_IN.get(name, 99), name))
        if ACTIVE_THROUGH < pr:
            pytest.skip(f"{name} check activates at P{pr:02d}")
        name = legacy
    return importlib.import_module(name)


def _subclasses(base: type) -> list[type]:
    out = []
    for sub in base.__subclasses__():
        out.append(sub)
        out.extend(_subclasses(sub))
    return out


def test_every_domain_error_subclass_has_an_error_row() -> None:
    DomainError = _module("phaseeval.kernel.errors").DomainError
    errors = _module("phaseeval.web.errors").ERRORS
    missing = [cls.__name__ for cls in _subclasses(DomainError) if cls not in errors]
    assert missing == []


def test_every_error_row_names_a_domain_error() -> None:
    DomainError = _module("phaseeval.kernel.errors").DomainError
    errors = _module("phaseeval.web.errors").ERRORS
    assert all(issubclass(cls, DomainError) for cls in errors)


def test_rules_cover_every_op() -> None:
    access = _module("phaseeval.tree.access")
    assert set(access.RULES) == set(access.Op)


def test_archives_cover_every_platform() -> None:
    program = _module("phaseeval.bgmn.program")
    assert set(program.ARCHIVES) == set(get_args(program.Platform))


def test_check_order_is_the_check_set_fields() -> None:
    checks = _module("phaseeval.review.checks")
    policy = _module("phaseeval.review.policy")
    assert checks.CHECK_ORDER == tuple(name.upper() for name in checks.CheckSet.model_fields if name != "policy")
    assert set(checks.CHECK_ORDER) == set(get_args(policy.CheckCode))
