from __future__ import annotations

import importlib
import pkgutil
import types
import typing
from typing import Annotated, get_args, get_origin

import pytest
from pydantic import BaseModel

import phaseeval
from phaseeval.kernel.layers import ACTIVE_MODULES

if "phaseeval.kernel.records" not in ACTIVE_MODULES:
    pytest.skip("record/audience checks activate at P11, before importing their dependencies", allow_module_level=True)

from phaseeval.kernel.errors import AudienceViolation, Denied
from phaseeval.kernel.ids import LINKABLE
from phaseeval.kernel.layers import LANDS_IN, LEGACY
from phaseeval.kernel.records import Audience, AudienceRecord, check_audiences

BLIND = {Audience.RATER_BLIND}
ROOTS = ("phaseeval.review.views:BlindView", "phaseeval.publish.dataset:JudgeInputV2")


def _root(spec: str) -> type[AudienceRecord]:
    module, name = spec.split(":")
    if LEGACY and module not in ACTIVE_MODULES:
        pytest.skip(f"{module} lands in {LANDS_IN.get(module.split('.')[1], 'a later PR')}")
    return getattr(importlib.import_module(module), name)


def _import_everything() -> None:
    for info in pkgutil.walk_packages(phaseeval.__path__, "phaseeval."):
        if info.name.rsplit(".", 1)[-1] != "__main__":
            if not LEGACY or info.name in ACTIVE_MODULES:
                importlib.import_module(info.name)


def _walk(annotation: object):
    """Every node of an annotation tree: Annotated metadata, union members, container args."""
    stack = [annotation]
    while stack:
        item = stack.pop()
        yield item
        if get_origin(item) is Annotated:
            args = get_args(item)
            stack.append(args[0])
            stack.extend(args[1:])
        elif get_origin(item) in (types.UnionType, typing.Union, tuple, list, dict, typing.Mapping, typing.Sequence) or (
            get_origin(item) is not None and get_args(item)
        ):
            stack.extend(get_args(item))


def _models_in(annotation: object) -> list[type[BaseModel]]:
    return [n for n in _walk(annotation) if isinstance(n, type) and issubclass(n, BaseModel)]


def _linkable_in(annotation: object, metadata: list) -> bool:
    return any(m is LINKABLE for m in metadata) or any(n is LINKABLE for n in _walk(annotation))


def _subclasses(base: type) -> list[type]:
    out = []
    for sub in base.__subclasses__():
        out.append(sub)
        out.extend(_subclasses(sub))
    return out


def _audiences(info) -> set[Audience]:
    return {m for m in info.metadata if isinstance(m, Audience)}


def test_views_are_audience_typed_all_the_way_down() -> None:
    _import_everything()
    bad = []
    for cls in _subclasses(AudienceRecord):
        for name, info in cls.model_fields.items():
            if not _audiences(info):
                bad.append(f"{cls.__name__}.{name}: no Audience")
            for nested in _models_in(info.annotation):
                if not issubclass(nested, AudienceRecord):
                    bad.append(f"{cls.__name__}.{name}: nested {nested.__name__} is not an AudienceRecord")
    assert bad == []


def test_a_field_without_an_audience_fails_at_class_creation() -> None:
    with pytest.raises(TypeError):

        class Leaky(AudienceRecord):
            secret: str


def test_no_blind_field_carries_a_linkable_id() -> None:
    _import_everything()
    bad = [
        f"{cls.__name__}.{name}"
        for cls in _subclasses(AudienceRecord)
        for name, info in cls.model_fields.items()
        if _audiences(info) & BLIND and _linkable_in(info.annotation, info.metadata)
    ]
    assert bad == []


@pytest.mark.parametrize("spec", ROOTS)
def test_the_view_type_holds_no_linkable_id(spec: str) -> None:
    root = _root(spec)
    seen: set[type] = set()
    todo = [root]
    bad = []
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        for name, info in cls.model_fields.items():
            if _linkable_in(info.annotation, info.metadata):
                bad.append(f"{cls.__name__}.{name}")
            todo.extend(_models_in(info.annotation))
    assert bad == []




class _Inner(AudienceRecord):
    shown: Annotated[str, Audience.RATER_BLIND]


class _Root(AudienceRecord):
    inner: Annotated[tuple[_Inner, ...], Audience.RATER_BLIND]
    name: Annotated[str, Audience.RATER_BLIND]


class _RootWithExpertField(_Root):
    model_name: Annotated[str, Audience.EXPERT]


def test_check_audiences_accepts_a_root_inside_the_permitted_set() -> None:
    check_audiences(_Root(inner=(_Inner(shown="x"),), name="n"), frozenset(BLIND))


def test_check_audiences_refuses_a_field_outside_the_permitted_set() -> None:
    leaky = _RootWithExpertField(inner=(_Inner(shown="x"),), name="n", model_name="gpt")
    with pytest.raises(AudienceViolation) as info:
        check_audiences(leaky, frozenset(BLIND))
    assert "model_name" in str(info.value)
    check_audiences(leaky, frozenset({Audience.RATER_BLIND, Audience.EXPERT}))


def test_check_audiences_walks_nested_containers() -> None:
    class _InnerExpert(AudienceRecord):
        secret: Annotated[str, Audience.EXPERT]

    class _Holder(AudienceRecord):
        items: Annotated[tuple[_InnerExpert, ...], Audience.RATER_BLIND]

    with pytest.raises(AudienceViolation) as info:
        check_audiences(_Holder(items=(_InnerExpert(secret="s"),)), frozenset(BLIND))
    assert "items[0].secret" in str(info.value)


def test_a_proof_cannot_be_minted_outside_stage() -> None:
    stage = importlib.import_module(_root("phaseeval.review.stage:IdentityProof").__module__)

    with pytest.raises(TypeError):
        stage.IdentityProof(principal_id="r", snapshot="a" * 64)
    with pytest.raises(TypeError):
        stage.ReferenceProof(principal_id="r", family="a" * 64)


def test_a_proof_is_bound_to_its_snapshot() -> None:
    stage = importlib.import_module(_root("phaseeval.review.stage:IdentityProof").__module__)

    proof = stage.IdentityProof(principal_id="r", snapshot="a" * 64, _mint=stage._MINT)
    proof.check("a" * 64)
    with pytest.raises(Denied):
        proof.check("b" * 64)


def test_permitted_audiences_follow_the_stage() -> None:
    module = importlib.import_module(_root("phaseeval.review.stage:ReviewStage").__module__)
    ReviewStage, permitted_audiences = module.ReviewStage, module.permitted_audiences

    assert permitted_audiences(ReviewStage.BLIND) == frozenset({Audience.RATER_BLIND})
    assert permitted_audiences(ReviewStage.SIGNED) == frozenset({Audience.RATER_BLIND, Audience.IDENTITY_OPEN})
    assert Audience.REVEALED in permitted_audiences(ReviewStage.REVEALED)
    assert permitted_audiences(ReviewStage.REVEALED, judge=True) == frozenset({Audience.RATER_BLIND})
