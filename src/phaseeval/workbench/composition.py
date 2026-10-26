"""Typed composition measurements and the composition an answer implies (N9, plan U14).

Target path: src/phaseeval/workbench/composition.py (public).

- A CompositionMeasurement is what a person typed from a real XRF, ICP or EDS result: element
  fractions on a stated basis, the method, and the text it came from. The store stamps who
  entered it from the authenticated principal. Nothing is prefilled or estimated, and a record
  without its source is refused. Which elements a method detects and quantifies depends on the
  instrument and method (light elements especially), and an element a measurement does not
  report is not zero, so the fractions need not sum to 1.
- implied_composition is pure: the answer's phase formulas times its weight fractions give
  element weight and atomic fractions. Its basis is the crystalline phases the answer models,
  renormalized to 100 wt%; it is not the specimen's composition (amorphous content and
  unmodeled phases are absent from it). When a phase has no weight fraction or no readable
  formula, nothing is computed and the result names those phases, because renormalizing over
  the rest would silently drop them.

Store layout (under the WorkbenchStore root, one file per record, nothing rewritten):

    compositions/<root>/<utc>-<sha256>.json          CompositionMeasurement
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pymatgen.core import Composition
from pymatgen.core import Element as PeriodicElement

from phaseeval.api.auth import AuthorizationError, CaseOperation, Principal, PrincipalType

from .labels import LabelStore, rater_kind_for
from .model import AttemptRef, Element, Record, SafeID
from .store import WorkbenchStore
from phaseeval.kernel.ledger import KeyShape, Mode

IMPLIED_BASIS = (
    "the crystalline phases the answer models, weight fractions renormalized to 100; "
    "not the specimen's composition"
)
# Typed fractions may round; anything past this is a typing error, not rounding.
SUM_TOLERANCE = 0.005


class CompositionMeasurementRequest(Record):
    basis: Literal["atomic", "weight"]
    method: Literal["XRF", "ICP", "EDS"]
    fractions: dict[Element, float] = Field(min_length=1)  # element -> fraction of the sample
    # Where the numbers came from: report id, instrument file, notebook page.
    source: str = Field(max_length=4000)
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def sourced_and_bounded(self) -> Self:
        if not self.source.strip():
            raise ValueError("a composition measurement names its source")
        if not self.fractions:
            raise ValueError("a composition measurement reports at least one element")
        if any(not (math.isfinite(f) and 0 <= f <= 1) for f in self.fractions.values()):
            raise ValueError("each fraction is a finite number between 0 and 1")
        if sum(self.fractions.values()) > 1 + SUM_TOLERANCE:
            raise ValueError("fractions sum past 1")
        return self


class CompositionMeasurement(Record):
    schema_version: Literal["composition-measurement.v1"] = "composition-measurement.v1"
    root: AttemptRef
    basis: Literal["atomic", "weight"]
    method: Literal["XRF", "ICP", "EDS"]
    fractions: dict[Element, float]
    source: str
    note: str
    entered_by: SafeID  # the principal, stamped by the store
    created_at_utc: datetime


class ImpliedPhase(Record):
    phase_id: SafeID
    formula: str
    weight_percent: float  # as the answer reports it, before renormalizing


class ImpliedComposition(Record):
    schema_version: Literal["implied-composition.v1"] = "implied-composition.v1"
    status: Literal["computed", "not_computable"]
    reason: str
    basis: str  # what the fractions are normalized over; the page shows it beside them
    phases: tuple[ImpliedPhase, ...]  # the phases computed over; empty when not computable
    phases_without_weight: tuple[SafeID, ...]
    phases_without_formula: tuple[SafeID, ...]  # no formula, or one pymatgen cannot read
    weight_fractions: dict[str, float]
    atomic_fractions: dict[str, float]
    # Set when renormalized over these elements only, e.g. the ones a measurement reports.
    restricted_to: tuple[str, ...] = ()


def implied_composition(
    phases: Sequence[tuple[SafeID, str | None, float | None]],
    restrict_to: Sequence[str] = (),
) -> ImpliedComposition:
    """phases: (phase_id, formula, weight_percent) per answered phase. Raises ValueError on a
    weight percent that is negative, above 100 or not finite: that is an invalid analysis, not
    a composition."""
    for phase_id, _, w in phases:
        if w is not None and not (math.isfinite(w) and 0 <= w <= 100):
            raise ValueError(f"{phase_id}: weight percent {w} is not a finite value between 0 and 100")
    without_weight = tuple(p for p, _, w in phases if w is None)
    without_formula = tuple(p for p, f, _ in phases if not f or not _readable(f))
    used = [
        ImpliedPhase(phase_id=p, formula=f, weight_percent=w)
        for p, f, w in phases
        if p not in without_weight and p not in without_formula
    ]
    total = sum(p.weight_percent for p in used)
    reasons = []
    if without_weight:
        reasons.append(f"no weight fraction for {', '.join(without_weight)}")
    if without_formula:
        reasons.append(f"no readable formula for {', '.join(without_formula)}")
    if not reasons and total <= 0:
        reasons.append("no answered phase with a positive weight fraction")
    if reasons:
        return ImpliedComposition(
            status="not_computable", reason="not computed: " + "; ".join(reasons), basis=IMPLIED_BASIS, phases=(),
            phases_without_weight=without_weight, phases_without_formula=without_formula,
            weight_fractions={}, atomic_fractions={},
        )
    weight: dict[str, float] = {}
    for phase in used:
        composition = Composition(phase.formula)
        for element in composition.elements:
            share = phase.weight_percent / total * composition.get_wt_fraction(element)
            weight[element.symbol] = weight.get(element.symbol, 0.0) + share
    basis = IMPLIED_BASIS
    if restrict_to:
        weight = {e: w for e, w in weight.items() if e in set(restrict_to)}
        basis = (
            f"{IMPLIED_BASIS}; renormalized over the reported elements {', '.join(sorted(restrict_to))} only, "
            "other elements are left out, not zero"
        )
    moles = {e: w / float(PeriodicElement(e).atomic_mass) for e, w in weight.items()}
    return ImpliedComposition(
        status="computed", reason=f"computed over {len(used)} phases", basis=basis, phases=tuple(used),
        phases_without_weight=(), phases_without_formula=(),
        weight_fractions=_normalized(weight), atomic_fractions=_normalized(moles),
        restricted_to=tuple(sorted(restrict_to)),
    )


def _readable(formula: str) -> bool:
    try:
        elements = Composition(formula).elements
    except (ValueError, KeyError, AttributeError):
        return False
    # pymatgen reads an unknown symbol such as "Qq" as a dummy species, not an error.
    return bool(elements) and all(isinstance(e, PeriodicElement) for e in elements)


def _normalized(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {e: v / total for e, v in sorted(values.items())} if total else {}


class CompositionStore:
    """Writes and reads measurements under the WorkbenchStore root, with its ACL and custody
    primitives. Only USER principals with access to the root enter or read them: an agent
    never supplies a measurement (R29). Blindness (R32): a rater reads only the measurements
    they entered until their reveal opens on the root, because another rater's entry is part of
    that rater's judgment; the admin owner reads every entry."""

    def __init__(self, store: WorkbenchStore) -> None:
        self._store = store
        self._labels = LabelStore(store)
        with store.ledger.transaction():
            if not hasattr(store, '_compositions'):
                store._compositions = store.ledger.collection('compositions', CompositionMeasurement, Mode.LOG, KeyShape(('hex64',)), owner=__name__)
        self._measurements = store._compositions

    def record(
        self, principal: Principal, root: AttemptRef, request: CompositionMeasurementRequest
    ) -> CompositionMeasurement:
        self._authorized(principal, root, CaseOperation.ACTION)
        measurement = CompositionMeasurement(
            root=root,
            entered_by=principal.principal_id,
            created_at_utc=self._store._clock(),
            **request.model_dump(),
        )
        with self._store.ledger.transaction():
            self._measurements.append((root,), measurement)
        return measurement

    def measurements(self, principal: Principal, root: AttemptRef) -> tuple[CompositionMeasurement, ...]:
        self._authorized(principal, root, CaseOperation.INSPECT)
        found = self._measurements.entries((root,))
        if _is_rater(principal) and not self._labels.reveal_state(principal, root).revealed:
            return tuple(m for m in found if m.entered_by == principal.principal_id)
        return found

    def _authorized(self, principal: Principal, root: AttemptRef, operation: CaseOperation) -> Path:
        if principal.principal_type is not PrincipalType.USER:
            raise PermissionError("only a person enters or reads a composition measurement")
        if not re.fullmatch(r"[0-9a-f]{64}", root):
            raise ValueError("a root is 64 hex")
        self._store._authorize(principal, root, operation)
        return self._store.root / "compositions" / root


def _is_rater(principal: Principal) -> bool:
    """A label-mode credential (labels.rater_kind_for); the admin owner is not one."""
    try:
        rater_kind_for(principal)
    except AuthorizationError:
        return False
    return True
