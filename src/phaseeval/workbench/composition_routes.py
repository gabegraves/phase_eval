"""HTTP routes for typed composition input (N9): a root's measurements and the composition an
answer implies. Every route takes the /api/attempts bearer authentication (R29).

Include with `app.include_router(composition_routes.router)`; it reads the WorkbenchStore from
`app.state.workbench_store`, as the attempt routes do.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from phaseeval.api.auth import AuthorizationError, Principal
from phaseeval.api.routes.attempts import authenticate_principal

from .composition import (
    CompositionMeasurementRequest,
    CompositionStore,
    implied_composition,
)
from .model import Element, Record, SafeID
from .store import WorkbenchStore

router = APIRouter(prefix="/api/attempts", tags=["composition"])


def get_composition_store(request: Request) -> CompositionStore:
    store = getattr(request.app.state, "workbench_store", None)
    if not isinstance(store, WorkbenchStore):
        raise HTTPException(status_code=503, detail="workbench store is not configured")
    return CompositionStore(store)


Compositions = Annotated[CompositionStore, Depends(get_composition_store)]
Authenticated = Annotated[Principal, Depends(authenticate_principal)]


class AnswerPhase(Record):
    phase_id: SafeID
    formula: str | None
    weight_percent: float | None = Field(ge=0, le=100, allow_inf_nan=False)


class ImpliedRequest(Record):
    phases: tuple[AnswerPhase, ...] = Field(min_length=1, max_length=200)
    restrict_to: tuple[Element, ...] = ()


def _call[R](function: Any, /, *args: Any) -> R:
    try:
        return function(*args)
    except (AuthorizationError, PermissionError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/roots/{root}/composition")
def get_measurements(root: str, compositions: Compositions, principal: Authenticated) -> list[dict[str, Any]]:
    """Every measurement entered for the root, oldest first; empty when none was entered."""
    measurements = _call(compositions.measurements, principal, root)
    return [m.model_dump(mode="json") for m in measurements]


@router.post("/roots/{root}/composition", status_code=201)
def post_measurement(
    root: str, request: CompositionMeasurementRequest, compositions: Compositions, principal: Authenticated
) -> dict[str, Any]:
    return _call(compositions.record, principal, root, request).model_dump(mode="json")


@router.post("/implied-composition")
def post_implied(request: ImpliedRequest, principal: Authenticated) -> dict[str, Any]:
    """Pure: the composition the posted answer phases imply. Stores nothing."""
    phases = [(p.phase_id, p.formula, p.weight_percent) for p in request.phases]
    return _call(implied_composition, phases, request.restrict_to).model_dump(mode="json")
