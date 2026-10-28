"""HTTP routes for snapshots, rubric labels, peak calls, sign-offs, reveal, step feedback and
permission records (R29, R32, R33, N8). Thin: authenticate, call the LabelStore, map errors.

Target path: src/phaseeval/workbench/label_routes.py (public), mounted under /api/attempts. Every
route depends on the attempt routes' bearer authentication; an unauthenticated call is refused.

A snapshot route's body is its labels.py request type without `snapshot`, which travels in the
path. No body carries a rater, rater_kind or exposure: extra="forbid" answers 422 (W2 labels).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import Field, ValidationError

from phaseeval.api.auth import Principal
from phaseeval.api.routes.attempts import Store, authenticate_principal

from .labels import (
    LabelRequest,
    LabelStore,
    PeakCallRequest,
    PermissionRequest,
    SignOffRequest,
    StepFeedbackRequest,
    TrajectoryReviewRequest,
)
from .model import AttemptRef, Digest, Record, SafeID
from .snapshot import SnapshotKind
from . import snapshot as snapshots
from .store import CustodyError, WorkbenchStore

router = APIRouter(prefix="/api/attempts", tags=["labels"])


class SnapshotRequest(Record):
    origin: AttemptRef
    kind: SnapshotKind


class AssignmentRequest(Record):
    raters: tuple[SafeID, ...] = Field(min_length=1)
    snapshots: tuple[Digest, ...] = Field(min_length=1)  # the review cohort (Build D12)


def get_label_store(request: Request) -> LabelStore:
    store = getattr(request.app.state, "workbench_store", None)
    if not isinstance(store, WorkbenchStore):
        raise HTTPException(status_code=503, detail="the workbench store is not configured")
    return LabelStore(store)


Labels = Annotated[LabelStore, Depends(get_label_store)]
Authenticated = Annotated[Principal, Depends(authenticate_principal)]
JsonBody = Annotated[dict[str, Any], Body()]


def _call[R](function: Callable[..., R], /, *args: Any) -> R:
    try:
        return function(*args)
    except PermissionError as exc:  # unknown digest, missing grant, wrong principal: one answer
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except CustodyError:
        raise
    except ValidationError as exc:
        detail = exc.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=422, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail="snapshot building is not available yet") from exc


def _with_snapshot[R: Record](kind: type[R], digest: str, body: dict[str, Any]) -> R:
    if body.get("snapshot", digest) != digest:
        raise HTTPException(status_code=422, detail="the body names another snapshot than the path")
    return _call(kind.model_validate, body | {"snapshot": digest})


def _json(records: Any) -> Any:
    if isinstance(records, tuple):
        return [record.model_dump(mode="json") for record in records]
    return records.model_dump(mode="json")


@router.post("/snapshots", status_code=201)
def post_snapshot(request: SnapshotRequest, labels: Labels, principal: Authenticated) -> dict[str, Any]:
    digest = _call(labels.freeze, principal, request.origin, request.kind)
    return {"digest": digest, "snapshot": _json(_call(labels.snapshot, principal, digest))}


@router.get("/snapshots/{digest}")
def get_snapshot(digest: str, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.snapshot, principal, digest))


@router.post("/snapshots/{digest}/labels", status_code=201)
def post_label(digest: str, body: JsonBody, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.label, principal, _with_snapshot(LabelRequest, digest, body)))


@router.get("/snapshots/{digest}/labels")
def get_labels(digest: str, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.visible_labels, principal, digest))


@router.post("/snapshots/{digest}/peak-calls", status_code=201)
def post_peak_call(digest: str, body: JsonBody, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.peak_call, principal, _with_snapshot(PeakCallRequest, digest, body)))


@router.post("/snapshots/{digest}/signoff", status_code=201)
def post_signoff(digest: str, body: JsonBody, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.sign_off, principal, _with_snapshot(SignOffRequest, digest, body)))


@router.get("/roots/{root}/reveal-state")
def get_reveal_state(root: str, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.reveal_state, principal, root))


@router.post("/roots/{root}/assignment", status_code=201)
def post_assignment(root: str, request: AssignmentRequest, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.assign, principal, root, request.raters, request.snapshots))


@router.post("/step-feedback", status_code=201)
def post_step_feedback(request: StepFeedbackRequest, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.step_feedback, principal, request))


@router.get("/step-feedback")
def get_step_feedback(attempt: str, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.visible_step_feedback, principal, attempt))


@router.post("/permissions", status_code=201)
def post_permission(request: PermissionRequest, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.record_permission, principal, request))


@router.get("/snapshots/{digest}/trajectory-review")
def get_trajectory_review(digest: str, store: Store, labels: Labels, principal: Authenticated) -> Any:
    reviews = _call(labels.own_trajectory_reviews, principal, digest)
    trace = _call(snapshots.method_trace, store, labels, principal, digest)
    limits = ["Recorded observations are historical evidence; retained context after compaction may be unknown."]
    if not trace:
        limits.append("No recorded conversation is available. Defer unless the record supports the judgment.")
    return {"snapshot": digest, "protocol_version": "xrd-method-v1", "trace": trace,
            "reviews": _json(reviews), "limitations": limits}


@router.post("/snapshots/{digest}/trajectory-review", status_code=201)
def post_trajectory_review(digest: str, body: JsonBody, labels: Labels, principal: Authenticated) -> Any:
    return _json(_call(labels.review_trajectory, principal, _with_snapshot(TrajectoryReviewRequest, digest, body)))
