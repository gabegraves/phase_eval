"""The workbench page and the review data it reads (U4, R12, R32, plan KTD4 and KTD12).

Target path: src/phaseeval/workbench/page_routes.py (public). W3 view.

- GET /workbench and /workbench/assets/{name} serve the page's static files. They carry no
  data, so they need no token; every /api route here requires bearer authentication (R29).
- GET /api/attempts/queue lists this rater's snapshot digests in review order, from assignment
  markers rebuilt when the label store opens. It names no run.
- GET /api/attempts/snapshots/{digest}/review is snapshot.blind_payload.
- GET /api/attempts/{digest}/review is snapshot.attempt_payload: the same blind shape for an
  attempt that is not a snapshot, such as a rater's own refit child beside its parent (D10).
  Both review routes read through the rater projection (labels.StageProjection.admits, CR-07).
- GET /api/attempts/snapshots/{digest}/revealed is snapshot.revealed_payload: 403 until the
  label store holds this rater's sign-off on the snapshot's root.
- GET /api/attempts/snapshots/{digest}/files.zip is snapshot.analysis_files: the same gate; the
  pattern, CIFs and the cited fit's BGMN files for the rater's records (AXA-43zwk.18).

Queue layout under the WorkbenchStore root: rater-items/<principal>/<snapshot>, one empty marker
per assigned snapshot. Legacy queues/ files are ignored.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from phaseeval.api.auth import AuthorizationError
from phaseeval.api.routes.attempts import Authenticated, Store

from . import snapshot
from .label_routes import Labels
from .store import CustodyError

ASSETS = Path(__file__).resolve().parents[1] / "api" / "assets"
PAGE_ASSETS = {
    **{f"{name}.js": "text/javascript" for name in ("workbench", "scorecard", "launcher", "report")},
    **{f"{name}.css": "text/css" for name in ("workbench", "scorecard", "launcher", "fonts", "presentation", "report")},
    "phaseeval-mark.svg": "image/svg+xml",
    **{f"fonts/{name}.woff2": "font/woff2" for name in (
        "IBMPlexSans-Regular", "IBMPlexSans-Medium", "IBMPlexSans-SemiBold",
        "IBMPlexMono-Regular", "IBMPlexMono-Medium",
    )},
}

router = APIRouter(tags=["workbench"])


def _call[R](function: Callable[..., R], /, *args: Any, **kwargs: Any) -> R:
    try:
        return function(*args, **kwargs)
    except (AuthorizationError, PermissionError) as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "refused") from exc
    except (FileNotFoundError, LookupError) as exc:
        raise HTTPException(status_code=404, detail="snapshot not found") from exc
    except (CustodyError, snapshot.SnapshotError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/workbench", include_in_schema=False)
def get_page() -> FileResponse:
    return FileResponse(ASSETS / "workbench.html", media_type="text/html")


@router.get("/workbench/assets/{name:path}", include_in_schema=False)
def get_page_asset(name: str) -> FileResponse:
    if name not in PAGE_ASSETS:
        raise HTTPException(status_code=404, detail="no such asset")
    return FileResponse(ASSETS / name, media_type=PAGE_ASSETS[name])


@router.get("/api/attempts/queue")
def get_queue(store: Store, labels: Labels, principal: Authenticated) -> dict[str, Any]:
    """This rater's snapshots in review order: digest, case and kind, nothing about the run."""
    items = []
    for digest in store._rater_items.members((principal.principal_id,)):
        try:
            snap = _call(labels.snapshot, principal, digest)
        except HTTPException as exc:
            if exc.status_code == 403:
                continue
            raise
        items.append({"digest": digest, "case_id": snap.case_id, "kind": snap.kind})
    return {"items": items}


@router.get("/api/attempts/snapshots/{digest:hex64}/review")
def get_review(
    digest: str, store: Store, labels: Labels, principal: Authenticated
) -> dict[str, Any]:
    return _call(snapshot.blind_payload, store, labels, principal, digest)


@router.get("/api/attempts/{digest:hex64}/review")
def get_attempt_review(digest: str, store: Store, principal: Authenticated) -> dict[str, Any]:
    """D10: the blind payload of an attempt that is not a snapshot, e.g. a refit child."""
    return _call(snapshot.attempt_payload, store, principal, digest)


@router.get("/api/attempts/snapshots/{digest:hex64}/files.zip")
def get_analysis_files(digest: str, store: Store, labels: Labels, principal: Authenticated) -> Response:
    """snapshot.analysis_files: 403 until this rater signed off, as /revealed."""
    data = _call(snapshot.analysis_files, store, labels, principal, digest)
    name = f"phaseeval-{digest[:12]}.zip"
    return Response(content=data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/api/attempts/snapshots/{digest:hex64}/revealed")
def get_revealed(
    digest: str, request: Request, store: Store, labels: Labels, principal: Authenticated
) -> dict[str, Any]:
    return _call(snapshot.revealed_payload, store, labels, principal, digest, custody=request.app.state.settings.custody)
