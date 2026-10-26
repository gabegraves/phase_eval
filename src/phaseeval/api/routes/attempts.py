"""HTTP routes over the attempt tree. Thin: authenticate, call the store, map errors (R12 to R16).

Every route depends on `authenticate_principal`, the bearer authentication; an unauthenticated
call is refused (R29). The projection is derived from the principal inside the store, never
from a request field. Digests in paths match only 64 lowercase hex, so routers from other units
mounted under /api/attempts (labels, page, chemistry) never collide with `/{digest}`.

Kept, not changed: /api/analyze/tools/* and /api/diffract/rank-polymorphs, which the
detrix-core Pi bridge calls (KTD1). Their calls are unrecorded diagnostics; nothing unrecorded
reaches an export or the dashboard, because both derive from the tree alone.
"""

import re
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import Field, TypeAdapter
from starlette.convertors import Convertor, register_url_convertor

from phaseeval.api.auth import (
    AuthenticationError,
    AuthorizationError,
    Principal,
    PrincipalAuthenticator,
)
from phaseeval.workbench.model import Action, Record
from phaseeval.workbench.review import RecordRequest
from phaseeval.workbench.store import (
    ActionRefused,
    ActionRequest,
    CustodyError,
    WorkbenchStore,
)

DEMO_BIND_HOST = "127.0.0.1"


class _Hex64(Convertor[str]):
    regex = "[0-9a-f]{64}"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        if not re.fullmatch(self.regex, value):
            raise ValueError("a digest is 64 lowercase hex")
        return value


register_url_convertor("hex64", _Hex64())

router = APIRouter(prefix="/api/attempts", tags=["attempts"])

# GET /verbs: the action union's JSON schema. The page's action bar and the MCP tool list are
# generated from the same classes, so no verb list is hand-synced (R2).
ACTION_SCHEMA: dict[str, Any] = TypeAdapter(Action).json_schema()


class GestureRequest(Record):
    """One POST, one or two attempts: a phase or context edit and the refine after it (R13).
    Step two's parent is step one's child."""

    steps: tuple[ActionRequest, ...] = Field(min_length=1, max_length=2)


def get_workbench_store(request: Request) -> WorkbenchStore:
    store = getattr(request.app.state, "workbench_store", None)
    if not isinstance(store, WorkbenchStore):
        raise HTTPException(status_code=503, detail="workbench store is not configured")
    return store


def get_attempt_authenticator(request: Request) -> PrincipalAuthenticator:
    authenticator = getattr(request.app.state, "attempt_authenticator", None)
    if not isinstance(authenticator, PrincipalAuthenticator):
        raise HTTPException(status_code=503, detail="attempt authentication is not configured")
    return authenticator


def authenticate_principal(
    authorization: Annotated[str | None, Header()] = None,
    authenticator: PrincipalAuthenticator = Depends(get_attempt_authenticator),
) -> Principal:
    try:
        return authenticator.authenticate(authorization)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


Store = Annotated[WorkbenchStore, Depends(get_workbench_store)]
Authenticated = Annotated[Principal, Depends(authenticate_principal)]


def _call[R](function: Callable[..., R], /, *args: Any, **kwargs: Any) -> R:
    try:
        return function(*args, **kwargs)
    except (AuthorizationError, PermissionError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ActionRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (CustodyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    return value.model_dump(mode="json")


@router.get("/verbs")
def get_verbs(principal: Authenticated) -> dict[str, Any]:
    return ACTION_SCHEMA


@router.get("/whoami")
def get_whoami(principal: Authenticated) -> dict[str, Any]:
    """The caller's own identity, so a client can refuse the wrong credential (U9)."""
    return principal.identity()


@router.get("/roots")
def get_roots(store: Store, principal: Authenticated) -> list[str]:
    return list(_call(store.roots, principal))


@router.get("/roots/{root:hex64}/tree")
def get_tree(root: str, store: Store, principal: Authenticated) -> Any:
    """Humans and services; 403 for agents."""
    return _json(_call(store.tree, principal, root))


@router.get("/compare")
def get_compare(a: str, b: str, store: Store, principal: Authenticated) -> Any:
    """Humans only."""
    return _json(_call(store.compare, principal, a, b))


@router.post("/{parent:hex64}/actions", status_code=201)
def post_actions(parent: str, request: GestureRequest, store: Store, principal: Authenticated) -> Any:
    """{"attempts": [review.GestureStep, ...]}, one per step that ran: blind summaries only (D10).
    Runs synchronously: a DEV006 refit takes about 140 s. 409 on a key reused with different
    arguments; 422 for a human parse failure."""
    return {"attempts": _json(_call(store.gesture, principal, parent, request.steps))}


@router.get("/{digest:hex64}")
def get_attempt(digest: str, store: Store, principal: Authenticated) -> Any:
    return _json(_call(store.view, principal, digest))


@router.get("/{digest:hex64}/path")
def get_path(digest: str, store: Store, principal: Authenticated) -> Any:
    return _json(_call(store.path, principal, digest))


@router.post("/{digest:hex64}/records", status_code=201)
def post_record(digest: str, request: RecordRequest, store: Store, principal: Authenticated) -> Any:
    """Notes and preferences (humans; 403 for agents and services, R29). Rubric labels are
    labels.py routes."""
    return _json(_call(store.record, principal, digest, request))


@router.get("/{digest:hex64}/artifacts/{sha256:hex64}")
def get_artifact(digest: str, sha256: str, store: Store, principal: Authenticated) -> Response:
    media_type, data = _call(store.blob, principal, digest, sha256)
    return Response(
        data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{sha256}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
