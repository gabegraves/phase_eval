"""Main FastAPI application for AgentXRD GUI backend."""

import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from phaseeval.workbench import composition_routes, dashboard, label_routes, page_routes, scorecard
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.labels import ExposureLedger
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import WorkbenchStore

from .auth import PrincipalAuthenticator
from .models import HealthResponse
from .routes import attempts, capabilities
from .state import app_state

# This repository ships the PhaseEval workbench only. The AgentXRD analysis pipeline's
# routes are not part of it.
pipeline_routers: tuple = ()

# Create FastAPI app
app = FastAPI(
    title="AgentXRD API",
    description="FastAPI backend for AgentXRD multiphase XRD analysis",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)


def configure_workbench(store: WorkbenchStore, authenticator: PrincipalAuthenticator) -> None:
    """Install the attempt-tree store and bearer authentication every /api/attempts route uses."""
    app.state.workbench_store = store
    app.state.attempt_authenticator = authenticator


def configure_workbench_from_environment(
    environment: Mapping[str, str] | None = None,
) -> None:
    """AXA_API_PRINCIPALS_FILE and AXA_WORKBENCH_ROOT, both absolute, configured together. The
    store executes refits live through the BGMN registry; the server runs in bgmn.slice
    (scripts/serve_workbench.sh) so every refit inherits the slice."""
    configured = os.environ if environment is None else environment
    principal_file = configured.get("AXA_API_PRINCIPALS_FILE")
    store_root = configured.get("AXA_WORKBENCH_ROOT")
    if not principal_file and not store_root:
        return
    if not principal_file or not store_root:
        raise ValueError("AXA_API_PRINCIPALS_FILE and AXA_WORKBENCH_ROOT must be configured together")
    root = Path(store_root)
    if not root.is_absolute():
        raise ValueError("AXA_WORKBENCH_ROOT must be absolute")
    store = WorkbenchStore(
        root,
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=Path(configured.get("AXA_WORKBENCH_IMPORT_ROOT", str(root / "imports"))),
        exposure=ExposureLedger(root),
    )
    configure_workbench(store, PrincipalAuthenticator.from_file(principal_file))


SECURITY_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
}


@app.middleware("http")
async def private_inspector_responses(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    if request.url.path == "/api/attempts" or request.url.path.startswith("/api/attempts/"):
        response.headers["Cache-Control"] = "no-store"
        vary = response.headers.get("Vary")
        response.headers["Vary"] = f"{vary}, Authorization" if vary else "Authorization"
    return response


app.include_router(page_routes.router)  # W3 view: /workbench and the review data routes
app.include_router(attempts.router)
for pipeline_router in pipeline_routers:
    app.include_router(pipeline_router)

# Workbench routers from other units (labels, page, chemistry) are included here, each with
# bearer auth on every route through attempts.authenticate_principal (R29). They may share the
# /api/attempts prefix: attempts.router matches digests only as 64 lowercase hex.
app.include_router(label_routes.router)  # W2 labels: snapshots, labels, reveal, step feedback
app.include_router(composition_routes.router)  # W5: typed composition input (N9)
app.include_router(dashboard.router)  # W6: /dashboard redirects to /scorecard; the admin-only dashboard data
app.include_router(scorecard.router)  # eval scorecard: /scorecard and the admin-only scorecard data
app.include_router(capabilities.router)  # what this server can run: refit needs BGMN


# Mount static files for frontend (if build exists)
FRONTEND_BUILD_DIR = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

if FRONTEND_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="static")


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        - status: "ok" if service is running
        - version: API version
        - pipeline_ready: Whether pipeline dependencies are available
        - num_uploaded_patterns: Number of patterns in memory
        - num_active_jobs: Number of running analysis jobs
    """
    stats = app_state.get_stats()

    # Check if pipeline dependencies are available
    pipeline_ready = False  # the analysis pipeline is not shipped

    return HealthResponse(
        status="ok",
        version="1.0.0",
        pipeline_ready=pipeline_ready,
        num_uploaded_patterns=stats["num_patterns"],
        num_active_jobs=stats["num_active_jobs"],
    )


# Root endpoint
@app.get("/api")
async def root():
    """API root endpoint."""
    return {
        "name": "AgentXRD API",
        "version": "1.0.0",
        "description": "FastAPI backend for multiphase XRD analysis",
        "endpoints": {
            "health": "/api/health",
            "upload": "/api/upload",
            "analyze": "/api/analyze",
            "results": "/api/results",
            "export": "/api/export",
            "phases": "/api/phases",
            "viewer": "/api/viewer",
            "diffract": "/api/diffract",
            "attempts": "/api/attempts",
            "websocket": "/api/ws/progress/{job_id}",
        },
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    if not hasattr(app.state, "workbench_store") and not hasattr(app.state, "attempt_authenticator"):
        configure_workbench_from_environment()


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Clean up any temp files
    pass


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    if request.url.path == "/api/attempts" or request.url.path.startswith("/api/attempts/"):
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "status_code": 500},
            headers={**SECURITY_HEADERS, "Cache-Control": "no-store", "Vary": "Authorization"},
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc), "status_code": 500},
        headers=dict(SECURITY_HEADERS),
    )


# 404 handler for API routes
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "detail": f"Endpoint {request.url.path} not found",
            "status_code": 404,
        },
    )
