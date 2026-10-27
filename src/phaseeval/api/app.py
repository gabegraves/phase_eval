"""Main FastAPI application for AgentXRD GUI backend."""

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from phaseeval.workbench import composition_routes, dashboard, label_routes, page_routes, scorecard
from phaseeval.workbench.cli import Settings, open_store
from phaseeval.workbench.store import WorkbenchStore

from .auth import PrincipalAuthenticator
from .errors import install_error_handlers
from .models import HealthResponse
from .routes import attempts, capabilities
from .security import BodyLimit

SECURITY_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
}


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


# Health check endpoint
async def health_check():
    """Health check endpoint.

    Returns:
        - status: "ok" if service is running
        - version: API version
        - pipeline_ready: Whether pipeline dependencies are available
        - num_uploaded_patterns: Number of patterns in memory
        - num_active_jobs: Number of running analysis jobs
    """
    # Check if pipeline dependencies are available
    pipeline_ready = False  # the analysis pipeline is not shipped

    return HealthResponse(
        status="ok",
        version="1.0.0",
        pipeline_ready=pipeline_ready,
        num_uploaded_patterns=0,
        num_active_jobs=0,
    )


# Root endpoint
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


# Global exception handler
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


def create_app(
    settings: Settings | None = None,
    *,
    store: WorkbenchStore | None = None,
    authenticator: PrincipalAuthenticator | None = None,
    extra_routers: Sequence[APIRouter] = (),
) -> FastAPI:
    settings = settings or Settings.from_env(os.environ)
    if (store is None) != (authenticator is None):
        raise ValueError("store and authenticator must be supplied together")
    if store is None and settings.store is not None:
        authenticator = PrincipalAuthenticator.from_file(settings.principals)
        store = open_store(settings.store, settings.import_root or settings.store / "imports", settings=settings)
    app = FastAPI(
        title="AgentXRD API",
        description="FastAPI backend for AgentXRD multiphase XRD analysis",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.workbench_store = store
    app.state.attempt_authenticator = authenticator
    install_error_handlers(app)
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)
    app.middleware("http")(private_inspector_responses)
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(404, not_found_handler)
    for router in (*extra_routers, page_routes.router, attempts.router, label_routes.router,
                   composition_routes.router, dashboard.router, scorecard.router, capabilities.router):
        app.include_router(router)
    app.add_api_route("/api/health", health_check, response_model=HealthResponse)
    app.add_api_route("/api", root)
    frontend = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="static")
    return app
