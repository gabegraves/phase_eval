from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from phaseeval.kernel.errors import (
    AudienceViolation,
    Busy,
    Conflict,
    Denied,
    DomainError,
    Refused,
    TooLarge,
    Unavailable,
)
from phaseeval.kernel.records import Record


class ErrorRow(Record):
    status: int
    message: str
    headers: tuple[str, ...] = ()


ERRORS: Mapping[type[DomainError], ErrorRow] = {
    Denied: ErrorRow(status=403, message=Denied.public_message),
    Refused: ErrorRow(status=422, message=Refused.public_message),
    Conflict: ErrorRow(status=409, message=Conflict.public_message),
    TooLarge: ErrorRow(status=413, message=TooLarge.public_message),
    Busy: ErrorRow(status=429, message=Busy.public_message, headers=("Retry-After",)),
    Unavailable: ErrorRow(status=503, message=Unavailable.public_message),
    AudienceViolation: ErrorRow(status=500, message=AudienceViolation.public_message),
}


def problem_response(error: DomainError) -> Any:
    from starlette.responses import JSONResponse
    row = ERRORS[type(error)]
    body = {"error": row.message}
    if isinstance(error, Refused):
        body["problems"] = [p.model_dump(mode="json") for p in error.problems]
    headers = {"Retry-After": str(error.retry_after_seconds)} if isinstance(error, Busy) else None
    return JSONResponse(body, status_code=row.status, headers=headers)


def install_error_handlers(app: Any) -> None:
    from fastapi.exceptions import RequestValidationError
    from starlette.responses import JSONResponse

    async def handle(request: Any, error: DomainError) -> Any:
        return problem_response(error)

    async def validation(request: Any, error: RequestValidationError) -> Any:
        problems = [{key: item[key] for key in ('loc', 'msg', 'type')} for item in error.errors()]
        return JSONResponse({'detail': problems}, status_code=422)

    app.add_exception_handler(DomainError, handle)
    app.add_exception_handler(RequestValidationError, validation)
