from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel


class DomainError(Exception):
    public_message: ClassVar[str]


class Denied(DomainError):
    public_message = "not found or not permitted"


class Problem(BaseModel):
    loc: tuple[str | int, ...]
    msg: str


class Refused(DomainError):
    public_message = "refused"

    def __init__(self, problems: tuple[Problem, ...]) -> None:
        super().__init__(problems)
        self.problems = problems


class Conflict(DomainError):
    public_message = "conflict"


class TooLarge(DomainError):
    public_message = "too large"


class Busy(DomainError):
    public_message = "busy, retry later"

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(retry_after_seconds)
        self.retry_after_seconds = retry_after_seconds


class Unavailable(DomainError):
    public_message = "unavailable"


class AudienceViolation(DomainError):
    public_message = "internal error"
