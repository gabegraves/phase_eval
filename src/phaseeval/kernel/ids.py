from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, Field

SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
HEX64 = r"^[0-9a-f]{64}$"


class _LinkableMarker:
    pass


LINKABLE = _LinkableMarker()

SafeID = Annotated[str, Field(pattern=SAFE_ID)]
Digest = Annotated[str, Field(pattern=HEX64)]
AttemptRef = Annotated[str, Field(pattern=HEX64), LINKABLE]
BlobRef = Annotated[str, Field(pattern=HEX64), LINKABLE]
SnapshotRef = Annotated[str, Field(pattern=HEX64), LINKABLE]
CandidateId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
Element = Annotated[str, Field(pattern=r"^[A-Z][a-z]?$")]


Family = Annotated[str, Field(pattern=HEX64)]


Handle = Annotated[str, Field(pattern=r"^h[a-z2-7]{26}$")]


RunLabel = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


def _relative_name(value: str) -> str:
    parts = value.split("/")
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError("a relative path inside its folder")
    return value


RelativeName = Annotated[str, AfterValidator(_relative_name)]


class _FreeTextMarker:
    pass


FREE_TEXT = _FreeTextMarker()
FreeText = Annotated[str, Field(max_length=4000), FREE_TEXT]
