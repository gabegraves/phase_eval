"""What this server can run, for pages that offer an action only when it can (GET /api/capabilities).

Refit needs a BGMN install the executor can find (refinement.bgmn.installer: $AXA_BGMN_DIR, else
~/.axa/bgmn; the launcher downloads it there on first use). Without one the review page disables its refit buttons and shows `note`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from phaseeval.refinement.bgmn.installer import BGMNInstaller

from .attempts import Authenticated

BGMN_MISSING = (
    "Refit is off: no BGMN on this computer. Relaunch PhaseEval online to download it, "
    "or install Profex-BGMN (profex.doebelin.org)."
)

router = APIRouter(tags=["capabilities"])


@router.get("/api/capabilities")
def get_capabilities(principal: Authenticated) -> dict[str, Any]:
    available = BGMNInstaller().is_installed()
    return {"refit": {"available": available, "note": "" if available else BGMN_MISSING}}
