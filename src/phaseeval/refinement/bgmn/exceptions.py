"""
BGMN refinement exceptions.

Specialized exceptions for BGMN-specific failure modes that callers
can handle independently (e.g., skip a CIF, try next candidate).
"""

from __future__ import annotations

from pathlib import Path


class SymmetryDeterminationError(Exception):
    """Raised when spacegroup analysis fails for a CIF.

    This replaces the silent P1 (triclinic) fallback that previously
    caused over-parameterization, noise-fitting, and Rietveld divergence.

    Callers should skip this CIF and try the next candidate rather than
    proceeding with degraded symmetry.

    Attributes:
        cif_path: Path to the CIF that failed symmetry analysis.
        tolerances_tried: Tolerance levels attempted before giving up.
        original_error: The underlying pymatgen exception.
    """

    def __init__(
        self,
        cif_path: Path | None = None,
        tolerances_tried: list[float] | None = None,
        original_error: Exception | None = None,
    ):
        self.cif_path = cif_path
        self.tolerances_tried = tolerances_tried or []
        self.original_error = original_error
        tols = (
            ", ".join(f"{t}" for t in self.tolerances_tried) if self.tolerances_tried else "default"
        )
        msg = f"Spacegroup analysis failed for {cif_path} (tolerances tried: {tols})"
        if original_error:
            msg += f": {original_error}"
        super().__init__(msg)
