"""
CIF-to-STR conversion for BGMN refinement.

Adapted from DARA's cif2str.py (378 lines) with axa-specific improvements:
- Integrated CIF validation before conversion
- Hash-based phase name collision avoidance
- All 7 crystal systems + Rhombohedral handled
- Lattice parameters always in nanometers (BGMN convention)

References:
- DARA cif2str.py: Original converter with Wyckoff handling
- DARA utils.py: load_symmetrized_structure, process_phase_name, standardize_coords
- BGMN manual: .str file format specification
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from asteval import Interpreter
from pymatgen.core.periodic_table import Element

from phaseeval.refinement.bgmn.validator import POSSIBLE_SPECIES

if TYPE_CHECKING:
    from pymatgen.core import Lattice
    from pymatgen.core.periodic_table import DummySpecie, Element, Specie
    from pymatgen.symmetry.structure import SymmetrizedStructure

logger = logging.getLogger(__name__)

# Path to the spacegroup database (bundled)
_SPG_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "spg.json"

# Heuristic element-specific defaults for B_iso (Å^2) when CIF lacks values.
_BISO_DEFAULTS = {
    "H": 1.5,
    "C": 1.2,
    "N": 1.2,
    "O": 1.2,
    "F": 1.0,
    "Cl": 1.0,
    "Br": 0.9,
    "I": 0.9,
    "Na": 1.0,
    "Mg": 0.9,
    "Al": 0.9,
    "Si": 0.9,
    "P": 0.9,
    "S": 0.9,
    "K": 0.8,
    "Ca": 0.8,
    "Ti": 0.7,
    "V": 0.7,
    "Cr": 0.7,
    "Mn": 0.7,
    "Fe": 0.7,
    "Co": 0.7,
    "Ni": 0.7,
    "Cu": 0.7,
    "Zn": 0.7,
    "Sn": 0.7,
    "Cs": 0.7,
    "Ba": 0.7,
    "Pb": 0.7,
}


class CIFToStrError(Exception):
    """Raised when CIF-to-STR conversion fails."""


# ---------------------------------------------------------------------------
# Utility functions (adapted from DARA utils.py)
# ---------------------------------------------------------------------------


def _standardize_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Adjust coordinates to exact fractional values if close.

    Handles common fractions (1/3, 2/3, 1/6, etc.) that get rounded
    in CIF files. Same logic as DARA's standardize_coords.
    """
    fractions = {
        0.3333: 1 / 3,
        0.6667: 2 / 3,
        0.1667: 1 / 6,
        0.8333: 5 / 6,
        0.0833: 1 / 12,
        0.4167: 5 / 12,
        0.5833: 7 / 12,
        0.9167: 11 / 12,
    }

    for key, value in fractions.items():
        if abs(x - key) < 0.0001:
            x = value
        if abs(y - key) < 0.0001:
            y = value
        if abs(z - key) < 0.0001:
            z = value

    return x, y, z


def _fuzzy_compare(a: float, b: float) -> bool:
    """Fuzzy comparison for fractional coordinates.

    Normalizes values to [0, 1) and checks if they match within tolerance.
    Handles special fractions explicitly. Same logic as DARA's fuzzy_compare.
    """
    fa = round(a, 6)
    fb = round(b, 6)

    while fa < 0.0:
        fa += 1.0
    while fb < 0.0:
        fb += 1.0
    while fa >= 1.0:
        fa -= 1.0
    while fb >= 1.0:
        fb -= 1.0

    # Check specific fractions
    fraction_ranges = [
        (0.3333, 0.3334),
        (0.6666, 0.6667),
        (0.1666, 0.1667),
        (0.8333, 0.8334),
        (0.0833, 0.0834),
        (0.4166, 0.4167),
        (0.5833, 0.5834),
        (0.9166, 0.9167),
    ]

    for lower, upper in fraction_ranges:
        if lower <= fa <= upper and lower <= fb <= upper:
            return True

    return abs(fa - fb) <= 1e-3


def _get_site_b_iso(site) -> float | None:
    """Extract B_iso (Å^2) from a pymatgen site, if present."""
    props = getattr(site, "properties", {}) or {}
    for key in ("B_iso_or_equiv", "B_iso", "Biso", "b_iso", "biso"):
        if key in props:
            try:
                return float(props[key])
            except (TypeError, ValueError):
                return None
    for key in ("U_iso_or_equiv", "U_iso", "Uiso", "u_iso", "uiso"):
        if key in props:
            try:
                u_iso = float(props[key])
            except (TypeError, ValueError):
                return None
            return 8.0 * math.pi * math.pi * u_iso
    return None


def _default_b_iso_for_element(symbol: str) -> float:
    if symbol in _BISO_DEFAULTS:
        return _BISO_DEFAULTS[symbol]
    try:
        z = Element(symbol).Z
    except Exception:
        return 1.0
    if z <= 10:
        return 1.2
    if z <= 20:
        return 1.0
    if z <= 40:
        return 0.8
    if z <= 60:
        return 0.7
    return 0.6


def _resolve_site_b_iso(site) -> float:
    b_iso = _get_site_b_iso(site)
    if b_iso is not None:
        return max(b_iso, 0.001)
    if site.is_ordered:
        symbol = getattr(site.specie, "symbol", str(site.specie))
        return _default_b_iso_for_element(symbol)
    total = 0.0
    for specie, occu in site.species.items():
        symbol = getattr(specie, "symbol", str(specie))
        total += float(occu) * _default_b_iso_for_element(symbol)
    return max(total, 0.001)


def _format_tds_param(b_iso: float) -> str:
    """Format B_iso in Angstrom^2 as BGMN TDS in nm^2.

    Input B_iso is Angstrom^2; BGMN TDS is nm^2, and reference dara/cif2str.py
    writes 0.01 nm^2 for B_iso = 1.0 Angstrom^2.
    """
    init = max(b_iso / 100.0, 0.00001)
    return f"{init:.6f}"


def _process_phase_name(phase_name: str) -> str:
    """Process phase name to remove special characters for BGMN.

    Same logic as DARA's process_phase_name.
    """
    import random

    processed = re.sub(r"[\s()_/\\+\u2013\-*.:]", "", phase_name)  # ":" is illegal in BGMN identifiers (AXA-tdpz.29.8)
    if processed.isdigit():
        processed = f"P{processed}"
    if not processed:
        processed = "P" + "".join(
            random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", k=5)
        )
    return processed


def _load_symmetrized_structure(cif_path: Path):
    """Load symmetrized structure from a CIF file.

    Adapted from DARA's load_symmetrized_structure. Tries two tolerance
    levels for robustness.

    Returns:
        (SymmetrizedStructure, SpacegroupAnalyzer)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            structure = SpacegroupAnalyzer(
                Structure.from_file(str(cif_path), site_tolerance=1e-3, occupancy_tolerance=100)
            ).get_refined_structure()
            spg = SpacegroupAnalyzer(structure)
            sym_structure = spg.get_symmetrized_structure()
            return sym_structure, spg
        except Exception:
            try:
                structure = Structure.from_file(
                    str(cif_path), site_tolerance=1e-2, occupancy_tolerance=100
                )
                structure = SpacegroupAnalyzer(structure).get_refined_structure()
                spg = SpacegroupAnalyzer(structure)
                sym_structure = spg.get_symmetrized_structure()
                return sym_structure, spg
            except Exception:
                # P1 fallback: construct a trivial SymmetrizedStructure
                import numpy as np
                from pymatgen.core.operations import SymmOp
                from pymatgen.symmetry.analyzer import SpacegroupOperations
                from pymatgen.symmetry.structure import SymmetrizedStructure

                structure = Structure.from_file(
                    str(cif_path), site_tolerance=1e-2, occupancy_tolerance=100
                )
                symmop = SymmOp.from_rotation_and_translation(np.eye(3), [0, 0, 0])
                spg_ops = SpacegroupOperations("P1", 1, [symmop])
                equiv = list(range(len(structure)))
                wyckoffs = ["a"] * len(structure)
                sym_structure = SymmetrizedStructure(structure, spg_ops, equiv, wyckoffs)
                logger.warning(
                    "Spacegroup analysis failed for %s; falling back to P1",
                    cif_path.name,
                )
                return sym_structure, None


# ---------------------------------------------------------------------------
# Species processing
# ---------------------------------------------------------------------------


def process_specie_string(sp: str | Specie | Element | DummySpecie) -> str:
    """Reverse charge notation of a species string for BGMN.

    Examples:
        Fe3+ -> FE+3
        O2-  -> O-2
        Na+  -> NA+1

    Raises:
        CIFToStrError: If species is not in POSSIBLE_SPECIES after fallback.
    """
    specie = re.sub(r"(\d+)([+-])", r"\2\1", str(sp))
    if specie.endswith(("+", "-")):
        specie += "1"
    specie = specie.upper()

    if specie not in POSSIBLE_SPECIES:
        # Remove valence and try bare element
        match = re.search(r"[A-Z]+", specie)
        if match:
            specie = match.group(0)
        if specie not in POSSIBLE_SPECIES:
            raise CIFToStrError(f"Unknown species {specie}, the original specie string is {sp}")
    return specie


# ---------------------------------------------------------------------------
# Lattice parameters (crystal system dependent, Angstroms -> nanometers)
# ---------------------------------------------------------------------------


def get_lattice_parameters_from_lattice(
    lattice: Lattice,
    crystal_system: str,
) -> dict[str, float]:
    """Get lattice parameters for BGMN based on crystal system.

    CRITICAL: Returns lengths in nanometers (pymatgen uses Angstroms).
    Division by 10 converts A -> nm.

    Angles remain in degrees (BGMN convention).

    Crystal system constraints (per BGMN/DARA):
    - Cubic: only A (B=C=A, all angles 90)
    - Tetragonal: A and C (B=A, all angles 90)
    - Orthorhombic: A, B, C (all angles 90)
    - Hexagonal/Trigonal: A and C (B=A, gamma=120)
    - Monoclinic: A, B, C, BETA
    - Triclinic: all 6 parameters
    - Rhombohedral: A and GAMMA (alpha)
    """
    if crystal_system == "Triclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "ALPHA": lattice.alpha,
            "BETA": lattice.beta,
            "GAMMA": lattice.gamma,
        }
    if crystal_system == "Monoclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "BETA": lattice.beta,
        }
    if crystal_system == "Orthorhombic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Tetragonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Rhombohedral":
        return {
            "A": lattice.a / 10,
            "GAMMA": lattice.alpha,
        }
    if crystal_system in ("Hexagonal", "Trigonal"):
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Cubic":
        return {
            "A": lattice.a / 10,
        }

    raise CIFToStrError(f"Unknown crystal system: {crystal_system}")


# ---------------------------------------------------------------------------
# Wyckoff position handling
# ---------------------------------------------------------------------------


def _get_std_position(
    spacegroup_setting: dict[str, Any],
    wyckoff_letter: str,
    positions: list[list[float]],
) -> tuple[list[float], bool]:
    """Get the standard position of a site based on Wyckoff letter.

    Uses asteval to evaluate Wyckoff notation expressions and match
    against actual positions.
    """
    wyckoff = spacegroup_setting["wyckoffs"].get(wyckoff_letter, {})
    if not wyckoff:
        raise CIFToStrError(f"Cannot find the wyckoff letter {wyckoff_letter}")

    std_notations = wyckoff["std_notations"]
    positions = [list(_standardize_coords(*pos)) for pos in positions]

    for position in positions:
        variable_dict = {"x": position[0], "y": position[1], "z": position[2]}
        for std_notation in std_notations:
            constraints = std_notation.split(" ")
            aeval = Interpreter(use_numpy=False, symtable=variable_dict)
            wx, wy, wz = (aeval.eval(constraint) for constraint in constraints)
            if (
                _fuzzy_compare(wx, position[0])
                and _fuzzy_compare(wy, position[1])
                and _fuzzy_compare(wz, position[2])
            ):
                return position, True

    logger.debug(
        f"Cannot find standard position for {wyckoff_letter}, using first position. "
        f"Positions: {positions}"
    )
    return positions[0], False


def _check_wyckoff(
    spacegroup_setting: dict[str, Any],
    structure: SymmetrizedStructure,
) -> tuple[list[dict[str, Any]], int]:
    """Check Wyckoff positions for all sites in structure.

    Returns:
        (element_settings, error_count) where element_settings is a list of
        dicts with keys E, Wyckoff, x, y, z, TDS.
    """
    element_settings: list[dict[str, Any]] = []
    error_count = 0

    for site_idx in structure.equivalent_indices:
        idx = site_idx[0]
        site = structure[idx]
        wyckoff_letter = structure.wyckoff_letters[idx]
        if wyckoff_letter == "A":
            wyckoff_letter = "alpha"

        std_position, ok = _get_std_position(
            spacegroup_setting,
            wyckoff_letter,
            [structure[i].frac_coords.tolist() for i in site_idx],
        )

        if not ok:
            error_count += 1

        if site.is_ordered:
            species_string = process_specie_string(str(next(iter(site.species))))
        else:
            sorted_species = sorted(site.species)
            species_string = ",".join(
                f"{process_specie_string(str(ssp))}({site.species[ssp]:.6f})"
                for ssp in sorted_species
            )
            species_string = f"({species_string})"

        tds = _format_tds_param(_resolve_site_b_iso(site))
        element_setting = {
            "E": species_string,
            "Wyckoff": wyckoff_letter,
            "x": f"{std_position[0]:.6f}",
            "y": f"{std_position[1]:.6f}",
            "z": f"{std_position[2]:.6f}",
            "TDS": tds,
        }
        element_settings.append(element_setting)

    return element_settings, error_count


# ---------------------------------------------------------------------------
# STR file construction helpers
# ---------------------------------------------------------------------------


def _make_spacegroup_setting_str(spacegroup_setting: dict[str, Any]) -> str:
    """Format spacegroup setting line for .str file."""
    return " ".join(f"{k}={v}" for k, v in spacegroup_setting["setting"].items()) + " //"


def _make_lattice_parameters_str(
    spacegroup_setting: dict[str, Any],
    structure: SymmetrizedStructure,
    lattice_range: float | str,
    lattice_shift: float = 0.0,
) -> str:
    """Format lattice parameters line for .str file.

    If lattice_range is 'fixed', parameters are exact values.
    Otherwise, PARAM= format with +/- lattice_range fraction.
    """
    crystal_system = spacegroup_setting["setting"]["Lattice"]
    lattice_params = get_lattice_parameters_from_lattice(structure.lattice, crystal_system)
    shift = 0.0
    try:
        parsed = float(lattice_shift)
        if math.isfinite(parsed):
            shift = max(-0.05, min(0.05, parsed))
    except (TypeError, ValueError):
        shift = 0.0

    if shift != 0.0:
        lattice_params = {
            key: (value * (1.0 + shift) if key in {"A", "B", "C"} else value)
            for key, value in lattice_params.items()
        }

    if lattice_range == "fixed":
        parts = [f"{k}={v:.5f}" for k, v in lattice_params.items()]
    else:
        parts = [
            f"PARAM={k}={v:.5f}_{v * (1 - lattice_range):.5f}^{v * (1 + lattice_range):.5f}"
            for k, v in lattice_params.items()
        ]
    return " ".join(parts) + " //"


def _make_peak_parameter_str(k1: str, k2: str, b1: str, gewicht: str, rp: int) -> str:
    """Format peak parameters line for .str file."""
    return (
        f"RP={rp} "
        + (f"PARAM=k1={k1} " if k1 != "fixed" else "k1=0 ")
        + (f"PARAM=k2={k2} " if k2 != "fixed" else "k2=0 ")
        + (f"PARAM=B1={b1} " if b1 != "fixed" else "B1=0 ")
        + (f"GEWICHT={gewicht} //" if gewicht != "0_0" else "PARAM=GEWICHT=0_0 //")
    )


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------


def cif_to_str(
    cif_path: Path,
    output_dir: Path,
    phase_name: str = "",
    *,
    lebail: bool = True,
    lattice_range: float = 0.1,
    gewicht: str = "0_0",
    rp: int = 4,
    k1: str = "0_0^0.01",
    k2: str = "0_0^0.01",
    b1: str = "0_0^0.01",
    lattice_shift: float = 0.0,
    **phase_params: Any,
) -> Path:
    """Convert a CIF file to BGMN .str format.

    Handles all 7 crystal systems with correct BGMN lattice parameter
    constraints. Lattice parameters are automatically converted from
    pymatgen Angstroms to BGMN nanometers (divide by 10).

    Args:
        cif_path: Path to the input CIF file.
        output_dir: Directory for the output .str file.
        phase_name: Optional phase name override. If empty, derived from filename.
        lebail: If True, set GEWICHT=0_0 for Le Bail / Pawley mode (default True).
        lattice_range: Fractional range for lattice parameter refinement (default 0.1 = +/-10%).
            Use "fixed" for fixed lattice parameters.
        gewicht: Weight fraction parameter. Options: '0_0', 'SPHAR0', 'SPHAR2', 'SPHAR4'.
        rp: Peak function number (default 4).
        k1: First peak shape parameter (default '0_0^0.01').
        k2: Second peak shape parameter (default '0_0^0.01').
        b1: Broadening parameter (default '0_0^0.01').
        lattice_shift: Fractional shift applied to lattice lengths (A/B/C)
            before bounds are generated (clamped to +/-5%).
        **phase_params: Additional overrides for lattice_range, k1, k2, b1, rp, gewicht.

    Returns:
        Path to the generated .str file.

    Raises:
        CIFToStrError: If conversion fails (bad CIF, unknown species, etc.).
    """
    cif_path = Path(cif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply phase_params overrides
    lattice_range = phase_params.pop("lattice_range", lattice_range)
    gewicht = phase_params.pop("gewicht", gewicht)
    rp = phase_params.pop("rp", rp)
    k1 = phase_params.pop("k1", k1)
    k2 = phase_params.pop("k2", k2)
    b1 = phase_params.pop("b1", b1)

    # Force Le Bail mode
    if lebail:
        gewicht = "0_0"

    # Load symmetrized structure
    try:
        structure, spg = _load_symmetrized_structure(cif_path)
    except Exception as e:
        raise CIFToStrError(f"Failed to load CIF: {cif_path}: {e}") from e

    # Load spacegroup database
    if not _SPG_DB_PATH.exists():
        raise CIFToStrError(
            f"Spacegroup database not found at {_SPG_DB_PATH}. "
            "Ensure the bundled data directory is intact."
        )

    with _SPG_DB_PATH.open("r", encoding="utf-8") as f:
        spg_group_db = json.load(f)

    if spg is not None:
        try:
            hall_number = str(spg.get_symmetry_dataset().hall_number)
        except Exception as e:
            from phaseeval.refinement.bgmn.exceptions import SymmetryDeterminationError

            raise SymmetryDeterminationError(
                cif_path=cif_path,
                tolerances_tried=[],
                original_error=e,
            ) from e
    else:
        # P1 fallback when SpacegroupAnalyzer is unavailable
        hall_number = "1"
    if hall_number not in spg_group_db:
        raise CIFToStrError(f"Hall number {hall_number} not found in spacegroup database")

    settings = spg_group_db[hall_number]["settings"]

    # Find best Wyckoff setting
    best_setting = None
    for spacegroup_setting in settings:
        element_settings, error_count = _check_wyckoff(spacegroup_setting, structure)
        if best_setting is None or error_count < best_setting[2]:
            best_setting = (spacegroup_setting, element_settings, error_count)
        if error_count == 0:
            break

    if best_setting is None:
        raise CIFToStrError(f"No spacegroup settings found for {cif_path}")

    spacegroup_setting, element_settings, error_count = best_setting

    if error_count > 0:
        raise CIFToStrError(f"Cannot find a valid lattice symmetry setting for {cif_path}.")

    # Build phase name
    if not phase_name:
        phase_name = cif_path.stem

    phase_name = _process_phase_name(phase_name)

    # Overwrite any existing .str in output_dir (may be from a prior stage).
    # Never mangle the phase name — it must match the original CIF stem
    # for downstream phase_id resolution.
    str_path = output_dir / f"{phase_name}.str"

    # Construct .str file content
    lines: list[str] = []

    # Metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"PHASE={phase_name} // generated by axa {timestamp}")
    formula = structure.composition.reduced_formula
    lines.append(f"FORMULA={formula} //")

    # Spacegroup setting
    lines.append(_make_spacegroup_setting_str(spacegroup_setting))

    # Lattice parameters (in nm)
    lines.append(
        _make_lattice_parameters_str(
            spacegroup_setting,
            structure,
            lattice_range,
            lattice_shift=lattice_shift,
        )
    )

    # Peak parameters
    lines.append(_make_peak_parameter_str(k1, k2, b1, gewicht, rp))

    # Le Bail flag
    if lebail:
        lines.append("LeBail=1")

    # Goals
    lines.append(f"GOAL:{phase_name}=GEWICHT*ifthenelse(ifdef(d),exp(my*d*3/4),1) //")
    lines.append("GOAL=GrainSize(1,1,1) //")

    # Wyckoff positions
    for elem_setting in element_settings:
        parts = [f"{k}={v}" for k, v in elem_setting.items()]
        lines.append(" ".join(parts))

    str_text = "\n".join(lines)

    with open(str_path, "w") as f:
        f.write(str_text)

    return str_path
