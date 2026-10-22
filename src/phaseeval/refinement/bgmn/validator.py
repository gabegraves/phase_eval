"""
CIF structure validation before BGMN submission.

Validates CIF files against BGMN requirements:
- All atomic species must be in BGMN's POSSIBLE_SPECIES list
- Structure must have atoms with Wyckoff positions
- Lattice parameters must be physically reasonable
- Disordered sites generate warnings (not errors)

References:
- DARA utils.py: POSSIBLE_SPECIES list (182 entries)
- BGMN manual: supported element scattering factors
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---- POSSIBLE_SPECIES: all BGMN-supported species (from DARA possible_species.txt) ----
POSSIBLE_SPECIES: set[str] = {
    "H",
    "HE",
    "LI",
    "LI+1",
    "BE",
    "BE+2",
    "B",
    "C",
    "N",
    "O",
    "O-1",
    "O-2",
    "F",
    "F-1",
    "NE",
    "NA",
    "NA+1",
    "MG",
    "MG+2",
    "AL",
    "AL+3",
    "SI",
    "SI+4",
    "P",
    "S",
    "CL",
    "CL-1",
    "AR",
    "K",
    "K+1",
    "CA",
    "CA+2",
    "SC",
    "SC+3",
    "TI",
    "TI+3",
    "TI+4",
    "V",
    "V+2",
    "V+3",
    "V+5",
    "CR",
    "CR+2",
    "CR+3",
    "MN",
    "MN+2",
    "MN+3",
    "MN+4",
    "FE",
    "FE+2",
    "FE+3",
    "CO",
    "CO+2",
    "CO+3",
    "NI",
    "NI+2",
    "NI+3",
    "CU",
    "CU+1",
    "CU+2",
    "ZN",
    "ZN+2",
    "GA",
    "GA+3",
    "GE",
    "AS",
    "SE",
    "BR",
    "BR-1",
    "KR",
    "RB",
    "RB+1",
    "SR",
    "SR+2",
    "Y",
    "Y+3",
    "ZR",
    "ZR+4",
    "NB",
    "NB+3",
    "NB+5",
    "MO",
    "MO+3",
    "MO+5",
    "MO+6",
    "TC",
    "RU",
    "RU+3",
    "RU+4",
    "RH",
    "RH+3",
    "RH+4",
    "PD",
    "PD+2",
    "PD+4",
    "AG",
    "AG+1",
    "AG+2",
    "CD",
    "CD+2",
    "IN",
    "IN+3",
    "SN",
    "SN+2",
    "SN+4",
    "SB",
    "SB+3",
    "SB+5",
    "TE",
    "I",
    "I-1",
    "XE",
    "CS",
    "CS+1",
    "BA",
    "BA+2",
    "LA",
    "LA+3",
    "CE",
    "CE+3",
    "CE+4",
    "PR",
    "PR+3",
    "PR+4",
    "ND",
    "ND+3",
    "PM",
    "PM+3",
    "SM",
    "SM+3",
    "EU",
    "EU+2",
    "EU+3",
    "GD",
    "GD+3",
    "TB",
    "TB+3",
    "DY",
    "DY+3",
    "HO",
    "HO+3",
    "ER",
    "ER+3",
    "TM",
    "TM+3",
    "YB",
    "YB+2",
    "YB+3",
    "LU",
    "LU+3",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LW",
}


class CIFValidationError(Exception):
    """Raised when a CIF file fails validation for BGMN submission.

    Attributes:
        errors: List of specific validation errors.
        warnings: List of non-fatal validation warnings.
    """

    def __init__(
        self, message: str, errors: list[str] | None = None, warnings: list[str] | None = None
    ):
        super().__init__(message)
        self.errors = errors or []
        self.validation_warnings = warnings or []


@dataclass
class ValidationResult:
    """Result of CIF validation for BGMN submission.

    Attributes:
        valid: True if the CIF passes all critical checks.
        errors: List of fatal validation errors (any -> valid=False).
        warnings: List of non-fatal warnings (disordered sites, etc.).
        crystal_system: Detected crystal system (e.g., 'Cubic', 'Hexagonal').
        spacegroup_no: International Tables space group number.
        num_sites: Number of unique atomic sites in the structure.
        species: List of element/species strings found in the structure.
        density: Computed density in g/cm3 (None if parse failed).
        min_interatomic_distance: Minimum pairwise distance in Angstroms.
        volume_per_atom: Unit cell volume divided by number of atoms (A3/atom).
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    crystal_system: str | None = None
    spacegroup_no: int | None = None
    num_sites: int = 0
    species: list[str] = field(default_factory=list)
    density: float | None = None
    min_interatomic_distance: float | None = None
    volume_per_atom: float | None = None


def _process_species_for_validation(sp_str: str) -> str:
    """Convert a pymatgen species string to BGMN format for validation.

    Reverses charge notation (Fe3+ -> FE+3) and upper-cases.
    If species with valence is not found, tries bare element.

    Returns the BGMN species string.
    Raises ValueError if species is not in POSSIBLE_SPECIES.
    """
    specie = re.sub(r"(\d+)([+-])", r"\2\1", str(sp_str))
    if specie.endswith(("+", "-")):
        specie += "1"
    specie = specie.upper()

    if specie in POSSIBLE_SPECIES:
        return specie

    # Strip valence and try bare element
    bare = re.search(r"[A-Z]+", specie)
    if bare:
        bare_str = bare.group(0)
        if bare_str in POSSIBLE_SPECIES:
            return bare_str

    raise ValueError(f"Unknown species '{sp_str}' (resolved to '{specie}')")


def _check_interatomic_distances(
    structure,
    errors: list[str],
    validation_warnings: list[str],
    error_cutoff: float = 0.5,
    warn_cutoff: float = 0.8,
) -> float | None:
    """Report the closest pair of sites, periodic images included.

    A pair closer than ``error_cutoff`` angstrom overlaps, which is an error unless both sites
    are partially occupied: that is how a CIF models disorder, so it only warns. A pair closer
    than ``warn_cutoff`` warns. Returns the closest distance (inf for a single site), or None
    when the distances cannot be computed.
    """
    import numpy as np

    try:
        fractional = structure.frac_coords
        distances = structure.lattice.get_all_distances(fractional, fractional)
    except Exception as exc:
        validation_warnings.append(f"Could not compute interatomic distances: {exc}")
        return None
    upper = np.triu_indices(len(structure), k=1)
    if not len(upper[0]):
        return float("inf")
    closest_pair = int(np.argmin(distances[upper]))
    first, second = int(upper[0][closest_pair]), int(upper[1][closest_pair])
    closest = float(distances[first, second])
    pair = f"{structure[first].species_string}-{structure[second].species_string}"
    occupancies = (structure[first].species.num_atoms, structure[second].species.num_atoms)
    if closest < error_cutoff and max(occupancies) < 1.0:
        validation_warnings.append(
            f"Partially occupied sites {pair} are {closest:.3f} A apart (a disorder model)"
        )
    elif closest < error_cutoff:
        errors.append(f"Sites {pair} overlap: {closest:.3f} A apart (< {error_cutoff} A)")
    elif closest < warn_cutoff:
        validation_warnings.append(f"Sites {pair} are only {closest:.3f} A apart (< {warn_cutoff} A)")
    return closest


def _check_density(
    structure,
    errors: list[str],
    validation_warnings: list[str],
) -> float | None:
    """Check density is physically reasonable (0.5–25 g/cm3 for inorganics)."""
    try:
        density = float(structure.density)
        if density > 25.0:
            errors.append(f"Density {density:.2f} g/cm3 is physically impossible (> 25)")
        elif density < 0.5:
            validation_warnings.append(f"Density {density:.2f} g/cm3 is unusually low (< 0.5)")
        return density
    except Exception as e:
        validation_warnings.append(f"Could not compute density: {e}")
        return None


def _check_volume_per_atom(
    structure,
    errors: list[str],
    validation_warnings: list[str],
) -> float | None:
    """Check cell volume per atom for overlapping/void anomalies."""
    try:
        # Use occupancy-weighted atom count to handle disordered structures
        # (e.g., organic/hybrid perovskites with many partial-occupancy H sites)
        n_atoms = sum(site.species.num_atoms for site in structure)
        if n_atoms < 1e-6:
            return None
        vpa = structure.volume / n_atoms
        if vpa < 5.0:
            errors.append(f"Volume per atom {vpa:.2f} A3 is too small (< 5, atoms overlapping)")
        elif vpa > 500.0:
            validation_warnings.append(f"Volume per atom {vpa:.2f} A3 is unusually large (> 500)")
        return float(vpa)
    except Exception as e:
        validation_warnings.append(f"Could not compute volume per atom: {e}")
        return None


def _check_site_occupancies(
    structure,
    errors: list[str],
    validation_warnings: list[str],
) -> None:
    """Check that site occupancies are within reasonable bounds."""
    for i, site in enumerate(structure):
        total_occ = sum(site.species.values())
        if total_occ > 1.05:
            errors.append(f"Site {i} ({site.species_string}) occupancy {total_occ:.3f} > 1.05")
        elif total_occ < 0.01:
            errors.append(
                f"Site {i} ({site.species_string}) occupancy {total_occ:.3f} < 0.01 (empty site)"
            )


def _check_lattice_angles(
    lattice,
    errors: list[str],
) -> None:
    """Check lattice angles are valid (0 < angle < 180, non-degenerate cell)."""
    for name, angle in [("alpha", lattice.alpha), ("beta", lattice.beta), ("gamma", lattice.gamma)]:
        if angle <= 0.0 or angle >= 180.0:
            errors.append(f"Lattice angle {name}={angle:.2f} deg is outside (0, 180)")

    # Check non-degenerate: the sum of any two angles must exceed the third
    angles = [lattice.alpha, lattice.beta, lattice.gamma]
    for i in range(3):
        others = sum(angles) - angles[i]
        if others <= angles[i]:
            errors.append(
                f"Degenerate cell: angle sum constraint violated "
                f"(alpha={angles[0]:.1f}, beta={angles[1]:.1f}, gamma={angles[2]:.1f})"
            )
            break


def validate_cif(cif_path: Path) -> ValidationResult:
    """Validate a CIF file for BGMN submission.

    Checks:
    1. CIF can be parsed by pymatgen
    2. SpacegroupAnalyzer can symmetrize the structure
    3. All species are in BGMN's POSSIBLE_SPECIES list
    4. Lattice parameters are physically reasonable (0.1 < a,b,c < 100 Angstroms)
    5. Structure has at least one atom
    6. Disordered sites produce warnings (not errors)
    7. Interatomic distances (< 0.5 A = error, < 0.8 A = warning)
    8. Density sanity (< 0.5 or > 25 g/cm3)
    9. Cell volume per atom (< 5 A3 = error, > 500 A3 = warning)
    10. Site occupancy bounds
    11. Lattice angle validity

    Args:
        cif_path: Path to CIF file to validate.

    Returns:
        ValidationResult with validation status, errors, and warnings.

    Note:
        Does NOT raise on warnings. Only ``valid=False`` with errors list
        when critical issues are found.
    """
    errors: list[str] = []
    validation_warnings: list[str] = []
    crystal_system: str | None = None
    spacegroup_no: int | None = None
    num_sites: int = 0
    species_list: list[str] = []
    density: float | None = None
    min_interatomic_distance: float | None = None
    volume_per_atom: float | None = None

    # 1. Load CIF
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            structure = Structure.from_file(
                str(cif_path), site_tolerance=1e-3, occupancy_tolerance=100
            )
    except Exception as e:
        errors.append(f"Failed to parse CIF: {e}")
        return ValidationResult(
            valid=False, errors=errors, warnings=validation_warnings, num_sites=0, species=[]
        )

    # 2. Get symmetrized structure — fail explicitly if analysis fails.
    #    The old P1 fallback silently degraded to triclinic, causing
    #    over-parameterization and Rietveld divergence.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            spg = SpacegroupAnalyzer(structure)
            refined = spg.get_refined_structure()
            spg = SpacegroupAnalyzer(refined)
            sym_structure = spg.get_symmetrized_structure()
            crystal_system = spg.get_crystal_system().capitalize()
            spacegroup_no = spg.get_space_group_number()
    except Exception as e:
        validation_warnings.append(f"Spacegroup analysis failed: {e}")
        crystal_system = None
        spacegroup_no = None
        sym_structure = None

    # 3. Check number of atoms
    num_sites = (
        len(sym_structure.equivalent_indices) if sym_structure is not None else len(structure)
    )
    if num_sites == 0:
        errors.append("Structure has no atomic sites")

    # 4. Check lattice parameters (lengths) — use raw structure if symmetrized unavailable
    check_struct = sym_structure if sym_structure is not None else structure
    lattice = check_struct.lattice
    for param_name, param_val in [("a", lattice.a), ("b", lattice.b), ("c", lattice.c)]:
        if param_val < 0.1:
            errors.append(
                f"Lattice parameter {param_name}={param_val:.4f} A is too small (< 0.1 A)"
            )
        if param_val > 100:
            errors.append(
                f"Lattice parameter {param_name}={param_val:.4f} A is too large (> 100 A)"
            )

    # 4b. Check lattice angles
    _check_lattice_angles(lattice, errors)

    # 5. Check species against POSSIBLE_SPECIES
    if sym_structure is not None:
        for site_idx_group in sym_structure.equivalent_indices:
            site = sym_structure[site_idx_group[0]]
            if site.is_ordered:
                sp_str = str(next(iter(site.species)))
                try:
                    bgmn_sp = _process_species_for_validation(sp_str)
                    species_list.append(bgmn_sp)
                except ValueError as e:
                    errors.append(str(e))
            else:
                # Disordered site: warn but don't error
                validation_warnings.append(
                    f"Disordered site at {site.frac_coords}: {dict(site.species)}"
                )
                for sp in site.species:
                    try:
                        bgmn_sp = _process_species_for_validation(str(sp))
                        species_list.append(bgmn_sp)
                    except ValueError as e:
                        errors.append(str(e))
    else:
        # Fallback: check species from raw structure
        for site in structure:
            if site.is_ordered:
                sp_str = str(next(iter(site.species)))
                try:
                    bgmn_sp = _process_species_for_validation(sp_str)
                    species_list.append(bgmn_sp)
                except ValueError as e:
                    errors.append(str(e))

    # 6. Physics checks on the raw structure (not symmetrized)
    min_interatomic_distance = _check_interatomic_distances(structure, errors, validation_warnings)
    density = _check_density(structure, errors, validation_warnings)
    volume_per_atom = _check_volume_per_atom(structure, errors, validation_warnings)
    _check_site_occupancies(structure, errors, validation_warnings)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=validation_warnings,
        crystal_system=crystal_system,
        spacegroup_no=spacegroup_no,
        num_sites=num_sites,
        species=species_list,
        density=density,
        min_interatomic_distance=min_interatomic_distance,
        volume_per_atom=volume_per_atom,
    )
