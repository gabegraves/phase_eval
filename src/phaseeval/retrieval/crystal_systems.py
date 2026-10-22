"""
Crystal system enumeration and space group mapping.

Defines the 7 crystal systems as an IntEnum with space group ranges,
supporting classification of crystal structures by their space group number.

The 7-class granularity is the base case for the CNN router. Future
ablations (Phase 7) may expand to 32-class Bravais lattice or 230-class
space group classification by changing num_classes config.

Space group ranges per ITA (International Tables for Crystallography):
    Triclinic:    1-2
    Monoclinic:   3-15
    Orthorhombic: 16-74
    Tetragonal:   75-142
    Trigonal:     143-167
    Hexagonal:    168-194
    Cubic:        195-230

References:
    - Kosmos report: CNN router achieves 98.57% on 7-class crystal system
    - DARA: Crystal system narrows tree search space
"""

from enum import IntEnum


class CrystalSystem(IntEnum):
    """Seven crystal systems mapped to integer labels (0-6).

    Used as classification targets for the CNN router model.
    Integer values enable direct use as tensor indices.
    """

    TRICLINIC = 0
    MONOCLINIC = 1
    ORTHORHOMBIC = 2
    TETRAGONAL = 3
    TRIGONAL = 4
    HEXAGONAL = 5
    CUBIC = 6


# Space group number ranges for each crystal system (inclusive)
CRYSTAL_SYSTEM_SG_RANGES: dict[CrystalSystem, tuple[int, int]] = {
    CrystalSystem.TRICLINIC: (1, 2),
    CrystalSystem.MONOCLINIC: (3, 15),
    CrystalSystem.ORTHORHOMBIC: (16, 74),
    CrystalSystem.TETRAGONAL: (75, 142),
    CrystalSystem.TRIGONAL: (143, 167),
    CrystalSystem.HEXAGONAL: (168, 194),
    CrystalSystem.CUBIC: (195, 230),
}

# Human-readable names for display and logging
CRYSTAL_SYSTEM_NAMES: dict[CrystalSystem, str] = {
    CrystalSystem.TRICLINIC: "Triclinic",
    CrystalSystem.MONOCLINIC: "Monoclinic",
    CrystalSystem.ORTHORHOMBIC: "Orthorhombic",
    CrystalSystem.TETRAGONAL: "Tetragonal",
    CrystalSystem.TRIGONAL: "Trigonal",
    CrystalSystem.HEXAGONAL: "Hexagonal",
    CrystalSystem.CUBIC: "Cubic",
}


def crystal_system_from_spacegroup(sg: int) -> CrystalSystem:
    """Map a space group number to its crystal system.

    Args:
        sg: Space group number (1-230).

    Returns:
        The corresponding CrystalSystem enum member.

    Raises:
        ValueError: If sg is outside the valid range [1, 230].

    Examples:
        >>> crystal_system_from_spacegroup(225)
        <CrystalSystem.CUBIC: 6>
        >>> crystal_system_from_spacegroup(14)
        <CrystalSystem.MONOCLINIC: 1>
        >>> crystal_system_from_spacegroup(1)
        <CrystalSystem.TRICLINIC: 0>
    """
    if not isinstance(sg, int) or sg < 1 or sg > 230:
        raise ValueError(f"Space group number must be an integer in [1, 230], got {sg}")

    for system, (sg_min, sg_max) in CRYSTAL_SYSTEM_SG_RANGES.items():
        if sg_min <= sg <= sg_max:
            return system

    # Should never reach here given valid ranges cover 1-230
    raise ValueError(f"No crystal system found for space group {sg}")


def crystal_system_matches(sg_no: int, predicted: CrystalSystem | str) -> bool:
    """Check if a spacegroup number belongs to the predicted crystal system.

    Args:
        sg_no: International Tables spacegroup number (1-230).
        predicted: Predicted crystal system (enum or capitalized name).

    Returns:
        True if the spacegroup falls within the predicted system's range.
    """
    if isinstance(predicted, str):
        # Convert capitalized name to enum
        name_to_enum = {v: k for k, v in CRYSTAL_SYSTEM_NAMES.items()}
        predicted_enum = name_to_enum.get(predicted)
        if predicted_enum is None:
            return False
    else:
        predicted_enum = predicted
    try:
        actual = crystal_system_from_spacegroup(sg_no)
    except ValueError:
        return False
    return actual == predicted_enum


def crystal_systems_for_label(label: int) -> list[int]:
    """Get the space group range for a crystal system label.

    Args:
        label: Crystal system integer label (0-6).

    Returns:
        List of space group numbers in this crystal system.

    Raises:
        ValueError: If label is outside [0, 6].
    """
    system = CrystalSystem(label)
    sg_min, sg_max = CRYSTAL_SYSTEM_SG_RANGES[system]
    return list(range(sg_min, sg_max + 1))
