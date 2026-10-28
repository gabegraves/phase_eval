"""Read the case manifest, the candidate manifest and CIF headers for the physics tools."""

from __future__ import annotations

import json
from pathlib import Path
import re

from pymatgen.core import Composition

INPUT_DIR = Path("/workspace/input")
OUTPUT_DIR = Path("/workspace/output")

SYSTEMS = ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"),
           (167, "trigonal"), (194, "hexagonal"), (230, "cubic"))
SYMBOL_TAGS = ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M")
NUMBER_TAGS = ("_space_group_IT_number", "_symmetry_Int_Tables_number")
CELL = ("a", "b", "c", "alpha", "beta", "gamma")


def load_case(case_id: str, input_dir: Path = INPUT_DIR) -> dict:
    cases = {case["case_id"]: case for case in json.loads((input_dir / "manifest.json").read_text())["cases"]}
    return cases[case_id]


def load_candidates(case_id: str, input_dir: Path = INPUT_DIR) -> list[dict]:
    """Manifest rows, each with its CIF path and header facts added."""
    library = input_dir / "candidates" / case_id
    rows = json.loads((library / "manifest.json").read_text())["candidates"]
    for row in rows:
        row["path"] = library / row["file"]
        row.update(cif_header(row["path"]))
        row["formula"] = reduced_formula(row.get("formula_sum") or "")
    return rows


def reduced_formula(formula_sum: str) -> str:
    try:
        return Composition(formula_sum).reduced_formula
    except Exception:
        return formula_sum.replace(" ", "")


def _tag(text: str, tag: str) -> str | None:
    match = re.search(rf"^{re.escape(tag)}\s+(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip().strip("'\"").strip() if match else None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(re.sub(r"\(.*?\)", "", value))
    except ValueError:
        return None


def crystal_system(number: int | None) -> str:
    if number is None:
        return "unknown"
    return next(name for top, name in SYSTEMS if number <= top)


def cif_header(path: Path) -> dict:
    """Space group and cell as the CIF states them."""
    text = path.read_text(errors="replace")
    from pymatgen.symmetry.groups import SpaceGroup

    symbol = next((value for tag in SYMBOL_TAGS if (value := _tag(text, tag))), None)
    number = next((int(value) for tag in NUMBER_TAGS if (value := _number(_tag(text, tag)))), None)
    try:
        if number is None and symbol:
            number = SpaceGroup(symbol.replace(" ", "")).int_number
        if number is not None:
            symbol = re.sub(r"^([PCIABF])1(.+)1$", r"\1\2", SpaceGroup.from_int_number(number).symbol)
    except Exception:
        pass
    cell = {key: _number(_tag(text, f"_cell_length_{key}" if key in "abc" else f"_cell_angle_{key}")) for key in CELL}
    return {"sg_symbol": (symbol or "?").replace(" ", ""), "sg_number": number,
            "system": crystal_system(number), "cell": cell}


def describe(row: dict) -> str:
    number = f" ({row['sg_number']})" if row["sg_number"] else ""
    return f"{row['candidate_id']} {row['formula']} {row['sg_symbol']}{number}"


def actor_context(case: dict, manifest_sha256: str) -> dict:
    """Known scientific fields only, with actor-safe provenance checked before derivation."""
    from phaseeval.workbench.importer import SOURCE_CHANNELS

    scalar = None
    schema = {
        "case_id": scalar, "chemistry_elements": [scalar],
        "measurement": dict.fromkeys(("instrument", "detector", "wavelength_angstrom",
            "kalpha2_wavelength_angstrom", "kalpha2_kalpha1_ratio", "two_theta_start",
            "two_theta_end", "point_count", "profile_caveat")),
        "instrument_profile": {**dict.fromkeys(("profile", "declared_by_source", "resolver_confidence",
            "instrument_profile_confidence", "quantitative_claims")), "warnings": [scalar]},
        "material_context": {
            "specimen": scalar, "precursor_amounts": scalar, "element_evidence": [scalar],
            "precursors": [dict.fromkeys(("formula", "name", "cas_number", "purity", "actual_mass_g", "target_mass_g"))],
            "processing": [{**dict.fromkeys(("action", "equipment", "furnace", "temperature_c",
                "duration_min", "atmosphere", "ramp_rate_c_per_min", "met_target_mass",
                "mass_recovered_mg", "mass_for_xrd_mg")), "temperature_profile": {
                    "time_minutes": [scalar], "temperature_celsius": [scalar]}}],
            "declared_target": {"formula": scalar, "stoichiometry": scalar},
        },
    }
    sources = case.get("field_sources") or {}
    projected_sources = {}
    missing = object()

    def project(value, shape, path):
        parts = path.split(".")
        ancestors = [".".join(parts[:i]) for i in range(1, len(parts) + 1)]
        source = None
        for ancestor in ancestors:
            if ancestor not in sources:
                continue
            candidate = sources[ancestor]
            if (not isinstance(candidate, dict) or candidate.get("class") not in SOURCE_CHANNELS
                    or not isinstance(candidate.get("locator"), str)
                    or re.search(r"custody|case_labels|reference_answer|operator[_ -]note", candidate["locator"], re.I)):
                return missing
            source = candidate
        if path == "material_context.declared_target" and isinstance(value, str):
            shape = None
        if isinstance(shape, dict):
            if not isinstance(value, dict):
                return missing
            result = {}
            for key, child_shape in shape.items():
                if key in value:
                    child = project(value[key], child_shape, f"{path}.{key}" if path else key)
                    if child is not missing:
                        result[key] = child
            return result
        if isinstance(shape, list):
            if not isinstance(value, list):
                return missing
            result = [project(item, shape[0], f"{path}.{i}") for i, item in enumerate(value)]
            # A redacted array element stays null so its source indexes do not shift.
            return [None if item is missing else item for item in result]
        if not isinstance(value, (str, int, float, bool, type(None))):
            return missing
        projected_sources[path] = {"document": "manifest.json", "document_sha256": manifest_sha256,
            **({"class": source["class"], "locator": source["locator"], "source_status": "recorded"} if source else {
                "class": "bundled_manifest", "locator": path, "source_status": "unknown"})}
        return value

    result = project(case, schema, "")
    result["field_sources"] = projected_sources
    # ponytail: the bundle has no calibration schema; add sourced fields when one is defined.
    result["calibration"] = None
    return result
