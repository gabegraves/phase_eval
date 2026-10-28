"""Short view of the case manifest: chemistry, specimen, precursors and their molar ratios, processing, declared target."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pymatgen.core import Composition

from casedata import INPUT_DIR, load_case, actor_context

SKIP_RATIO_ELEMENTS = {"O", "H", "C", "N"}


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "not recorded"
    return str(value)


def precursor_lines(precursors: list[dict]) -> list[str]:
    lines, moles = [], {}
    for row in precursors:
        if not isinstance(row, dict) or not row.get("formula"):
            continue
        mass = row.get("actual_mass_g", row.get("target_mass_g"))
        parts = [row["formula"]]
        if row.get("name"):
            parts.append(f"({row['name']})")
        if row.get("purity") is not None:
            parts.append(f"purity {row['purity']:.5g}")
        if mass is not None:
            mol = mass / Composition(row["formula"]).weight
            moles[row["formula"]] = mol
            weighed = f"weighed {fmt(row.get('actual_mass_g'))} g" if "actual_mass_g" in row else ""
            target = f"target {fmt(row.get('target_mass_g'))} g" if "target_mass_g" in row else ""
            parts.append(", ".join(p for p in (weighed, target) if p) + f" = {mol * 1000:.4f} mmol")
        lines.append("  " + " ".join(parts))
    if len(moles) >= 2:
        low = min(moles.values())
        lines.append("  Precursor molar ratio: " + " : ".join(f"{formula} {mol / low:.3f}" for formula, mol in moles.items()))
        elements: dict[str, float] = {}
        for formula, mol in moles.items():
            for element, count in Composition(formula).get_el_amt_dict().items():
                if element not in SKIP_RATIO_ELEMENTS:
                    elements[element] = elements.get(element, 0.0) + count * mol
        if len(elements) >= 2:
            first = next(iter(elements.values()))
            lines.append("  Element molar ratio from these masses (O, H, C, N left out): "
                         + " : ".join(f"{element} {mol / first:.3f}" for element, mol in elements.items()))
    return lines


def trace_summary(profile: dict) -> str:
    times, temps = profile.get("time_minutes") or [], profile.get("temperature_celsius") or []
    if (not times or len(times) != len(temps)
            or any(type(value) not in (int, float) for value in [*times, *temps])):
        return "furnace trace present but unreadable"
    peak = max(range(len(temps)), key=temps.__getitem__)
    return (f"furnace trace {len(temps)} points: start {temps[0]:g} C at {times[0]:.1f} min, "
            f"peak {temps[peak]:g} C at {times[peak]:.1f} min, end {temps[-1]:g} C at {times[-1]:.1f} min")


def step_line(index: int, step: dict) -> str:
    parts = [step.get("action", "step")]
    if "temperature_c" in step or "duration_min" in step:
        parts.append(f"{fmt(step.get('temperature_c'))} C for {fmt(step.get('duration_min'))} min")
    if "atmosphere" in step:
        parts.append(f"atmosphere {fmt(step['atmosphere'])}")
    for key, value in step.items():
        if key in ("action", "temperature_c", "duration_min", "atmosphere", "temperature_profile"):
            continue
        if not isinstance(value, (dict, list)):
            parts.append(f"{key} {fmt(value)}")
    if isinstance(step.get("temperature_profile"), dict):
        parts.append(trace_summary(step["temperature_profile"]))
    return f"  {index}. " + "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    case = actor_context(load_case(args.case, args.input_dir),
                         hashlib.sha256((args.input_dir / "manifest.json").read_bytes()).hexdigest())
    measurement = case.get("measurement") or {}
    profile = case.get("instrument_profile") or {}
    context = dict(case.get("material_context") or {})

    print(f"Case {case['case_id']}")
    print("Elements: " + " ".join(case.get("chemistry_elements") or []))
    print(f"Measurement: {measurement.get('instrument', '?')}, {measurement.get('detector', '?')}; "
          f"K-alpha1 {fmt(measurement.get('wavelength_angstrom'))} A, K-alpha2 {fmt(measurement.get('kalpha2_wavelength_angstrom'))} A, "
          f"ratio {fmt(measurement.get('kalpha2_kalpha1_ratio'))}; 2theta {fmt(measurement.get('two_theta_start'))} to "
          f"{fmt(measurement.get('two_theta_end'))} deg, {fmt(measurement.get('point_count'))} points")
    print(f"Instrument profile: {profile.get('profile', '?')} (confidence {profile.get('instrument_profile_confidence', '?')}); "
          f"quantities: {profile.get('quantitative_claims', '?')}")
    if measurement.get("profile_caveat"):
        print(f"Profile caveat: {measurement['profile_caveat']}")
    print("Scan-specific calibration: not recorded")
    print("Full actor context and field sources: read_evidence(handle=\"context\")")
    print(f"Specimen: {fmt(context.pop('specimen', None))}")
    precursors = context.pop("precursors", None) or []
    print("Precursors:" + ("" if precursors else " none recorded"))
    for line in precursor_lines(precursors):
        print(line)
    if "precursor_amounts" in context:
        print(f"  Precursor amounts: {str(context.pop('precursor_amounts')).split(':')[0]}")
    steps = context.pop("processing", None) or []
    print("Processing:" + ("" if steps else " none recorded"))
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            print(step_line(index, step))
    target = context.pop("declared_target", None)
    if isinstance(target, dict):
        print("Declared target: " + ", ".join(f"{key} {fmt(value)}" for key, value in target.items()))
    elif target is not None:
        print(f"Declared target: {fmt(target)}")
    context.pop("element_evidence", None)
    for key, value in context.items():
        print(f"{key}: {fmt(value) if not isinstance(value, (dict, list)) else value}")


if __name__ == "__main__":
    main()
