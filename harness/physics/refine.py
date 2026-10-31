"""Run the existing BGMN backend on a candidate set from the actor bundle."""

import argparse
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from phaseeval.refinement.bgmn.backend import BGMNBackend
from phaseeval.refinement.bgmn.result_parser import NUMBER, displacement_text, grain_range_warning
from phaseeval.refinement.bgmn.str_cache import StrCache
from phaseeval.refinement.config import RefinementConfig, RietveldConfig
from phaseeval.refinement.protocol import (
    RefinementInput,
    calculate_stage_degradation_evidence,
    texture_dependence,
    texture_dependence_text,
)
from casedata import cif_header, reduced_formula

CELL = ("a", "b", "c", "alpha", "beta", "gamma")
PROFILE_CAVEAT = "Bundled AERIS instrument model; calibration for this individual scan is not established."
QUANTITY_BASIS = "Relative modeled crystalline weight fractions, percent. No absolute amorphous reference."
LST_GOAL = re.compile(r"^Q(?P<phase>\S+?)=(?P<value>[^+\s]+)\+-(?P<esd>\S+)\s*$", re.MULTILINE)
TEXTURE_ORDERS = (0, 2, 4, 6, 8, 10)
EPS1_BOUND = "0_-0.01^0.01"
EPS3_BOUND = "0_-0.01^0.01"
MAX_GRAIN_DIAMETER_UM = 1000.0


@dataclass(frozen=True)
class FitOptions:
    """What an agent or a scientist may choose about a fit. The default is the staged recipe."""

    texture_order: int = 4
    refine_zero: bool = False
    refine_transparency: bool = False
    grain_diameter_um: float | None = None

    def __post_init__(self) -> None:
        if self.texture_order not in TEXTURE_ORDERS:
            raise ValueError(f"texture_order must be one of {TEXTURE_ORDERS}")
        diameter = self.grain_diameter_um
        if diameter is not None and not (math.isfinite(diameter) and 0 < diameter <= MAX_GRAIN_DIAMETER_UM):
            raise ValueError(f"grain_diameter_um must be a number above 0 and at most {MAX_GRAIN_DIAMETER_UM:g}")

    def config_fields(self) -> dict[str, Any]:
        """The RefinementConfig fields these options set."""
        order = self.texture_order
        return {
            "rietveld": RietveldConfig(gewicht=f"SPHAR{order}" if order else "0_0"),
            "eps1": EPS1_BOUND if self.refine_zero else 0.0,
            "eps3": EPS3_BOUND if self.refine_transparency else None,
            "grain_diameter_um": self.grain_diameter_um,
        }


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def checked_file(directory: Path, name: str, expected_digest: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"File outside input directory or missing: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError(f"Input digest mismatch: {name}")
    return path


def number(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "n/a"


def weight_esd(fit_dir: Path, stage: str) -> dict[str, float | None]:
    """BGMN's mass-fraction esd per phase from the stage .lst GOAL lines, in percent. None when
    the line prints no finite number."""
    matches = sorted(fit_dir.glob(f"*_{stage}.lst"))
    if len(matches) != 1:
        return {}
    return {
        m["phase"]: float(m["esd"]) * 100 if re.fullmatch(NUMBER, m["esd"]) else None
        for m in LST_GOAL.finditer(matches[0].read_text(errors="replace"))
    }


def cell_text(refined: dict, initial: dict) -> str:
    parts = []
    for key in CELL:
        value, start = refined.get(key), initial.get(key)
        if not isinstance(value, (int, float)):
            continue
        if key not in "abc" and abs(value - 90) < 1e-6 and (start is None or abs(start - 90) < 1e-6):
            continue
        shift = f" ({(value - start) / start * 100:+.2f}%)" if isinstance(start, (int, float)) and start else ""
        parts.append(f"{key} {value:.5g}{shift}")
    return " ".join(parts)


def displacement_line(stage: dict, scan_two_theta_deg: tuple[float, float] | None, eps2_bound: str | None) -> str:
    eps2 = (stage.get("angle_corrections") or {}).get("eps2")
    if eps2 is None:
        return f"EPS2 at final stage {stage['stage']}: not listed (BGMN lists only refined parameters)."
    return displacement_text(eps2["value"], eps2["esd"], scan_two_theta_deg, eps2_bound)


def other_corrections_line(stage: dict) -> str | None:
    corrections = stage.get("angle_corrections") or {}
    refined = [(name, corrections.get(key)) for name, key in (("EPS1 (zero point)", "eps1"), ("EPS3 (transparency)", "eps3"))]
    parts = [f"{name} {number(c['value'], 6)}" + (f" +- {c['esd']:.6f}" if c["esd"] is not None else "")
             for name, c in refined if c is not None]
    return "; ".join(parts) + "." if parts else None


def texture_line(dependence: dict | None, texture_order: int, phase_count: int) -> str:
    if dependence is not None:
        return texture_dependence_text(dependence)
    if texture_order == 0:
        return "Texture dependence: not applicable, the final stage has no texture model (texture order 0)."
    if phase_count == 1:
        return "Texture dependence: not applicable, a single phase has no amount to move."
    return "Texture dependence: not available (needs a converged final Rietveld stage and its texture control fit)."


def summary_lines(
    output_dir: Path,
    payload: dict,
    selected: list[dict],
    cifs: list[Path],
    evidence,
    *,
    scan_two_theta_deg: tuple[float, float] | None = None,
    eps2_bound: str | None = None,
    texture_order: int = FitOptions.texture_order,
    grain_diameter_um: float | None = None,
) -> list[str]:
    lines = [f"Run {output_dir.name}: {output_dir / 'result.json'}; curves {output_dir}/<stage>.csv "
             "(two_theta,observed,calculated,difference)",
             f"Model assumptions: Cu radiation. {PROFILE_CAVEAT}",
             f"Quantity basis: {QUANTITY_BASIS}",
             "Printed +/- values are conditional fit estimates, not total uncertainty."]
    if payload["refinement_failed"]:
        lines.append(f"Refinement failed: {payload['failure_reason']}")
    stages = [payload[name] for name in ("pawley", "rietveld_lattice", "rietveld_profile", "rietveld") if payload.get(name)]
    for stage in stages:
        lines.append(f"Stage {stage['stage']}: Rwp {number(stage.get('rwp'))}% Rexp {number(stage.get('rexp'))}% "
                     f"GoF {number(stage.get('gof'))} converged {'yes' if stage.get('converged') else 'no'}")
    if stages:
        final = stages[-1]
        pawley = final["stage"] == "pawley"
        esd = weight_esd(output_dir / "fit", final["stage"])
        rows = {row["candidate_id"]: (row, cif) for row, cif in zip(selected, cifs)}
        lines.append(f"Phases at final stage {final['stage']} (cell shift in % from the CIF cell):")
        for phase in final.get("phases") or []:
            row, cif = rows.get(phase["phase_id"], ({}, None))
            header = cif_header(cif) if cif else {"sg_symbol": "?", "sg_number": None, "cell": {}}
            sg = f"{header['sg_symbol']} ({header['sg_number']})" if header["sg_number"] else header["sg_symbol"]
            if pawley:
                weight = "no weight fraction (Pawley)"
            else:
                error = f" +- {number(esd[phase['name']])}" if phase["name"] in esd else ""
                weight = f"{number(phase.get('weight_fraction'))}{error} wt%"
                my = phase.get("linear_attenuation_per_um")
                if grain_diameter_um is not None:
                    weight += f"; my*d {number(None if my is None else my * grain_diameter_um)}"
            initial = phase.get("lattice_params_initial") or header["cell"]
            lines.append(f"  {phase['phase_id']} {reduced_formula(row.get('formula_sum', ''))} {sg}: {weight}; "
                         f"{cell_text(phase.get('lattice_params') or {}, initial)}")
        if final.get("failure_reason"):
            lines.append(final["failure_reason"])
        if grain_diameter_um is not None and not pawley:
            my_d = {p["phase_id"]: p["linear_attenuation_per_um"] * grain_diameter_um
                    for p in final.get("phases") or [] if p.get("linear_attenuation_per_um") is not None}
            if (warning := grain_range_warning(my_d)) is not None:
                lines.append(warning)
        lines.append(displacement_line(final, scan_two_theta_deg, eps2_bound))
        if (others := other_corrections_line(final)) is not None:
            lines.append(others)
    lines.append(texture_line(payload.get("texture_dependence"), texture_order, len(selected)))
    if evidence.comparison_available:
        flag = "true" if evidence.rietveld_degraded_after_good_pawley else "false"
        lines.append(f"rietveld_degraded_after_good_pawley: {flag} (Pawley Rwp {number(evidence.pawley_rwp)}%, Rietveld Rwp "
                     f"{number(evidence.rietveld_rwp)}%; true when Pawley Rwp <= 15 and Rietveld minus Pawley >= 10)")
        if evidence.rietveld_degraded_after_good_pawley:
            lines.append("  The peak-position fit is much better than the structure model: suspect a wrong phase, "
                         "polymorph or structure model, or a specimen effect: preferred orientation beyond the "
                         "texture model, microabsorption or poor particle statistics.")
    else:
        lines.append("rietveld_degraded_after_good_pawley: not available (needs converged Pawley and Rietveld stages)")
    lines.append("Convergence and a false value for the degradation flag do not establish phase identity or completeness; "
                 "check residuals and competing hypotheses where informative.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--pawley-only", action="store_true")
    parser.add_argument("--texture-order", type=int, choices=TEXTURE_ORDERS, default=FitOptions.texture_order)
    parser.add_argument("--refine-zero", action="store_true")
    parser.add_argument("--refine-transparency", action="store_true")
    parser.add_argument("--grain-diameter-um", type=float)
    args = parser.parse_args()
    try:
        options = FitOptions(args.texture_order, args.refine_zero, args.refine_transparency, args.grain_diameter_um)
    except ValueError as error:
        parser.error(str(error))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_label):
        parser.error("run-label must be 1-64 characters from A-Z, a-z, 0-9, _ and -")
    if len(args.candidates) > 5 or len(set(args.candidates)) != len(args.candidates):
        parser.error("choose one to five distinct candidate IDs")

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    case = cases[args.case]
    pattern = checked_file(args.input_dir, case["pattern_file"], case["pattern_sha256"])
    library = args.input_dir / "candidates" / args.case
    entries = json.loads((library / "manifest.json").read_text())["candidates"]
    candidates = {entry["candidate_id"]: entry for entry in entries}
    selected = [candidates[identifier] for identifier in args.candidates]
    cifs = [checked_file(library, row["file"], row["sha256"]) for row in selected]
    output_dir = args.output_dir / args.run_label
    output_dir.mkdir(parents=True, exist_ok=False)

    config = RefinementConfig(
        n_threads=1,
        working_dir=output_dir,
        prefer_tmpfs=False,
        preserve_temp_files=True,
        enable_ml_pawley_bypass=False,
        pawley_only=args.pawley_only,
        **options.config_fields(),
    )
    request = RefinementInput(
        pattern_path=pattern,
        cif_paths=cifs,
        phase_ids=args.candidates,
        working_dir=output_dir,
        hypothesis_id="fit",
        instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
        wavelength="Cu",
    )
    receipt = {
        "case_id": args.case,
        "candidates": selected,
        "pattern_sha256": case["pattern_sha256"],
        "config": config.to_dict(),
        "profile_caveat": PROFILE_CAVEAT,
        "quantity_basis": QUANTITY_BASIS,
    }
    (output_dir / "request.json").write_text(json.dumps(json_value(receipt), indent=2) + "\n")
    start = time.monotonic()
    result = BGMNBackend(str_cache=StrCache(output_dir / "str_cache")).refine(request, config)
    payload = asdict(result)
    curve_files = []
    for stage in result.stages:
        if stage.dia_data is not None:
            arrays = stage.dia_data
            columns = [arrays[key] for key in ("two_theta", "observed", "calculated", "difference")]
            path = output_dir / f"{stage.stage.value}.csv"
            np.savetxt(path, np.column_stack(columns), delimiter=",", header="two_theta,observed,calculated,difference", comments="")
            curve_files.append(str(path))
    for stage_payload in payload.values():
        if isinstance(stage_payload, dict) and "dia_data" in stage_payload:
            arrays = stage_payload.pop("dia_data")
            if arrays is not None:
                stage_payload["curve_file"] = str(output_dir / f"{json_value(stage_payload['stage'])}.csv")
                stage_payload["curve_points"] = len(arrays["two_theta"])
    payload = json_value(payload)
    evidence = calculate_stage_degradation_evidence(result)
    payload["elapsed_seconds"] = time.monotonic() - start
    payload["curve_files"] = curve_files
    payload["stage_degradation"] = json_value(evidence.to_dict())
    dependence = texture_dependence(result)
    payload["texture_dependence"] = json_value(asdict(dependence)) if dependence is not None else None
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    two_theta = np.loadtxt(pattern, usecols=0)
    print("\n".join(summary_lines(output_dir, payload, selected, cifs, evidence,
                                  scan_two_theta_deg=(float(two_theta.min()), float(two_theta.max())),
                                  eps2_bound=config.eps2, texture_order=options.texture_order,
                                  grain_diameter_um=options.grain_diameter_um)))


if __name__ == "__main__":
    main()
