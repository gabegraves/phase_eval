"""Run the existing BGMN backend on a candidate set from the actor bundle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from phaseeval.refinement.bgmn.str_cache import StrCache
from phaseeval.refinement.config import RefinementConfig
from phaseeval.refinement.protocol import RefinementInput, calculate_stage_degradation_evidence
from casedata import cif_header, reduced_formula

CELL = ("a", "b", "c", "alpha", "beta", "gamma")
LST_GOAL = re.compile(r"^Q(?P<phase>\S+?)=(?P<value>[^+\s]+)\+-(?P<esd>\S+)\s*$", re.MULTILINE)


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
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def weight_esd(fit_dir: Path, stage: str) -> dict[str, float]:
    """BGMN's mass-fraction esd per phase from the stage .lst GOAL lines, in percent."""
    matches = sorted(fit_dir.glob(f"*_{stage}.lst"))
    if len(matches) != 1:
        return {}
    return {m["phase"]: float(m["esd"]) * 100 for m in LST_GOAL.finditer(matches[0].read_text(errors="replace"))}


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


def summary_lines(output_dir: Path, payload: dict, selected: list[dict], cifs: list[Path], evidence) -> list[str]:
    lines = [f"Run {output_dir.name}: {output_dir / 'result.json'}; curves {output_dir}/<stage>.csv "
             "(two_theta,observed,calculated,difference)"]
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
                error = esd.get(phase["phase_id"])
                weight = f"{number(phase.get('weight_fraction'))}" + (f" +- {error:.2f}" if error is not None else "") + " wt%"
            initial = phase.get("lattice_params_initial") or header["cell"]
            lines.append(f"  {phase['phase_id']} {reduced_formula(row.get('formula_sum', ''))} {sg}: {weight}; "
                         f"{cell_text(phase.get('lattice_params') or {}, initial)}")
    if evidence.comparison_available:
        flag = "true" if evidence.rietveld_degraded_after_good_pawley else "false"
        lines.append(f"rietveld_degraded_after_good_pawley: {flag} (Pawley Rwp {number(evidence.pawley_rwp)}%, Rietveld Rwp "
                     f"{number(evidence.rietveld_rwp)}%; true when Pawley Rwp <= 15 and Rietveld minus Pawley >= 10)")
        if evidence.rietveld_degraded_after_good_pawley:
            lines.append("  The peak-position fit is much better than the structure model: suspect a wrong phase, "
                         "polymorph or structure model.")
    else:
        lines.append("rietveld_degraded_after_good_pawley: not available (needs converged Pawley and Rietveld stages)")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/output"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--pawley-only", action="store_true")
    args = parser.parse_args()
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
        "profile_caveat": "Bundled AERIS instrument model; calibration for this individual scan is not established.",
        "quantity_basis": "Relative modeled crystalline weight fractions, percent. No absolute amorphous reference.",
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
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("\n".join(summary_lines(output_dir, payload, selected, cifs, evidence)))


if __name__ == "__main__":
    main()
