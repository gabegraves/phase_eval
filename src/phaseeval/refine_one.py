"""Refine one powder pattern against a fixed phase set with BGMN.

BGMN is not shipped. The first run downloads it with ``ensure_bgmn``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from phaseeval.data.xrdml import parse_xrdml, write_xy
from phaseeval.refinement.bgmn.backend import BGMNBackend
from phaseeval.refinement.bgmn.installer import BGMNInstaller, ensure_bgmn
from phaseeval.refinement.bgmn.str_cache import StrCache
from phaseeval.refinement.config import RefinementConfig
from phaseeval.refinement.protocol import RefinementInput, StageResult


def _number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _stage_summary(stage: StageResult) -> dict:
    return {
        "stage": stage.stage.value,
        "rwp_percent": _number(stage.rwp),
        "rexp_percent": _number(stage.rexp),
        "gof": _number(stage.gof),
        "converged": stage.converged,
        "elapsed_seconds": stage.elapsed_seconds,
        "phases": [
            {
                "phase_id": phase.phase_id,
                "weight_fraction_percent": _number(phase.weight_fraction),
                "lattice_params": phase.lattice_params,
            }
            for phase in stage.phases
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", type=Path, required=True, help=".xrdml or two-column .xy file")
    parser.add_argument("--cif", type=Path, nargs="+", required=True, help="one CIF per phase")
    parser.add_argument("--out", type=Path, required=True, help="new directory for this execution")
    parser.add_argument(
        "--bgmn-dir", type=Path, default=None, help="BGMN install directory (default ~/.axa/bgmn)"
    )
    parser.add_argument("--instrument-profile", default="Aeris-fds-Pixcel1d-Medipix3")
    parser.add_argument("--wavelength", default="Cu")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=False)
    fetch_start = time.monotonic()
    bgmn_dir = ensure_bgmn(install_dir=args.bgmn_dir)
    fetch_seconds = time.monotonic() - fetch_start

    pattern = args.pattern
    if pattern.suffix.lower() == ".xrdml":
        data = parse_xrdml(pattern)
        pattern = write_xy(args.out / f"{pattern.stem}.xy", data.two_theta, data.intensities)

    config = RefinementConfig(
        n_threads=1,
        working_dir=args.out,
        prefer_tmpfs=False,
        preserve_temp_files=True,
        enable_ml_pawley_bypass=False,
    )
    request = RefinementInput(
        pattern_path=pattern,
        cif_paths=list(args.cif),
        phase_ids=[cif.stem for cif in args.cif],
        working_dir=args.out,
        hypothesis_id="fit",
        instrument_profile=args.instrument_profile,
        wavelength=args.wavelength,
    )
    backend = BGMNBackend(
        installer=BGMNInstaller(install_dir=args.bgmn_dir),
        str_cache=StrCache(args.out / "str_cache"),
    )
    refine_start = time.monotonic()
    result = backend.refine(request, config)
    summary = {
        "bgmn_dir": str(bgmn_dir),
        "bgmn_fetch_seconds": fetch_seconds,
        "refine_wall_seconds": time.monotonic() - refine_start,
        "refinement_failed": result.refinement_failed,
        "failure_reason": result.failure_reason,
        "stages": [_stage_summary(stage) for stage in result.stages],
    }
    text = json.dumps(summary, indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(text + "\n")
    print(text)
    return 1 if result.refinement_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
