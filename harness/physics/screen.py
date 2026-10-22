"""Screen candidate structures against the observed pattern before refining.

Default: rank every candidate by how much of its simulated Cu K-alpha1 line intensity lands on an
observed peak, list strong observed peaks that a greedy covering set does not explain, then run a
tree phase search (BGMN-backed) with the declared precursors pinned.
--residual LABEL: list the largest residual peaks of a saved refine.py run and the candidates with
lines near them.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from casedata import INPUT_DIR, OUTPUT_DIR, describe, load_candidates, load_case, reduced_formula

STAGE_ORDER = ("rietveld", "rietveld_profile", "rietveld_lattice", "pawley")
CELL_SCALES = np.round(np.linspace(-0.01, 0.01, 21), 4)


def two_theta_for(two_theta: np.ndarray, from_wavelength: float, to_wavelength: float) -> np.ndarray:
    ratio = np.clip(to_wavelength / from_wavelength * np.sin(np.radians(two_theta / 2)), -1, 1)
    return 2 * np.degrees(np.arcsin(ratio))


def d_spacing(two_theta: float, wavelength: float) -> float:
    return wavelength / (2 * np.sin(np.radians(two_theta / 2)))


def observed_peaks(two_theta: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Peak positions and heights above a rolling-minimum background, in percent of the strongest."""
    step = float(np.median(np.diff(two_theta)))
    window = max(3, int(round(3.0 / step)))
    background = uniform_filter1d(minimum_filter1d(counts, window), window)
    net = uniform_filter1d(np.clip(counts - background, 0, None), 3)
    index, _ = find_peaks(net, prominence=0.01 * net.max(), distance=max(1, int(0.08 / step)))
    relative = net[index] / net.max() * 100
    keep = relative >= 1.0
    return two_theta[index][keep], relative[keep]


def simulate(rows: list[dict], wavelength: float, two_theta_range: tuple[float, float]) -> dict[str, tuple]:
    """Simulated K-alpha1 lines (two_theta, intensity scaled to 100) per candidate, or the error."""
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core import Structure

    calculator = XRDCalculator(wavelength=wavelength)
    lines = {}
    for row in rows:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pattern = calculator.get_pattern(Structure.from_file(row["path"]), two_theta_range=two_theta_range)
            lines[row["candidate_id"]] = (np.asarray(pattern.x), np.asarray(pattern.y))
        except Exception as error:
            lines[row["candidate_id"]] = (np.array([]), np.array([]), f"{type(error).__name__}")
    return lines


def near(positions: np.ndarray, targets: np.ndarray, tolerance: float) -> np.ndarray:
    if positions.size == 0 or targets.size == 0:
        return np.zeros(positions.size, dtype=bool)
    return np.abs(positions[:, None] - targets[None, :]).min(axis=1) <= tolerance


def scaled(two_theta: np.ndarray, scale: float, wavelength: float) -> np.ndarray:
    """Line positions after multiplying every d-spacing by (1 + scale)."""
    ratio = np.clip(np.sin(np.radians(two_theta / 2)) / (1 + scale), -1, 1)
    return 2 * np.degrees(np.arcsin(ratio))


def match_lines(lines: tuple, peaks_tt: np.ndarray, peaks_rel: np.ndarray, ka1: float, ka2: float, tolerance: float) -> dict:
    """Line coverage at the isotropic cell scale (within +-1%) that puts the most line intensity on observed peaks;
    fom = coverage * sqrt(explained)."""
    x, y = lines[0], lines[1]
    if x.size == 0:
        return {"fom": 0.0, "coverage": 0.0, "explained": 0.0, "missed": "-", "scale": 0.0, "positions": x,
                "explains": np.zeros(peaks_tt.size, dtype=bool)}
    strong = y >= 5

    def coverage_at(scale: float) -> float:
        hit = near(scaled(x[strong], scale, ka1), peaks_tt, tolerance)
        return float(y[strong][hit].sum() / y[strong].sum()) if strong.any() else 0.0

    coverage, scale = max((coverage_at(s), -abs(s), s) for s in CELL_SCALES)[::2]
    x = scaled(x, scale, ka1)
    major = y >= 30
    missed = int((~near(x[major], peaks_tt, tolerance)).sum())
    visible = x[strong]
    explains = near(peaks_tt, np.concatenate([visible, two_theta_for(visible, ka1, ka2)]), tolerance)
    explained = float(peaks_rel[explains].sum() / peaks_rel.sum()) if peaks_rel.size else 0.0
    return {"fom": coverage * explained ** 0.5, "coverage": coverage, "explained": explained, "missed": f"{missed}/{int(major.sum())}", "scale": scale,
            "positions": x, "explains": explains}


def greedy_cover(matches_by_id: dict, by_id: dict, peaks_rel: np.ndarray, limit: int = 5) -> tuple[list[str], np.ndarray]:
    """Pick phases one formula at a time by not-yet-explained peak intensity times coverage, until the gain is small."""
    remaining = peaks_rel.copy()
    chosen: list[str] = []
    for _ in range(limit):
        formulas = {by_id[cid]["formula"] for cid in chosen}
        options = [(remaining[s["explains"]].sum() * s["coverage"], cid) for cid, s in matches_by_id.items()
                   if by_id[cid]["formula"] not in formulas]
        gain, best = max(options, default=(0.0, None))
        if best is None or gain < 0.02 * peaks_rel.sum():
            break
        chosen.append(best)
        remaining[matches_by_id[best]["explains"]] = 0
    return chosen, remaining > 0


@contextlib.contextmanager
def quiet(log_path: Path):
    """Send this process's stdout and stderr to a file (the search's workers are verbose)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with log_path.open("ab") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)


def phase_search(case: dict, rows: list[dict], matches_by_id: dict, two_theta, counts, wavelength: float) -> list[str]:
    precursors = [reduced_formula(p["formula"]) for p in (case.get("material_context") or {}).get("precursors") or []]
    pins = []
    for formula in precursors:
        same = [row for row in rows if row["formula"] == formula]
        if same:
            pins.append(max(same, key=lambda row: (matches_by_id[row["candidate_id"]]["fom"], matches_by_id[row["candidate_id"]]["explained"])))
    out = ["", f"Phase search: tree search with quick BGMN fits over all {len(rows)} candidates, at most 5 phases per "
           "hypothesis. Declared precursors are pinned in every hypothesis; for each, the pinned structure is the candidate "
           "of that formula with the highest fom above, and another structure of the same formula can still fit better."]
    log_path = Path(tempfile.gettempdir()) / "screen-search.log"
    os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
    # Pinned phases need express mode off: this search build fails on pins inside express-mode grouping.
    attempts = [(pins, False), ([], True)] if pins else [([], True)]
    result = None
    for pinned, express in attempts:
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as folder, quiet(log_path):
                logging.disable(logging.INFO)
                from axa.data.schema import XRDPattern
                from axa.refinement.dara_adapter import DaraAdapter

                paths = {}
                for row in rows:
                    paths[row["candidate_id"]] = Path(folder) / f"{row['candidate_id']}.cif"
                    shutil.copyfile(row["path"], paths[row["candidate_id"]])
                pinned_ids = {row["candidate_id"] for row in pinned}
                pattern = XRDPattern(sample_id=case["case_id"], two_theta=list(map(float, two_theta)),
                                     intensities=list(map(float, counts)), wavelength=wavelength)
                result = DaraAdapter().search(
                    pattern, cif_paths=[path for cid, path in paths.items() if cid not in pinned_ids], max_phases=5,
                    top_k=5, express_mode=express, pinned_phases=[paths[cid] for cid in pinned_ids] or None)
            out.append("Pinned: " + (", ".join(describe(row) for row in pinned) or "none") + ".")
            break
        except Exception as error:
            out.append(f"Search with {'pins' if pinned else 'no pins'} failed after {time.monotonic() - started:.0f} s: "
                       f"{type(error).__name__}: {str(error)[:200]} (log {log_path}).")
    if result is None:
        out.append("No search result. Use the ranking above.")
        return out
    by_id = {row["candidate_id"]: row for row in rows}
    out.append(f"{len(result.hypotheses)} hypotheses in {time.monotonic() - started:.0f} s. Rwp and weights come from "
               "the search's own quick fit, which may use a shortened 2theta range; confirm with refine.py.")
    for hypothesis in result.hypotheses:
        phases = []
        for phase in hypothesis.phases:
            cid = Path(phase.cif_path).stem
            weight = f" {phase.weight_fraction:.1f}%" if phase.weight_fraction is not None else ""
            phases.append(f"{describe(by_id[cid]) if cid in by_id else cid}{weight}")
        out.append(f"  H{hypothesis.rank}: Rwp {hypothesis.metrics.rwp:.2f}%, search fom {hypothesis.fom:.3f}: " + "; ".join(phases))
    return out


def match_all(args, case: dict, rows: list[dict]):
    measurement = case.get("measurement") or {}
    ka1 = float(measurement.get("wavelength_angstrom") or 1.540598)
    ka2 = float(measurement.get("kalpha2_wavelength_angstrom") or 1.544426)
    data = np.loadtxt(args.input_dir / case["pattern_file"])
    two_theta, counts = data[:, 0], data[:, 1]
    peaks_tt, peaks_rel = observed_peaks(two_theta, counts)
    lines = simulate(rows, ka1, (float(two_theta.min()), float(two_theta.max())))
    matches = {cid: match_lines(line, peaks_tt, peaks_rel, ka1, ka2, args.tolerance) for cid, line in lines.items()}
    return two_theta, counts, ka1, peaks_tt, peaks_rel, lines, matches


def screen(args, case: dict, rows: list[dict]) -> list[str]:
    two_theta, counts, ka1, peaks_tt, peaks_rel, lines, matches = match_all(args, case, rows)
    by_id = {row["candidate_id"]: row for row in rows}
    ranked = sorted(rows, key=lambda row: (-matches[row["candidate_id"]]["fom"], -matches[row["candidate_id"]]["explained"]))
    out = [f"{case['case_id']}: {peaks_tt.size} observed peaks above 1% of the strongest; match window +-{args.tolerance} deg 2theta.",
           "cov (coverage) = share of a candidate's simulated K-alpha1 line intensity (lines >= 5% of its strongest) that "
           "falls on an observed peak, at the cell scale s (all d-spacings times 1+s, |s| <= 1%) that maximizes it. expl "
           "(explained) = share of observed peak intensity those lines (K-alpha1 or K-alpha2) can account for. "
           "fom = cov * sqrt(expl). "
           "missed = its lines >= 30% with no observed peak / all such lines. A high fom says the lines are there, not that "
           "the phase is; overlapping and dense patterns rank high by chance, and weak minor phases rank low.",
           f"Ranking (top {min(args.top, len(rows))} of {len(rows)}; --top {len(rows)} lists all):"]
    for rank, row in enumerate(ranked[: args.top], start=1):
        m = matches[row["candidate_id"]]
        error = f" (could not simulate: {lines[row['candidate_id']][2]})" if len(lines[row["candidate_id"]]) == 3 else ""
        out.append(f"{rank}. {describe(row)}: fom {m['fom']:.2f} cov {m['coverage']:.2f} s {m['scale'] * 100:+.1f}% "
                   f"expl {m['explained'] * 100:.0f}% missed {m['missed']}{error}")
    chosen, unexplained = greedy_cover(matches, by_id, peaks_rel)
    out.append("Greedy covering set (one structure per formula, picked by not-yet-explained peak intensity times coverage): "
               + ("; ".join(describe(by_id[cid]) for cid in chosen) or "none"))
    strong = np.flatnonzero(unexplained & (peaks_rel >= 2))
    strong = strong[np.argsort(-peaks_rel[strong])][:12]
    out.append("Strong observed peaks (>= 2%) the covering set does not explain: "
               + (", ".join(f"{peaks_tt[i]:.2f} deg (d {d_spacing(peaks_tt[i], ka1):.3f} A, {peaks_rel[i]:.0f}%)"
                            for i in sorted(strong, key=lambda i: peaks_tt[i])) or "none"))
    if not args.no_search:
        out += phase_search(case, rows, matches, two_theta, counts, ka1)
    return out


def residual(args, case: dict, rows: list[dict]) -> list[str]:
    run = args.output_dir / args.residual
    in_fit = {row["candidate_id"] for row in json.loads((run / "request.json").read_text())["candidates"]}
    stage = next((name for name in STAGE_ORDER if (run / f"{name}.csv").is_file()), None)
    if stage is None:
        return [f"No stage curve (.csv) in {run}; the run has no finished stage."]
    *_, ka1, _, _, lines, matches = match_all(args, case, rows)
    two_theta, observed, calculated, difference = np.loadtxt(run / f"{stage}.csv", delimiter=",", skiprows=1).T
    step = float(np.median(np.diff(two_theta)))
    smooth = uniform_filter1d(difference, 5)
    noise = 1.4826 * float(np.median(np.abs(difference - np.median(difference))))
    top = float(observed.max())
    by_id = {row["candidate_id"]: row for row in rows}
    out = [f"Run {args.residual}, stage {stage}, phases in fit: {', '.join(sorted(in_fit))}. Residual = observed - calculated; "
           f"robust noise {noise:.1f} counts; heights in % of the highest observed count ({top:.0f}). Candidate lines are "
           f"placed at each candidate's best cell scale from the screen; window +-{args.tolerance} deg."]
    for sign, title in ((1, "Largest positive residual peaks (observed intensity the fit misses)"),
                        (-1, "Largest negative residual peaks (fit puts intensity where little is observed)")):
        index, props = find_peaks(sign * smooth, height=4 * noise, prominence=3 * noise, distance=max(1, int(0.1 / step)))
        order = index[np.argsort(-props["peak_heights"])][: args.peaks if sign > 0 else 5]
        out.append(f"{title}:" + ("" if order.size else " none above 4x noise"))
        for i in sorted(order, key=lambda i: two_theta[i]):
            text = f"  {two_theta[i]:.2f} deg (d {d_spacing(two_theta[i], ka1):.3f} A): {sign * smooth[i] / top * 100:.1f}%"
            if sign > 0:
                near_lines = []
                for cid, line in lines.items():
                    close = (np.abs(matches[cid]["positions"] - two_theta[i]) <= args.tolerance) & (line[1] >= 3)
                    if close.any():
                        near_lines.append((float(line[1][close].max()), cid))
                near_lines.sort(reverse=True)
                text += "; lines nearby: " + ("; ".join(
                    f"{describe(by_id[cid])}{' [in fit]' if cid in in_fit else ''} ({height:.0f}%)"
                    for height, cid in near_lines[:5]) or "none in the candidate set")
            out.append(text)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--case", required=True)
    parser.add_argument("--top", type=int, default=20, help="show the N best-ranked candidates (default 20)")
    parser.add_argument("--tolerance", type=float, default=0.15, help="match window in degrees 2theta")
    parser.add_argument("--no-search", action="store_true", help="skip the tree phase search (about 1-3 minutes)")
    parser.add_argument("--residual", metavar="RUN_LABEL", help="inspect the residual of a saved refine.py run")
    parser.add_argument("--peaks", type=int, default=10, help="residual peaks to list")
    args = parser.parse_args()
    if not (args.residual or args.no_search) and importlib.util.find_spec("axa") is None:
        sys.exit("screen.py: the tree phase search is not in this release. Rerun with --no-search.")
    case = load_case(args.case, args.input_dir)
    rows = load_candidates(args.case, args.input_dir)
    print("\n".join(residual(args, case, rows) if args.residual else screen(args, case, rows)))


if __name__ == "__main__":
    main()
