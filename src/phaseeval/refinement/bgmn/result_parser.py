"""
Parse BGMN .lst and .dia result files into StageResult objects.

Extracts global R-factors (Rwp, Rp, Rpb, Rexp, rho), per-phase lattice
parameters, weight fractions, and spacegroup data from BGMN .lst output.
Lattice parameters are converted from nanometers (BGMN convention) to
Angstroms (*10) for our internal representation.

Key parsing patterns (from DARA result.py reference):
- Global R-factors: Rp=X.XX%, Rpb=X.XX%, Rwp=X.XX%, Rexp=X.XX%
- Background correlation: 1-rho=X.X%
- Iteration count: N iteration steps
- Phase sections: Local parameters and GOALs for phase NAME
- Per-phase values: KEY=value+-error
- GEWICHT: weight fraction (may have phase name prefix)
- Lattice: A, B, C in nm; ALPHA, BETA, GAMMA in degrees
- Rpb=100% indicates false convergence

Reference: dara-main/src/dara/result.py
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phaseeval.refinement.protocol import (
    PhaseRefinementResult,
    RefinementStage,
    StageResult,
)
from phaseeval.retrieval.crystal_systems import crystal_system_from_spacegroup

logger = logging.getLogger(__name__)


class BGMNParseError(Exception):
    """Raised when parsing a BGMN result file fails."""


# Measured on DEV006 with the lattice fixed, and Connolly's -114.59 s cos(theta) / R: EPS2 is s/R.
TWO_THETA_DEG_PER_EPS2 = -114.59
AT_BOUND_FRACTION = 0.98
_BOUND_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_BOUND = re.compile(rf"_({_BOUND_NUMBER})\^({_BOUND_NUMBER})$")


@dataclass(frozen=True)
class ValueEsd:
    """A refined value and its esd as the .lst printed them. NaN: printed without a usable number."""

    value: float
    esd: float | None


@dataclass(frozen=True)
class AngleCorrections:
    """BGMN's refined angle corrections for one stage: zero point, sample displacement s/R and
    transparency. None means not refined, because BGMN does not list a fixed parameter."""

    eps1: ValueEsd | None = None
    eps2: ValueEsd | None = None
    eps3: ValueEsd | None = None


def two_theta_shift_deg(eps2: float, two_theta_deg: float) -> float:
    """The peak shift EPS2 implies at this angle, in degrees 2theta. Positive EPS2 lowers the angle."""
    return TWO_THETA_DEG_PER_EPS2 * eps2 * math.cos(math.radians(two_theta_deg / 2))


def eps2_at_bound(value: float, bound: str | None) -> bool | None:
    """Whether |EPS2| is within 2% of its limit. None when the bound is not a low^high range."""
    match = _BOUND.search(bound) if bound else None
    if match is None:
        return None
    limit = float(match[2]) if value >= 0 else float(match[1])
    return abs(value) >= AT_BOUND_FRACTION * abs(limit)


def displacement_text(
    value: float | None,
    esd: float | None,
    scan_two_theta_deg: tuple[float, float] | None,
    bound: str | None,
) -> str:
    """One line on the refined EPS2: the shift at both scan ends and whether it hit its bound."""
    if value is None:
        return "EPS2 (sample displacement s/R) is listed without a usable value."
    error = f" +- {esd:.6f}" if esd is not None else ""
    text = f"EPS2 (sample displacement s/R) {value:.6f}{error}"
    if scan_two_theta_deg is not None:
        low, high = scan_two_theta_deg
        text += (
            f": shifts peaks {two_theta_shift_deg(value, low):+.3f} deg 2theta at {low:.2f} deg"
            f" and {two_theta_shift_deg(value, high):+.3f} deg at {high:.2f} deg"
        )
    text += "."
    saturated = eps2_at_bound(value, bound)
    if saturated is None:
        return text + " Its limit is not recorded, so the bound was not checked."
    if saturated:
        text += (
            f" WARNING: EPS2 is at its bound ({bound}). The displacement model is saturated: "
            "position error it could not absorb stays in the cell and profile parameters."
        )
    return text


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _complete_lattice_params(
    params: dict[str, float],
    *,
    lattice_system: str | None = None,
    spacegroup_no: int | None = None,
    unique_axis: str | None = None,
    initial: dict[str, float] | None = None,
) -> dict[str, float]:
    """Fill omitted cell parameters using BGMN's setting and fixed input values.

    Explicit observations always win, including zero or non-finite values.
    BGMN uses A+C for hexagonal axes and A+GAMMA for rhombohedral axes;
    a trigonal space-group number alone does not distinguish those settings.
    """
    lattice = params.copy()
    initial = initial or {}
    system = (lattice_system or "").lower()
    if not system and spacegroup_no is not None:
        try:
            system = crystal_system_from_spacegroup(spacegroup_no).name.lower()
        except ValueError:
            pass

    if system == "trigonal":
        if lattice_system is not None:
            system = "hexagonal"  # Explicit BGMN Lattice=Trigonal uses A+C.
        elif "gamma" in params and "c" not in params:
            system = "rhombohedral"
        elif "gamma" in initial and initial.get("alpha") == initial.get("beta") == initial["gamma"]:
            system = "rhombohedral"
        elif "c" in params or initial.get("gamma") == 120.0:
            system = "hexagonal"

    if "a" not in lattice and "a" in initial:
        lattice["a"] = initial["a"]
    if "a" in lattice:
        if system in ("cubic", "tetragonal", "hexagonal", "trigonal", "rhombohedral"):
            lattice.setdefault("b", lattice["a"])
        if system in ("cubic", "rhombohedral"):
            lattice.setdefault("c", lattice["a"])

    if system == "rhombohedral":
        angle = lattice.get("gamma", initial.get("gamma"))
        if angle is not None:
            for key in ("alpha", "beta", "gamma"):
                lattice.setdefault(key, angle)
    elif system in ("cubic", "tetragonal", "orthorhombic", "hexagonal"):
        lattice.setdefault("alpha", 90.0)
        lattice.setdefault("beta", 90.0)
        lattice.setdefault("gamma", 120.0 if system == "hexagonal" else 90.0)
    elif system == "monoclinic":
        angles = {"a": "alpha", "b": "beta", "c": "gamma"}
        if unique_axis is None:
            unique_axis = next(
                (axis for axis, key in angles.items() if key in params or initial.get(key, 90.0) != 90.0),
                "b",
            )
        for axis, key in angles.items():
            if axis != unique_axis.lower():
                lattice.setdefault(key, 90.0)

    for key, value in initial.items():
        lattice.setdefault(key, value)
    return lattice


def _parse_value(raw: str) -> float | tuple[float, float] | int | str | None:
    """Parse a single value string from BGMN .lst output.

    Handles:
    - Percentage values: "4.14%" -> 4.14
    - Values with errors: "0.5898+-0.0013" -> (0.5898, 0.0013)
    - Integers: "166" -> 166
    - Error/undefined markers: "ERROR", "UNDEF" -> None
    - Plain strings returned as-is

    Args:
        raw: Raw value string from .lst file.

    Returns:
        Parsed value as float, tuple, int, str, or None.
    """
    raw = raw.strip()
    raw_clean = raw.rstrip("%")

    if raw_clean in ("ERROR", "UNDEF"):
        return None

    try:
        if "+-" in raw_clean:
            center, uncertainty = raw_clean.split("+-", maxsplit=1)
            # Preserve a parseable center when BGMN truncates the uncertainty.
            # Admission rejects the resulting non-finite error instead of
            # silently carrying the center into the next refinement stage.
            return (float(center), float(uncertainty) if uncertainty else float("nan"))
        val = float(raw_clean)
        if val.is_integer() and "." not in raw_clean:
            return int(val)
        return val
    except ValueError:
        return raw


def _get_number(val: float | tuple[float, float] | int | str | None) -> float:
    """Extract a plain float from a parsed value.

    Tuples (value, error) -> value. Ints and floats pass through.
    None -> 0.0.
    """
    if val is None:
        return 0.0
    if isinstance(val, tuple):
        return float(val[0])
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_section(text: str) -> dict[str, Any]:
    """Parse key=value pairs from a BGMN text section.

    Args:
        text: Multi-line text block from .lst file.

    Returns:
        Dict of parsed key-value pairs.
    """
    pairs = re.findall(r"^(\w+)=(.+?)$", text, re.MULTILINE)
    return {k: _parse_value(v) for k, v in pairs}


def _angle_correction(section: dict[str, Any], key: str) -> ValueEsd | None:
    if key not in section:
        return None
    raw = section[key]
    if isinstance(raw, tuple):
        return ValueEsd(value=raw[0], esd=raw[1])
    if isinstance(raw, (int, float)):
        return ValueEsd(value=float(raw), esd=None)
    return ValueEsd(value=math.nan, esd=None)


def _extract_gewicht(section: dict[str, Any]) -> float | None:
    """Extract weight fraction from a phase section.

    BGMN GEWICHT may be prefixed with phase name, e.g.:
    'GEWICHT=NiO0.3827+-0.0049' or 'GEWICHT=0.3827+-0.0049'
    Texture-enabled stages can also report a model token plus the refined
    mean value, e.g. 'GEWICHT=SPHAR2, MeanValue(GEWICHT)=1.46506'.

    Returns the numeric value, or None when the section prints no number for it.
    """
    raw = section.get("GEWICHT")
    if raw is None:
        return None
    if isinstance(raw, str):
        mean_match = re.search(
            r"MeanValue\(GEWICHT\)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
            raw,
        )
        if mean_match:
            return float(mean_match.group(1))

        prefixed_match = re.search(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*\+-",
            raw,
        )
        if prefixed_match:
            return float(prefixed_match.group(1))
        return None

    return _get_number(raw)


def _grain_corrected_percent(
    phases: dict[str, dict[str, Any]], grain_diameter_um: float
) -> tuple[dict[str, float] | None, str | None]:
    """Mass percent per phase from GEWICHT x exp(my d 3/4), normalized over the phases.

    BGMN rounds a GOAL to its esd, so a Q GOAL can print 0+-14 for a phase with a real amount.
    Returns (None, reason) when any phase lacks a usable my or GEWICHT.
    """
    unusable = [
        name
        for name, pdata in phases.items()
        if not (0 < pdata.get("my", math.nan) < math.inf)
        or pdata.get("gewicht") is None
        or not (0 <= pdata["gewicht"] < math.inf)
    ]
    if unusable:
        return None, (
            f"Amounts not available: with grain diameter {grain_diameter_um} um each needs my and "
            f"GEWICHT, and the .lst prints no usable pair for {', '.join(unusable)}."
        )
    exponents = {name: pdata["my"] * grain_diameter_um * 3 / 4 for name, pdata in phases.items()}
    # Relative to the largest exponent, so that a large d cannot overflow exp.
    largest = max(exponents.values(), default=0.0)
    corrected = {
        name: pdata["gewicht"] * math.exp(exponents[name] - largest) for name, pdata in phases.items()
    }
    total = sum(corrected.values())
    if phases and not total > 0:
        return None, (
            f"Amounts not available: with grain diameter {grain_diameter_um} um the corrected "
            "GEWICHT values sum to 0."
        )
    return {name: value / total * 100.0 for name, value in corrected.items()}, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_lst(lst_path: Path) -> dict[str, Any]:
    """Parse a BGMN .lst output file.

    Extracts global R-factors and per-phase refinement results.
    Lattice parameters are converted from nm to Angstroms (*10).

    Args:
        lst_path: Path to the .lst file.

    Returns:
        Dict with keys:
        - rwp, rp, rpb, rexp, rho: Global R-factors (float, %)
        - iteration_count: Number of refinement iterations
        - angle_corrections: AngleCorrections from the global section
        - phases: Dict mapping phase_name -> {
            spacegroup_no, lattice_params {a,b,c,alpha,beta,gamma},
            weight_fraction, ...
          }

    Raises:
        BGMNParseError: If the .lst file cannot be parsed.
        FileNotFoundError: If the .lst file does not exist.
    """
    if not lst_path.exists():
        raise FileNotFoundError(f".lst file not found: {lst_path}")
    return parse_lst_text(lst_path.read_text())


def parse_lst_text(text: str) -> dict[str, Any]:
    """parse_lst over the file's text."""
    result: dict[str, Any] = {"raw_lst": text}

    # Iteration count
    m = re.search(r"(\d+) iteration steps?", text)
    if m:
        result["iteration_count"] = int(m.group(1))
    else:
        result["iteration_count"] = 0

    # Global R-factors
    for var in ("Rp", "Rpb", "Rwp", "Rexp"):
        m = re.search(rf"{var}=(\d+(?:\.\d+)?)%", text)
        result[var.lower()] = float(m.group(1)) if m else 0.0

    # R (unweighted) -- optional
    m = re.search(r"(?<!\w)R=(\d+(?:\.\d+)?)%", text)
    result["r"] = float(m.group(1)) if m else 0.0

    # Durbin-Watson d -- optional
    m = re.search(r"Durbin-Watson d=(\d+(?:\.\d+)?)", text)
    result["d"] = float(m.group(1)) if m else 0.0

    # Background correlation (1-rho)
    m = re.search(r"1-rho=(\d+(?:\.\d+)?)%", text)
    result["rho"] = float(m.group(1)) if m else 0.0

    global_section = _parse_section(text.split("Local parameters and GOALs for phase", 1)[0])
    result["angle_corrections"] = AngleCorrections(
        eps1=_angle_correction(global_section, "EPS1"),
        eps2=_angle_correction(global_section, "EPS2"),
        eps3=_angle_correction(global_section, "EPS3"),
    )

    # Extract phase names from STRUC or Local parameters sections
    phase_names_from_sections = re.findall(r"Local parameters and GOALs for phase (\S+)", text)

    # Parse per-phase sections
    phase_sections = re.findall(
        r"Local parameters and GOALs for phase \S+\n\*+\n(.*?)(?=\n\n|\nLocal parameters|\Z)",
        text,
        re.DOTALL,
    )

    phases: dict[str, dict[str, Any]] = {}
    for phase_name, section_text in zip(phase_names_from_sections, phase_sections, strict=False):
        section = _parse_section(section_text)
        phase_data: dict[str, Any] = {}

        # Spacegroup
        phase_data["spacegroup_no"] = (
            int(_get_number(section.get("SpacegroupNo")))
            if section.get("SpacegroupNo") is not None
            else None
        )
        phase_data["hermann_mauguin"] = section.get("HermannMauguin")
        phase_data["lattice_system"] = section.get("Lattice")
        phase_data["unique_axis"] = section.get("UniqueAxis")

        # Lattice parameters: convert nm -> Angstroms for a, b, c
        for param in ("A", "B", "C"):
            val = section.get(param)
            if param in section and val is None:
                phase_data[param.lower()] = float("nan")
            elif val is not None:
                # *10 to convert nm to Angstroms
                if isinstance(val, tuple):
                    phase_data[param.lower()] = val[0] * 10.0
                    phase_data[f"{param.lower()}_err"] = val[1] * 10.0
                else:
                    phase_data[param.lower()] = _get_number(val) * 10.0

        # Angular lattice parameters: degrees (no conversion)
        for param in ("ALPHA", "BETA", "GAMMA"):
            val = section.get(param)
            if param in section and val is None:
                phase_data[param.lower()] = float("nan")
            elif val is not None:
                if isinstance(val, tuple):
                    phase_data[param.lower()] = val[0]
                    phase_data[f"{param.lower()}_err"] = val[1]
                else:
                    phase_data[param.lower()] = _get_number(val)

        # Weight fraction
        phase_data["gewicht"] = _extract_gewicht(section)
        phase_data["weight_fraction"] = phase_data["gewicht"] or 0.0

        # Linear attenuation, printed when the .str asks for GOAL=my
        my = section.get("my")
        if my is not None:
            phase_data["my"] = _get_number(my)

        # Rphase (per-phase R-factor)
        rphase = section.get("Rphase")
        if rphase is not None:
            phase_data["rphase"] = _get_number(rphase)

        # X-ray density
        xrd = section.get("XrayDensity")
        if xrd is not None:
            phase_data["xray_density"] = _get_number(xrd)

        # Store full parsed section for downstream use
        phase_data["_raw_section"] = section

        phases[phase_name] = phase_data

    result["phases"] = phases
    return result


def parse_dia(dia_path: Path) -> dict[str, Any] | None:
    """Parse a BGMN .dia file for calculated/observed/difference patterns.

    The .dia file format:
    - First line: header (column names)
    - Subsequent lines: [2theta, iObs, iCalc, iBkgr, struct1, struct2, ...]

    Args:
        dia_path: Path to the .dia file.

    Returns:
        Dict with numpy arrays for 'two_theta', 'observed', 'calculated',
        'difference', 'background'. Returns None if file does not exist.
    """
    if not dia_path.exists():
        return None

    try:
        text = dia_path.read_text()
        lines = text.strip().split("\n")

        if len(lines) < 2:
            return None

        # Skip header line (first line)
        data = np.loadtxt(lines[1:])

        if data.ndim != 2 or data.shape[1] < 4:
            logger.error(
                "Unexpected .dia format: shape=%s, expected at least 4 columns",
                data.shape,
            )
            return None

        two_theta = data[:, 0]
        observed = data[:, 1]
        calculated = data[:, 2]
        background = data[:, 3]
        difference = observed - calculated

        return {
            "two_theta": two_theta,
            "observed": observed,
            "calculated": calculated,
            "difference": difference,
            "background": background,
        }
    except Exception as e:
        logger.error("Failed to parse .dia file %s: %s", dia_path, e)
        return None


def to_stage_result(
    lst_data: dict[str, Any],
    dia_data: dict[str, Any] | None,
    stage: RefinementStage,
    initial_lattice_params: dict[str, dict[str, float]],
    elapsed: float,
    *,
    grain_diameter_um: float | None = None,
) -> StageResult:
    """Convert parsed BGMN data into a StageResult.

    Maps per-phase data to PhaseRefinementResult objects and computes
    convergence status.

    Args:
        lst_data: Parsed .lst data from parse_lst().
        dia_data: Parsed .dia data from parse_dia() (may be None).
        stage: Which refinement stage produced this result.
        initial_lattice_params: Dict mapping phase_name -> {a,b,c,alpha,beta,gamma}
            in Angstroms/degrees (from CIF, for FoM calculation downstream).
        elapsed: Wall-clock time in seconds for this stage.
        grain_diameter_um: The fit's d, when set. GEWICHT omits the factor exp(my*d*3/4), so the
            weights apply it. A phase without a usable my or GEWICHT leaves every weight None and
            says so in failure_reason.

    Returns:
        StageResult with all metrics and per-phase results.
    """
    rwp = lst_data.get("rwp", 0.0)
    rp = lst_data.get("rp", 0.0)
    rpb = lst_data.get("rpb", 0.0)
    rexp = lst_data.get("rexp", 0.0)
    rho = lst_data.get("rho", 0.0)
    durbin_watson = lst_data.get("d")
    iteration_count = lst_data.get("iteration_count", 0)

    # Goodness of fit
    gof = rwp / rexp if rexp > 0 else 0.0

    # Convergence: Rpb=100 means false convergence per research
    converged = rpb < 100.0

    # Normalize weight fractions to 0-100%
    phases_data = lst_data.get("phases", {})
    total_weight = sum(p.get("weight_fraction", 0.0) for p in phases_data.values())
    grain_percent: dict[str, float] | None = None
    failure_reason: str | None = None
    if grain_diameter_um is not None:
        grain_percent, failure_reason = _grain_corrected_percent(phases_data, grain_diameter_um)
        if failure_reason is not None:
            logger.warning(failure_reason)

    phase_results: list[PhaseRefinementResult] = []
    for phase_name, pdata in phases_data.items():
        raw_weight = pdata.get("weight_fraction", 0.0)
        normalized_weight: float | None = (raw_weight / total_weight * 100.0) if total_weight > 0 else 0.0
        if grain_diameter_um is not None:
            normalized_weight = grain_percent[phase_name] if grain_percent is not None else None

        # Build lattice params and uncertainty dicts. Carry-forward admission
        # needs both; discarding ``*_err`` here can turn an unconstrained BGMN
        # result into an apparently valid lattice center.
        lattice = {}
        lattice_errors = {}
        for key in ("a", "b", "c", "alpha", "beta", "gamma"):
            if key in pdata:
                lattice[key] = pdata[key]
            error_key = f"{key}_err"
            if error_key in pdata:
                lattice_errors[key] = pdata[error_key]

        lattice = _complete_lattice_params(
            lattice,
            lattice_system=pdata.get("lattice_system"),
            spacegroup_no=pdata.get("spacegroup_no"),
            unique_axis=pdata.get("unique_axis"),
            initial=initial_lattice_params.get(phase_name),
        )

        # Get initial lattice params for this phase
        initial = initial_lattice_params.get(phase_name, lattice.copy())

        phase_results.append(
            PhaseRefinementResult(
                phase_id=phase_name,
                cif_path=Path(f"{phase_name}.cif"),  # placeholder
                name=phase_name,
                weight_fraction=normalized_weight,
                lattice_params=lattice,
                lattice_params_initial=initial,
                spacegroup_no=pdata.get("spacegroup_no"),
                lattice_param_errors=lattice_errors or None,
            )
        )

    return StageResult(
        stage=stage,
        phases=phase_results,
        rwp=rwp,
        rp=rp,
        rpb=rpb,
        rexp=rexp,
        gof=gof,
        rho=rho,
        durbin_watson=durbin_watson,
        converged=converged,
        elapsed_seconds=elapsed,
        iteration_count=iteration_count,
        dia_data=dia_data,
        failure_reason=failure_reason,
        angle_corrections=lst_data.get("angle_corrections"),
    )
