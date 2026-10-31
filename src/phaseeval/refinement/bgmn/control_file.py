"""
.sav control file generation for BGMN refinement.

Generates BGMN control files (.sav) for Pawley and Rietveld stages with
stage-specific parameter constraints. Follows DARA's control file format
closely (VERZERR, STRUC, VAL, LIST, OUTPUT, DIAGRAMM, etc.) but integrates
with our RefinementConfig/PawleyConfig/RietveldConfig model.

Key BGMN syntax conventions:
- VERZERR= for instrument profile (not DEVICE=)
- STRUC[n]= for phases (not PHASE[n]=)
- VAL[1]= for measured data
- LAMBDA=CU (uppercase) for lab sources
- SYNCHROTRON=value for numeric wavelengths
- UNIT=NM: lattice params in nanometers in .lst output

Reference: dara-main/src/dara/generate_control_file.py
"""

from __future__ import annotations

import copy
import logging
import math
import re
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from phaseeval.refinement.bgmn.str_generator import _process_phase_name

if TYPE_CHECKING:
    from phaseeval.refinement.config import PawleyConfig, RefinementConfig, RietveldConfig
    from phaseeval.refinement.protocol import RefinementStage

logger = logging.getLogger(__name__)

# Default instrument profile search path (bundled profiles)
_DEVICES_DIR = Path(__file__).resolve().parent.parent / "data" / "devices"
_GEQ_V1_DEPENDENCY_START = 24
_GEQ_V1_DEPENDENCY_END = 128
_GER_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_GER_THETA_RECORD = re.compile(
    rf"THETA=(?P<theta>{_GER_FLOAT})\s+"
    rf"N=(?P<count>\d+)\s+"
    rf"GSUM=(?P<gsum>{_GER_FLOAT})"
)


class ProfileReachabilityError(ValueError):
    """Typed deterministic failure proving a profile cannot cover a pattern."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def parse_geq_ger_dependency(data: bytes) -> str:
    """Extract the null/space-padded GER filename from a pinned GEQ.V1 header."""
    if not data.startswith(b"GEQ.V1") or len(data) < _GEQ_V1_DEPENDENCY_END:
        raise ProfileReachabilityError(
            "profile_geq_malformed",
            "GEQ entrypoint is not a complete GEQ.V1 header",
        )
    raw = data[_GEQ_V1_DEPENDENCY_START:_GEQ_V1_DEPENDENCY_END]
    raw = raw.split(b"\0", 1)[0].strip(b" ")
    try:
        dependency = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ProfileReachabilityError(
            "profile_geq_malformed",
            "GEQ GER dependency is not ASCII",
        ) from exc
    relative = Path(dependency)
    if (
        not dependency
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != dependency
        or relative.suffix.casefold() != ".ger"
    ):
        raise ProfileReachabilityError(
            "profile_dependency_outside_bundle",
            f"GEQ GER dependency must be one bundle-local filename: {dependency!r}",
        )
    return dependency


def parse_ger_theta_knots(text: str) -> tuple[float, ...]:
    """Parse finite, strictly increasing THETA knots from GER records."""
    knots: list[float] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line.startswith("THETA"):
            continue
        match = _GER_THETA_RECORD.fullmatch(line)
        if match is None:
            raise ProfileReachabilityError(
                "profile_ger_malformed",
                f"invalid THETA record at line {line_number}",
            )
        theta = float(match.group("theta"))
        count = int(match.group("count"))
        gsum = float(match.group("gsum"))
        if not math.isfinite(theta) or not math.isfinite(gsum) or count <= 0:
            raise ProfileReachabilityError(
                "profile_ger_malformed",
                f"non-finite or empty THETA record at line {line_number}",
            )
        if knots and theta <= knots[-1]:
            raise ProfileReachabilityError(
                "profile_theta_non_monotonic",
                f"THETA knot {theta} does not follow {knots[-1]}",
            )
        knots.append(theta)
    if not knots:
        raise ProfileReachabilityError(
            "profile_ger_malformed",
            "GER contains no THETA records",
        )
    return tuple(knots)


def read_phase_name_from_str(str_path: Path) -> str:
    """Extract the PHASE= name from a .str file's first line.

    Args:
        str_path: Path to the .str structure file.

    Returns:
        Phase name string (e.g., "BaSnO3").

    Raises:
        ValueError: If PHASE= directive not found in the file.
    """
    text = str_path.read_text()
    match = re.search(r"PHASE=(\S+)", text)
    if match is None:
        raise ValueError(f"Could not find PHASE= in {str_path}. Content:\n{text[:200]}")
    return match.group(1)


def copy_instrument_profile(
    profile_name: str,
    working_dir: Path,
    search_dirs: list[Path] | None = None,
    *,
    declared_companions: tuple[str, ...] | None = None,
) -> str:
    """Copy instrument profile (.geq file) to the working directory.

    Searches for the .geq file in the following order:
    1. If profile_name is an absolute path or path with .geq suffix, use directly
    2. Check each directory in search_dirs
    3. Check bundled DARA profiles directory

    Args:
        profile_name: Instrument profile name or path to .geq file.
        working_dir: Directory to copy the profile into.
        search_dirs: Additional directories to search for profiles.
        declared_companions: Exact bundle-local companion filenames to copy.
            ``None`` preserves legacy implicit GER and wildcard LAM behavior.

    Returns:
        Instrument name (stem of the .geq file, without extension).

    Raises:
        FileNotFoundError: If the .geq file cannot be found.
    """

    def _copy_profile_companions(geq_path: Path) -> None:
        if declared_companions is not None:
            for companion_name in declared_companions:
                relative = Path(companion_name)
                if (
                    relative.is_absolute()
                    or len(relative.parts) != 1
                    or relative.name != companion_name
                ):
                    raise ProfileReachabilityError(
                        "profile_dependency_outside_bundle",
                        f"declared companion must be one filename: {companion_name!r}",
                    )
                companion = geq_path.parent / relative
                if not companion.is_file():
                    raise ProfileReachabilityError(
                        "profile_companion_missing",
                        f"declared companion is missing: {companion}",
                    )
                shutil.copy(companion, working_dir)
            return

        # Legacy v1/v2 behavior: preserve the historical implicit GER and LAM copy.
        resolved_geq = geq_path.resolve()
        for companion in (geq_path.with_suffix(".ger"), resolved_geq.with_suffix(".ger")):
            if companion.exists():
                shutil.copy(companion, working_dir)
                break
        for lam_dir in (geq_path.parent, resolved_geq.parent):
            copied_any = False
            for lam_file in lam_dir.glob("*.lam"):
                shutil.copy(lam_file, working_dir)
                copied_any = True
            if copied_any:
                break

    instrument_path = Path(profile_name)

    # Direct path: if it has .geq suffix and exists, use it
    if instrument_path.suffix == ".geq" and instrument_path.exists():
        shutil.copy(instrument_path, working_dir)
        _copy_profile_companions(instrument_path)
        return instrument_path.stem

    # Strip .geq suffix if present for name-based search
    profile_base = instrument_path.name.removesuffix(".geq")

    # Build search path list
    dirs_to_check: list[Path] = []
    if search_dirs:
        dirs_to_check.extend(search_dirs)
    if _DEVICES_DIR.is_dir():
        dirs_to_check.append(_DEVICES_DIR)

    # Search for the profile
    for search_dir in dirs_to_check:
        candidate = search_dir / f"{profile_base}.geq"
        if candidate.exists():
            shutil.copy(candidate, working_dir)
            _copy_profile_companions(candidate)
            return candidate.stem

    raise FileNotFoundError(
        f"Could not find instrument profile '{profile_name}'. "
        f"Searched: {[str(d) for d in dirs_to_check]}"
    )


def prepare_pattern(pattern_path: Path, working_dir: Path) -> Path:
    """Copy and clean XY pattern for BGMN processing.

    Copies the pattern to working_dir (if not already there), trims
    negative/zero intensities to 1e-6, and removes points below 1.0 deg
    2-theta.

    Args:
        pattern_path: Path to the source XY pattern file.
        working_dir: Directory to place the prepared pattern.

    Returns:
        Path to the prepared pattern file in working_dir.

    Raises:
        ValueError: If pattern file cannot be loaded.
    """
    try:
        xy_content = np.loadtxt(pattern_path)
    except ValueError as e:
        raise ValueError(f"Could not load pattern file {pattern_path}") from e

    # Shift negative/zero intensities to a positive floor. Clipping instead
    # would give every clipped point ~infinite 1/y weight; background-subtracted
    # patterns (EMPA: half the points negative) then fail Rwp on an artifact.
    y_min = xy_content[:, 1].min()
    if y_min <= 0:
        warnings.warn(
            f"Pattern contains negative or zero intensities (min {y_min:.4g}). "
            "Shifting so the minimum is 1e-6.",
            stacklevel=2,
        )
        xy_content[:, 1] = xy_content[:, 1] - y_min + 1e-6

    # Remove points below 1.0 deg 2-theta
    if xy_content[:, 0].min() < 1.0:
        warnings.warn(
            "Pattern contains 2-theta values below 1.0 deg. Removing them.",
            stacklevel=2,
        )
        xy_content = xy_content[xy_content[:, 0] >= 1.0]

    # Write cleaned pattern to working_dir
    dest = working_dir / pattern_path.name
    np.savetxt(dest, xy_content, fmt="%.6f")
    return dest


def _format_phase_params(
    phase_name: str,
    stage_config: PawleyConfig | RietveldConfig,
    is_pawley: bool,
    ml_weight_prior: float | None = None,
) -> str:
    """Format per-phase parameter line for a .str file.

    Args:
        phase_name: The BGMN phase name from the .str file.
        stage_config: Stage-specific config (PawleyConfig or RietveldConfig).
        is_pawley: Whether this is a Pawley/Le Bail stage.

    Returns:
        Single RP/k1/k2/B1/GEWICHT line with BGMN syntax.
    """
    # These are composition fractions, not BGMN's raw GEWICHT scale.  Keep
    # them available to the control-file receipt, but never serialize them as
    # scale-factor bounds here.
    _ = phase_name, ml_weight_prior

    # Peak profile function
    parts: list[str] = [f"RP={stage_config.rp}"]

    # Crystallite-size distribution width (k1)
    k1 = stage_config.k1
    if k1 == "fixed":
        parts.append("k1=0")
    else:
        parts.append(f"PARAM=k1={k1}")

    # Square of rms microstrain (k2)
    k2 = stage_config.k2
    if k2 == "fixed":
        parts.append("k2=0")
    else:
        parts.append(f"PARAM=k2={k2}")

    # Crystallite size broadening (B1)
    b1 = stage_config.b1
    if b1 == "fixed":
        parts.append("B1=0")
    else:
        parts.append(f"PARAM=B1={b1}")

    # Weight scheme
    gewicht = "0_0" if is_pawley else stage_config.gewicht
    if gewicht == "0_0":
        parts.append("PARAM=GEWICHT=0_0")
    else:
        parts.append(f"GEWICHT={gewicht}")

    return " ".join(parts) + " //"


def _apply_stage_phase_params_to_str(
    str_path: Path,
    phase_name: str,
    stage_config: PawleyConfig | RietveldConfig,
    is_pawley: bool,
    ml_weight_prior: float | None = None,
) -> None:
    """Rewrite stage-sensitive lines in a copied .str file.

    This enforces per-stage RP/k1/k2/B1/GEWICHT and LeBail behavior even when
    the same input .str is reused across Pawley and Rietveld calls.
    """
    phase_param_line = _format_phase_params(
        phase_name, stage_config, is_pawley, ml_weight_prior=ml_weight_prior
    )

    text = str_path.read_text()
    lines = text.splitlines()
    had_trailing_newline = text.endswith("\n")

    updated: list[str] = []
    replaced_peak_line = False
    has_lebail_line = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("RP="):
            updated.append(phase_param_line)
            replaced_peak_line = True
            continue

        if stripped.startswith("LeBail="):
            if is_pawley:
                updated.append("LeBail=1")
                has_lebail_line = True
            # For Rietveld, explicitly drop LeBail line.
            continue

        updated.append(line)

    if not replaced_peak_line:
        insert_at = next(
            (i for i, line in enumerate(updated) if line.strip().startswith("GOAL:")),
            len(updated),
        )
        updated.insert(insert_at, phase_param_line)

    if is_pawley and not has_lebail_line:
        insert_after_peak = next(
            (i for i, line in enumerate(updated) if line.strip().startswith("RP=")),
            None,
        )
        if insert_after_peak is None:
            updated.append("LeBail=1")
        else:
            updated.insert(insert_after_peak + 1, "LeBail=1")

    new_text = "\n".join(updated)
    if had_trailing_newline:
        new_text += "\n"
    str_path.write_text(new_text)


def generate_sav(
    pattern_path: Path,
    str_paths: list[Path],
    config: RefinementConfig,
    working_dir: Path,
    stage: RefinementStage,
    *,
    wmin: float | None = None,
    wmax: float | None = None,
    ml_weight_fractions: list[float] | None = None,
) -> Path:
    """Generate a BGMN .sav control file for the given refinement stage.

    Creates a complete control file with instrument profile, wavelength,
    phases, and stage-appropriate parameter constraints. All phases are
    included in a single .sav file (BGMN handles simultaneous multi-phase
    refinement natively).

    Args:
        pattern_path: Path to the XY diffraction pattern.
        str_paths: Paths to the .str structure files.
        config: Top-level RefinementConfig.
        working_dir: Directory for BGMN working files.
        stage: Refinement stage (PAWLEY or RIETVELD).
        wmin: Optional minimum 2-theta bound.
        wmax: Optional maximum 2-theta bound.
        ml_weight_fractions: Optional ML priors aligned with ``str_paths``.

    Returns:
        Path to the generated .sav control file.

    Raises:
        FileNotFoundError: If instrument profile or .str files missing.
        ValueError: If pattern cannot be loaded.
    """
    from phaseeval.refinement.protocol import RefinementStage

    # Ensure working directory exists
    working_dir.mkdir(parents=True, exist_ok=True)

    # Copy and prepare pattern
    prepared_pattern = prepare_pattern(pattern_path, working_dir)

    # Copy instrument profile
    instrument_name = copy_instrument_profile(config.instrument_profile, working_dir)

    # Copy .str files to working_dir if not already there
    working_str_paths: list[Path] = []
    for str_path in str_paths:
        dest = working_dir / str_path.name
        if str_path.parent != working_dir:
            shutil.copy(str_path, dest)
        else:
            dest = str_path
        working_str_paths.append(dest)

    # Read phase names from .str files
    phase_names = [read_phase_name_from_str(p) for p in working_str_paths]

    # Select stage-specific config
    stage_config: Any
    if stage == RefinementStage.PAWLEY:
        stage_config = config.pawley
        is_pawley = True
    elif stage == RefinementStage.RIETVELD_LATTICE:
        stage_config = config.rietveld_lattice
        is_pawley = False
    elif stage == RefinementStage.RIETVELD_PROFILE:
        stage_config = config.rietveld_profile
        is_pawley = False
    else:
        stage_config = config.rietveld
        is_pawley = False

    normalized_ml_priors: list[float] | None = None
    if not is_pawley and ml_weight_fractions is not None:
        if len(ml_weight_fractions) != len(working_str_paths):
            raise ValueError(
                "ml_weight_fractions must align with str_paths length "
                f"({len(working_str_paths)}), got {len(ml_weight_fractions)}"
            )
        cleaned = [max(float(v), 0.0) for v in ml_weight_fractions]
        total = sum(cleaned)
        if total <= 0:
            raise ValueError("ml_weight_fractions must contain at least one positive value")
        normalized_ml_priors = [v / total for v in cleaned]

    # Enforce stage-specific phase parameters in copied .str files.
    for idx, (str_path, phase_name) in enumerate(zip(working_str_paths, phase_names, strict=False)):
        ml_prior = normalized_ml_priors[idx] if normalized_ml_priors is not None else None
        phase_stage_config = stage_config
        for configured_phase_id, stage_overrides in config.phase_stage_overrides.items():
            if _process_phase_name(configured_phase_id) != phase_name:
                continue
            phase_override = stage_overrides.get(stage.value, {})
            if phase_override:
                phase_stage_config = copy.copy(stage_config)
                for key, value in phase_override.items():
                    if hasattr(phase_stage_config, key):
                        setattr(phase_stage_config, key, value)
            break
        _apply_stage_phase_params_to_str(
            str_path,
            phase_name,
            phase_stage_config,
            is_pawley,
            ml_weight_prior=ml_prior,
        )

    # Build control file content
    lines: list[str] = []

    # Header
    lines.append(f"% Control file generated by AgentXRD ({stage.value} stage)")

    # Instrument profile
    lines.append("% Theoretical instrumental function")
    lines.append(f"VERZERR={instrument_name}.geq")

    # Wavelength
    lines.append("% Wavelength")
    wavelength = config.wavelength
    if isinstance(wavelength, str):
        lines.append(f"LAMBDA={wavelength.upper()}")
    else:
        # AgentXRD wavelengths are in Angstroms; BGMN SYNCHROTRON expects nanometres.
        lines.append(f"SYNCHROTRON={wavelength / 10.0:.6f}")

    # Background polynomial degree (optional; BGMN default when unset)
    if config.background_order is not None:
        lines.append(f"RU={int(config.background_order)}")

    # 2-theta range bounds (optional)
    if wmin is not None:
        lines.append(f"WMIN={wmin}")
    if wmax is not None:
        lines.append(f"WMAX={wmax}")

    # Phase structure files
    lines.append("% Phases")
    for i, str_path in enumerate(working_str_paths, start=1):
        lines.append(f"STRUC[{i}]={str_path.name}")

    # Measured data
    lines.append("% Measured data")
    lines.append(f"VAL[1]={prepared_pattern.name}")

    # Output files
    sav_stem = f"{pattern_path.stem}_{stage.value}"
    lines.append("% Result list output")
    lines.append(f"LIST={sav_stem}.lst")
    lines.append("% Peak list output")
    lines.append(f"OUTPUT={sav_stem}.par")
    lines.append("% Diagram output")
    lines.append(f"DIAGRAMM={sav_stem}.dia")

    # Global parameters
    lines.append("% Global parameters for zero point and sample displacement")
    eps1 = config.eps1
    if isinstance(eps1, str):
        lines.append(f"PARAM[1]=EPS1={eps1}")
    else:
        lines.append(f"EPS1={eps1}")

    eps2 = config.eps2
    if isinstance(eps2, str):
        param_idx = 2 if isinstance(eps1, str) else 1
        lines.append(f"PARAM[{param_idx}]=EPS2={eps2}")
    else:
        lines.append(f"EPS2={eps2}")

    lines.append(f"NTHREADS={config.n_threads}")
    lines.append("PROTOKOLL=Y")

    if normalized_ml_priors is not None:
        lines.append("% ML GEWICHT priors (normalized)")
        for phase_name, prior in zip(phase_names, normalized_ml_priors, strict=False):
            lines.append(f"% ML_PRIOR_{phase_name}={prior:.4f}")

    # Phase fraction normalization (sum, Q-values, GOALs)
    lines.append(f"sum={'+'.join(phase_names)}")

    for phase_name in phase_names:
        lines.append(f"Q{phase_name}={phase_name}/sum")

    for i, phase_name in enumerate(phase_names, start=1):
        lines.append(f"GOAL[{i}]=Q{phase_name}")

    # Write control file
    sav_path = working_dir / f"{sav_stem}.sav"
    content = "\n".join(lines) + "\n"

    with open(sav_path, "w") as f:
        f.write(content)

    logger.info("Generated %s control file: %s", stage.value, sav_path)
    return sav_path
