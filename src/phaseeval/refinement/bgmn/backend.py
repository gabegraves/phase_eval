"""
BGMNBackend: Two-stage Pawley-then-Rietveld refinement orchestrator.

Implements the RefinementBackend protocol using BGMN subprocess execution.
Ties together all BGMN components (str_generator, control_file, runner,
result_parser) into a coherent pipeline.

Locked decisions:
- Pawley runs before Rietveld by default (optional bypass at >99% ML confidence)
- preserve_temp_files is always True (working dirs kept)
- Rpb=100 is false convergence (never treated as success)
- Exceptions NEVER propagate — always return RefinementOutput with failure info

Reference:
- DARA search/core.py: DEFAULT_PHASE_PARAMS, two-stage logic
- Kosmos report: 55pp Rwp improvement from Pawley-first
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from phaseeval.refinement.bgmn.control_file import (
    copy_instrument_profile,
    generate_sav,
    read_phase_name_from_str,
)
from phaseeval.refinement.bgmn.convergence_monitor import (
    ConvergenceMonitor,
    parse_rwp_from_line,
)
from phaseeval.refinement.bgmn.exceptions import SymmetryDeterminationError
from phaseeval.refinement.bgmn.installer import BGMNInstaller, BGMNNotInstalledError
from phaseeval.refinement.bgmn.result_parser import (
    _complete_lattice_params,
    parse_dia,
    parse_lst,
    to_stage_result,
)
from phaseeval.refinement.bgmn.runner import BGMNCrashError, BGMNRunner, BGMNTimeoutError
from phaseeval.refinement.bgmn.str_cache import StrCache
from phaseeval.refinement.bgmn.str_generator import CIFToStrError, _process_phase_name
from phaseeval.refinement.bgmn.validator import validate_cif
from phaseeval.refinement.config import RefinementConfig
from phaseeval.refinement.protocol import (
    PhaseRefinementResult,
    RefinementInput,
    RefinementOutput,
    RefinementStage,
    StageResult,
)

logger = logging.getLogger(__name__)

_EASY_SAMPLE_PAWLEY_RWP_THRESHOLD = 10.0
_EASY_SAMPLE_ML_CONFIDENCE_THRESHOLD = 0.90

_PAWLEY_MIN_TIMEOUT_SECONDS = 45.0  # Keep aligned with calibrated minimum Pawley timeout
_MAX_LATTICE_CARRY_RELATIVE_UNCERTAINTY = 0.5


def _phase_map(input: RefinementInput) -> dict[str, tuple[str, Path]]:
    """Bind native BGMN names to the exact requested IDs and source CIFs."""
    phase_ids = input.phase_ids if input.phase_ids is not None else [p.stem for p in input.cif_paths]
    if len(phase_ids) != len(input.cif_paths):
        raise ValueError("phase_ids must match cif_paths")
    mapping = {}
    for phase_id, cif_path in zip(phase_ids, input.cif_paths, strict=True):
        native_name = _process_phase_name(phase_id)
        if native_name in mapping:
            raise ValueError(f"BGMN phase name collision: {native_name}")
        mapping[native_name] = (phase_id, cif_path)
    return mapping


def _bind_phase_ids(result: StageResult, input: RefinementInput) -> StageResult:
    mapping = _phase_map(input)
    phases = []
    for phase in result.phases:
        if phase.phase_id not in mapping:
            raise ValueError(f"Unknown BGMN phase: {phase.phase_id}")
        phase_id, cif_path = mapping[phase.phase_id]
        # Keep name as the native key used by STR files and lattice carry-forward.
        phases.append(replace(phase, phase_id=phase_id, cif_path=cif_path))
    return replace(result, phases=phases)


def _pattern_point_count(pattern_path: Path) -> int:
    """Read the numeric XY pattern for timeout estimation without guessing its size."""
    try:
        data = np.loadtxt(pattern_path, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load pattern file {pattern_path}: {exc}") from exc
    if data.shape[0] == 0 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError(
            f"Invalid pattern file {pattern_path}: expected finite XY data with at least two columns"
        )
    return data.shape[0]


def _adaptive_rietveld_timeout(
    pawley_elapsed: float,
    pawley_estimate: float,
    n_phases: int,
    n_points: int,
) -> float:
    """Compute adaptive Rietveld timeout based on Pawley performance.

    If Pawley converged faster than estimated, Rietveld likely will too.
    Scale Rietveld timeout by the same ratio, with a floor of 45s.
    """
    base_soft = BGMNRunner.estimate_runtime(n_phases, n_points)
    if pawley_estimate > 0 and pawley_elapsed > 0:
        speed_ratio = min(pawley_elapsed / pawley_estimate, 1.5)
        adjusted = base_soft * speed_ratio * 1.5
    else:
        adjusted = base_soft * 1.5
    return max(adjusted, _PAWLEY_MIN_TIMEOUT_SECONDS)


def _scale_bgmn_bound(bound_str: str, multiplier: float) -> str:
    """Scale a BGMN parameter bound string by a multiplier.

    Handles formats like "0_0^0.01" (scales the upper bound),
    "fixed" (unchanged), and plain numbers.

    Args:
        bound_str: BGMN bound string (e.g., "0_0^0.01", "fixed").
        multiplier: Scaling factor (e.g., 0.5 for tighter, 2.0 for wider).

    Returns:
        Scaled bound string, or original if not scalable.
    """
    if multiplier == 1.0 or bound_str == "fixed":
        return bound_str
    if "^" in str(bound_str):
        parts = str(bound_str).split("^")
        base_part = parts[0]
        upper = float(parts[1])
        return f"{base_part}^{upper * multiplier}"
    return bound_str


class BGMNBackend:
    """Two-stage Pawley-then-Rietveld refinement using BGMN.

    Implements the ``RefinementBackend`` protocol. Orchestrates:

    1. CIF validation and .str generation
    2. Pawley (Le Bail) constrained fit with divergence recovery
    3. Quality gate (skip Rietveld on poor Pawley)
    4. Rietveld structural refinement
    5. Result assembly with normalized weight fractions

    All exceptions are caught internally; ``refine()`` always returns
    a ``RefinementOutput`` (never raises).

    Args:
        installer: BGMNInstaller for binary management. Created lazily if None.
        str_cache: StrCache for .str file caching. Created lazily if None.

    Example::

        backend = BGMNBackend()
        if backend.is_available():
            output = backend.refine(input, config)
    """

    def __init__(
        self,
        installer: BGMNInstaller | None = None,
        str_cache: StrCache | None = None,
    ) -> None:
        # Lazy init: don't check BGMN until first refine() call
        self._installer = installer
        self._str_cache = str_cache
        self._runner: BGMNRunner | None = None

    def _ensure_installer(self) -> BGMNInstaller:
        """Get or create the BGMNInstaller."""
        if self._installer is None:
            self._installer = BGMNInstaller()
        return self._installer

    def _ensure_str_cache(self) -> StrCache:
        """Get or create the StrCache."""
        if self._str_cache is None:
            self._str_cache = StrCache()
        return self._str_cache

    def _ensure_runner(self) -> BGMNRunner:
        """Get or create the BGMNRunner."""
        if self._runner is None:
            self._runner = BGMNRunner(installer=self._ensure_installer())
        return self._runner

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return "bgmn"

    def is_available(self) -> bool:
        """Check if BGMN binary is installed and ready."""
        return self._ensure_installer().is_installed()

    # ------------------------------------------------------------------
    # Full refinement pipeline
    # ------------------------------------------------------------------

    def refine(self, input: RefinementInput, config: RefinementConfig) -> RefinementOutput:
        """Run two-stage Pawley-then-Rietveld refinement.

        NEVER raises exceptions. All failures are captured in the
        returned RefinementOutput.

        Args:
            input: Pattern path, CIF paths, hypothesis ID, etc.
            config: Stage params, quality gate settings, etc.

        Returns:
            RefinementOutput with pawley and/or rietveld StageResults.
        """
        restart_count = 0
        try:
            # STEP 0: Check BGMN availability
            if not self.is_available():
                raise BGMNNotInstalledError()

            # STEP 1: Prepare working directory
            work_dir = self._prepare_working_dir(input, config)

            # STEP 2: Generate .str files
            str_paths, initial_lattice_params = self._generate_str_files(input, config, work_dir)

            # Optional fast path: high-confidence ML prior skips Pawley.
            if self._should_bypass_pawley(input, config):
                logger.info(
                    "Bypassing Pawley for %s (ml_confidence=%.4f > %.4f)",
                    input.hypothesis_id,
                    float(input.ml_confidence),
                    config.pawley_bypass_confidence,
                )
                rl, rp, rf = self._run_rietveld(
                    input, config, work_dir, str_paths, initial_lattice_params
                )
                best = rf or rp or rl
                if best is None:
                    return RefinementOutput(
                        hypothesis_id=input.hypothesis_id,
                        working_dir=work_dir,
                        pawley=None,
                        refinement_failed=True,
                        needs_human_review=True,
                        failure_reason=(
                            "Rietveld failed after Pawley bypass (high-confidence ML prior path)"
                        ),
                        restart_count=restart_count,
                    )

                failed = not best.converged
                return RefinementOutput(
                    hypothesis_id=input.hypothesis_id,
                    working_dir=work_dir,
                    pawley=None,
                    rietveld_lattice=rl,
                    rietveld_profile=rp,
                    rietveld=rf,
                    refinement_failed=failed,
                    needs_human_review=(best.rpb >= 100.0),
                    failure_reason=("Rietveld unconverged after Pawley bypass" if failed else None),
                    restart_count=restart_count,
                )

            # STEP 3: Run Pawley
            pawley_result, restart_count = self._run_pawley(
                input, config, work_dir, str_paths, initial_lattice_params
            )

            # STEP 4: Pawley quality gate
            skip_rietveld, failed, failure_reason, needs_review = self._pawley_quality_gate(
                pawley_result, config
            )

            if failed:
                return RefinementOutput(
                    hypothesis_id=input.hypothesis_id,
                    working_dir=work_dir,
                    pawley=pawley_result,
                    rietveld=None,
                    refinement_failed=True,
                    needs_human_review=needs_review,
                    failure_reason=failure_reason,
                    restart_count=restart_count,
                )

            if config.pawley_only:
                return RefinementOutput(
                    hypothesis_id=input.hypothesis_id,
                    working_dir=work_dir,
                    pawley=pawley_result,
                    rietveld=None,
                    refinement_failed=False,
                    needs_human_review=needs_review,
                    failure_reason=(
                        config.pawley_only_reason
                        or "Pawley-only diagnostic policy skipped Rietveld"
                    ),
                    restart_count=restart_count,
                )

            if skip_rietveld:
                return RefinementOutput(
                    hypothesis_id=input.hypothesis_id,
                    working_dir=work_dir,
                    pawley=pawley_result,
                    rietveld=None,
                    refinement_failed=False,
                    needs_human_review=needs_review,
                    failure_reason=failure_reason,
                    restart_count=restart_count,
                )

            # STEP 4.5: Optionally carry forward Pawley's converged lattice params
            if config.enable_pawley_lattice_carry_forward and pawley_result is not None:
                pawley_converged = self._extract_converged_lattice_from_result(pawley_result)
                if pawley_converged:
                    self._inject_converged_lattice_into_str(work_dir, pawley_converged)
                    logger.info(
                        "Carried forward %d converged Pawley lattice params for %s",
                        len(pawley_converged),
                        input.hypothesis_id,
                    )

            # STEP 5: Run staged Rietveld
            rl, rp, rf = self._run_rietveld(
                input,
                config,
                work_dir,
                str_paths,
                initial_lattice_params,
                pawley_result=pawley_result,
            )

            # STEP 6: Assemble output
            best_rietveld = rf or rp or rl
            both_failed = (
                pawley_result is not None
                and not pawley_result.converged
                and best_rietveld is not None
                and not best_rietveld.converged
            )
            any_rpb_100 = (pawley_result is not None and pawley_result.rpb >= 100.0) or (
                best_rietveld is not None and best_rietveld.rpb >= 100.0
            )

            return RefinementOutput(
                hypothesis_id=input.hypothesis_id,
                working_dir=work_dir,
                pawley=pawley_result,
                rietveld_lattice=rl,
                rietveld_profile=rp,
                rietveld=rf,
                refinement_failed=both_failed,
                needs_human_review=any_rpb_100,
                failure_reason="Both stages unconverged" if both_failed else None,
                restart_count=restart_count,
            )

        except BGMNNotInstalledError as e:
            logger.error("BGMN not installed: %s", e)
            return self._failure_output(input, config, "BGMN not installed", restart_count)
        except (CIFToStrError, SymmetryDeterminationError) as e:
            logger.error("CIF conversion failed: %s", e)
            return self._failure_output(input, config, f"CIF conversion failed: {e}", restart_count)
        except BGMNTimeoutError as e:
            logger.error("BGMN timeout: %s", e)
            return self._failure_output(input, config, "BGMN timeout", restart_count)
        except BGMNCrashError as e:
            logger.error("BGMN crash: %s", e)
            return self._failure_output(input, config, "BGMN crash", restart_count)
        except Exception as e:
            logger.error("Unexpected error in BGMNBackend.refine(): %s", e)
            return self._failure_output(input, config, f"Unexpected error: {e}", restart_count)

    def _should_bypass_pawley(self, input: RefinementInput, config: RefinementConfig) -> bool:
        """Return True when high-confidence ML priors permit Pawley bypass."""
        if not config.enable_ml_pawley_bypass:
            return False
        if input.ml_confidence is None:
            return False
        try:
            confidence = float(input.ml_confidence)
        except (TypeError, ValueError):
            return False
        return confidence > float(config.pawley_bypass_confidence)

    # ------------------------------------------------------------------
    # Pawley-only and continue-Rietveld workflows
    # ------------------------------------------------------------------

    def refine_pawley_only(
        self, input: RefinementInput, config: RefinementConfig
    ) -> RefinementOutput:
        """Run Pawley (STEPS 0-4) only. No Rietveld.

        Used by batch callers for adaptive top-k: run Pawley on ALL
        candidates, filter by score, then continue_rietveld() on survivors.

        Args:
            input: Refinement input specification.
            config: Refinement configuration.

        Returns:
            RefinementOutput with pawley populated, rietveld=None.
        """
        restart_count = 0
        try:
            if not self.is_available():
                raise BGMNNotInstalledError()

            work_dir = self._prepare_working_dir(input, config)
            str_paths, initial_lattice_params = self._generate_str_files(input, config, work_dir)
            pawley_result, restart_count = self._run_pawley(
                input, config, work_dir, str_paths, initial_lattice_params
            )

            skip_rietveld, failed, failure_reason, needs_review = self._pawley_quality_gate(
                pawley_result, config
            )

            return RefinementOutput(
                hypothesis_id=input.hypothesis_id,
                working_dir=work_dir,
                pawley=pawley_result,
                rietveld=None,
                refinement_failed=failed,
                needs_human_review=needs_review,
                failure_reason=failure_reason,
                restart_count=restart_count,
            )

        except BGMNNotInstalledError as e:
            logger.error("BGMN not installed: %s", e)
            return self._failure_output(input, config, "BGMN not installed", restart_count)
        except (CIFToStrError, SymmetryDeterminationError) as e:
            logger.error("CIF conversion failed: %s", e)
            return self._failure_output(input, config, f"CIF conversion failed: {e}", restart_count)
        except BGMNTimeoutError as e:
            logger.error("BGMN timeout: %s", e)
            return self._failure_output(input, config, "BGMN timeout", restart_count)
        except BGMNCrashError as e:
            logger.error("BGMN crash: %s", e)
            return self._failure_output(input, config, "BGMN crash", restart_count)
        except Exception as e:
            logger.error("Unexpected error in refine_pawley_only(): %s", e)
            return self._failure_output(input, config, f"Unexpected error: {e}", restart_count)

    def continue_rietveld(
        self,
        pawley_output: RefinementOutput,
        input: RefinementInput,
        config: RefinementConfig,
    ) -> RefinementOutput:
        """Continue from Pawley-only output to run Rietveld.

        Reuses existing working_dir and .str files from pawley_output.
        Runs STEP 5 (Rietveld) + STEP 6 (assembly).

        Args:
            pawley_output: Previous Pawley-only RefinementOutput.
            input: Original refinement input.
            config: Refinement configuration.

        Returns:
            Complete RefinementOutput with both pawley and rietveld.
        """
        restart_count = pawley_output.restart_count
        try:
            if not self.is_available():
                raise BGMNNotInstalledError()

            work_dir = pawley_output.working_dir

            # Collect existing .str files from working directory
            str_paths = sorted(work_dir.glob("*.str"))
            if not str_paths:
                return RefinementOutput(
                    hypothesis_id=input.hypothesis_id,
                    working_dir=work_dir,
                    pawley=pawley_output.pawley,
                    rietveld=None,
                    refinement_failed=True,
                    failure_reason="No .str files found in working directory",
                    restart_count=restart_count,
                )

            # Reconstruct initial lattice params from Pawley results
            initial_lattice_params: dict[str, dict[str, float]] = {}
            if pawley_output.pawley is not None:
                for phase in pawley_output.pawley.phases:
                    initial_lattice_params[phase.name] = phase.lattice_params_initial

            # Optionally carry forward Pawley's converged lattice params
            if config.enable_pawley_lattice_carry_forward and pawley_output.pawley is not None:
                pawley_converged = self._extract_converged_lattice_from_result(pawley_output.pawley)
                if pawley_converged:
                    self._inject_converged_lattice_into_str(work_dir, pawley_converged)

            rl, rp, rf = self._run_rietveld(
                input,
                config,
                work_dir,
                str_paths,
                initial_lattice_params,
                pawley_result=pawley_output.pawley,
            )

            best_rietveld = rf or rp or rl
            both_failed = (
                pawley_output.pawley is not None
                and not pawley_output.pawley.converged
                and best_rietveld is not None
                and not best_rietveld.converged
            )
            any_rpb_100 = (
                pawley_output.pawley is not None and pawley_output.pawley.rpb >= 100.0
            ) or (best_rietveld is not None and best_rietveld.rpb >= 100.0)

            return RefinementOutput(
                hypothesis_id=input.hypothesis_id,
                working_dir=work_dir,
                pawley=pawley_output.pawley,
                rietveld_lattice=rl,
                rietveld_profile=rp,
                rietveld=rf,
                refinement_failed=both_failed,
                needs_human_review=any_rpb_100,
                failure_reason="Both stages unconverged" if both_failed else None,
                restart_count=restart_count,
            )

        except Exception as e:
            logger.error("Unexpected error in continue_rietveld(): %s", e)
            return RefinementOutput(
                hypothesis_id=input.hypothesis_id,
                working_dir=pawley_output.working_dir,
                pawley=pawley_output.pawley,
                rietveld=None,
                refinement_failed=True,
                failure_reason=f"Rietveld continuation failed: {e}",
                restart_count=restart_count,
            )

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def dry_run(self, input: RefinementInput, config: RefinementConfig) -> dict:
        """Validate inputs and estimate runtime without running BGMN.

        Checks:
        - BGMN availability
        - CIF validation for all phases
        - .str generation (CIF-to-STR conversion)
        - .sav generation (control file syntax)
        - Runtime estimation

        Args:
            input: Refinement input specification.
            config: Refinement configuration.

        Returns:
            Dict with bgmn_available, cif_validation, str_generated,
            sav_generated, estimated_runtime, errors. Invalid pattern data leaves
            estimated_runtime as None and adds an explicit error.
        """
        errors: list[str] = []
        result: dict[str, Any] = {
            "bgmn_available": False,
            "cif_validation": {},
            "str_generated": False,
            "sav_generated": False,
            "estimated_runtime": None,
            "errors": errors,
        }

        # Check BGMN availability
        try:
            result["bgmn_available"] = self.is_available()
            if not result["bgmn_available"]:
                errors.append("BGMN is not installed")
        except Exception as e:
            errors.append(f"BGMN check failed: {e}")

        # Validate all CIFs
        cif_validations: dict[str, Any] = {}
        for cif_path in input.cif_paths:
            try:
                vr = validate_cif(cif_path)
                cif_validations[str(cif_path)] = {
                    "valid": vr.valid,
                    "errors": vr.errors,
                    "warnings": vr.warnings,
                    "crystal_system": vr.crystal_system,
                    "spacegroup_no": vr.spacegroup_no,
                }
                if not vr.valid:
                    errors.extend(f"CIF {cif_path.name}: {e}" for e in vr.errors)
            except Exception as e:
                cif_validations[str(cif_path)] = {
                    "valid": False,
                    "errors": [str(e)],
                }
                errors.append(f"CIF validation error for {cif_path.name}: {e}")
        result["cif_validation"] = cif_validations

        # Try .str generation
        if not errors:
            try:
                work_dir = self._prepare_working_dir(input, config)
                str_paths, _ = self._generate_str_files(input, config, work_dir)
                result["str_generated"] = True

                # Try .sav generation
                try:
                    generate_sav(
                        input.pattern_path,
                        str_paths,
                        config,
                        work_dir,
                        RefinementStage.PAWLEY,
                    )
                    result["sav_generated"] = True
                except Exception as e:
                    errors.append(f"SAV generation failed: {e}")

            except Exception as e:
                errors.append(f"STR generation failed: {e}")

        # Estimate runtime
        n_phases = len(input.cif_paths)
        try:
            n_points = _pattern_point_count(input.pattern_path)
            result["estimated_runtime"] = BGMNRunner.estimate_runtime(n_phases, n_points)
        except ValueError as e:
            errors.append(f"Runtime estimation failed: {e}")

        return result

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _prepare_working_dir(self, input: RefinementInput, config: RefinementConfig) -> Path:
        """Create working directory for this hypothesis.

        Returns:
            Path to the working directory.
        """
        if config.working_dir is not None:
            base = config.working_dir
        else:
            tmpfs_dir = Path("/dev/shm")
            if config.prefer_tmpfs and tmpfs_dir.is_dir():
                base = Path(tempfile.mkdtemp(prefix="axa_bgmn_", dir=str(tmpfs_dir)))
            else:
                base = Path(tempfile.mkdtemp(prefix="axa_bgmn_"))

        work_dir = base / input.hypothesis_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _extract_initial_lattice_params(self, str_path: Path) -> dict[str, float] | None:
        """Extract initial lattice parameters from generated .str content.

        Returns lengths in Angstroms and angles in degrees. If parsing fails,
        returns None and downstream code falls back to refined values.
        """
        text = str_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(
            r"\b(A|B|C|ALPHA|BETA|GAMMA)=([-+]?\d+(?:\.\d+)?)(?=[_\^\s/]|$)",
            text,
            flags=re.IGNORECASE,
        )
        if not matches:
            return None

        params_nm_or_deg: dict[str, float] = {}
        for key, value_str in matches:
            key_upper = key.upper()
            if key_upper not in params_nm_or_deg:
                params_nm_or_deg[key_upper] = float(value_str)

        a_nm = params_nm_or_deg.get("A")
        if a_nm is None or a_nm <= 0:
            return None

        # BGMN .str uses nm for lengths. Convert to Angstroms for FoM pipeline.
        params = {
            key.lower(): value * 10.0 if key in ("A", "B", "C") else value
            for key, value in params_nm_or_deg.items()
        }
        lattice_match = re.search(r"\bLattice=(\w+)", text, flags=re.IGNORECASE)
        axis_match = re.search(r"\bUniqueAxis=([abc])\b", text, flags=re.IGNORECASE)
        return _complete_lattice_params(
            params,
            lattice_system=lattice_match.group(1) if lattice_match else None,
            unique_axis=axis_match.group(1) if axis_match else None,
        )

    def _generate_str_files(
        self,
        input: RefinementInput,
        config: RefinementConfig,
        work_dir: Path,
        stage: RefinementStage = RefinementStage.PAWLEY,
        *,
        lattice_range_multiplier: float = 1.0,
        b1_upper_multiplier: float = 1.0,
    ) -> tuple[list[Path], dict[str, dict[str, float]]]:
        """Generate .str files for all phases. Validate CIFs first.

        Args:
            input: Refinement input with cif_paths.
            config: Refinement config.
            work_dir: Working directory for .str output.
            stage: Which stage parameters to use (Pawley or Rietveld).
            lattice_range_multiplier: Multiply lattice_range by this (for recovery).
            b1_upper_multiplier: Multiply b1 upper bound by this (for recovery).

        Returns:
            Tuple of (str_paths, initial_lattice_params dict).

        Raises:
            CIFToStrError: If any CIF-to-STR conversion fails.
        """
        _phase_map(input)  # Reject ambiguous names before STR files can overwrite each other.
        str_cache = self._ensure_str_cache()
        str_paths: list[Path] = []
        initial_lattice_params: dict[str, dict[str, float]] = {}

        if stage == RefinementStage.PAWLEY:
            stage_config = config.pawley
            lebail = config.pawley.lebail
        elif stage == RefinementStage.RIETVELD_LATTICE:
            stage_config = config.rietveld_lattice
            lebail = False
        elif stage == RefinementStage.RIETVELD_PROFILE:
            stage_config = config.rietveld_profile
            lebail = False
        else:
            stage_config = config.rietveld
            lebail = False

        lattice_range = stage_config.lattice_range * lattice_range_multiplier
        lattice_shift_prior = self._normalize_lattice_shift_prior(input.lattice_shift_prior)

        # Compute b1 with multiplier (divergence recovery).
        # If b1 is "fixed" (Pawley strict mode), promote to refinable
        # during recovery so crystallite broadening can compensate.
        b1 = stage_config.b1
        if b1_upper_multiplier != 1.0:
            if b1 == "fixed":
                # Unlock B1 for recovery with DARA default bounds
                b1 = "0_0^0.005"
                logger.info("Divergence recovery: promoting b1 from fixed to %s", b1)
            elif "^" in str(b1):
                parts = str(b1).split("^")
                base_part = parts[0]
                upper = float(parts[1])
                b1 = f"{base_part}^{upper * b1_upper_multiplier}"

        # Also unlock k1 during divergence recovery if fixed
        k1 = stage_config.k1
        if b1_upper_multiplier != 1.0 and k1 == "fixed":
            k1 = "0_0^0.01"
            logger.info("Divergence recovery: promoting k1 from fixed to %s", k1)

        for i, cif_path in enumerate(input.cif_paths):
            # Validate CIF first
            vr = validate_cif(cif_path)
            if vr.warnings:
                for w in vr.warnings:
                    logger.warning("CIF %s: %s", cif_path.name, w)
            if not vr.valid:
                raise CIFToStrError(
                    f"CIF validation failed for {cif_path.name}: " + "; ".join(vr.errors)
                )

            # Per-phase elastic strain bounds override global lattice_range
            phase_lattice_range = lattice_range
            if input.elastic_strain_bounds and input.phase_ids:
                phase_id = input.phase_ids[i] if i < len(input.phase_ids) else None
                if phase_id and phase_id in input.elastic_strain_bounds:
                    phase_lattice_range = (
                        input.elastic_strain_bounds[phase_id] * lattice_range_multiplier
                    )
                    logger.debug(
                        "Using elastic strain bound %.4f for phase %s (global: %.4f)",
                        phase_lattice_range,
                        phase_id,
                        lattice_range,
                    )

            # Apply symmetry-aware bounds scaling per crystal system.
            # High-symmetry → tighter bounds (fewer DOF), low-symmetry → wider.
            phase_k1 = k1
            phase_b1 = b1
            if config.symmetry_bounds and vr.crystal_system:
                sbp = config.symmetry_bounds
                cs = vr.crystal_system
                lr_mult = sbp.get_multiplier(cs, "lattice_range")
                phase_lattice_range *= lr_mult
                phase_k1 = _scale_bgmn_bound(k1, sbp.get_multiplier(cs, "k1"))
                phase_b1 = _scale_bgmn_bound(b1, sbp.get_multiplier(cs, "b1"))
                if lr_mult != 1.0:
                    logger.debug(
                        "Symmetry bounds for %s (%s): lr×%.2f, k1×%.2f, b1×%.2f",
                        cif_path.name,
                        cs,
                        lr_mult,
                        sbp.get_multiplier(cs, "k1"),
                        sbp.get_multiplier(cs, "b1"),
                    )

            phase_id = (
                input.phase_ids[i]
                if input.phase_ids and i < len(input.phase_ids)
                else cif_path.stem
            )
            phase_override = config.phase_stage_overrides.get(phase_id, {}).get(
                stage.value, {}
            )
            if "lattice_range" in phase_override:
                phase_lattice_range = float(phase_override["lattice_range"])

            # Generate .str via cache
            str_kwargs: dict[str, Any] = {
                "lebail": lebail,
                "lattice_range": phase_lattice_range,
                "k1": phase_override.get("k1", phase_k1),
                "k2": phase_override.get("k2", stage_config.k2),
                "b1": phase_override.get("b1", phase_b1),
                "rp": phase_override.get("rp", stage_config.rp),
                "gewicht": phase_override.get("gewicht", stage_config.gewicht),
            }
            if lattice_shift_prior is not None:
                str_kwargs["lattice_shift"] = lattice_shift_prior

            # Pass explicit phase name from original phase_id so STR phase
            # names are deterministic and independent of CIF temp filenames.
            if input.phase_ids and i < len(input.phase_ids):
                str_kwargs["phase_name"] = _process_phase_name(input.phase_ids[i])

            str_path = str_cache.get_or_generate(
                cif_path,
                work_dir,
                **str_kwargs,
            )
            str_paths.append(str_path)

            # Store initial lattice params for FoM calculation
            phase_name = read_phase_name_from_str(str_path)
            initial = self._extract_initial_lattice_params(str_path)
            if initial is not None:
                initial_lattice_params[phase_name] = initial
            else:
                logger.warning(
                    "Could not parse initial lattice params from %s; "
                    "will fall back to refined lattice values for FoM.",
                    str_path.name,
                )

        return str_paths, initial_lattice_params

    def _run_pawley(
        self,
        input: RefinementInput,
        config: RefinementConfig,
        work_dir: Path,
        str_paths: list[Path],
        initial_lattice_params: dict[str, dict[str, float]],
    ) -> tuple[StageResult | None, int]:
        """Run Pawley stage with divergence recovery.

        STEP 3 + STEP 3.5 from the plan.

        Returns:
            Tuple of (StageResult or None, restart_count).
        """
        runner = self._ensure_runner()
        restart_count = 0

        # Generate .sav for Pawley
        sav_path = generate_sav(
            input.pattern_path,
            str_paths,
            config,
            work_dir,
            RefinementStage.PAWLEY,
        )

        # Copy instrument profile
        copy_instrument_profile(config.instrument_profile, work_dir)

        # Compute timeout
        n_phases = len(input.cif_paths)
        _, hard_timeout = runner.compute_timeout(
            n_phases,
            _pattern_point_count(input.pattern_path),
            multiplier=config.timeout_multiplier,
        )
        # Empirically, difficult in-dist Pawley fits require >18s even for
        # low phase counts; enforce a practical floor to avoid false timeouts.
        hard_timeout = max(hard_timeout, _PAWLEY_MIN_TIMEOUT_SECONDS)

        # First attempt
        start_time = time.monotonic()
        pawley_failed = False
        pawley_rpb_100 = False
        lst_data = None

        conv_monitor = ConvergenceMonitor(config.convergence)

        def _on_stdout(line: str) -> bool:
            rwp = parse_rwp_from_line(line)
            if rwp is not None:
                conv_monitor.update(rwp)
            return conv_monitor.should_terminate()

        try:
            lst_path = runner.run_with_monitor(
                sav_path,
                timeout_seconds=hard_timeout,
                cpu_affinity=config.cpu_affinity,
                on_stdout_line=_on_stdout,
            )
        except BGMNTimeoutError:
            elapsed = time.monotonic() - start_time
            dia_only_result = self._pawley_dia_only_result(
                sav_path,
                initial_lattice_params,
                elapsed,
                failure_reason=(
                    f"BGMN timeout: Pawley exceeded {hard_timeout:.0f}s hard cap "
                    f"after {elapsed:.0f}s; DIA-only evidence preserved"
                ),
            )
            if dia_only_result is not None:
                logger.warning(
                    "Pawley produced DIA-only evidence for hypothesis %s after timeout",
                    input.hypothesis_id,
                )
                return _bind_phase_ids(dia_only_result, input), restart_count + 1
            lst_path = None
            pawley_failed = True
            restart_count += 1

        if lst_path is not None:
            try:
                lst_data = parse_lst(lst_path)
            except FileNotFoundError:
                dia_only_result = self._pawley_dia_only_result(
                    sav_path,
                    initial_lattice_params,
                    time.monotonic() - start_time,
                    failure_reason="Pawley LST missing; DIA-only evidence preserved",
                )
                if dia_only_result is not None:
                    logger.warning(
                        "Pawley produced DIA-only evidence for hypothesis %s with missing LST",
                        input.hypothesis_id,
                    )
                    return _bind_phase_ids(dia_only_result, input), restart_count
                raise
            if lst_data.get("rpb", 0.0) >= 100.0:
                pawley_rpb_100 = True
        elif not pawley_failed:
            # run_with_restart returned None (all attempts failed)
            pawley_failed = True
            restart_count += 1

        # STEP 3.5: False-convergence recovery
        #
        # Timeouts/crashes have already consumed run_with_restart()'s retries.
        # Re-running a second relaxed Pawley pass turns hard failures into a
        # runtime hotspot on difficult EMPA iodides without creating a usable
        # basin. Reserve the extra relaxed pass for explicit Rpb=100 false
        # convergence, where BGMN did finish and we have evidence that wider
        # bounds may rescue the fit.
        if pawley_rpb_100:
            logger.info(
                "Pawley false-convergence recovery: rpb_100=%s, failed=%s. "
                "Retrying with relaxed constraints (2x lattice_range, 2x b1).",
                pawley_rpb_100,
                pawley_failed,
            )
            restart_count += 1

            # Generate NEW .str files with relaxed constraints
            relaxed_str_paths, _ = self._generate_str_files(
                input,
                config,
                work_dir,
                stage=RefinementStage.PAWLEY,
                lattice_range_multiplier=2.0,
                b1_upper_multiplier=2.0,
            )

            # Generate NEW .sav with relaxed .str files
            sav_path = generate_sav(
                input.pattern_path,
                relaxed_str_paths,
                config,
                work_dir,
                RefinementStage.PAWLEY,
            )

            start_time = time.monotonic()
            lst_data = None  # Reset for fresh parse after recovery
            try:
                lst_path = runner.run_with_restart(
                    sav_path,
                    timeout_seconds=hard_timeout,
                    max_restarts=0,  # no more restarts for recovery attempt
                    cpu_affinity=config.cpu_affinity,
                )
            except BGMNTimeoutError:
                lst_path = None

            if lst_path is not None:
                lst_data = parse_lst(lst_path)

            str_paths = relaxed_str_paths

        # Build Pawley result from parsed data
        if lst_data is not None:
            elapsed = time.monotonic() - start_time

            # Parse .dia
            dia_stem = sav_path.stem
            dia_path = work_dir / f"{dia_stem}.dia"
            dia_data = parse_dia(dia_path)

            pawley_result = to_stage_result(
                lst_data,
                dia_data,
                RefinementStage.PAWLEY,
                initial_lattice_params,
                elapsed,
            )
            return _bind_phase_ids(pawley_result, input), restart_count

        # Complete failure
        logger.error(
            "Pawley failed completely for hypothesis %s",
            input.hypothesis_id,
        )
        return None, restart_count

    def _pawley_dia_only_result(
        self,
        sav_path: Path,
        initial_lattice_params: dict[str, dict[str, float]],
        elapsed: float,
        *,
        failure_reason: str,
    ) -> StageResult | None:
        """Preserve fresh Pawley ``.dia`` evidence when BGMN wrote no ``.lst``.

        BGMN can leave a useful Pawley diagram when a monitored run is
        terminated or times out before the list file is available. Treat that as
        failed diagnostic evidence, not as a complete absence of Pawley output.
        """
        dia_path = sav_path.with_suffix(".dia")
        if not dia_path.exists():
            return None
        try:
            if dia_path.stat().st_mtime < sav_path.stat().st_mtime:
                return None
        except OSError:
            return None

        try:
            first_line = dia_path.read_text(errors="replace").splitlines()[0]
        except (IndexError, OSError):
            return None

        metrics: dict[str, float] = {}
        for key, value in re.findall(r"\b(Rp|R|Rwp)=([0-9]+(?:\.[0-9]+)?)", first_line):
            metrics[key.lower()] = float(value)
        rwp = metrics.get("rwp")
        if rwp is None or not math.isfinite(rwp):
            return None

        dia_data = parse_dia(dia_path)
        phases = [
            PhaseRefinementResult(
                phase_id=phase_name,
                cif_path=Path(f"{phase_name}.cif"),
                name=phase_name,
                weight_fraction=0.0,
                lattice_params=lattice.copy(),
                lattice_params_initial=lattice.copy(),
                spacegroup_no=None,
            )
            for phase_name, lattice in initial_lattice_params.items()
        ]
        return StageResult(
            stage=RefinementStage.PAWLEY,
            phases=phases,
            rwp=rwp,
            rp=metrics.get("rp", 0.0),
            rpb=100.0,
            rexp=0.0,
            gof=0.0,
            rho=0.0,
            converged=False,
            elapsed_seconds=elapsed,
            iteration_count=0,
            dia_data=dia_data,
            durbin_watson=None,
            failure_reason=failure_reason,
        )

    def _pawley_quality_gate(
        self,
        pawley_result: StageResult | None,
        config: RefinementConfig,
    ) -> tuple[bool, bool, str | None, bool]:
        """Evaluate Pawley result for quality gate.

        Returns:
            Tuple of (skip_rietveld, refinement_failed, failure_reason,
            needs_human_review).
        """
        if pawley_result is None:
            return True, True, "Pawley stage failed completely", True

        rpb = pawley_result.rpb
        rwp = pawley_result.rwp

        if pawley_result.iteration_count == 0 and pawley_result.rexp == 0.0 and rpb >= 100.0:
            reason = (
                pawley_result.failure_reason or "Pawley LST missing; DIA-only evidence preserved"
            )
            return True, True, reason, True

        # Rpb=100 means false convergence
        if rpb >= 100.0:
            logger.warning(
                "Pawley Rpb=%.1f%% (false convergence). Marking as failed.",
                rpb,
            )
            return True, True, f"Pawley false convergence (Rpb={rpb:.1f}%)", True

        # Quality gate: skip Rietveld if Rpb improvement is insufficient
        if config.pawley_quality_gate:
            # Rpb improvement threshold: if Rpb is still very high,
            # the background-only fit is not meaningfully improved
            if rpb >= (100.0 - config.rpb_improvement_threshold):
                logger.info(
                    "Pawley quality gate: Rpb=%.1f%% (improvement < %.1fpp). Skipping Rietveld.",
                    rpb,
                    config.rpb_improvement_threshold,
                )
                return (
                    True,
                    False,
                    (
                        f"Quality gate: Rpb={rpb:.1f}% "
                        f"(improvement < {config.rpb_improvement_threshold}pp)"
                    ),
                    False,
                )
            if config.pawley_rwp_threshold is not None and rwp >= config.pawley_rwp_threshold:
                logger.info(
                    "Pawley quality gate: Rwp=%.1f%% (>= %.1f%%). Skipping Rietveld.",
                    rwp,
                    config.pawley_rwp_threshold,
                )
                return (
                    True,
                    False,
                    (f"Quality gate: Rwp={rwp:.1f}% (>= {config.pawley_rwp_threshold:.1f}%)"),
                    True,
                )

        # Pawley passed quality gate: proceed to Rietveld
        logger.info("Pawley quality gate passed: Rpb=%.1f%%", rpb)
        return False, False, None, False

    def _run_single_rietveld_stage(
        self,
        input: RefinementInput,
        config: RefinementConfig,
        work_dir: Path,
        str_paths: list[Path],
        initial_lattice_params: dict[str, dict[str, float]],
        stage: RefinementStage,
        *,
        ml_weight_fractions: list[float] | None = None,
        lattice_start_params: dict[str, dict[str, float]] | None = None,
    ) -> StageResult | None:
        """Run a single Rietveld-family stage.

        Generates .str + .sav for the given stage, executes BGMN, parses results.

        Args:
            stage: One of RIETVELD_LATTICE, RIETVELD_PROFILE, or RIETVELD.
            ml_weight_fractions: Only applied for RIETVELD (final) stage.
            lattice_start_params: Lattice values carried from the preceding
                stage. Applied after stage-specific STR cache generation so
                cache materialization cannot overwrite the carry-forward.

        Returns:
            StageResult or None if the stage fails.
        """
        runner = self._ensure_runner()

        stage_str_paths, _ = self._generate_str_files(
            input,
            config,
            work_dir,
            stage=stage,
        )
        if not stage_str_paths:
            stage_str_paths = str_paths
        # StrCache returns cache-owned paths.  Materialize stage-local copies
        # before applying carried values so neither the cache nor the carry is
        # lost when generate_sav later prepares its working files.
        local_stage_str_paths: list[Path] = []
        for str_path in stage_str_paths:
            local_path = work_dir / str_path.name
            if str_path != local_path:
                shutil.copy(str_path, local_path)
            local_stage_str_paths.append(local_path)
        stage_str_paths = local_stage_str_paths
        if lattice_start_params:
            self._inject_converged_lattice_into_str(work_dir, lattice_start_params)

        sav_path = generate_sav(
            input.pattern_path,
            stage_str_paths,
            config,
            work_dir,
            stage,
            ml_weight_fractions=ml_weight_fractions if stage == RefinementStage.RIETVELD else None,
        )

        n_phases = len(input.cif_paths)
        _, hard_timeout = runner.compute_timeout(
            n_phases,
            _pattern_point_count(input.pattern_path),
            multiplier=config.timeout_multiplier,
        )

        conv_monitor = ConvergenceMonitor(config.convergence)

        def _on_stdout(line: str) -> bool:
            rwp = parse_rwp_from_line(line)
            if rwp is not None:
                conv_monitor.update(rwp)
            return conv_monitor.should_terminate()

        start_time = time.monotonic()
        try:
            lst_path = runner.run_with_monitor(
                sav_path,
                timeout_seconds=hard_timeout,
                cpu_affinity=config.cpu_affinity,
                on_stdout_line=_on_stdout,
            )
        except (BGMNTimeoutError, BGMNCrashError) as e:
            logger.error("%s failed: %s", stage.value, e)
            return None

        if lst_path is None:
            logger.error("%s returned no result for %s", stage.value, input.hypothesis_id)
            return None

        elapsed = time.monotonic() - start_time
        lst_data = parse_lst(lst_path)

        dia_stem = sav_path.stem
        dia_path = work_dir / f"{dia_stem}.dia"
        dia_data = parse_dia(dia_path)

        result = to_stage_result(
            lst_data,
            dia_data,
            stage,
            initial_lattice_params,
            elapsed,
        )
        return _bind_phase_ids(result, input)

    def _extract_converged_lattice_from_result(
        self,
        stage_result: StageResult,
    ) -> dict[str, dict[str, float]]:
        """Extract converged lattice params from a stage result.

        Returns dict mapping phase_name -> {a, b, c, alpha, beta, gamma} in Angstroms/degrees.
        Used to carry forward converged values to subsequent stages.
        """
        converged: dict[str, dict[str, float]] = {}
        for phase in stage_result.phases:
            params = dict(phase.lattice_params)
            lengths = [params.get(axis) for axis in ("a", "b", "c")]
            if any(
                value is None or not math.isfinite(float(value)) or float(value) <= 0.0
                for value in lengths
            ):
                logger.debug(
                    "Rejecting invalid lattice carry for %s: %s",
                    phase.phase_id,
                    params,
                )
                continue
            uncertainties = phase.lattice_param_errors or {}
            invalid_uncertainties = {
                axis: uncertainty
                for axis, uncertainty in uncertainties.items()
                if axis in ("a", "b", "c")
                and (
                    not math.isfinite(float(uncertainty))
                    or float(uncertainty) < 0.0
                    or float(uncertainty)
                    > abs(float(params[axis])) * _MAX_LATTICE_CARRY_RELATIVE_UNCERTAINTY
                )
            }
            if invalid_uncertainties:
                logger.debug(
                    "Rejecting uncertain lattice carry for %s: values=%s errors=%s",
                    phase.phase_id,
                    params,
                    invalid_uncertainties,
                )
                continue
            converged[phase.name] = params
        return converged

    def _inject_converged_lattice_into_str(
        self,
        work_dir: Path,
        converged_params: dict[str, dict[str, float]],
    ) -> None:
        """Rewrite lattice parameter lines in .str files with converged values.

        Reads each .str in work_dir, replaces PARAM=A=... lines with
        converged values (keeping the same fractional range format).
        Lattice lengths are converted from Angstroms back to nm (÷10).
        """
        for str_path in sorted(work_dir.glob("*.str")):
            phase_name = read_phase_name_from_str(str_path)
            if phase_name not in converged_params:
                continue

            params = converged_params[phase_name]
            text = str_path.read_text()
            lines = text.splitlines()
            updated: list[str] = []

            for line in lines:
                stripped = line.strip()
                # Match lattice parameter lines: PARAM=A=val_lo^hi // or A=val //
                lattice_match = re.match(
                    r"(PARAM=)?(A|B|C|ALPHA|BETA|GAMMA)="
                    r"([\d.]+)(?:_([\d.]+)\^([\d.]+))?",
                    stripped,
                )
                if lattice_match:
                    is_param = lattice_match.group(1) is not None
                    key = lattice_match.group(2)
                    key_lower = key.lower()

                    if key_lower in params:
                        new_val = params[key_lower]
                        # Convert lengths from Angstroms to nm for .str
                        if key in ("A", "B", "C"):
                            new_val_nm = new_val / 10.0
                        else:
                            new_val_nm = new_val

                        if is_param:
                            # Preserve fractional range from original
                            old_center = float(lattice_match.group(3))
                            old_lo = (
                                float(lattice_match.group(4))
                                if lattice_match.group(4)
                                else old_center * 0.99
                            )
                            old_hi = (
                                float(lattice_match.group(5))
                                if lattice_match.group(5)
                                else old_center * 1.01
                            )
                            # Compute fractional range from original
                            if old_center > 0:
                                frac_lo = (old_center - old_lo) / old_center
                                frac_hi = (old_hi - old_center) / old_center
                            else:
                                frac_lo = frac_hi = 0.01
                            new_lo = new_val_nm * (1 - frac_lo)
                            new_hi = new_val_nm * (1 + frac_hi)
                            # Reconstruct the rest of the line after the param
                            rest = stripped[lattice_match.end() :]
                            updated.append(
                                f"PARAM={key}={new_val_nm:.5f}_{new_lo:.5f}^{new_hi:.5f}{rest}"
                            )
                        else:
                            rest = stripped[lattice_match.end() :]
                            updated.append(f"{key}={new_val_nm:.5f}{rest}")
                        continue

                updated.append(line)

            new_text = "\n".join(updated)
            if text.endswith("\n"):
                new_text += "\n"
            str_path.write_text(new_text)

        logger.debug(
            "Injected converged lattice params for %d phases into %s",
            len(converged_params),
            work_dir,
        )

    def _run_rietveld(
        self,
        input: RefinementInput,
        config: RefinementConfig,
        work_dir: Path,
        str_paths: list[Path],
        initial_lattice_params: dict[str, dict[str, float]],
        pawley_result: StageResult | None = None,
    ) -> tuple[StageResult | None, StageResult | None, StageResult | None]:
        """Run staged Rietveld refinement (STEP 5).

        Thesis-adapted staged parameter release:
        1. RIETVELD_LATTICE: Release atomic coords, keep profiles tight
        2. RIETVELD_PROFILE: Release peak width (crystallite size and microstrain)
        3. RIETVELD: Full refinement with texture (SPHAR4)

        Each stage carries forward converged lattice params from the
        previous stage. If an intermediate stage fails, falls through
        to the next stage (best-effort).

        Returns:
            Tuple of (lattice_result, profile_result, full_result).
            Each may be None if that stage failed.
        """
        carried_lattice = (
            self._extract_converged_lattice_from_result(pawley_result)
            if config.enable_pawley_lattice_carry_forward and pawley_result is not None
            else {}
        )

        if not config.enable_staged_rietveld:
            full_result = self._run_single_rietveld_stage(
                input,
                config,
                work_dir,
                str_paths,
                initial_lattice_params,
                RefinementStage.RIETVELD,
                ml_weight_fractions=input.ml_weight_fractions,
                lattice_start_params=carried_lattice,
            )
            if full_result is not None:
                logger.info(
                    "Rietveld (single-stage) done for %s: Rwp=%.2f%%, converged=%s",
                    input.hypothesis_id,
                    full_result.rwp,
                    full_result.converged,
                )
            else:
                logger.warning(
                    "Rietveld (single-stage) failed for %s",
                    input.hypothesis_id,
                )
            return None, None, full_result

        if self._should_skip_intermediate_rietveld(input, pawley_result):
            full_result = self._run_single_rietveld_stage(
                input,
                config,
                work_dir,
                str_paths,
                initial_lattice_params,
                RefinementStage.RIETVELD,
                ml_weight_fractions=input.ml_weight_fractions,
                lattice_start_params=carried_lattice,
            )
            if full_result is not None:
                logger.info(
                    "Skipping staged Rietveld intermediates for %s: "
                    "Pawley Rwp=%.2f%% and ml_confidence=%.3f",
                    input.hypothesis_id,
                    float(pawley_result.rwp),
                    float(input.ml_confidence),
                )
            else:
                logger.warning(
                    "Full Rietveld failed after stage skip for %s",
                    input.hypothesis_id,
                )
            return None, None, full_result

        # Stage 2: Rietveld-Lattice (tight profiles, no LeBail)
        lattice_result = self._run_single_rietveld_stage(
            input,
            config,
            work_dir,
            str_paths,
            initial_lattice_params,
            RefinementStage.RIETVELD_LATTICE,
            lattice_start_params=carried_lattice,
        )
        if lattice_result is not None:
            converged = self._extract_converged_lattice_from_result(lattice_result)
            if converged:
                carried_lattice.update(converged)
                self._inject_converged_lattice_into_str(work_dir, converged)
            logger.info(
                "Rietveld-Lattice done for %s: Rwp=%.2f%%, converged=%s",
                input.hypothesis_id,
                lattice_result.rwp,
                lattice_result.converged,
            )
        else:
            logger.warning(
                "Rietveld-Lattice failed for %s, continuing to profile stage",
                input.hypothesis_id,
            )

        # Stage 3: Rietveld-Profile (release peak shape)
        profile_result = self._run_single_rietveld_stage(
            input,
            config,
            work_dir,
            str_paths,
            initial_lattice_params,
            RefinementStage.RIETVELD_PROFILE,
            lattice_start_params=carried_lattice,
        )
        if profile_result is not None:
            converged = self._extract_converged_lattice_from_result(profile_result)
            if converged:
                carried_lattice.update(converged)
                self._inject_converged_lattice_into_str(work_dir, converged)
            logger.info(
                "Rietveld-Profile done for %s: Rwp=%.2f%%, converged=%s",
                input.hypothesis_id,
                profile_result.rwp,
                profile_result.converged,
            )
        else:
            logger.warning(
                "Rietveld-Profile failed for %s, continuing to full Rietveld",
                input.hypothesis_id,
            )

        # Stage 4: Full Rietveld (texture + final optimization)
        full_result = self._run_single_rietveld_stage(
            input,
            config,
            work_dir,
            str_paths,
            initial_lattice_params,
            RefinementStage.RIETVELD,
            ml_weight_fractions=input.ml_weight_fractions,
            lattice_start_params=carried_lattice,
        )
        if full_result is not None:
            logger.info(
                "Rietveld-Full done for %s: Rwp=%.2f%%, converged=%s",
                input.hypothesis_id,
                full_result.rwp,
                full_result.converged,
            )
        else:
            logger.warning(
                "Rietveld-Full failed for %s, using last good intermediate result",
                input.hypothesis_id,
            )

        return lattice_result, profile_result, full_result

    def _should_skip_intermediate_rietveld(
        self,
        input: RefinementInput,
        pawley_result: StageResult | None,
    ) -> bool:
        """Return True for easy samples that can jump straight to full Rietveld."""
        if pawley_result is None or input.ml_confidence is None:
            return False
        try:
            ml_confidence = float(input.ml_confidence)
            pawley_rwp = float(pawley_result.rwp)
        except (TypeError, ValueError):
            return False
        return (
            ml_confidence >= _EASY_SAMPLE_ML_CONFIDENCE_THRESHOLD
            and pawley_rwp < _EASY_SAMPLE_PAWLEY_RWP_THRESHOLD
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _failure_output(
        self,
        input: RefinementInput,
        config: RefinementConfig,
        reason: str,
        restart_count: int,
    ) -> RefinementOutput:
        """Construct a failure RefinementOutput."""
        # Determine working dir (may not exist yet)
        if config.working_dir is not None:
            work_dir = config.working_dir / input.hypothesis_id
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="axa_bgmn_fail_"))

        work_dir.mkdir(parents=True, exist_ok=True)

        return RefinementOutput(
            hypothesis_id=input.hypothesis_id,
            working_dir=work_dir,
            pawley=None,
            rietveld=None,
            refinement_failed=True,
            needs_human_review=True,
            failure_reason=reason,
            restart_count=restart_count,
        )

    @staticmethod
    def _normalize_lattice_shift_prior(
        lattice_shift_prior: float | None,
    ) -> float | None:
        """Normalize lattice shift prior to finite +/-5% strain."""
        if lattice_shift_prior is None:
            return None
        try:
            parsed = float(lattice_shift_prior)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return max(-0.05, min(0.05, parsed))
