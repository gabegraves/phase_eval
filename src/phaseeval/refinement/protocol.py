"""
RefinementBackend protocol and data models for AgentXRD.

Defines the pluggable interface for refinement engines (BGMN, GSAS-II, etc.)
and the data models for refinement input/output.

Key types:
- RefinementBackend: Protocol that any refinement engine must implement
- RefinementInput: What goes into a refinement run
- StageResult: Result of a single refinement stage (Pawley or Rietveld)
- RefinementOutput: Combined result of a full refinement pipeline
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from phaseeval.refinement.bgmn.result_parser import AngleCorrections
    from phaseeval.refinement.config import RefinementConfig


class RefinementStage(enum.Enum):
    """Refinement stage identifier.

    Four-stage approach (thesis-adapted for BGMN):
    - PAWLEY: Constrained Le Bail fit (Stage 1)
    - RIETVELD_LATTICE: Atomic coords released, tight profiles (Stage 2)
    - RIETVELD_PROFILE: Peak shape released (Stage 3)
    - RIETVELD: Full refinement with texture (Stage 4)
    - RIETVELD_TEXTURE_CONTROL: Stage 4 again with texture off. Evidence about the final stage,
      never a stage of the fit: RefinementOutput.stages and best_result do not hold it.
    """

    PAWLEY = "pawley"
    RIETVELD_LATTICE = "rietveld_lattice"
    RIETVELD_PROFILE = "rietveld_profile"
    RIETVELD = "rietveld"
    RIETVELD_TEXTURE_CONTROL = "rietveld_texture_control"


@dataclass
class RefinementInput:
    """Input specification for a refinement run.

    Attributes:
        pattern_path: Path to the XY diffraction pattern file.
        cif_paths: List of CIF file paths, one per candidate phase.
        str_paths: Pre-generated .str structure files (optional, generated if absent).
        instrument_profile: Instrument profile name or path to .geq file.
        wavelength: X-ray wavelength as element symbol ("Cu") or float (Angstroms).
        working_dir: Directory for BGMN working files.
        hypothesis_id: Tracking ID for this candidate hypothesis.
        ml_weight_fractions: Optional per-phase ML priors aligned with ``cif_paths``.
        ml_confidence: Optional ML confidence for conditional Pawley bypass.
        lattice_shift_prior: Optional hypothesis-level lattice shift prior
            (fractional, e.g. 0.01 = +1%).
        elastic_strain_bounds: Per-phase max elastic strain (fractional).
            Maps phase_id → max_strain.  Overrides config.lattice_range for
            the matching phase.  Use literature elastic limits (sigma_y / E)
            to prevent BGMN from accepting wrong phases via unphysical
            lattice distortion.  Defaults to config.lattice_range when absent.
    """

    pattern_path: Path
    cif_paths: list[Path]
    working_dir: Path
    hypothesis_id: str
    str_paths: list[Path] | None = None
    instrument_profile: str = "Aeris-fds-Pixcel1d-Medipix3"
    wavelength: str | float = "Cu"
    ml_weight_fractions: list[float] | None = None
    ml_confidence: float | None = None
    lattice_shift_prior: float | None = None
    phase_ids: list[str] | None = None
    elastic_strain_bounds: dict[str, float] | None = None


@dataclass
class PhaseRefinementResult:
    """Refined parameters for a single phase.

    Attributes:
        phase_id: Unique identifier for this phase (e.g., COD ID).
        cif_path: Path to the CIF file used.
        name: Human-readable phase name.
        weight_fraction: Weight fraction in percent (0-100), when GSAS-II or
            the active backend reports one. Missing extraction stays missing.
        lattice_params: Refined lattice parameters {a, b, c, alpha, beta, gamma}.
        lattice_param_errors: Optional one-sigma errors for reported lattice parameters.
        lattice_params_initial: Initial lattice parameters from CIF (reference).
        spacegroup_no: International Tables space group number.
        linear_attenuation_per_um: BGMN's my, printed when the fit sets a grain diameter.
        gewicht_usable: The .lst prints a positive GEWICHT larger than its esd. weight_fraction
            alone cannot say, because it reads UNDEF, ERROR and an absent line as zero.
    """

    phase_id: str
    cif_path: Path
    name: str
    weight_fraction: float | None
    lattice_params: dict[str, float]
    lattice_params_initial: dict[str, float]
    spacegroup_no: int | None = None
    lattice_param_errors: dict[str, float] | None = None
    weight_fraction_error: float | None = None
    weight_fraction_unit: str = "percent"
    linear_attenuation_per_um: float | None = None
    gewicht_usable: bool = False


@dataclass
class StageResult:
    """Result from a single refinement stage (Pawley or Rietveld).

    Attributes:
        stage: Which refinement stage produced this result.
        phases: Per-phase refined results.
        rwp: Weighted profile R-factor (%).
        rp: Profile R-factor (%).
        rpb: Background-subtracted R-factor (%).
        rexp: Expected R-factor (%).
        gof: Goodness of fit (Rwp/Rexp).
        rho: BGMN "1-rho" value (percent) for residual correlation.
        converged: Whether the refinement converged.
        elapsed_seconds: Wall-clock time for this stage.
        iteration_count: Number of refinement iterations.
        dia_data: Parsed .dia file data with keys 'observed', 'calculated',
            'difference', 'two_theta' (numpy arrays). Preserved for every
            stage result so residual patterns are always available.
        durbin_watson: Durbin-Watson statistic from the BGMN .lst output.
        failure_reason: Cause of an incomplete stage, such as a DIA-only timeout.
        angle_corrections: The EPS values this stage's .lst lists.
    """

    stage: RefinementStage
    phases: list[PhaseRefinementResult]
    rwp: float
    rp: float | None
    rpb: float | None
    rexp: float | None
    gof: float | None
    rho: float | None
    converged: bool | None
    elapsed_seconds: float
    iteration_count: int
    dia_data: dict[str, Any] | None = None
    durbin_watson: float | None = None
    metric_name: str = "Rwp"
    refinement_settings: dict[str, Any] | None = None
    uncertainty_diagnostics: dict[str, Any] | None = None
    failure_reason: str | None = None
    angle_corrections: AngleCorrections | None = None


@dataclass
class RefinementOutput:
    """Combined output from a full refinement pipeline.

    Contains results from both Pawley and Rietveld stages, plus metadata
    about the refinement process.

    Attributes:
        pawley: Result from Pawley/Le Bail stage (None if skipped).
        rietveld: Result from Rietveld stage (None if skipped or gated).
        hypothesis_id: Tracking ID matching the input hypothesis.
        refinement_failed: True if refinement completely failed.
        needs_human_review: True if results need expert review.
        failure_reason: Description of why refinement failed (if applicable).
        working_dir: Directory containing all refinement files.
        restart_count: Number of restarts needed (telemetry).
        backend_name: Refinement backend that produced the output, when known.
        backend_version: Backend version or source revision, when available.
        calibration_status: Backend-specific calibration status. Consumers must
            not treat uncalibrated backend metrics as equivalent to BGMN metrics.
        reported_rwp: Optional externally reported summary Rwp. Native stage
            Rwps remain unchanged and authoritative for stage comparisons.
        rietveld_texture_control: The final stage refit with texture off, for TextureDependence.
            None when it does not apply or failed.
    """

    hypothesis_id: str
    working_dir: Path
    pawley: StageResult | None = None
    rietveld_lattice: StageResult | None = None
    rietveld_profile: StageResult | None = None
    rietveld: StageResult | None = None
    refinement_failed: bool = False
    needs_human_review: bool = False
    failure_reason: str | None = None
    restart_count: int = 0
    backend_name: str | None = None
    backend_version: str | None = None
    calibration_status: str | None = None
    reported_rwp: float | None = None
    artifact_paths: dict[str, Path] | None = None
    backend_identity: dict[str, Any] | None = None
    refinement_settings: dict[str, Any] | None = None
    parent_project_path: Path | None = None
    rietveld_texture_control: StageResult | None = None

    @property
    def best_result(self) -> StageResult | None:
        """Return the best available stage result.

        Prefers later stages (more refined). Returns None if
        no stage produced results.
        """
        for stage_result in (
            self.rietveld,
            self.rietveld_profile,
            self.rietveld_lattice,
            self.pawley,
        ):
            if stage_result is not None:
                return stage_result
        return None

    @property
    def stages(self) -> list[StageResult]:
        """Return populated stage results in execution order."""
        return [
            stage_result
            for stage_result in (
                self.pawley,
                self.rietveld_lattice,
                self.rietveld_profile,
                self.rietveld,
            )
            if stage_result is not None
        ]

    @property
    def final_rwp(self) -> float | None:
        """Return the reported summary Rwp, otherwise the best native stage Rwp."""
        reported = _finite_float(self.reported_rwp)
        if reported is not None:
            return reported
        best = self.best_result
        return best.rwp if best is not None else None

    @property
    def lattice_params(self) -> dict[str, dict[str, float]]:
        """Return final per-phase lattice parameters keyed by phase ID."""
        best = self.best_result
        if best is None:
            return {}
        return {phase.phase_id: phase.lattice_params for phase in best.phases}

    @property
    def weight_fractions(self) -> dict[str, float | None]:
        """Return final per-phase weight fractions keyed by phase ID."""
        best = self.best_result
        if best is None:
            return {}
        return {phase.phase_id: phase.weight_fraction for phase in best.phases}

    @property
    def residual_pattern(self) -> dict[str, Any] | None:
        """Return dia_data from the best result.

        Convenience property for downstream consumers that need residual
        patterns for every hypothesis.
        """
        best = self.best_result
        if best is not None:
            return best.dia_data
        return None


@dataclass(frozen=True)
class StageDegradationEvidence:
    """Native Pawley/Rietveld comparison with conservative interpretation."""

    pawley_rwp: float | None
    rietveld_rwp: float | None
    rietveld_minus_pawley_rwp: float | None
    good_pawley_rwp_threshold: float | None
    rietveld_degradation_delta_rwp_threshold: float | None
    selected_stage: str | None
    comparison_available: bool
    rietveld_degraded_after_good_pawley: bool
    reason_code: str | None
    interpretation: str | None
    alternative_explanations: tuple[str, ...]
    backend_name: str | None
    backend_version: str | None
    calibration_status: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pawley_rwp": self.pawley_rwp,
            "rietveld_rwp": self.rietveld_rwp,
            "rietveld_minus_pawley_rwp": self.rietveld_minus_pawley_rwp,
            "good_pawley_rwp_threshold": self.good_pawley_rwp_threshold,
            "rietveld_degradation_delta_rwp_threshold": (
                self.rietveld_degradation_delta_rwp_threshold
            ),
            "selected_stage": self.selected_stage,
            "comparison_available": self.comparison_available,
            "rietveld_degraded_after_good_pawley": (
                self.rietveld_degraded_after_good_pawley
            ),
            "reason_code": self.reason_code,
            "interpretation": self.interpretation,
            "alternative_explanations": list(self.alternative_explanations),
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "calibration_status": self.calibration_status,
        }


def calculate_stage_degradation_evidence(
    output: RefinementOutput,
    *,
    good_pawley_rwp_threshold: float = 15.0,
    rietveld_degradation_delta_rwp_threshold: float = 10.0,
    selected_stage: str | None = None,
) -> StageDegradationEvidence:
    """Compare native finite stages without choosing a diagnosis or remedy."""

    pawley_rwp = _finite_stage_rwp(output.pawley)
    rietveld_rwp = _finite_stage_rwp(output.rietveld)
    good_threshold = _finite_float(good_pawley_rwp_threshold)
    degradation_threshold = _finite_float(rietveld_degradation_delta_rwp_threshold)
    comparison_available = bool(
        not output.refinement_failed
        and output.pawley is not None
        and output.rietveld is not None
        and output.pawley.converged
        and output.rietveld.converged
        and pawley_rwp is not None
        and rietveld_rwp is not None
        and good_threshold is not None
        and degradation_threshold is not None
    )
    if comparison_available:
        assert pawley_rwp is not None
        assert rietveld_rwp is not None
        assert good_threshold is not None
        assert degradation_threshold is not None
        delta = rietveld_rwp - pawley_rwp
        degraded = pawley_rwp <= good_threshold and delta >= degradation_threshold
    else:
        delta = None
        degraded = False
    if selected_stage is None and output.best_result is not None:
        selected_stage = output.best_result.stage.value

    return StageDegradationEvidence(
        pawley_rwp=pawley_rwp,
        rietveld_rwp=rietveld_rwp,
        rietveld_minus_pawley_rwp=delta,
        good_pawley_rwp_threshold=good_threshold,
        rietveld_degradation_delta_rwp_threshold=degradation_threshold,
        selected_stage=selected_stage,
        comparison_available=comparison_available,
        rietveld_degraded_after_good_pawley=degraded,
        reason_code="rietveld_degraded_after_good_pawley" if degraded else None,
        interpretation=(
            "The peak-position/cell fit is substantially better than the supplied "
            "structural intensity model."
            if degraded
            else None
        ),
        alternative_explanations=(
            "texture",
            "occupancy",
            "microstructure",
            "profile_mismatch",
            "structure_or_cif_choice",
        ),
        backend_name=output.backend_name,
        backend_version=output.backend_version,
        calibration_status=output.calibration_status,
    )


def texture_order_of(gewicht: Any) -> int | None:
    """The spherical harmonic order of a stage's GEWICHT setting; 0 for no texture model, None
    when the setting is neither."""
    if gewicht == "0_0":
        return 0
    match = re.fullmatch(r"SPHAR(\d{1,2})", gewicht) if isinstance(gewicht, str) else None
    return int(match[1]) if match else None


# Connolly expects repeat runs to agree only within several percent. Pre-registered: do not tune.
TEXTURE_DEPENDENT_WT_POINTS = 5.0


@dataclass(frozen=True)
class TextureDependence:
    """The amounts from the texture control fit (same fit, texture off) to the final fit."""

    rwp_control: float
    rwp_final: float
    weight_percent_control: dict[str, float | None]
    weight_percent_final: dict[str, float | None]
    largest_change_wt_points: float
    largest_change_phase: str
    exceeds_threshold: bool


def texture_dependence(output: RefinementOutput) -> TextureDependence | None:
    """None without a converged control fit and a final stage, or when any phase lacks a usable
    amount in either fit."""
    control, final = output.rietveld_texture_control, output.rietveld
    if control is None or final is None or not control.converged:
        return None
    if any(p.weight_fraction is None or not p.gewicht_usable for p in (*control.phases, *final.phases)):
        return None
    weights_control = {p.phase_id: p.weight_fraction for p in control.phases}
    weights_final = {p.phase_id: p.weight_fraction for p in final.phases}
    if not weights_control or set(weights_control) != set(weights_final):
        return None
    changes = {phase: abs(weights_final[phase] - weight) for phase, weight in weights_control.items()}
    phase = max(changes, key=lambda name: changes[name])
    return TextureDependence(
        rwp_control=control.rwp,
        rwp_final=final.rwp,
        weight_percent_control=weights_control,
        weight_percent_final=weights_final,
        largest_change_wt_points=changes[phase],
        largest_change_phase=phase,
        exceeds_threshold=changes[phase] > TEXTURE_DEPENDENT_WT_POINTS,
    )


def texture_dependence_text(dependence: Mapping[str, Any]) -> str:
    """One line on a TextureDependence in its result.json form."""
    text = (
        f"Texture dependence, refit with texture off against the final fit: Rwp {dependence['rwp_control']:.2f}% off, "
        f"{dependence['rwp_final']:.2f}% on; largest amount change {dependence['largest_change_wt_points']:.2f} "
        f"wt% points on {dependence['largest_change_phase']}."
    )
    if dependence["exceeds_threshold"]:
        text += (
            f" Over {TEXTURE_DEPENDENT_WT_POINTS} points: the amounts depend on the texture model. "
            "This check does not affect the phase identification."
        )
    return text


def _finite_stage_rwp(stage: StageResult | None) -> float | None:
    return _finite_float(stage.rwp) if stage is not None else None


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@runtime_checkable
class RefinementBackend(Protocol):
    """Protocol defining the interface for refinement engines.

    Any refinement backend (BGMN, GSAS-II, etc.) must implement this
    protocol. Decorated with @runtime_checkable for isinstance() checks.

    Example:
        class BGMNBackend:
            @property
            def name(self) -> str:
                return "bgmn"

            def is_available(self) -> bool:
                return BGMNInstaller().is_installed()

            def refine(self, input, config) -> RefinementOutput:
                ...

            def dry_run(self, input, config) -> dict:
                ...

        assert isinstance(BGMNBackend(), RefinementBackend)
    """

    @property
    def name(self) -> str:
        """Human-readable name of this refinement backend."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is ready to use.

        Returns True if the backend executable is installed and accessible.
        """
        ...

    def refine(self, input: RefinementInput, config: RefinementConfig) -> RefinementOutput:
        """Run a full refinement (Pawley + optional Rietveld).

        Args:
            input: Refinement input specification.
            config: Refinement configuration parameters.

        Returns:
            RefinementOutput with stage results and metadata.
        """
        ...

    def dry_run(self, input: RefinementInput, config: RefinementConfig) -> dict:
        """Validate inputs and estimate runtime without executing refinement.

        Args:
            input: Refinement input specification.
            config: Refinement configuration parameters.

        Returns:
            Dict with validation results, estimated runtime, etc.
        """
        ...
