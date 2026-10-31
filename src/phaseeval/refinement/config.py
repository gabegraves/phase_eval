"""
Refinement configuration for AgentXRD.

Four-stage sequential refinement following the thesis insight that peak
positions must be established before widths.  Stages progressively unlock
parameters:

  Stage 1 (Pawley):          Lattice + EPS2 only.  k1/k2/B1 fixed at 0.
  Stage 2 (Rietveld-Lattice): + atoms + tight k1/B1 (DARA defaults).
  Stage 3 (Rietveld-Profile): Wider k1/B1, k2 refinable.
  Stage 4 (Rietveld-Full):    Widest k1/B1, GEWICHT=SPHAR4 texture.

DARA reference values (dara-main/src/dara/search/core.py):
    DEFAULT_PHASE_PARAMS = {
        "gewicht": "0_0",
        "lattice_range": 0.01,
        "k1": "0_0^0.01",
        "k2": "fixed",
        "b1": "0_0^0.005",
        "rp": 4,
    }
    DEFAULT_REFINEMENT_PARAMS = {"n_threads": 8, "eps1": 0, "eps2": "0_-0.05^0.05"}
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# Crystal system multipliers: high-symmetry → tighter bounds, low-symmetry → wider.
# Cubic has fewest real DOF (a=b=c, α=β=γ=90°), so bounds should be tightest.
# Triclinic has the most DOF (all 6 independent), so bounds need the most room.
_DEFAULT_SYMMETRY_MULTIPLIERS: dict[str, float] = {
    "Cubic": 0.5,
    "Hexagonal": 0.75,
    "Trigonal": 0.75,
    "Tetragonal": 0.75,
    "Orthorhombic": 1.0,
    "Monoclinic": 1.5,
    "Triclinic": 2.0,
}


@dataclass
class SymmetryBoundsPolicy:
    """Per-crystal-system multipliers for profile/lattice bounds.

    High-symmetry (cubic) → tighter bounds (fewer real DOF).
    Low-symmetry (triclinic) → wider bounds (more DOF need room).

    Reference: MILK (BSD-3, LANL) parameterized refinement bounds pattern.

    Attributes:
        lattice_range_multiplier: Crystal-system → multiplier for lattice_range.
        k1_max_multiplier: Crystal-system → multiplier for k1 upper bound.
        b1_max_multiplier: Crystal-system → multiplier for b1 upper bound.
    """

    lattice_range_multiplier: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SYMMETRY_MULTIPLIERS)
    )
    k1_max_multiplier: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SYMMETRY_MULTIPLIERS)
    )
    b1_max_multiplier: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SYMMETRY_MULTIPLIERS)
    )

    @classmethod
    def default(cls) -> SymmetryBoundsPolicy:
        return cls()

    def get_multiplier(self, crystal_system: str, param: str) -> float:
        """Get the multiplier for a crystal system and parameter.

        Args:
            crystal_system: Capitalized crystal system name (e.g., "Cubic").
            param: One of "lattice_range", "k1", "b1".

        Returns:
            Multiplier (1.0 if crystal_system not recognized).
        """
        table = {
            "lattice_range": self.lattice_range_multiplier,
            "k1": self.k1_max_multiplier,
            "b1": self.b1_max_multiplier,
        }.get(param, {})
        return table.get(crystal_system, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lattice_range_multiplier": dict(self.lattice_range_multiplier),
            "k1_max_multiplier": dict(self.k1_max_multiplier),
            "b1_max_multiplier": dict(self.b1_max_multiplier),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymmetryBoundsPolicy:
        return cls(
            lattice_range_multiplier=d.get(
                "lattice_range_multiplier", dict(_DEFAULT_SYMMETRY_MULTIPLIERS)
            ),
            k1_max_multiplier=d.get("k1_max_multiplier", dict(_DEFAULT_SYMMETRY_MULTIPLIERS)),
            b1_max_multiplier=d.get("b1_max_multiplier", dict(_DEFAULT_SYMMETRY_MULTIPLIERS)),
        )


@dataclass
class PawleyConfig:
    """Stage 1: Strict Pawley/Le Bail — lattice parameters only.

    Fixes peak shape parameters (k1, k2, B1) at zero so BGMN focuses
    exclusively on refining lattice parameters and sample displacement
    (EPS2).  This follows the thesis insight that "the biggest chance to
    cause peak overlap is not getting the peaks centered from the start":
    locking widths forces all degrees of freedom into peak *positions*.

    Peak profile parameters are unlocked in Stage 2 (RietveldLatticeConfig)
    and progressively widened in Stages 3-4.

    Attributes:
        lattice_range: Fractional strain range (0.01 = 1%).
        k1: Crystallite-size distribution width range (tight; small k1 is a wide distribution).
        k2: Square of rms microstrain, fixed at zero.
        b1: Crystallite broadening range (tight, allows minimal size broadening).
        rp: Peak profile function (4 = Cauchy^2).
        gewicht: Weight scheme ("0_0" = Le Bail mode).
        lebail: Whether this is a Le Bail fit (always True for Pawley stage).
    """

    lattice_range: float = 0.01
    k1: str = "0_0^0.01"
    k2: str = "fixed"
    b1: str = "0_0^0.005"
    rp: int = 4
    gewicht: str = "0_0"
    lebail: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lattice_range": self.lattice_range,
            "k1": self.k1,
            "k2": self.k2,
            "b1": self.b1,
            "rp": self.rp,
            "gewicht": self.gewicht,
            "lebail": self.lebail,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PawleyConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RietveldLatticeConfig:
    """Stage 2: Rietveld — unlock peak shape with tight constraints.

    Releases atomic coordinates (LEBAIL=no) and introduces tight peak
    profile refinement (k1, B1) that was locked at zero in Pawley.
    Equivalent to thesis step: positions + basic peak shapes.

    This is the first stage where k1 and B1 are refinable, inheriting
    the DARA DEFAULT_PHASE_PARAMS values that Pawley previously used.

    Attributes:
        lattice_range: Fractional strain range (same as Pawley).
        k1: Crystallite-size distribution width range (tight, first stage refinable).
        k2: Square of rms microstrain (fixed).
        b1: Crystallite size broadening range (tight, first stage refinable).
        rp: Peak profile function (4 = Cauchy^2).
        gewicht: Weight scheme ("0_0" = no preferred orientation yet).
    """

    lattice_range: float = 0.01
    k1: str = "0_0^0.01"
    k2: str = "fixed"
    b1: str = "0_0^0.005"
    rp: int = 4
    gewicht: str = "0_0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lattice_range": self.lattice_range,
            "k1": self.k1,
            "k2": self.k2,
            "b1": self.b1,
            "rp": self.rp,
            "gewicht": self.gewicht,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RietveldLatticeConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RietveldProfileConfig:
    """Stage 3: Rietveld with released peak shape parameters.

    Releases the crystallite-size distribution width, crystallite size and microstrain.
    Equivalent to thesis step (2): scale widths/broadening.

    Attributes:
        lattice_range: Fractional strain range (same as Pawley).
        k1: Crystallite-size distribution width range (wider).
        k2: Square of rms microstrain (refinable).
        b1: Crystallite size broadening range (moderate).
        rp: Peak profile function (4 = Cauchy^2).
        gewicht: Weight scheme ("0_0" = no preferred orientation yet).
    """

    lattice_range: float = 0.01
    k1: str = "0_0^0.1"
    k2: str = "0_0^0.01"
    b1: str = "0_0^0.01"
    rp: int = 4
    gewicht: str = "0_0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lattice_range": self.lattice_range,
            "k1": self.k1,
            "k2": self.k2,
            "b1": self.b1,
            "rp": self.rp,
            "gewicht": self.gewicht,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RietveldProfileConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RietveldConfig:
    """Stage 4: Full Rietveld refinement parameters.

    Wider parameter ranges after intermediate stages establish fit.
    Key differences from earlier stages:
    - gewicht="SPHAR4": enables spherical harmonics for preferred orientation
    - k1="0_0^1": crystallite-size distribution width range
    - k2="0_0^0.01": square of rms microstrain, refinable
    - b1="0_0^0.05": 10x wider crystallite broadening range

    Attributes:
        lattice_range: Fractional strain range (same as Pawley).
        k1: Crystallite-size distribution width range.
        k2: Square of rms microstrain (refinable in Rietveld).
        b1: Crystallite size broadening range (10x wider than Pawley).
        rp: Peak profile function (4 = Cauchy^2).
        gewicht: Preferred orientation model (e.g. "SPHAR4").
    """

    lattice_range: float = 0.01
    k1: str = "0_0^1"
    k2: str = "0_0^0.01"
    b1: str = "0_0^0.05"
    rp: int = 4
    gewicht: str = "SPHAR4"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "lattice_range": self.lattice_range,
            "k1": self.k1,
            "k2": self.k2,
            "b1": self.b1,
            "rp": self.rp,
            "gewicht": self.gewicht,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RietveldConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RefinementConfig:
    """Top-level refinement configuration.

    Combines stage-specific parameters with global refinement settings.
    Sensible defaults match DARA's validated parameters.

    Attributes:
        pawley: Stage 1 (Pawley/Le Bail) parameters.
        rietveld: Stage 2 (Rietveld) parameters.
        sample_type: Sample preparation type affecting texture models.
        max_phases: Maximum simultaneous phases in refinement.
        n_threads: BGMN thread count (per DARA default: 8).
        eps1: BGMN zero-point angle correction. A number fixes it; a bound string refines it.
        eps2: BGMN sample displacement s/R, refined within this bound.
        max_restarts: Maximum restart attempts per candidate.
            Defaults to 0 because recent real-data failures are deterministic
            bad inputs / unstable stages, not transient subprocess hiccups.
            False-convergence recovery remains handled separately via the
            explicit Pawley Rpb=100 retry path.
        rpb_improvement_threshold: Minimum Rpb improvement (pp) vs background-only.
        pawley_rwp_threshold: Maximum Pawley Rwp (%) to attempt Rietveld.
        pawley_quality_gate: Skip Rietveld if Pawley fails.
        pawley_only: Stop after Pawley when the caller only needs diagnostic
            Pawley evidence.
        pawley_only_reason: Optional reason recorded when pawley_only stops
            before Rietveld.
        enable_ml_pawley_bypass: Allow direct Rietveld when ML confidence is high.
        pawley_bypass_confidence: Confidence threshold for bypass (0.99 = 99%).
        enable_staged_rietveld: Run intermediate lattice/profile stages
            before full Rietveld.
        enable_pawley_lattice_carry_forward: Carry converged Pawley lattice
            values into Rietveld input STR files.
        instrument_profile: Instrument profile name or path to .geq file.
        wavelength: X-ray wavelength as element symbol or float (Angstroms).
        working_dir: Base directory for refinement working files.
        prefer_tmpfs: Prefer /dev/shm temp directories when working_dir is unset.
        cpu_affinity: Optional Linux taskset CPU range (e.g. "0-7").
        preserve_temp_files: Keep BGMN temp files for debugging.
        verbose: Enable verbose logging.
        debug: Enable debug-level logging.
        dry_run: Validate without executing refinement.
        gsasii_failure_wr_threshold: Optional native GSAS-II wR threshold that
            marks GSAS-II refinements as failed without raising.
        gsasii_refine_instrument_profile: Refine U/V/W instrument terms. Keep
            false when assessing sample broadening or using a supplied profile.
    """

    pawley: PawleyConfig = field(default_factory=PawleyConfig)
    rietveld_lattice: RietveldLatticeConfig = field(default_factory=RietveldLatticeConfig)
    rietveld_profile: RietveldProfileConfig = field(default_factory=RietveldProfileConfig)
    rietveld: RietveldConfig = field(default_factory=RietveldConfig)
    phase_stage_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    sample_type: Literal["powder", "thin_film", "bulk"] = "powder"
    max_phases: int = 5
    n_threads: int = 8
    eps1: float = 0.0
    eps2: str = "0_-0.05^0.05"
    # BGMN background polynomial degree (RU=). None keeps BGMN's default.
    background_order: int | None = None
    max_restarts: int = 0
    rpb_improvement_threshold: float = 2.0
    pawley_rwp_threshold: float | None = 35.0
    pawley_quality_gate: bool = True
    pawley_only: bool = False
    pawley_only_reason: str | None = None
    enable_ml_pawley_bypass: bool = True
    pawley_bypass_confidence: float = 0.90
    convergence: Any = field(
        default_factory=lambda: __import__(
            "phaseeval.refinement.bgmn.convergence_monitor", fromlist=["ConvergenceConfig"]
        ).ConvergenceConfig()
    )
    timeout_multiplier: float = 1.5  # Hard timeout = soft * multiplier (was 2.0)
    enable_staged_rietveld: bool = True
    enable_pawley_lattice_carry_forward: bool = True
    symmetry_bounds: SymmetryBoundsPolicy | None = field(
        default_factory=SymmetryBoundsPolicy.default
    )
    instrument_profile: str = "Aeris-fds-Pixcel1d-Medipix3"
    wavelength: str | float = "Cu"
    working_dir: Path | None = None
    prefer_tmpfs: bool = True
    cpu_affinity: str | None = None
    preserve_temp_files: bool = True
    verbose: bool = False
    debug: bool = False
    dry_run: bool = False
    gsasii_failure_wr_threshold: float | None = None
    gsasii_refine_instrument_profile: bool = False
    gsasii_refine_atomic_coordinates: bool = False
    gsasii_refine_sample_microstrain: bool = False
    gsasii_refine_sample_shift: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Paths are converted to strings for YAML/JSON compatibility.
        """
        d: dict[str, Any] = {
            "pawley": self.pawley.to_dict(),
            "rietveld_lattice": self.rietveld_lattice.to_dict(),
            "rietveld_profile": self.rietveld_profile.to_dict(),
            "rietveld": self.rietveld.to_dict(),
            "phase_stage_overrides": copy.deepcopy(self.phase_stage_overrides),
            "sample_type": self.sample_type,
            "max_phases": self.max_phases,
            "n_threads": self.n_threads,
            "eps1": self.eps1,
            "eps2": self.eps2,
            "background_order": self.background_order,
            "max_restarts": self.max_restarts,
            "rpb_improvement_threshold": self.rpb_improvement_threshold,
            "pawley_rwp_threshold": self.pawley_rwp_threshold,
            "pawley_quality_gate": self.pawley_quality_gate,
            "pawley_only": self.pawley_only,
            "pawley_only_reason": self.pawley_only_reason,
            "enable_ml_pawley_bypass": self.enable_ml_pawley_bypass,
            "pawley_bypass_confidence": self.pawley_bypass_confidence,
            "enable_staged_rietveld": self.enable_staged_rietveld,
            "enable_pawley_lattice_carry_forward": self.enable_pawley_lattice_carry_forward,
            "symmetry_bounds": self.symmetry_bounds.to_dict() if self.symmetry_bounds else None,
            "instrument_profile": self.instrument_profile,
            "wavelength": self.wavelength,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "prefer_tmpfs": self.prefer_tmpfs,
            "cpu_affinity": self.cpu_affinity,
            "preserve_temp_files": self.preserve_temp_files,
            "verbose": self.verbose,
            "debug": self.debug,
            "dry_run": self.dry_run,
            "gsasii_failure_wr_threshold": self.gsasii_failure_wr_threshold,
            "gsasii_refine_instrument_profile": self.gsasii_refine_instrument_profile,
            "gsasii_refine_atomic_coordinates": self.gsasii_refine_atomic_coordinates,
            "gsasii_refine_sample_microstrain": self.gsasii_refine_sample_microstrain,
            "gsasii_refine_sample_shift": self.gsasii_refine_sample_shift,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RefinementConfig:
        """Create from dictionary.

        Handles nested PawleyConfig and RietveldConfig dicts.
        """
        d = copy.deepcopy(d)

        # Parse nested configs
        if "pawley" in d and isinstance(d["pawley"], dict):
            d["pawley"] = PawleyConfig.from_dict(d["pawley"])
        if "rietveld_lattice" in d and isinstance(d["rietveld_lattice"], dict):
            d["rietveld_lattice"] = RietveldLatticeConfig.from_dict(d["rietveld_lattice"])
        if "rietveld_profile" in d and isinstance(d["rietveld_profile"], dict):
            d["rietveld_profile"] = RietveldProfileConfig.from_dict(d["rietveld_profile"])
        if "rietveld" in d and isinstance(d["rietveld"], dict):
            d["rietveld"] = RietveldConfig.from_dict(d["rietveld"])
        if "symmetry_bounds" in d:
            sb = d["symmetry_bounds"]
            if isinstance(sb, dict):
                d["symmetry_bounds"] = SymmetryBoundsPolicy.from_dict(sb)
            elif sb is None:
                d["symmetry_bounds"] = None

        # Convert working_dir string back to Path
        if "working_dir" in d and d["working_dir"] is not None:
            d["working_dir"] = Path(d["working_dir"])

        # Filter to known fields only
        known_fields = set(cls.__dataclass_fields__.keys())
        d = {k: v for k, v in d.items() if k in known_fields}

        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RefinementConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            RefinementConfig populated from the file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to write the YAML configuration file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def adjust_for_sample_type(self) -> RefinementConfig:
        """Adjust parameters based on sample_type.

        Widens lattice_range for thin_film and bulk samples where
        greater lattice distortion is expected (e.g., strain in thin films,
        HEA lattice distortion).

        Returns:
            Self (mutated) for chaining.
        """
        if self.sample_type == "thin_film":
            # Thin films: 2x wider lattice range for substrate strain.
            # 5x (0.05) caused BGMN timeout/divergence on all 48 EMPA
            # samples (2026-04-16); 2x is the proven safe bound (= bulk).
            self.pawley.lattice_range = 0.02
            self.rietveld_lattice.lattice_range = 0.02
            self.rietveld_profile.lattice_range = 0.02
            self.rietveld.lattice_range = 0.02
        elif self.sample_type == "bulk":
            # Bulk: 2x wider for potential HEA/solid-solution distortion
            self.pawley.lattice_range = 0.02
            self.rietveld_lattice.lattice_range = 0.02
            self.rietveld_profile.lattice_range = 0.02
            self.rietveld.lattice_range = 0.02
        # powder: keep defaults (0.01)

        return self
