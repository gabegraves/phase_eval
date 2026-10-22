"""Resolve instrument profiles for data sources.

Maps opXRD data sources (EMPA, USC, LBNL, ...) to the correct BGMN .geq
instrument profile.  When no source-specific profile is configured, falls
back to the default with a logged warning so profile mismatches are
immediately visible in triage output.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parent / "data" / "instrument_profiles.yaml"
_DEFAULT_PROFILE_METADATA = Path(__file__).parent / "data" / "profile_metadata.yaml"
_DEVICES_DIR = Path(__file__).parent / "data" / "devices"

ProfileConfidence = Literal["matched", "fallback", "missing", "inferred", "screened"]


@dataclass
class InstrumentSpec:
    """Resolved instrument specification for a data source."""

    profile: str
    wavelength: str | float
    is_default: bool  # True when fallback profile was used
    sample_type: str = "powder"  # "powder", "thin_film", or "bulk"


@dataclass
class ResolvedInstrumentPolicy:
    """Pure policy output shared by benchmark and product metrology callers."""

    source: str
    spec: InstrumentSpec
    profile_confidence: ProfileConfidence
    pattern_wavelength: float | None
    effective_wavelength_for_transform: float | None
    wavelength_provenance: str | None


@dataclass
class PreflightResult:
    """Result of instrument compatibility preflight checks."""

    ok: bool
    warnings: list[str]


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and return the instrument profile YAML config."""
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        logger.warning("Instrument config not found: %s — using hardcoded defaults", path)
        return {
            "default_profile": "Aeris-fds-Pixcel1d-Medipix3",
            "default_wavelength": "Cu",
            "sources": {},
        }
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_profile_metadata(metadata_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load per-.geq profile metadata keyed by profile basename."""
    path = metadata_path or _DEFAULT_PROFILE_METADATA
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict):
        return {}
    return {
        str(name): dict(meta or {})
        for name, meta in profiles.items()
        if isinstance(meta, dict)
    }


def available_profile_names(metadata_path: Path | None = None) -> set[str]:
    """Return known profile basenames from metadata and the devices directory."""
    names = set(load_profile_metadata(metadata_path).keys())
    if _DEVICES_DIR.exists():
        names.update(path.stem for path in _DEVICES_DIR.glob("*.geq"))
    return names


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _profile_from_metadata_entry(
    profile_name: str,
    profile_meta: dict[str, Any],
    metadata: dict[str, Any],
) -> InstrumentSpec:
    wavelength = (
        metadata.get("instrument_wavelength")
        or metadata.get("wavelength_angstrom")
        or profile_meta.get("wavelength_class")
        or "Cu"
    )
    return InstrumentSpec(
        profile=profile_name,
        wavelength=wavelength,
        is_default=False,
        sample_type=str(metadata.get("sample_type") or profile_meta.get("sample_type") or "powder"),
    )


def _representative_for_family(
    profiles: dict[str, dict[str, Any]],
    family: str,
) -> tuple[str, dict[str, Any]] | None:
    family_entries = [
        (name, meta)
        for name, meta in profiles.items()
        if str(meta.get("family", "")) == family
    ]
    if not family_entries:
        return None
    for name, meta in family_entries:
        if bool(meta.get("representative", False)):
            return name, meta
    return sorted(family_entries, key=lambda item: item[0])[0]


def infer_profile_from_metadata(
    metadata: dict[str, Any],
    *,
    metadata_path: Path | None = None,
) -> InstrumentSpec | None:
    """Infer a profile from instrument metadata without running refinement.

    This is intentionally deterministic and conservative. It only maps explicit
    profile names or recognizable manufacturer/model/detector tokens to a
    representative profile family; ambiguous source-type-only lab metadata is
    not enough to select a profile.
    """
    if not isinstance(metadata, dict):
        return None

    profiles = load_profile_metadata(metadata_path)
    if not profiles:
        return None

    explicit_profile = metadata.get("instrument_profile")
    if explicit_profile:
        profile_name = Path(str(explicit_profile)).stem
        if profile_name in profiles or profile_name in available_profile_names(metadata_path):
            return _profile_from_metadata_entry(profile_name, profiles.get(profile_name, {}), metadata)

    metadata_text = " ".join(
        _normalise_token(metadata.get(key))
        for key in (
            "measuring_instrument",
            "instrument_manufacturer",
            "instrument_model",
            "detector_type",
            "source_type",
        )
        if metadata.get(key)
    )
    if not metadata_text:
        return None

    if "synchrotron" in metadata_text:
        entry = _representative_for_family(profiles, "synchrotron")
        if entry is not None:
            return _profile_from_metadata_entry(entry[0], entry[1], metadata)

    family: str | None = None
    if "miniflex" in metadata_text:
        family = "rigaku_miniflex"
    elif "smartlab" in metadata_text:
        family = "rigaku_smartlab"
    elif "rigaku" in metadata_text and "ultima" in metadata_text:
        family = "rigaku_ultima"
    elif "bruker" in metadata_text and "d2" in metadata_text:
        family = "bruker_d2"
    elif "lynxeye" in metadata_text or ("bruker" in metadata_text and "d8" in metadata_text):
        family = "bruker_d8_lynxeye"
    elif "solxe" in metadata_text:
        family = "bruker_d8_solxe"
    elif "aeris" in metadata_text:
        family = "panalytical_aeris"
    elif "cubix" in metadata_text:
        family = "panalytical_cubix"
    elif "pw1800" in metadata_text:
        family = "panalytical_pw1800"
    elif (
        "pw3040" in metadata_text
        or "xcelerator" in metadata_text
        or "celerator" in metadata_text
        or "x cel" in metadata_text
    ):
        family = "panalytical_xcelerator"
    elif "pixcel" in metadata_text:
        family = "panalytical_xpert_pixcel"
    elif "siemens" in metadata_text or "d5000" in metadata_text:
        family = "siemens_d5000"

    if family is None:
        return None
    entry = _representative_for_family(profiles, family)
    if entry is None:
        return None
    return _profile_from_metadata_entry(entry[0], entry[1], metadata)


def resolve_instrument(
    source: str,
    config_path: Path | None = None,
    *,
    _config_cache: dict[str, Any] | None = None,
) -> InstrumentSpec:
    """Resolve instrument profile for a data source.

    Args:
        source: Data source identifier (e.g. "EMPA", "USC").
        config_path: Optional path to instrument_profiles.yaml.
        _config_cache: Pre-loaded config dict (avoids re-reading YAML).

    Returns:
        InstrumentSpec with the resolved profile and wavelength.
    """
    cfg = _config_cache if _config_cache is not None else _load_config(config_path)
    default_profile = cfg.get("default_profile", "Aeris-fds-Pixcel1d-Medipix3")
    default_wavelength = cfg.get("default_wavelength", "Cu")

    sources = cfg.get("sources", {})
    entry = sources.get(source)

    if entry is None:
        logger.warning(
            "No instrument config for source '%s' — using default profile '%s'",
            source,
            default_profile,
        )
        return InstrumentSpec(
            profile=default_profile,
            wavelength=default_wavelength,
            is_default=True,
        )

    profile = entry.get("profile")
    wavelength = entry.get("wavelength") or default_wavelength
    sample_type = entry.get("sample_type", "powder")

    if profile is None:
        logger.warning(
            "Source '%s' has no instrument profile — falling back to default '%s'",
            source,
            default_profile,
        )
        return InstrumentSpec(
            profile=default_profile,
            wavelength=wavelength,
            is_default=True,
            sample_type=sample_type,
        )

    return InstrumentSpec(
        profile=profile,
        wavelength=wavelength,
        is_default=False,
        sample_type=sample_type,
    )


def validate_instrument_compatibility(
    two_theta_range: tuple[float, float],
    pattern_wavelength: float | None,
    spec: InstrumentSpec,
) -> PreflightResult:
    """Preflight checks before refinement.

    Metadata-only checks (sub-millisecond):
    - 2theta range beyond typical lab calibration (>100°)
    - Wavelength mismatch between pattern and spec
    - Default profile fallback warning
    """
    warnings: list[str] = []

    # Check 2theta range
    if two_theta_range[1] > 100.0:
        warnings.append(
            f"2theta extends to {two_theta_range[1]:.1f}° — most lab profiles calibrated to ~90°"
        )

    # Check wavelength mismatch
    if pattern_wavelength is not None and isinstance(spec.wavelength, (int, float)):
        if abs(pattern_wavelength - float(spec.wavelength)) > 0.01:
            warnings.append(
                f"Pattern wavelength {pattern_wavelength:.4f}Å ≠ spec wavelength {spec.wavelength}Å"
            )

    # Flag default fallback
    if spec.is_default:
        warnings.append(
            f"Using default profile '{spec.profile}' — no source-specific instrument match"
        )

    return PreflightResult(ok=len(warnings) == 0, warnings=warnings)


def resolve_pattern_instrument_policy(
    pattern: Any,
    *,
    instrument_source: str | None = None,
    config_path: Path | None = None,
    _config_cache: dict[str, Any] | None = None,
) -> ResolvedInstrumentPolicy:
    """Resolve instrument/profile policy for a pattern without handler-specific side effects."""

    cfg = _config_cache if _config_cache is not None else _load_config(config_path)
    default_wavelength = cfg.get("default_wavelength", "Cu")

    pattern_metadata = getattr(pattern, "metadata", None)
    if not isinstance(pattern_metadata, dict):
        pattern_metadata = {}

    explicit_profile = pattern_metadata.get("instrument_profile")
    explicit_wavelength = (
        pattern_metadata.get("instrument_wavelength")
        or pattern_metadata.get("wavelength_angstrom")
        or getattr(pattern, "wavelength", None)
    )
    effective_wavelength = pattern_metadata.get("effective_wavelength_for_transform")
    wavelength_provenance = pattern_metadata.get("wavelength_provenance")
    inferred_spec: InstrumentSpec | None = None

    if explicit_profile:
        known_profiles = available_profile_names()
        if cfg.get("default_profile"):
            known_profiles.add(str(cfg.get("default_profile")))
        for entry in cfg.get("sources", {}).values():
            if isinstance(entry, dict) and entry.get("profile"):
                known_profiles.add(str(entry["profile"]))
        profile_str = str(explicit_profile)
        spec = InstrumentSpec(
            profile=profile_str,
            wavelength=explicit_wavelength if explicit_wavelength is not None else default_wavelength,
            is_default=profile_str not in known_profiles,
            sample_type=str(pattern_metadata.get("sample_type", "powder")),
        )
    else:
        source = (
            instrument_source
            or pattern_metadata.get("instrument_source")
            or getattr(pattern, "source", None)
            or str(pattern_metadata.get("source", "unknown"))
        )
        spec = resolve_instrument(source, config_path, _config_cache=cfg)
        inferred_spec = infer_profile_from_metadata(pattern_metadata)
        if spec.is_default and inferred_spec is not None:
            spec = inferred_spec

    profile_confidence: ProfileConfidence = "matched"
    if spec.is_default:
        profile_confidence = "fallback"
    if spec.profile is None:
        profile_confidence = "missing"
    if not spec.is_default and inferred_spec is not None and not explicit_profile:
        profile_confidence = "inferred"

    metadata_confidence = pattern_metadata.get("instrument_profile_confidence")
    if metadata_confidence == "proxy" and profile_confidence == "matched":
        profile_confidence = "fallback"

    source = (
        instrument_source
        or pattern_metadata.get("instrument_source")
        or getattr(pattern, "source", None)
        or str(pattern_metadata.get("source", "unknown"))
    )
    pattern_wavelength = getattr(pattern, "wavelength", None)
    return ResolvedInstrumentPolicy(
        source=str(source),
        spec=spec,
        profile_confidence=profile_confidence,
        pattern_wavelength=float(pattern_wavelength) if pattern_wavelength is not None else None,
        effective_wavelength_for_transform=(
            float(effective_wavelength) if isinstance(effective_wavelength, (int, float)) else None
        ),
        wavelength_provenance=(
            str(wavelength_provenance) if wavelength_provenance is not None else None
        ),
    )


def apply_bounded_candidate_wavelength(
    metadata: dict[str, Any] | None,
    *,
    enabled: bool = False,
    candidates: Sequence[float] | None = None,
    selected_wavelength: float | None = None,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    """Return metadata for a bounded, non-promotable candidate-wavelength decision."""

    resolved = dict(metadata or {})
    if not enabled:
        return resolved

    candidate_values = [float(item) for item in (candidates or [])]
    if not candidate_values:
        raise ValueError("candidate wavelength helper requires a non-empty candidate set")
    if len(candidate_values) > 4:
        raise ValueError("candidate wavelength helper exceeds bounded candidate limit")
    if selected_wavelength is None:
        raise ValueError("candidate wavelength helper requires a selected wavelength")
    selected_value = float(selected_wavelength)
    if all(abs(selected_value - candidate) > 1e-8 for candidate in candidate_values):
        raise ValueError("selected wavelength must be drawn from the bounded candidate set")

    resolved.update(
        {
            "effective_wavelength_for_transform": selected_value,
            "wavelength_candidates": candidate_values,
            "wavelength_selection_reason": selection_reason or "bounded_candidate_wavelength_helper",
            "wavelength_assumed": False,
            "wavelength_provenance": "candidate_selected_non_promotable",
            "candidate_wavelength_promotable": False,
        }
    )
    return resolved
