"""Minimal XRDML parsing helpers for external powder benchmark intake.

The DARA benchmark uses Malvern/PANalytical ``.xrdml`` files.  AgentXRD's
runtime works with in-memory two-theta / intensity arrays and BGMN-readable
``.xy`` files, so this module deliberately extracts only the stable pieces
needed for benchmark intake without depending on the external ``dara`` package.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class XRDMLPatternData:
    """Parsed XRDML pattern arrays and basic acquisition metadata."""

    two_theta: list[float]
    intensities: list[float]
    wavelength: float | None
    start_two_theta: float
    end_two_theta: float
    num_points: int
    counting_time_seconds: float | None
    sample_id: str | None


def _ns(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[0] + "}"
    return ""


def _text_float(node: ET.Element | None) -> float | None:
    if node is None or node.text is None:
        return None
    try:
        return float(node.text.strip())
    except ValueError:
        return None


def parse_xrdml(path: str | Path) -> XRDMLPatternData:
    """Parse two-theta, counts, and wavelength from a Malvern XRDML file."""

    path = Path(path)
    root = ET.parse(path).getroot()
    ns = _ns(root.tag)

    sample_id_node = root.find(f".//{ns}sample/{ns}id")
    sample_id = sample_id_node.text.strip() if sample_id_node is not None and sample_id_node.text else None

    k_alpha1 = _text_float(root.find(f".//{ns}usedWavelength/{ns}kAlpha1"))

    data_points = root.find(f".//{ns}scan/{ns}dataPoints")
    if data_points is None:
        raise ValueError(f"xrdml_missing_dataPoints:{path}")

    two_theta_positions = None
    for positions in data_points.findall(f"{ns}positions"):
        axis = (positions.attrib.get("axis") or "").strip().lower()
        if axis in {"2theta", "twotheta", "2-theta"}:
            two_theta_positions = positions
            break
    if two_theta_positions is None:
        positions = data_points.findall(f"{ns}positions")
        if positions:
            two_theta_positions = positions[0]
    if two_theta_positions is None:
        raise ValueError(f"xrdml_missing_twotheta_positions:{path}")

    start = _text_float(two_theta_positions.find(f"{ns}startPosition"))
    end = _text_float(two_theta_positions.find(f"{ns}endPosition"))
    if start is None or end is None:
        raise ValueError(f"xrdml_missing_twotheta_bounds:{path}")

    counts_node = data_points.find(f"{ns}counts")
    if counts_node is None or not counts_node.text:
        raise ValueError(f"xrdml_missing_counts:{path}")
    intensities = [float(token) for token in counts_node.text.split()]
    if not intensities:
        raise ValueError(f"xrdml_empty_counts:{path}")
    two_theta = np.linspace(start, end, len(intensities)).astype(float).tolist()

    return XRDMLPatternData(
        two_theta=two_theta,
        intensities=intensities,
        wavelength=k_alpha1,
        start_two_theta=float(start),
        end_two_theta=float(end),
        num_points=len(intensities),
        counting_time_seconds=_text_float(data_points.find(f"{ns}commonCountingTime")),
        sample_id=sample_id,
    )


def write_xy(path: str | Path, two_theta: list[float], intensities: list[float]) -> Path:
    """Write a BGMN-readable two-column XY pattern file."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for x_val, y_val in zip(two_theta, intensities, strict=False):
            handle.write(f"{float(x_val):.6f} {float(y_val):.6f}\n")
    return out
