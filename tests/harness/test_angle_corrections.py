"""What BGMN's angle corrections reach result.json and the agent, read from real BGMN output."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from phaseeval.refinement.bgmn.result_parser import (
    AngleCorrections,
    ValueEsd,
    displacement_text,
    eps2_at_bound,
    parse_lst,
    to_stage_result,
    two_theta_shift_deg,
)
from phaseeval.refinement.config import RefinementConfig
from phaseeval.refinement.protocol import RefinementOutput, RefinementStage

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/workbench/fixtures/dev006_bgmn/DEV006_rietveld.lst"
DEFAULT_BOUND = "0_-0.01^0.01"


@pytest.fixture
def refine(monkeypatch):
    physics = ROOT / "harness/physics"
    monkeypatch.syspath_prepend(str(physics))
    spec = importlib.util.spec_from_file_location("refine_angles", physics / "refine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_refined_correction_is_listed_and_kept_with_its_esd() -> None:
    corrections = parse_lst(FIXTURE)["angle_corrections"]
    assert corrections == AngleCorrections(eps1=None, eps2=ValueEsd(-0.0008327, 0.0000075), eps3=None)


def test_a_printed_error_is_listed_without_a_value_and_not_called_unrefined(tmp_path: Path) -> None:
    lst = tmp_path / "a.lst"
    lst.write_text("Rwp=3.52% Rexp=1.94%\nEPS2=ERROR\n")
    eps2 = parse_lst(lst)["angle_corrections"].eps2
    assert eps2 is not None and eps2.value != eps2.value and eps2.esd is None


def test_corrections_reach_result_json_through_the_stage_asdict_path(refine) -> None:
    stage = to_stage_result(parse_lst(FIXTURE), None, RefinementStage.RIETVELD, {}, 1.0)
    output = RefinementOutput(hypothesis_id="fit", working_dir=Path("."), rietveld=stage)
    payload = refine.json_value(asdict(output))
    json.dumps(payload, allow_nan=False)
    assert payload["rietveld"]["angle_corrections"] == {
        "eps1": None,
        "eps2": {"value": -0.0008327, "esd": 0.0000075},
        "eps3": None,
    }


def test_the_default_eps2_bound_is_tightened() -> None:
    assert RefinementConfig().eps2 == DEFAULT_BOUND


def test_positive_eps2_lowers_the_angle_by_the_measured_factor() -> None:
    assert two_theta_shift_deg(0.001, 0.0) == pytest.approx(-0.11459)
    assert two_theta_shift_deg(0.001, 60.0) == pytest.approx(-0.11459 * 0.8660254)
    assert two_theta_shift_deg(-0.001, 10.0) > 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0098, True), (-0.0098, True), (0.00979, False), (-0.00979, False), (0.0, False), (0.01, True)],
)
def test_at_bound_is_within_two_percent_of_the_limit(value: float, expected: bool) -> None:
    assert eps2_at_bound(value, DEFAULT_BOUND) is expected


def test_at_bound_reads_the_bound_the_run_recorded_not_the_current_default() -> None:
    assert eps2_at_bound(0.0099, "-8.3E-04_-0.05^0.05") is False
    assert eps2_at_bound(0.05, "-8.3E-04_-0.05^0.05") is True
    assert eps2_at_bound(0.05, None) is None
    assert eps2_at_bound(0.05, "0_e^e") is None
    assert eps2_at_bound(0.05, "fixed") is None


def test_the_line_names_both_scan_ends_and_warns_only_at_the_bound() -> None:
    inside = displacement_text(-0.0008327, 0.0000075, (10.0, 80.0), DEFAULT_BOUND)
    assert "-0.000833 +- 0.000008" in inside
    assert "+0.095 deg 2theta at 10.00 deg" in inside and "+0.073 deg at 80.00 deg" in inside
    assert "at its bound" not in inside

    saturated = displacement_text(0.00995, 0.00001, (10.0, 80.0), DEFAULT_BOUND)
    assert "at its bound (-0.01 to 0.01)" in saturated and "saturated" in saturated


def test_a_value_without_a_number_or_a_limit_says_so() -> None:
    assert "without a usable value" in displacement_text(None, None, (10.0, 80.0), DEFAULT_BOUND)
    assert "limit is not recorded" in displacement_text(0.001, None, None, None)


def _payload(angle_corrections: dict | None) -> dict:
    stage = {"stage": "rietveld", "rwp": 3.5, "rexp": 1.9, "gof": 1.8, "converged": True, "phases": [],
             "angle_corrections": angle_corrections}
    return {"refinement_failed": False, "rietveld": stage}


EVIDENCE = SimpleNamespace(comparison_available=False)


def test_the_agent_summary_prints_the_line_for_the_final_stage(refine, tmp_path: Path) -> None:
    eps2 = {"eps1": None, "eps2": {"value": 0.0099, "esd": 0.00001}, "eps3": None}
    text = "\n".join(refine.summary_lines(tmp_path, _payload(eps2), [], [], EVIDENCE,
                                          scan_two_theta_deg=(10.0, 80.0), eps2_bound=DEFAULT_BOUND))
    assert "EPS2 (sample displacement s/R) 0.009900 +- 0.000010" in text
    assert "WARNING: EPS2 is at its bound" in text


def test_the_agent_summary_says_when_the_final_stage_lists_no_eps2(refine, tmp_path: Path) -> None:
    text = "\n".join(refine.summary_lines(tmp_path, _payload(None), [], [], EVIDENCE))
    assert "EPS2 at final stage rietveld: not listed" in text
