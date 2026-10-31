"""How far the amounts move between the texture control fit and the final fit."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from phaseeval.refinement.protocol import (
    TEXTURE_DEPENDENT_WT_POINTS,
    PhaseRefinementResult,
    RefinementOutput,
    RefinementStage,
    StageResult,
    texture_dependence,
    texture_dependence_text,
)

ROOT = Path(__file__).resolve().parents[2]
PHYSICS = ROOT / "harness/physics"


def _stage(stage: RefinementStage, rwp: float, **weights: float | None) -> StageResult:
    phases = [
        PhaseRefinementResult(phase_id=name, cif_path=Path(f"{name}.cif"), name=name, weight_fraction=weight,
                              lattice_params={}, lattice_params_initial={})
        for name, weight in weights.items()
    ]
    return StageResult(stage=stage, phases=phases, rwp=rwp, rp=None, rpb=None, rexp=None, gof=None, rho=None,
                       converged=True, elapsed_seconds=1.0, iteration_count=1)


def _output(control: StageResult | None, final: StageResult | None) -> RefinementOutput:
    return RefinementOutput(hypothesis_id="fit", working_dir=Path("."), rietveld_texture_control=control, rietveld=final)


CONTROL = _stage(RefinementStage.RIETVELD_TEXTURE_CONTROL, 9.5, A=60.0, B=40.0)
FINAL = _stage(RefinementStage.RIETVELD, 6.0, A=52.0, B=48.0)


def test_it_reports_both_fits_and_the_largest_change_with_its_phase() -> None:
    result = texture_dependence(_output(CONTROL, _stage(RefinementStage.RIETVELD, 6.0, A=59.0, B=35.0)))
    assert (result.rwp_control, result.rwp_final) == (9.5, 6.0)
    assert result.weight_percent_control == {"A": 60.0, "B": 40.0}
    assert result.weight_percent_final == {"A": 59.0, "B": 35.0}
    assert (result.largest_change_wt_points, result.largest_change_phase) == (5.0, "B")


def test_the_threshold_is_pre_registered_at_five_and_exceeded_only_above_it() -> None:
    assert TEXTURE_DEPENDENT_WT_POINTS == 5.0
    exactly = _stage(RefinementStage.RIETVELD, 6.0, A=55.0, B=45.0)
    assert texture_dependence(_output(CONTROL, exactly)).exceeds_threshold is False
    assert texture_dependence(_output(CONTROL, FINAL)).exceeds_threshold is True


@pytest.mark.parametrize(
    "output",
    [
        _output(None, FINAL),
        _output(CONTROL, None),
        _output(_stage(RefinementStage.RIETVELD_TEXTURE_CONTROL, 9.5, A=None), _stage(RefinementStage.RIETVELD, 6.0, A=50.0)),
        _output(replace(CONTROL, converged=False), FINAL),
    ],
)
def test_it_returns_none_without_a_converged_control_fit_and_a_final_stage(output: RefinementOutput) -> None:
    assert texture_dependence(output) is None


def test_the_rietveld_profile_stage_is_not_a_comparison() -> None:
    profile = _stage(RefinementStage.RIETVELD_PROFILE, 9.5, A=60.0, B=40.0)
    output = RefinementOutput(hypothesis_id="fit", working_dir=Path("."), rietveld_profile=profile, rietveld=FINAL)
    assert texture_dependence(output) is None


def test_the_line_names_both_fits_and_stays_quiet_under_the_threshold() -> None:
    small = {"rwp_control": 9.5, "rwp_final": 9.0, "largest_change_wt_points": 1.0, "largest_change_phase": "A", "exceeds_threshold": False}
    text = texture_dependence_text(small)
    assert "refit with texture off against the final fit: Rwp 9.50% off, 9.00% on" in text
    assert "depend on the texture model" not in text


def test_over_the_threshold_the_line_scopes_the_caveat_to_the_amounts() -> None:
    large = {"rwp_control": 9.5, "rwp_final": 6.0, "largest_change_wt_points": 8.0, "largest_change_phase": "A", "exceeds_threshold": True}
    text = texture_dependence_text(large)
    assert "the amounts depend on the texture model" in text
    assert "does not affect the phase identification" in text


@pytest.fixture
def refine(monkeypatch):
    monkeypatch.syspath_prepend(str(PHYSICS))
    spec = importlib.util.spec_from_file_location("refine_texture", PHYSICS / "refine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(refine, monkeypatch, tmp_path: Path, capsys, *flags: str, phases: int = 2, control: StageResult | None = CONTROL) -> tuple[dict, str]:
    input_dir = ROOT / "harness/examples/dev006/input"
    library = json.loads((input_dir / "candidates/DEV006/manifest.json").read_text())["candidates"]
    candidates = [row["candidate_id"] for row in library][:phases]

    class Backend:
        def __init__(self, str_cache) -> None:
            pass

        def refine(self, request, config):
            return _output(control, FINAL)

    monkeypatch.setattr(refine, "BGMNBackend", Backend)
    monkeypatch.setattr("sys.argv", ["refine.py", "--input-dir", str(input_dir), "--output-dir", str(tmp_path), "--case", "DEV006",
                                     "--candidates", *candidates, "--run-label", "r1", *flags])
    refine.main()
    return json.loads((tmp_path / "r1/result.json").read_text()), capsys.readouterr().out


def test_the_result_reaches_result_json_and_the_agent(refine, monkeypatch, tmp_path: Path, capsys) -> None:
    payload, printed = _run(refine, monkeypatch, tmp_path, capsys)
    assert payload["texture_dependence"]["largest_change_wt_points"] == 8.0
    assert payload["texture_dependence"]["exceeds_threshold"] is True
    assert "the amounts depend on the texture model" in printed


def test_a_fit_without_a_texture_model_says_the_comparison_does_not_apply(refine, monkeypatch, tmp_path: Path, capsys) -> None:
    payload, printed = _run(refine, monkeypatch, tmp_path, capsys, "--texture-order", "0", control=None)
    assert payload["texture_dependence"] is None
    assert "not applicable, the final stage has no texture model" in printed


def test_a_fit_of_one_phase_says_the_comparison_does_not_apply(refine, monkeypatch, tmp_path: Path, capsys) -> None:
    _, printed = _run(refine, monkeypatch, tmp_path, capsys, phases=1, control=None)
    assert "not applicable, a single phase has no amount to move" in printed


def test_a_fit_whose_control_is_absent_says_not_available(refine, monkeypatch, tmp_path: Path, capsys) -> None:
    _, printed = _run(refine, monkeypatch, tmp_path, capsys, control=None)
    assert "Texture dependence: not available (needs a converged final Rietveld stage and its texture control fit)" in printed


def test_the_working_order_names_the_comparison_and_its_limit() -> None:
    readme = (PHYSICS / "README.md").read_text()
    assert "compare the fit with `--texture-order 0`" in readme
    assert "Severe texture can also exceed what SPHAR separates" in readme
