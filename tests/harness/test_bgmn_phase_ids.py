"""Native BGMN identifiers must not replace the candidate IDs in fit receipts."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from phaseeval.refinement.bgmn import backend
from phaseeval.refinement.config import RefinementConfig
from phaseeval.refinement.protocol import RefinementInput, RefinementOutput, RefinementStage
from phaseeval.workbench.executors import _stage_payload, _stage_summary, read_lst_facts


@pytest.mark.parametrize("stage", [stage for stage in RefinementStage if stage != RefinementStage.RIETVELD_TEXTURE_CONTROL])
def test_native_stages_restore_requested_ids_and_summary(tmp_path, monkeypatch, stage):
    # Synthetic candidates reproduce the live result's underscore-stripped names.
    ids = ["phase_one", "phase_two", "phase_three"]
    names = ["phaseone", "phasetwo", "phasethree"]
    fractions = [20.0, 30.0, 50.0]
    formulas = ["Na Cl", "Si O2", "Ca F2"]
    groups = [225, 152, 225]
    cifs = [tmp_path / f"{cid}.cif" for cid in ids]
    for cif, group in zip(cifs, groups):
        cif.write_text(f"_space_group_IT_number {group}\n")
    pattern = tmp_path / "pattern.xy"
    pattern.write_text("10 1\n11 2\n")
    request = RefinementInput(pattern, cifs, tmp_path, "fit", phase_ids=ids)
    config = RefinementConfig()
    initial = {name: {"a": 4.0, "b": 4.0, "c": 4.0} for name in names}
    engine = backend.BGMNBackend()
    runner = SimpleNamespace(compute_timeout=lambda *a, **kw: (1, 1),
                             run_with_monitor=lambda *a, **kw: tmp_path / "fit.lst")
    monkeypatch.setattr(engine, "_ensure_runner", lambda: runner)
    monkeypatch.setattr(engine, "_generate_str_files", lambda *a, **kw: ([], initial))
    monkeypatch.setattr(backend, "generate_sav", lambda *a, **kw: tmp_path / "fit.sav")
    monkeypatch.setattr(backend, "copy_instrument_profile", lambda *a: None)
    monkeypatch.setattr(backend, "parse_dia", lambda *a: None)
    monkeypatch.setattr(backend, "parse_lst", lambda *a: {
        "rwp": 3.52, "rpb": 14.45, "rexp": 1.94,
        "phases": {name: {"weight_fraction": fraction, "a": 4.1, "b": 4.1, "c": 4.1}
                   for name, fraction in zip(names, fractions)},
    })
    if stage == RefinementStage.PAWLEY:
        result, _ = engine._run_pawley(request, config, tmp_path, [], initial)
    else:
        result = engine._run_single_rietveld_stage(request, config, tmp_path, [], initial, stage)
    assert [p.phase_id for p in result.phases] == ids
    assert [p.cif_path for p in result.phases] == cifs
    assert [p.name for p in result.phases] == names
    assert [p.weight_fraction for p in result.phases] == pytest.approx(fractions)
    assert set(engine._extract_converged_lattice_from_result(result)) == set(names)

    physics = Path(__file__).resolve().parents[2] / "harness/physics"
    monkeypatch.syspath_prepend(str(physics))
    spec = importlib.util.spec_from_file_location("refine_phase_ids", physics / "refine.py")
    refine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refine)
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / f"fit_{stage.value}.lst").write_text(
        "\n".join(f"Q{name}=0.2+-0.003" for name in names))
    payload = {"refinement_failed": False, stage.value: asdict(result)}
    payload[stage.value]["stage"] = stage.value
    selected = [{"candidate_id": cid, "formula_sum": formula} for cid, formula in zip(ids, formulas)]
    summary = "\n".join(refine.summary_lines(
        tmp_path, payload, selected, cifs, SimpleNamespace(comparison_available=False)))
    for cid, formula, group in zip(ids, ["NaCl", "SiO2", "CaF2"], groups):
        assert f"{cid} {formula} " in summary
        assert f"({group})" in summary
    if stage != RefinementStage.PAWLEY:
        assert summary.count("+- 0.30 wt%") == 3
    output = RefinementOutput("fit", tmp_path, **{stage.value: result})
    workbench_summary = _stage_summary(
        _stage_payload(output, stage.value),
        read_lst_facts((fit / f"fit_{stage.value}.lst").read_text()), None)
    assert [p.candidate_id for p in workbench_summary.phases] == ids
    assert [p.weight_percent_error for p in workbench_summary.phases] == (
        [None] * 3 if stage == RefinementStage.PAWLEY else [0.3] * 3)


def test_collisions_rejected_before_generating_str(tmp_path):
    request = RefinementInput(tmp_path / "pattern.xy", [tmp_path / "one.cif", tmp_path / "two.cif"],
                              tmp_path, "fit", phase_ids=["phase_one", "phaseone"])
    with pytest.raises(ValueError, match="phase name collision"):
        backend.BGMNBackend()._generate_str_files(request, RefinementConfig(), tmp_path)


def test_unknown_native_phase_is_not_guessed(tmp_path):
    request = RefinementInput(tmp_path / "pattern.xy", [tmp_path / "phase.cif"],
                              tmp_path, "fit", phase_ids=["phase_one"])
    result = backend.to_stage_result({"phases": {"phaseunknown": {}}}, None,
                                     RefinementStage.RIETVELD, {}, 0)
    with pytest.raises(ValueError, match="Unknown BGMN phase: phaseunknown"):
        backend._bind_phase_ids(result, request)


@pytest.mark.parametrize("missing_lst", [False, True])
def test_dia_only_pawley_also_restores_ids(tmp_path, monkeypatch, missing_lst):
    request = RefinementInput(tmp_path / "pattern.xy", [tmp_path / "phase.cif"],
                              tmp_path, "fit", phase_ids=["phase_one"])
    request.pattern_path.write_text("10 1\n11 2\n")
    native = backend.to_stage_result({"phases": {"phaseone": {}}}, None,
                                     RefinementStage.PAWLEY, {}, 0)
    engine = backend.BGMNBackend()

    def run(*args, **kwargs):
        if missing_lst:
            return tmp_path / "missing.lst"
        raise backend.BGMNTimeoutError("timeout")

    monkeypatch.setattr(engine, "_ensure_runner", lambda: SimpleNamespace(
        compute_timeout=lambda *a, **kw: (1, 1), run_with_monitor=run))
    monkeypatch.setattr(engine, "_pawley_dia_only_result", lambda *a, **kw: native)
    monkeypatch.setattr(backend, "generate_sav", lambda *a: tmp_path / "fit.sav")
    monkeypatch.setattr(backend, "copy_instrument_profile", lambda *a: None)
    result, _ = engine._run_pawley(request, RefinementConfig(), tmp_path, [], {})
    assert result.phases[0].phase_id == "phase_one"


def test_continuation_uses_native_initial_lattice_keys(tmp_path, monkeypatch):
    request = RefinementInput(tmp_path / "pattern.xy", [tmp_path / "phase.cif"],
                              tmp_path, "fit", phase_ids=["phase_one"])
    (tmp_path / "phaseone.str").touch()
    native = backend.to_stage_result({"phases": {"phaseone": {}}}, None,
                                     RefinementStage.PAWLEY, {"phaseone": {"a": 4.0}}, 0)
    pawley = backend._bind_phase_ids(native, request)
    engine = backend.BGMNBackend()
    captured = {}

    def rietveld(input, config, work_dir, str_paths, initial, **kwargs):
        captured.update(initial)
        return None, None, None, None

    monkeypatch.setattr(engine, "is_available", lambda: True)
    monkeypatch.setattr(engine, "_run_rietveld", rietveld)
    engine.continue_rietveld(RefinementOutput("fit", tmp_path, pawley=pawley), request,
                             RefinementConfig(enable_pawley_lattice_carry_forward=False))
    assert captured == {"phaseone": {"a": 4.0}}
