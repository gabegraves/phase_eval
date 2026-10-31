"""The function-only agent must receive the assumptions saved with a fit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_fit_limits_reach_the_agent_summary(tmp_path, monkeypatch):
    physics = Path(__file__).resolve().parents[2] / 'harness/physics'
    monkeypatch.syspath_prepend(str(physics))
    spec = importlib.util.spec_from_file_location('refine_limits', physics / 'refine.py')
    refine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refine)
    evidence = SimpleNamespace(comparison_available=True,
                               rietveld_degraded_after_good_pawley=False,
                               pawley_rwp=3.3, rietveld_rwp=3.5)
    text = '\n'.join(refine.summary_lines(
        tmp_path, {'refinement_failed': False}, [], [], evidence))
    assert refine.PROFILE_CAVEAT in text
    assert refine.QUANTITY_BASIS in text
    assert 'Cu radiation' in text
    assert 'conditional fit estimates, not total uncertainty' in text
    assert 'false value for the degradation flag do not establish phase identity' in text


def test_a_degraded_fit_names_specimen_effects_beside_a_wrong_phase(tmp_path, monkeypatch):
    physics = Path(__file__).resolve().parents[2] / 'harness/physics'
    monkeypatch.syspath_prepend(str(physics))
    spec = importlib.util.spec_from_file_location('refine_causes', physics / 'refine.py')
    refine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refine)
    evidence = SimpleNamespace(comparison_available=True,
                               rietveld_degraded_after_good_pawley=True,
                               pawley_rwp=3.3, rietveld_rwp=20.0)
    text = '\n'.join(refine.summary_lines(tmp_path, {'refinement_failed': False}, [], [], evidence))
    readme = (physics / 'README.md').read_text()
    for cause in ('wrong phase, polymorph or structure model', 'preferred orientation beyond the texture model',
                  'microabsorption', 'poor particle statistics'):
        assert cause in text and cause in readme
