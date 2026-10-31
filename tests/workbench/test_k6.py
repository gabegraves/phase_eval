"""K6, the advisory check on displacement and texture dependence: the real refine.py output into
the real fit record, saved BGMN bundles through the import path, and absent evidence."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_import_external import BUNDLE, CANDIDATES, CASE_DIR, _case, _client, _queue, _restore, run_cli

from phaseeval.refinement.bgmn.backend import _bind_phase_ids
from phaseeval.refinement.bgmn.result_parser import parse_lst, to_stage_result
from phaseeval.refinement.protocol import RefinementOutput, RefinementStage
from phaseeval.workbench import cli
from phaseeval.workbench.labels import LabelStore
from phaseeval.workbench.snapshot import FitRecord, _checks, _recorded_fit, k6

ROOT = Path(__file__).resolve().parents[2]
PHYSICS = ROOT / "harness/physics"
LST = BUNDLE / "DEV006_rietveld.lst"
SCAN = (10.0, 80.0)
SCRIPT = ROOT / "src/phaseeval/api/assets/workbench.js"


def _fit(**fields) -> FitRecord:
    return FitRecord(label="r1", candidates=(), failed=False, stages={}, final=True, rwp=3.5, gof=1.8, phases=(), **fields)


INSIDE = {"eps1": None, "eps2": {"value": -0.0008, "esd": 0.00001}, "eps3": None}
SATURATED = {"eps1": None, "eps2": {"value": 0.0099, "esd": 0.00001}, "eps3": None}
SMALL = {"rwp_before": 9.0, "rwp_after": 8.0, "largest_change_wt_points": 1.0, "largest_change_phase": "A", "exceeds_threshold": False}
A_PHASE = {"phase_id": "A"}
LARGE = {"rwp_before": 9.0, "rwp_after": 6.0, "largest_change_wt_points": 8.0, "largest_change_phase": "A", "exceeds_threshold": True}
BOUND = "0_-0.01^0.01"


def test_it_passes_only_when_displacement_and_texture_were_both_checked() -> None:
    check = k6(_fit(angle_corrections=INSIDE, eps2_bound=BOUND, texture_dependence=SMALL, texture_order=4), SCAN)
    assert check["status"] == "pass" and check["advisory"] is True
    assert "+0.091 deg 2theta at 10.00 deg" in check["detail"] and "Not texture alone" in check["detail"]
    assert "not to which phases are present" in check["detail"]


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        ({"angle_corrections": SATURATED, "eps2_bound": BOUND, "texture_dependence": SMALL, "texture_order": 4}, "at its bound"),
        ({"angle_corrections": INSIDE, "eps2_bound": BOUND, "texture_dependence": LARGE, "texture_order": 4}, "the amounts depend on the texture model"),
        ({"angle_corrections": SATURATED, "eps2_bound": BOUND}, "at its bound"),
        ({"texture_dependence": LARGE, "texture_order": 4}, "the amounts depend on the texture model"),
    ],
)
def test_either_part_flags_it_even_when_the_other_is_absent(fields: dict, says: str) -> None:
    check = k6(_fit(**fields), SCAN)
    assert check["status"] == "flag" and says in check["detail"]


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        ({}, "Displacement: not available"),
        ({}, "Texture dependence: not available"),
        ({"angle_corrections": INSIDE, "eps2_bound": BOUND, "texture_order": 4}, "Texture dependence: not available"),
        ({"texture_dependence": SMALL, "texture_order": 4}, "Displacement: not available"),
        ({"angle_corrections": INSIDE, "texture_dependence": SMALL, "texture_order": 4}, "limit is not recorded"),
    ],
)
def test_absent_evidence_says_not_available_rather_than_passing(fields: dict, says: str) -> None:
    check = k6(_fit(**fields), SCAN)
    assert check["status"] == "not_computable" and says in check["detail"]


def test_a_fit_with_no_texture_model_can_pass_on_its_displacement() -> None:
    check = k6(_fit(angle_corrections=INSIDE, eps2_bound=BOUND, texture_order=0), SCAN)
    assert check["status"] == "pass" and "not applicable" in check["detail"]


def test_no_fit_is_not_available() -> None:
    check = _checks(None, None, (), None)[-1]
    assert (check["code"], check["status"]) == ("K6", "not_computable") and check["advisory"] is True
    assert check["detail"].startswith("Not available")


def test_a_result_recorded_before_the_check_existed_is_not_available() -> None:
    result = {"refinement_failed": False, "rietveld": {"rwp": 3.5, "gof": 1.8, "converged": True, "phases": []}}
    request = {"candidates": [], "config": {"eps2": "0_-0.05^0.05", "rietveld": {"gewicht": "SPHAR4"}}}
    assert k6(_recorded_fit("r1", result, request), SCAN)["status"] == "not_computable"


def test_the_recorded_bound_decides_not_the_current_default() -> None:
    stage = {"rwp": 3.5, "gof": 1.8, "converged": True, "phases": [A_PHASE], "angle_corrections": SATURATED}
    result = {"refinement_failed": False, "rietveld": stage, "texture_dependence": SMALL}
    request = {"candidates": [{"candidate_id": "A"}], "config": {"eps2": "0_-0.05^0.05", "rietveld": {"gewicht": "SPHAR4"}}}
    assert k6(_recorded_fit("r1", result, request), SCAN)["status"] == "pass"


@pytest.mark.parametrize(
    ("corrections", "dependence"),
    [
        ("EPS2", "large"),
        ({"eps2": "-0.001"}, {"largest_change_phase": "A"}),
        ({"eps2": {"value": "-0.001", "esd": None}}, SMALL | {"rwp_after": "8.0"}),
        ({"eps2": {"value": True, "esd": None}}, SMALL | {"exceeds_threshold": "no"}),
        (["eps2"], SMALL | {"largest_change_phase": "qwen-run-name"}),
    ],
)
def test_a_recorded_field_of_another_shape_is_not_available_and_names_nothing(corrections: object, dependence: object) -> None:
    stage = {"rwp": 3.5, "gof": 1.8, "converged": True, "phases": [A_PHASE], "angle_corrections": corrections}
    result = {"refinement_failed": False, "rietveld": stage, "texture_dependence": dependence}
    request = {"candidates": [], "config": {"eps2": 5, "rietveld": {"gewicht": ["SPHAR4"]}}}
    check = k6(_recorded_fit("r1", result, request), SCAN)
    assert check["status"] == "not_computable"
    assert "Displacement: not available" in check["detail"] and "Texture dependence: not available" in check["detail"]
    assert "qwen" not in check["detail"]


def _refine_module(monkeypatch):
    monkeypatch.syspath_prepend(str(PHYSICS))
    spec = importlib.util.spec_from_file_location("refine_k6", PHYSICS / "refine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_what_refine_py_writes_is_what_the_fit_record_reads(monkeypatch, tmp_path: Path, capsys) -> None:
    refine = _refine_module(monkeypatch)
    library = json.loads((CASE_DIR / "candidates/DEV006/manifest.json").read_text())["candidates"]
    final = to_stage_result(parse_lst(LST), None, RefinementStage.RIETVELD, {}, 1.0)
    profile = to_stage_result(parse_lst(LST), None, RefinementStage.RIETVELD_PROFILE, {}, 1.0)

    class Backend:
        def __init__(self, str_cache) -> None:
            pass

        def refine(self, request, config):
            return RefinementOutput(hypothesis_id="fit", working_dir=request.working_dir,
                                    rietveld_profile=_bind_phase_ids(profile, request), rietveld=_bind_phase_ids(final, request))

    monkeypatch.setattr(refine, "BGMNBackend", Backend)
    monkeypatch.setattr("sys.argv", ["refine.py", "--input-dir", str(CASE_DIR), "--output-dir", str(tmp_path), "--case", "DEV006",
                                     "--candidates", *(row["candidate_id"] for row in library if row["candidate_id"] in CANDIDATES), "--run-label", "r1"])
    refine.main()
    capsys.readouterr()
    run = tmp_path / "r1"
    fit = _recorded_fit("r1", json.loads((run / "result.json").read_text()), json.loads((run / "request.json").read_text()))
    assert fit.eps2_bound == BOUND and fit.texture_order == 4
    check = k6(fit, SCAN)
    assert check["status"] == "pass" and "-0.000833 +- 0.000008" in check["detail"]
    assert "largest amount change 0.00 wt% points" in check["detail"]


# --- saved BGMN bundles, through import-bgmn and the rater's review route -------------------------


def _review_k6(tmp_path: Path, bundle: Path, *extra: str) -> dict:
    case, store = _case(tmp_path), tmp_path / "state/store"
    run_cli("import-bgmn", "--case-dir", str(case), "--run-dir", str(bundle), "--case", "DEV006", "--store", str(store), *extra)
    (digest,) = _queue(store)
    client, saved = _client(tmp_path, store)
    try:
        with client:
            review = client.get(f"/api/attempts/snapshots/{digest}/review")
            assert review.status_code == 200, review.text
    finally:
        _restore(saved)
    (check,) = [c for c in review.json()["checks"] if c["code"] == "K6"]
    (frozen,) = [c for _, snap in LabelStore(cli.open_store(store, tmp_path))._snapshots() for c in snap.checks if c.code == "K6"]
    assert (frozen.status, frozen.detail) == (check["status"], check["detail"])
    return check


def _bundle_with_profile_stage(tmp_path: Path, gewicht: dict[str, str]) -> Path:
    """The committed Rietveld stage plus a rietveld_profile stage: no texture model, its own Rwp
    and the given GEWICHT values, in the order of the phase sections."""
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    text = LST.read_text().replace("Rwp=3.52%", "Rwp=9.40%")
    for value in gewicht.values():
        text = re.sub(r"GEWICHT=SPHAR4, MeanValue\(GEWICHT\)=\S+", f"GEWICHT={value}+-0.0050", text, count=1)
    (bundle / "DEV006_rietveld_profile.lst").write_text(text)
    dia = (BUNDLE / "DEV006_rietveld.dia").read_text().splitlines()
    head = dia[0].replace("TITEL=DEV006_rietveld ", "TITEL=DEV006_rietveld_profile ").replace("Rwp=3.52", "Rwp=9.40")
    (bundle / "DEV006_rietveld_profile.dia").write_text(head + "\n 10.0094 1.0 1.0 1.0 0.1 0.1 0.1\n")
    sav = (BUNDLE / "DEV006_rietveld.sav").read_text().replace("DEV006_rietveld.", "DEV006_rietveld_profile.")
    (bundle / "DEV006_rietveld_profile.sav").write_text(sav)
    return bundle


CONTROL = ("--control", "DEV006_rietveld.sav")


def test_a_bundle_of_one_stage_shows_its_displacement_and_no_texture_comparison(tmp_path: Path) -> None:
    check = _review_k6(tmp_path, BUNDLE)
    assert check["status"] == "not_computable"
    assert "-0.000833 +- 0.000008" in check["detail"] and "+0.095 deg 2theta at 10.01 deg" in check["detail"]
    assert "at its bound" not in check["detail"] and "Texture dependence: not available" in check["detail"]


def test_a_bundle_with_its_profile_stage_compares_the_amounts(tmp_path: Path) -> None:
    gewicht = {"cod1010093": "0.3772", "cod9008283": "0.4228", "cod1526458": "0.2000"}
    check = _review_k6(tmp_path, _bundle_with_profile_stage(tmp_path, gewicht), *CONTROL)
    assert check["status"] == "pass"
    assert "Rwp 9.40% to 3.52%" in check["detail"]
    assert "largest amount change 2.00 wt% points on cod_1010093" in check["detail"]


def test_amounts_that_move_past_the_threshold_flag_it_and_scope_the_caveat(tmp_path: Path) -> None:
    gewicht = {"cod1010093": "0.4500", "cod9008283": "0.3500", "cod1526458": "0.2000"}
    check = _review_k6(tmp_path, _bundle_with_profile_stage(tmp_path, gewicht), *CONTROL)
    assert check["status"] == "flag"
    assert "the amounts depend on the texture model" in check["detail"]
    assert "does not affect the phase identification" in check["detail"]


def _bundle_with_grain_diameter(tmp_path: Path, *, prints_my: bool) -> Path:
    gewicht = {"cod1010093": "0.3772", "cod9008283": "0.4228", "cod1526458": "0.2000"}
    bundle = _bundle_with_profile_stage(tmp_path, gewicht)
    for sav in bundle.glob("*.sav"):
        sav.write_text(sav.read_text().replace("NTHREADS=", "d=5.0\nNTHREADS=", 1))
    if prints_my:
        for lst in bundle.glob("*.lst"):
            lst.write_text(lst.read_text().replace("GEWICHT=", "my=0.02+-0.000001\nGEWICHT="))
    return bundle


def test_a_bundle_fitted_with_a_grain_diameter_compares_the_corrected_amounts(tmp_path: Path) -> None:
    check = _review_k6(tmp_path, _bundle_with_grain_diameter(tmp_path, prints_my=True), *CONTROL)
    assert check["status"] == "pass" and "largest amount change 2.00 wt% points on cod_1010093" in check["detail"]


def test_a_bundle_fitted_with_a_grain_diameter_that_prints_no_my_has_no_comparison(tmp_path: Path) -> None:
    check = _review_k6(tmp_path, _bundle_with_grain_diameter(tmp_path, prints_my=False), *CONTROL)
    assert check["status"] == "not_computable" and "Texture dependence: not available" in check["detail"]


def test_the_bound_comes_from_the_bundles_own_control_file(tmp_path: Path) -> None:
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    sav = bundle / "DEV006_rietveld.sav"
    sav.write_text(sav.read_text().replace("_-0.05^0.05", "_-0.00084^0.00084"))
    check = _review_k6(tmp_path, bundle)
    assert check["status"] == "flag" and "at its bound (-0.00084 to 0.00084)" in check["detail"]


def test_a_saved_fit_with_no_texture_model_says_the_comparison_does_not_apply(tmp_path: Path) -> None:
    bundle = tmp_path / "fit"
    shutil.copytree(BUNDLE, bundle)
    lst = bundle / "DEV006_rietveld.lst"
    lst.write_text(lst.read_text().replace("GEWICHT=SPHAR4, MeanValue(GEWICHT)=", "GEWICHT="))
    check = _review_k6(tmp_path, bundle)
    assert check["status"] == "pass" and "not applicable" in check["detail"]


# --- the page ------------------------------------------------------------------------------------


PAGE = """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const ctx = vm.createContext({ console });
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);
process.stdout.write(JSON.stringify(vm.runInContext(`(() => {
  const check = (status) => ({ code: "K6", name: "Displacement and texture dependence", status, detail: "the detail" });
  const e = { checks: [check("flag")] };
  S.ep = e;
  return {
    absent: checkView(check("not_computable"), e),
    flagged: checkView(check("flag"), e),
    passed: checkView(check("pass"), e),
    plain: CHECK_PLAIN.K6,
    panel: peakPanel(e),
  };
})()`, ctx)));
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_the_page_shows_k6_and_says_not_available_when_evidence_is_absent(tmp_path: Path) -> None:
    (tmp_path / "page.js").write_text(PAGE)
    run = subprocess.run(["node", str(tmp_path / "page.js"), str(SCRIPT)], capture_output=True, text=True, timeout=60, check=True)
    page = json.loads(run.stdout)
    assert page["absent"]["label"] == "not available"
    assert page["flagged"]["label"] == "check" and page["passed"]["label"] == "pass"
    assert "not to which phases are present" in page["plain"]
    assert "Displacement and texture dependence" in page["panel"] and "the detail" in page["panel"]
