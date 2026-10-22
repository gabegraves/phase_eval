"""Dogfood fixes on the review page and report (PI 21:11 UTC, AXA-43zwk.18): A4 a passing K1 names
what passed; F2 K2 never reads pass while it lists a weak peak; F5/A3 the GSAS-II claim gets the
engine note and a delta name on P2_1/c Bi2O3 is flagged; F4 an exact reference match that leaves a
residual peak says so."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"

RENDER = """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const ctx = vm.createContext({ console });
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);  // no DOM, no boot
ctx.e = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
process.stdout.write(JSON.stringify(vm.runInContext(`(() => {
  S.ep = e;
  const k1 = checkView({ code: "K1", name: "Rietveld degraded after a good Pawley fit", status: "pass" }, e);
  const k2 = checkView({ code: "K2", name: "Unexplained residual peaks", status: "pass" }, e);
  const k2none = checkView({ code: "K2", name: "Unexplained residual peaks", status: "pass" }, { ...e, residual_peaks: [] });
  const delta = claimMark({ text: "Monoclinic P2_1/c (delta-Bi2O3 type). Weight fraction 22.35 wt%." }, e);
  const gsas = claimMark({ text: "Staged BGMN Rietveld refinement using GSAS-II. Rietveld improved on Pawley (delta Rwp = -0.26%). Bi2O3 (P2_1/c). Atomic coordinates were NOT refined." }, e);
  const form = claimMark({ text: "CIF001783 (Bi2O3, P2_1/c, delta-form) preferred over CIF001779." }, e);
  const rv = { reference: { phases: [{ formula: "NiO" }, { formula: "Bi2O3" }] }, reference_match: { phase_list_exact: true } };
  return { k1, k2, k2none, delta, gsas, form, note: exactNote(e, rv), noteOther: exactNote(e, { ...rv, reference: { phases: [{ formula: "Bi2CO5" }] } }) };
})()`, ctx)));
"""


def payload() -> dict:
    return {
        "answer": {"phases": [{"candidate_id": "CIF001783", "formula": "Bi2O3", "space_group": "P2_1/c (14)"}, {"candidate_id": "CIF012075", "formula": "NiO"}]},
        "brief": {"precursors": [{"formula": "Bi2O3"}, {"formula": "NiO"}]},
        "residual_peaks": [{"peak_id": "p23.95", "two_theta_deg": 23.951, "height_percent": 3.18}],
        "steps": [],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_dogfood_wording(tmp_path: Path) -> None:
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(payload()))
    run = subprocess.run(["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")], capture_output=True, text=True, timeout=60, check=True)
    shown = json.loads(run.stdout)
    assert shown["k1"] == {"name": "Rietveld did not degrade after Pawley", "label": "no degradation", "cls": "pass"}
    assert shown["k2"]["label"] == "weak peak listed for review (below the 5% flag)" and shown["k2"]["cls"] != "pass"
    assert shown["k2none"]["label"] == "pass"
    assert shown["delta"][1] == "Polymorph misnamed" and "monoclinic alpha" in shown["delta"][2]
    assert shown["gsas"][1] == "Engine note"  # "delta Rwp" is not a polymorph name
    assert shown["form"][1] == "Polymorph misnamed"
    assert "The published reference lists the declared precursors only" in shown["note"] and "23.95°" in shown["note"]
    assert shown["noteOther"].startswith('<p style="margin:0 0 8px" id="exact-note">An exact match')
