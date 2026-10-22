"""The analysis report (AXA-43zwk.18): one button after this rater's own sign-off writes the whole
analysis as a print-ready page, built only from what the server already returned to this rater.
Before the sign-off there is no report; the report has no form or control but its print button,
and on paper the light variants of the phase colors replace the ones too faint on white."""

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
const out = vm.runInContext(`(() => {
  S.ep = e; S.me = "rater-a"; const V = view();
  const before = analysisReport(e);
  V.labels[labelKey("phase_supported", "CIF000001")] = { answer: "no", reason: "", note: "no NiO lines at 37 deg" };
  V.signoff = { verdict: "would_not_sign" }; V.comment = "NiO is not supported";
  V.revealed = { identity: { model: "model-x" }, signoff: { rater: { principal_id: "owner-rater", kind: "owner" }, exposure: "revealed", verdict: "would_not_sign", comment: "NiO is not supported" } };
  const owner = analysisReport(e);
  V.revealed = { signoff: { rater: { principal_id: "rater-a", kind: "expert" }, exposure: "blind", verdict: "would_not_sign" } };
  return { before, after: owner, expert: analysisReport(e) };
})()`, ctx);
process.stdout.write(JSON.stringify(out));
"""


def payload() -> dict:
    tt = [10 + 0.1 * i for i in range(400)]
    return {
        "digest": "d" * 12,
        "case_id": "DEV006",
        "kind": "agent_answer",
        "answer": {"phases": [{"candidate_id": "CIF000001", "formula": "NiO", "space_group": "Fm-3m"}, {"candidate_id": "CIF000002", "formula": "Li2CO3"}, {"candidate_id": "CIF000003", "formula": "Bi2O3"}, {"candidate_id": "CIF000004", "formula": "C"}, {"candidate_id": "CIF000005", "formula": "Bi2CO5"}]},
        "fit": {"two_theta": tt, "observed": [100.0] * 400, "calculated": [99.0] * 400, "difference": [1.0] * 400, "phase_curves": {}, "rwp": 3.5},
        "checks": [{"code": "K2", "name": "Unexplained residual peaks", "status": "pass", "detail": "none"}],
        "residual_peaks": [],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_report_exists_only_after_the_sign_off_and_holds_the_analysis(tmp_path: Path) -> None:
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(payload()))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")],
        capture_output=True, text=True, timeout=60, check=True,
    )
    shown = json.loads(run.stdout)
    assert shown["before"] == ""
    # CR21: the sign-off's server provenance is printed, so an owner record never reads as expert
    assert "owner-rater (owner, revealed: a revealed demonstration label, not expert calibration)" in shown["after"]
    assert "Recorded as" in shown["after"]
    assert "rater-a (expert, blind)" in shown["expert"] and "demonstration" not in shown["expert"]
    html = shown["after"]
    assert "PhaseEval analysis report" in html and "CIF000001" in html and "<svg" in html
    assert "no NiO lines at 37 deg" in html and "NiO is not supported" in html and "I would not sign it" in html
    assert "Unexplained residual peaks" in html and "Chemistry" in html and "After sign-off" in html
    # the one control is the print button; no form reaches paper
    assert html.count("<button") == 1 and "window.print()" in html and "<form" not in html
    # the fifth phase takes the palette's orange, printed as its light variant
    assert "#C38700" in html and "#E69F00" not in html
