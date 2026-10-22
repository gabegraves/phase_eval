"""After sign-off the review page sets the rater's answers beside every other rater's (AXA-43zwk.18).

A model judge's label is never presented as an expert label, and an owner label is marked as a
revealed demo label, not expert calibration. Agreement counts carry their n and leave out Can't tell.
"""

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
  S.ep = e; const V = view();
  V.labels[labelKey("phase_supported", "CIF000001")] = { answer: "yes" };
  V.labels[labelKey("phase_plausible", "CIF000001")] = { answer: "no" };
  V.labels[labelKey("fit_supports_conclusion", null)] = { answer: "cannot_assess" };
  const judge = { principal_id: "judge-opus", kind: "model_judge", judge: { model: "claude-opus" } };
  const owner = { principal_id: "owner-rater", kind: "owner" };
  V.others = [
    { rater: judge, item: "phase_supported", phase_id: "CIF000001", answer: "yes", created_at_utc: "1" },
    { rater: judge, item: "phase_plausible", phase_id: "CIF000001", answer: "yes", created_at_utc: "2" },
    { rater: judge, item: "fit_supports_conclusion", phase_id: null, answer: "no", created_at_utc: "3" },
    { rater: owner, item: "phase_supported", phase_id: "CIF000001", answer: "no", created_at_utc: "4" },
  ];
  return agreementTable(e, V);
})()`, ctx);
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_other_raters_are_shown_by_kind_with_counted_agreement(tmp_path: Path) -> None:
    body = {"answer": {"phases": [{"candidate_id": "CIF000001", "formula": "NiO"}]}}
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(body))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    html = json.loads(run.stdout)
    assert "Judge claude-opus (model, not an expert)" in html
    assert "owner-rater (owner, revealed demo label)" in html and "(expert)" not in html
    assert "NiO supported" in html and "NiO plausible" in html
    # judge: agrees on supported, differs on plausible; Can't tell on the fit is left out -> 1 of 2
    # owner: one comparable item, which differs -> 0 of 1
    assert '<td class="num">1 of 2</td>' in html and '<td class="num">0 of 1</td>' in html
    assert html.count('class="differs"') == 2
