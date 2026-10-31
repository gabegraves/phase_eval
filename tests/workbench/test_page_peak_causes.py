"""The review page offers the specimen-effect peak call: its plain text, its number key, and the
K1 text that names specimen effects beside a wrong phase."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from phaseeval.workbench.labels import PeakCall

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"

RENDER = """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const clicked = [];
const button = (selector) => ({ disabled: false, click: () => clicked.push(selector) });
const ctx = vm.createContext({
  console,
  CSS: { escape: (text) => text },
  document: { querySelector: button, querySelectorAll: () => [] },
});
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);
const out = vm.runInContext(`(() => ({
  causes: PEAK_CAUSES,
  plain: PEAKS_PLAIN,
  k1: CHECK_PLAIN.K1,
  press: (key) => {
    S.ep = { residual_peaks: [{ peak_id: "p23.95", two_theta_deg: 23.95, height_percent: 3 }] };
    S.cur = peakKey("p23.95");
    rubricKey({ key, metaKey: false, ctrlKey: false, altKey: false });
  },
}))()`, ctx);
const presses = ["1", "2", "3", "4", "5", "6"].map((key) => { clicked.length = 0; out.press(key); return clicked.slice(); });
process.stdout.write(JSON.stringify({ causes: out.causes, plain: out.plain, k1: out.k1, presses, source: src }));
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture
def page(tmp_path: Path) -> dict:
    (tmp_path / "render.js").write_text(RENDER)
    run = subprocess.run(["node", str(tmp_path / "render.js"), str(SCRIPT)], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(run.stdout)


@needs_node
def test_the_page_offers_every_cause_the_record_accepts_with_cant_tell_last(page: dict) -> None:
    values = [value for value, _ in page["causes"]]
    assert set(values) == set(PeakCall.model_fields["cause"].annotation.__args__)
    assert values[-1] == "cannot_assess"
    assert dict(page["causes"])["specimen_effect"] == "Specimen effect"
    assert "texture, grain statistics or another specimen effect" in page["plain"]


@needs_node
def test_number_keys_one_to_five_answer_a_peak_in_the_order_shown(page: dict) -> None:
    pressed = [p[0].split('data-val="')[1].rstrip('"]') if p else None for p in page["presses"]]
    assert pressed == [value for value, _ in page["causes"]] + [None]
    assert "Peaks take 1 to 5." in page["source"]


@needs_node
def test_k1_names_specimen_effects_beside_a_wrong_phase(page: dict) -> None:
    for cause in ("wrong phase, polymorph or structure model", "preferred orientation beyond the texture model", "microabsorption", "poor particle statistics"):
        assert cause in page["k1"]
