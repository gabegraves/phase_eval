"""The review page's "How the agent found it" (AXA-43zwk.18): per phase, the candidate screen that
ranked it, the refinement that first tried it and the same-formula candidates set aside, read from
the blind payload's screenings and steps."""

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
process.stdout.write(JSON.stringify(vm.runInContext(
  "S.ep = e; ({ nio: phaseOrigin(e, phasesOf(e)[0]), extras: screenExtras(e) })", ctx)));
"""


def payload() -> dict:
    lib = [("CIF012075", "NiO", "Fm-3m (225)"), ("CIF012076", "NiO", "Fm-3m (225)"), ("CIF001772", "Bi2CO5", "Pna2_1 (33)")]
    rank = lambda n, cid, fom: {"rank": n, "candidate_id": cid, "fom": fom, "cov": 1.0, "expl_percent": 20.0, "missed_lines": [0, 3]}  # noqa: E731
    return {
        "answer": {"phases": [{"candidate_id": "CIF012075", "formula": "NiO"}]},
        "candidates": [{"candidate_id": c, "formula": f, "space_group": g} for c, f, g in lib],
        "steps": [
            {"kind": "segment", "title": "Screened candidate phases", "t_start": 2.4, "calls": 7, "call_index": 0},
            {"kind": "refine", "run_label": "H1_precursors", "phases": [{"candidate_id": "CIF012075", "formula": "NiO"}], "t_start": 142.5, "rwp": 4.32, "call_index": 7},
            {"kind": "refine", "run_label": "H3_NiO76", "phases": [{"candidate_id": "CIF012076", "formula": "NiO"}], "t_start": 855.1, "rwp": 3.54, "call_index": 34},
        ],
        "screenings": [
            {"call_index": 4, "ranked_of": 60, "ranking": [rank(4, "CIF012075", 0.44), rank(5, "CIF012076", 0.44), rank(13, "CIF001772", 0.30)], "covering_set": ["CIF012075", "CIF001772"]}
        ],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_each_phase_says_how_the_agent_found_it(tmp_path: Path) -> None:
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(payload()))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")],
        capture_output=True, text=True, timeout=60, check=True,
    )
    shown = json.loads(run.stdout)
    nio = shown["nio"]
    assert "Candidate screen at 0:02 ranked it <b>#4</b> of 60" in nio and "in the screen's covering set" in nio
    assert 'First refined in <span class="mono">H1_precursors</span> at 2:23 (Rwp 4.32)' in nio
    # CR20: the screen's expl is a share of selected observed peak height, not of all intensity
    assert "match 20% of the selected observed peak height" in nio and "explains" not in nio
    assert "100% of its simulated strong-line intensity (lines at least 5% of its strongest) matches observed peaks" in nio
    assert "Set aside:" in nio and "CIF012076" in nio and "ranked #5" in nio and "H3_NiO76" in nio
    # a covering-set pick of another formula that the answer left out, never refined
    assert "CIF001772" in shown["extras"] and "never refined" in shown["extras"] and "(1)" in shown["extras"]
