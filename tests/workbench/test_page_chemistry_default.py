"""The chemistry panel's default view keeps the hull approximation visible (CR13, AXA-43zwk.8).

The hull value is a Materials Project value that Dara's metadata associates with the phase's formula
and space group, not this structure's computed stability. Decluttering may collapse the long basis,
but the header and a short qualifier stay outside every disclosure, and the source sits behind a
keyboard-focusable summary.
"""

from __future__ import annotations

import json
import re
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
process.stdout.write(JSON.stringify(vm.runInContext("S.ep = e; chemistryPanel(e)", ctx)));
"""

SOURCE = "dara 1.1.5 data/cod_filtered_info_2024.json.gz"


def payload() -> dict:
    hull = {
        "status": "found",
        "e_above_hull_ev_per_atom": 0.0,
        "entry_id": "COD 1010093",
        "space_group_number": 225,
        "source": SOURCE,
        "detail": "row for COD 1010093",
        "basis": "Materials Project e_above_hull that Dara matched by reduced formula and space group",
    }
    return {
        "answer": {"phases": [{"candidate_id": "CIF012075", "formula": "NiO"}]},
        "chemistry": {
            "phases": [
                {
                    "phase_id": "CIF012075",
                    "formula": "NiO",
                    "charge_balance": {"status": "balanced", "assignment": {"Ni": 2, "O": -2}},
                    "hull": hull,
                    "elements_not_in_precursors": [],
                    "identity": {"status": "consistent"},
                }
            ],
            "element_balance": {"status": "compared"},
            "precursor_elements_unaccounted": [],
            "precursors": ["NiO"],
        },
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_hull_approximation_is_visible_without_opening_anything(tmp_path: Path) -> None:
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(payload()))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    html = json.loads(run.stdout)
    # what a rater sees before opening any disclosure: every <details> body removed, title
    # attributes (hover only) removed
    shown = re.sub(r"<details[\s\S]*?</details>", "", html)
    shown = re.sub(r'\stitle="[^"]*"', "", shown)
    assert "Approx. hull (eV/atom)" in shown
    qualifier = re.search(r'<p[^>]*id="hull-qualifier"[^>]*>([^<]*)</p>', shown)
    assert qualifier, "the hull qualifier is hidden in the default view"
    assert "Materials Project" in qualifier.group(1) and "formula and space group" in qualifier.group(1)
    assert "not this structure" in qualifier.group(1)
    # the source is one keyboard-focusable summary away
    sources = re.search(r"<details[^>]*><summary>Hull sources</summary>([\s\S]*?)</details>", html)
    assert sources and SOURCE in sources.group(1) and "COD 1010093" in sources.group(1)
