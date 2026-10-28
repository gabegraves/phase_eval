"""Phase colors on the review page (AXA-43zwk.18).

Each phase in the answer keeps one Okabe-Ito color everywhere it is drawn: its contribution curve,
its tick row of calculated peak positions, the phase legend and its native windows. Residual peaks
carry a marker in the observed ink, a shape and color no phase uses. Phase names stay in the page
ink, because the first palette color is too dark for text on the dark page.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"
# The research brief's phase color spec order (overnight/research/scientific-interface.md).
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

RENDER = """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const ctx = vm.createContext({ console });
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);  // no DOM, no boot
ctx.e = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = vm.runInContext(`(() => {
  S.ep = e; S.peakSel = null; S.phaseSel = null;
  const shown = { plot: plot(e), legend: plotLegend(e), colors: phasesOf(e).map((p) => phaseColor(p.id)), extra: phaseColor("CIF009999") };
  S.phaseSel = "CIF000002"; S.phaseWin = 0; shown.zoom = zoomPlot(e);
  return shown;
})()`, ctx);
process.stdout.write(JSON.stringify(out));
"""


def _peak(x: float, center: float, height: float) -> float:
    return height * math.exp(-(((x - center) / 0.08) ** 2))


def payload() -> dict:
    tt = [round(10 + 0.05 * i, 2) for i in range(1001)]
    a = [_peak(x, 20, 1000) + _peak(x, 35, 400) for x in tt]
    b = [_peak(x, 27, 800) + _peak(x, 45, 300) for x in tt]
    calc = [100 + p + q for p, q in zip(a, b, strict=True)]
    obs = [c + _peak(x, 50, 60) for x, c in zip(tt, calc, strict=True)]
    near = [i for i, x in enumerate(tt) if abs(x - 27) <= 0.5]
    window = {
        "phase_id": "CIF000002",
        "center_deg": 27.0,
        "two_theta_deg": [tt[i] for i in near],
        "observed": [obs[i] for i in near],
        "calculated": [calc[i] for i in near],
        "difference": [obs[i] - calc[i] for i in near],
        "background": [100.0 for _ in near],
        "phase_curves": {"CIF000001": [a[i] for i in near], "CIF000002": [b[i] for i in near]},
    }
    return {
        "answer": {
            "phases": [
                {"candidate_id": "CIF000001", "formula": "NiO", "space_group": "Fm-3m"},
                {"candidate_id": "CIF000002", "formula": "Bi2O2CO3", "space_group": "I4/mmm"},
            ]
        },
        "fit": {
            "two_theta": tt,
            "observed": obs,
            "calculated": calc,
            "difference": [o - c for o, c in zip(obs, calc, strict=True)],
            "phase_curves": {"CIF000001": a, "CIF000002": b},
        },
        "residual_peaks": [{"peak_id": "p50.00", "two_theta_deg": 50.0, "height_percent": 6.0}],
        "residual_windows": [],
        "phase_windows": [window],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_each_phase_keeps_one_color_and_residual_peaks_use_none(tmp_path: Path) -> None:
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(payload()))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    shown = json.loads(run.stdout)
    nio, bi = shown["colors"]
    # the answer's phases take the first palette colors in answer order; a phase outside it the next
    assert [nio, bi] == OKABE_ITO[:2] and shown["extra"] == OKABE_ITO[2]
    plot = shown["plot"]
    names = re.findall(r'class="axis tickname"[^>]*>([^<]+)</text>', plot)
    assert names == ["NiO", "Bi2O2CO3"], "tick rows stack in answer order"
    for color in (nio, bi):
        # its contribution curve, and a 2 px tick row at its two calculated peaks
        assert f'stroke="{color}" stroke-width="1.1"' in plot
        ticks = re.search(rf'<path d="([^"]*)" stroke="{color}" stroke-width="2"', plot)
        assert ticks and ticks.group(1).count("M") == 2
    # the residual marker sits in the page ink, never in a phase color
    marker = plot[plot.index('data-peak="p50.00"') :]
    marker = marker[: marker.index("</g>")]
    assert 'stroke="var(--obs)"' in marker and not any(c in marker for c in OKABE_ITO)
    assert "unexplained peak" in shown["legend"] and "fitted-curve maxima" in shown["legend"]
    # the native window draws the same phase in the same color
    assert f'stroke="{bi}" stroke-width="2"' in shown["zoom"]
    assert f'style="background:{nio}"' in shown["zoom"] and "NaN" not in plot + shown["zoom"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_residual_axes_show_counts_magnitude_and_no_clipped_phase_overlay(tmp_path: Path) -> None:
    from test_page_method import run_js
    e = payload()
    e["brief"] = {"acquisition": {"wavelength_angstrom": 1.540598}}
    result = run_js(tmp_path, f"""(() => {{
      const e={json.dumps(e)}; S.ep=e; S.phaseSel='CIF000002';
      const first=plot(e), zoom=zoomPlot(e);
      e.fit.difference=e.fit.difference.map(x=>x*10);
      return [first,plot(e),zoom,dSpacing(e,43.38),dSpacing({{}},43.38)];
    }})()""")
    assert "60 counts (5.5%" in result[0] and "600 counts (54.5%" in result[1]
    for html in result[:3]:
        assert "Δ (counts)" in html and "Axes scale independently" in html
        assert 'clip-path="url(#band)"' not in html and "NaN" not in html
    assert "d 2.084 Å (λ₁)" in result[3]
    assert result[4] == ""
