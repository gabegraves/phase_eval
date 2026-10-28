"""Native windows around each reported phase's own calculated peaks (Science, AXA-43zwk.10 08:42).

A good fit with a small minor phase can leave no residual peak and so no residual window; the
rater and the judge still need observed against model where that phase puts intensity. The
windows come from the fit's .dia alone, by one fixed rule (snapshot.PHASE_WINDOW_RULE), and the
page draws them through "Show its peaks".
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phaseeval.workbench.snapshot import PHASE_WINDOWS, RESIDUAL_WINDOW_DEG, Dia, _phase_windows, k2

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"


def _peaks(
    tt: np.ndarray, peaks: tuple[tuple[float, float], ...], width: float = 0.02
) -> np.ndarray:
    return sum((h * np.exp(-0.5 * ((tt - c) / width) ** 2) for c, h in peaks), np.zeros_like(tt))


def no_residual_fit() -> Dia:
    """A major phase with a doublet stronger than its other reflections and a minor phase with
    its own doublet; observed is the model rounded to whole counts, so K2 lists no peak."""
    tt = np.round(np.arange(10.0, 40.0, 0.01), 4)
    background = np.full_like(tt, 50.0)
    major = _peaks(
        tt,
        (
            (15.0, 1000),
            (15.15, 900),
            (20.0, 800),
            (25.0, 600),
            (30.0, 400),
            (35.0, 300),
            (38.0, 200),
        ),
    )
    minor = _peaks(tt, ((22.0, 30), (22.15, 25), (27.0, 20), (33.0, 10)))
    calculated = background + major + minor
    rows = np.column_stack([tt, np.round(calculated), calculated, background, major, minor])
    return Dia(label="fit", stage="rietveld", phase_ids=("CIF000001", "CIF000002"), rows=rows)


def test_a_minor_phase_with_no_residual_gets_bounded_windows_of_exact_native_rows() -> None:
    dia = no_residual_fit()
    assert k2(dia)["peaks"] == []
    missing: list[str] = []
    windows = _phase_windows(dia, ["CIF000001", "CIF000002"], missing)
    centers = {
        cid: [w["center_deg"] for w in windows if w["phase_id"] == cid] for cid in dia.phase_ids
    }
    # strongest first; the 15.15 deg partner sits within 1 deg of 15.0 and takes no slot
    assert centers == {"CIF000001": [15.0, 20.0, 25.0, 30.0, 35.0], "CIF000002": [22.0, 27.0, 33.0]}
    assert len(centers["CIF000001"]) == PHASE_WINDOWS and missing == []
    for w in windows:
        near = dia.rows[np.abs(dia.rows[:, 0] - w["center_deg"]) <= RESIDUAL_WINDOW_DEG]
        assert w["center_deg"] in w["two_theta_deg"]  # a recorded x, not an interpolated one
        assert (
            w["two_theta_deg"] == near[:, 0].tolist() and len(near) == 101
        )  # every row, no stride
        assert w["observed"] == near[:, 1].tolist() and w["calculated"] == near[:, 2].tolist()
        assert w["difference"] == (near[:, 1] - near[:, 2]).tolist()
        assert w["background"] == near[:, 3].tolist()
        assert w["phase_curves"] == {
            "CIF000001": near[:, 4].tolist(),
            "CIF000002": near[:, 5].tolist(),
        }


def test_overlapping_maxima_resolve_the_same_way_every_time() -> None:
    tt = np.round(np.arange(10.0, 20.0, 0.01), 4)
    equal = _peaks(tt, ((12.6, 100), (12.0, 100), (16.0, 50), (16.9, 60)))
    rows = np.column_stack([tt, equal, equal, np.zeros_like(tt), equal])
    dia = Dia(label=None, stage="rietveld", phase_ids=("CIF000003",), rows=rows)
    first = _phase_windows(dia, ["CIF000003"], [])
    # a tie goes to the lower angle; 12.6 and 16.0 each sit within 1 deg of a stronger choice
    assert [w["center_deg"] for w in first] == [12.0, 16.9]
    assert _phase_windows(dia, ["CIF000003"], []) == first
    for a, b in zip(first, first[1:], strict=False):
        assert abs(a["center_deg"] - b["center_deg"]) > 2 * RESIDUAL_WINDOW_DEG


def test_a_phase_without_a_stored_contribution_gets_no_window_and_is_named_missing() -> None:
    dia = no_residual_fit()
    missing: list[str] = []
    windows = _phase_windows(dia, ["CIF000002", "CIF000009"], missing)
    assert {w["phase_id"] for w in windows} == {"CIF000002"}
    assert missing == ["fit.phase_curves.CIF000009"]
    missing = []
    assert _phase_windows(None, ["CIF000001"], missing) == [] and missing == []  # "fit.dia" says it
    zero = Dia(
        None,
        "rietveld",
        ("CIF000001",),
        np.column_stack([dia.rows[:, :4], np.zeros(len(dia.rows))]),
    )
    assert _phase_windows(zero, ["CIF000001"], missing) == [] and missing == []


def payload() -> dict:
    """A blind payload as snapshot._blind writes it, for a fit with no residual window."""
    dia = no_residual_fit()
    phases = [
        {
            "candidate_id": "CIF000001",
            "formula": "NiO",
            "name": "bunsenite",
            "space_group": "Fm-3m",
            "crystal_system": "cubic",
            "cif": {"formula": "NiO", "space_group": "Fm-3m (225)"},
        },
        {
            "candidate_id": "CIF000002",
            "formula": "Bi2O2CO3",
            "name": "bismutite",
            "space_group": "I4/mmm",
            "crystal_system": "tetragonal",
            "cif": {"formula": "Bi2CO5", "space_group": "I4/mmm (139)"},
        },
    ]
    return {
        "answer": {"phases": phases},
        "fit": {"phase_curves": {cid: [0.0] for cid in dia.phase_ids}},
        "residual_peaks": [],
        "residual_windows": [],
        "phase_windows": _phase_windows(dia, ["CIF000001", "CIF000002"], []),
        "phase_window_rule": "the fixture's rule",
    }


RENDER = """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const ctx = vm.createContext({ console });
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);  // no DOM, no boot
ctx.e = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = vm.runInContext(`(() => {
  const shown = {};
  S.peakSel = null; S.phaseSel = "CIF000002"; S.phaseWin = 0; shown.first = zoomPlot(e);
  S.phaseWin = 2; shown.third = zoomPlot(e);
  S.phaseSel = null; shown.none = zoomPlot(e);
  e.residual_peaks = [{ peak_id: "p30.00", two_theta_deg: 30.0, height_percent: 6.0 }];
  e.residual_windows = [{ peak_id: "p30.00", ...e.phase_windows[3], phase_id: undefined, center_deg: undefined }];
  S.phaseSel = "CIF000002"; S.peakSel = "p30.00"; shown.residual = zoomPlot(e);
  shown.names = phaseNames(phasesOf(e)[1]);
  return shown;
})()`, ctx);
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_show_its_peaks_draws_the_phase_windows_when_no_residual_peak_exists(
    tmp_path: Path,
) -> None:
    body = payload()
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(body))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    shown = json.loads(run.stdout)
    minor = [w for w in body["phase_windows"] if w["phase_id"] == "CIF000002"]
    assert shown["first"].count("<circle ") == len(minor[0]["two_theta_deg"]) == 101
    assert "Bi2O2CO3 peak 1 of 3</b> at 22.00°" in shown["first"]
    assert "a calculated peak of the selected phase" in shown["first"]
    assert 'title="Window rule: the fixture&#39;s rule"' in shown["first"]
    assert '<i class="sw bg"></i>background' in shown["first"] and "NaN" not in shown["first"]
    assert "Bi2O2CO3 peak 3 of 3</b> at 33.00°" in shown["third"]
    assert shown["none"] == ""
    # a selected residual peak keeps its own window and wording
    assert (
        "the selected residual peak" in shown["residual"] and "Window rule" not in shown["residual"]
    )
    # the answer's own words stay apart from the CIF's
    assert "bismutite · Bi2O2CO3" in shown["names"] and ">CIF</span> Bi2CO5" in shown["names"]
    assert "the answer's words differ from the CIF" in shown["names"]


@pytest.mark.parametrize('ambiguous', [False, True])
def test_review_binds_legacy_curve_aliases_without_rewriting_records(monkeypatch, ambiguous):
    from types import SimpleNamespace as NS
    from phaseeval.workbench import snapshot as s

    original = no_residual_fit()
    dia = s.replace(original, phase_ids=('CIF000001', 'CIF000002'))
    ids = ['CIF_000001', 'CIF_000002'] + (['CIF-000001'] if ambiguous else [])
    case = NS(candidates=[NS(candidate_id=cid) for cid in ids])
    checks = ({'code': 'recorded-check', 'status': 'flag'},)
    ev = s.Evidence(
        kind='agent_answer', case=case, root='r', origin='o', answered=False,
        fit=NS(label='legacy', phases=()), fit_role='saved fit', summary=None,
        curve_sha256=None, curve=np.column_stack((dia.rows[:, :3], dia.rows[:, 1] - dia.rows[:, 2])), dia=dia,
        answer_percent=None, answer_phases=(), claims=(), reported={}, descriptors={},
        quantity_basis=None, submitted=None, checks=checks, lattice=(), steps=(),
        missing=(), library=s.Library({}, {}),
    )
    snap = NS(kind='agent_answer', answered=False, phases=[s.SnapshotPhase(phase_id=cid, formula=None, space_group=None, weight_percent=None, weight_percent_error=None) for cid in ids],
              stage='rietveld', rwp_percent=1, gof=1, fit_curve_sha256=None,
              root='r', origin='o', case_id='DEV001', residual_peaks=(), chemistry=None)
    monkeypatch.setattr(s, '_brief', lambda *args: {})
    payload = s._blind(None, ev, snap, None)
    expected = set(dia.phase_ids) if ambiguous else set(ids)
    assert set(payload['fit']['phase_curves']) == expected
    assert {w['phase_id'] for w in payload['phase_windows']} == (set() if ambiguous else set(ids))
    assert payload['checks'] == list(checks)
    assert ev.dia is dia and ev.dia.phase_ids == original.phase_ids
    assert len([m for m in payload['missing'] if m.startswith('fit.phase_curves.')]) == (len(ids) if ambiguous else 0)
