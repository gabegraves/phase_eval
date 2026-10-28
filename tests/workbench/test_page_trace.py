"""The review page's trace (workbench.js traceView): the agent's steps in order, each with its mark
control, the agent's own words, every command it ran and what the environment returned, and where
a tool output first showed it each residual peak. Rendered from a blind payload's steps, trace and
peak_sightings (snapshot.trace, snapshot.peak_sightings)."""

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
ctx.timeOrder = process.argv[4] === "time";
process.stdout.write(JSON.stringify(vm.runInContext("S.ep = e; S.traceTime = timeOrder; traceView(e)", ctx)));
"""

ORIGIN = "a" * 64


def _call(index: int, step: int, command: str, output: str, **extra: object) -> dict:
    return {"call_index": index, "step": step, "tool": "exec_command", "command": command, "exit_code": 0, "running": False, "wall_s": 0.5, "output": output} | extra


def payload(with_trace: bool = True) -> dict:
    residual = "Largest positive residual peaks:\n  23.96 deg (d 3.711 A): 2.5%; lines nearby: CIF001772 Bi2CO5"
    return {
        "digest": "d" * 64,
        "origin": ORIGIN,
        "answer": {"cited_fit": "H1"},
        "fit": {"run_label": "H1", "role": "cited by the answer"},
        "checks": [
            {"code": "K1", "name": "Rietveld did not degrade", "status": "pass"},
            {"code": "K5", "name": "Answer equals the cited fit", "status": "pass"},
        ],
        "residual_peaks": [
            {"peak_id": "p23.95", "two_theta_deg": 23.95, "height_percent": 2.7, "residual_counts": 891.7},
            {"peak_id": "p50.00", "two_theta_deg": 50.0, "height_percent": 1.1, "residual_counts": 300.0},
        ],
        "steps": [
            {"kind": "segment", "title": "Read the case brief", "t_start": 10.0, "calls": 1, "call_index": 0, "attempt": ORIGIN, "step_kind": "segment"},
            {"kind": "refine", "run_label": "H1", "phases": [{"candidate_id": "CIF012075", "formula": "NiO"}], "t_start": 20.0, "rwp": 4.32, "call_index": 1, "attempt": ORIGIN, "step_kind": "refine"},
            {"kind": "segment", "title": "Screened residual peaks", "t_start": 45.0, "calls": 1, "call_index": 3, "attempt": ORIGIN, "step_kind": "segment"},
            {"kind": "answer", "t": 48.0, "calls": 1, "call_index": 4, "attempt": ORIGIN, "step_kind": "tool_call"},
        ],
        "trace": [
            {"source": "context", "t": 0.0, "step": None, "message": "You analyze powder XRD. Amounts are relative modeled crystalline fractions.", "calls": []},
            {"source": "task", "t": 1.0, "step": None, "message": "Analyze the real powder pattern DEV006.", "calls": []},
            {"source": "agent", "t": 10.0, "step": 0, "message": "I'll start by reading the case brief.", "calls": [_call(0, 0, "python3 /workspace/physics/case_brief.py --case DEV006", "DEV006: Bi Li Ni O <precursors>")]},
            {"source": "agent", "t": 20.0, "step": 1, "message": "Refining the three precursors.", "calls": [_call(1, 1, "python3 /workspace/physics/refine.py --run-label H1", "", exit_code=None, running=True, wall_s=10.0)]},
            {"source": "agent", "t": 40.0, "step": 1, "message": "", "calls": [_call(2, 1, "", "H1 rietveld Rwp 4.32%", tool="write_stdin")]},
            {"source": "agent", "t": 45.0, "step": 2, "message": "Checking what the fit misses.", "calls": [_call(3, 2, "python3 /workspace/physics/screen.py --residual H1", residual, exit_code=2, clipped=True, executor_chars=48000)]},
            {"source": "agent", "t": 48.0, "step": 3, "message": "", "calls": [_call(4, 3, "cat > /workspace/output/analysis.json", "")]},
            {"source": "agent", "t": 50.0, "step": 2, "message": "The 23.96 deg peak is minor; I keep the precursors.", "calls": []},
        ]
        if with_trace
        else [],
        "peak_sightings": {"p23.95": 3, "p50.00": None} if with_trace else {},
    }


def render(tmp_path: Path, data: dict, order: str = "steps") -> str:
    (tmp_path / "render.js").write_text(RENDER)
    (tmp_path / "payload.json").write_text(json.dumps(data))
    run = subprocess.run(
        ["node", str(tmp_path / "render.js"), str(SCRIPT), str(tmp_path / "payload.json"), order],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(run.stdout)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@needs_node
def test_agent_summary_reports_identified_screens_without_claiming_unrecorded_absence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(render.__globals__, "RENDER", RENDER.replace("traceView(e)", "agentLine(e)"))
    data = payload()
    html = render(tmp_path, data)
    assert "no candidate screen identified in the recorded steps" in html
    assert "ran no candidate screen" not in html
    data["steps"][0]["title"] = "Read the case brief and the candidate table, then screened candidate phases"
    html = render(tmp_path, data)
    assert "screened the candidate library" in html
    assert "ran 1 refinement" in html


@needs_node
def test_every_step_shows_its_mark_control_the_agents_words_and_each_command_with_its_output(tmp_path: Path) -> None:
    html = render(tmp_path, payload())
    order = ["Read the case brief", "Refine NiO", "Screened residual peaks"]
    assert [html.index(t) for t in order] == sorted(html.index(t) for t in order)
    assert html.count("Mark step") == 4
    assert "Analyze the real powder pattern DEV006." in html  # the task the agent was given
    # the system prompt and context the harness gave the agent, behind their own disclosure
    assert "What the harness gave the agent (1)" in html and "relative modeled crystalline fractions" in html
    assert "I&#39;ll start by reading the case brief." in html
    assert "python3 /workspace/physics/case_brief.py --case DEV006" in html
    assert "DEV006: Bi Li Ni O &lt;precursors&gt;" in html and "<precursors>" not in html
    # the agent's last words sit under the step it was in, after the output it answered
    assert html.index("lines nearby: CIF001772 Bi2CO5") < html.index("The 23.96 deg peak is minor")
    assert "exit 2" in html and "still running" in html


@needs_node
def test_a_session_poll_reads_as_more_output_of_the_command_it_waited_on(tmp_path: Path) -> None:
    html = render(tmp_path, payload())
    refine, residual = html.index("Refine NiO"), html.index("Screened residual peaks")
    poll = html.index("More output of the running command")
    assert refine < poll < residual and refine < html.index("H1 rietveld Rwp 4.32%") < residual


@needs_node
def test_each_residual_peak_says_where_a_tool_output_first_showed_it_to_the_agent(tmp_path: Path) -> None:
    html = render(tmp_path, payload())
    # a text match over screen.py's peak listings, worded as what it is
    assert "23.95° residual peak: first found in the recorded text at 0:45" in html
    assert "50.00° residual peak: not found in the recorded text (only screen.py peak listings are searched)" in html
    badge = html.index("first shows 23.95°")
    assert html.index("python3 /workspace/physics/screen.py --residual H1") < badge < html.index("lines nearby: CIF001772")


@needs_node
def test_a_message_to_the_agent_while_it_worked_reads_as_one(tmp_path: Path) -> None:
    data = payload()
    data["trace"].insert(3, {"source": "user", "t": 30.0, "step": 1, "message": "Please continue.", "calls": []})
    html = render(tmp_path, data)
    assert "Told the agent: Please continue." in html
    assert html.index("Refine NiO") < html.index("Told the agent") < html.index("Screened residual peaks")
    assert html.count("The task the agent was given") == 1


@needs_node
def test_without_a_recorded_trace_the_steps_still_list_with_their_marks(tmp_path: Path) -> None:
    html = render(tmp_path, payload(with_trace=False))
    assert html.count("Mark step") == 4 and "Read the case brief" in html
    assert "residual peak:" not in html and "<pre" not in html


@needs_node
def test_the_cited_fit_and_the_answer_are_tied_to_the_steps_that_made_them(tmp_path: Path) -> None:
    html = render(tmp_path, payload())
    refine, answer = html.index("Refine NiO"), html.index("Wrote an answer")
    cites = html.index("the answer cites this fit · K1 pass · K5 pass")
    assert refine < cites < html.index("Screened residual peaks")
    assert answer < html.index("wrote the submitted answer")
    assert "The answer cites fit <span class=\"mono\">H1</span>, made at 0:20" in html


@needs_node
def test_a_clipped_output_offers_its_full_record_and_says_the_tool_produced_more(tmp_path: Path) -> None:
    html = render(tmp_path, payload())
    assert html.count('data-act="trace-full"') == 1 and 'data-call="3"' in html
    assert "the tool produced 48000 characters" in html


@needs_node
def test_as_it_happened_lists_every_call_in_order_with_its_step(tmp_path: Path) -> None:
    html = render(tmp_path, payload(), order="time")
    order = ["case_brief.py", "refine.py --run-label H1", "More output of the running command", "screen.py --residual H1", "analysis.json"]
    assert [html.index(t) for t in order] == sorted(html.index(t) for t in order)
    assert "step 2 · Refine NiO" in html and "Mark step" not in html


@needs_node
def test_a_lowest_rwp_fit_shown_for_want_of_a_citation_is_never_called_cited(tmp_path: Path) -> None:
    data = payload()
    data["fit"]["role"] = "lowest-Rwp finished Rietveld fit (no cited fit)"
    html = render(tmp_path, data)
    assert "the answer cites this fit" not in html and "The answer cites fit" not in html
