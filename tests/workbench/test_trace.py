"""snapshot.trace and snapshot.peak_sightings: the agent's recorded trajectory as the review page's
trace (what the agent was given, what it said, what it ran, what the environment returned), and
the call whose output first showed the agent each K2 residual peak.

The trace is blind: every system prompt, injected context block, message, command and output
passes the run-identity scrub, with the container's /workspace paths kept so a command stays
readable. The agent's private reasoning is never part of it.
"""

from __future__ import annotations

import json
import re

import pytest

from phaseeval.workbench.snapshot import TRACE_OUTPUT_CHARS, peak_sightings, trace

IDENTITY = ("u6-dev006-qwen-006", "task-dev006__5SKiLMp", "qwen36-27b", "codex", "qwen")


def _framed(body: str, *, code: int | None = 0, session: int | None = None, wall: float = 0.5) -> str:
    status = f"Process exited with code {code}" if code is not None else f"Process running with session ID {session}"
    return f"Chunk ID: 1a2b3c\nWall time: {wall} seconds\n{status}\nOriginal token count: 12\nOutput:\n{body}"


def _call(cid: str, fn: str, **arguments: object) -> dict:
    return {"tool_call_id": cid, "function_name": fn, "arguments": arguments}


def _step(source: str, second: int, message: str = "", calls: list[dict] = (), outputs: dict[str, str] | None = None) -> dict:
    step = {"source": source, "timestamp": f"2026-10-21T10:00:{second:02d}Z", "message": message}
    if calls:
        step["tool_calls"] = list(calls)
        step["observation"] = {"results": [{"source_call_id": k, "content": v} for k, v in (outputs or {}).items()]}
    return step


RESIDUAL = (
    "Run H1, stage rietveld, phases in fit: CIF001779, CIF012075.\n"
    "Largest positive residual peaks (observed intensity the fit misses):\n"
    "  23.96 deg (d 3.711 A): 2.5%; lines nearby: CIF001772 Bi2CO5 Pna2_1 (33) (36%)\n"
    "  43.41 deg (d 2.083 A): 3.8%; lines nearby: CIF008438 Li0.3Ni0.7O1 Fm-3m (225)\n"
)

TRAJECTORY = {
    "steps": [
        _step("system", 0, "<skills_instructions>Codex skills at /tmp/codex-home/skills</skills_instructions>"),
        _step("user", 0, "<environment_context><cwd>/workspace</cwd></environment_context>"),
        _step("user", 1, "Analyze the real powder pattern DEV006. Run python3 /workspace/physics/case_brief.py --case DEV006."),
        _step(
            "agent", 10, "I'll start by reading the case brief.",
            [_call("a", "exec_command", cmd="python3 /workspace/physics/case_brief.py --case DEV006")],
            {"a": _framed("DEV006: Bi Li Ni O, written by qwen36-27b to /jobs/u6-dev006-qwen-006/task-dev006__5SKiLMp/brief.txt")},
        ),
        _step(
            "agent", 20, "Refining the three precursors.",
            [_call("b", "exec_command", cmd="python3 /workspace/physics/refine.py --case DEV006 --candidates CIF001779 CIF012075 --run-label H1")],
            {"b": _framed("", code=None, session=7, wall=10.0)},
        ),
        _step("agent", 40, "", [_call("c", "write_stdin", session_id=7, chars="")], {"c": _framed("H1 rietveld Rwp 4.32%")}),
        _step(
            "agent", 45, "Checking what the fit misses.",
            [_call("d", "exec_command", cmd="python3 /workspace/physics/screen.py --case DEV006 --residual H1")],
            {"d": _framed(RESIDUAL, wall=2.7)},
        ),
        _step("agent", 50, "The 23.96 deg peak is minor; I keep the precursors."),
    ]
}
# build_steps' shape: a segment of the brief read, the refinement (its session poll is its own),
# then a segment holding the residual screen.
STEPS = [
    {"kind": "segment", "title": "Read the case brief", "call_index": 0},
    {"kind": "refine", "run_label": "H1", "call_index": 1},
    {"kind": "segment", "title": "Screened residual peaks", "call_index": 3},
]


def test_the_trace_keeps_the_context_the_task_and_every_agent_turn_in_order() -> None:
    turns = trace(TRAJECTORY, STEPS, IDENTITY)
    assert [t["source"] for t in turns] == ["context", "context", "task", "agent", "agent", "agent", "agent", "agent"]
    # the harness's system prompt and context, scrubbed: what the agent was given before the task
    assert turns[0]["message"] == "<skills_instructions>[run identity] skills at [path]</skills_instructions>"
    assert turns[1]["message"] == "<environment_context><cwd>/workspace</cwd></environment_context>"
    assert turns[2]["message"].startswith("Analyze the real powder pattern DEV006")
    assert [t["t"] for t in turns] == [0.0, 0.0, 1.0, 10.0, 20.0, 40.0, 45.0, 50.0]  # seconds from the first step
    assert [t["message"] for t in turns[3:]] == [
        "I'll start by reading the case brief.",
        "Refining the three precursors.",
        "",
        "Checking what the fit misses.",
        "The 23.96 deg peak is minor; I keep the precursors.",
    ]
    assert "reasoning" not in json.dumps(turns)


def test_each_call_carries_its_command_the_environment_output_and_its_exit() -> None:
    calls = [c for t in trace(TRAJECTORY, STEPS, IDENTITY) for c in t["calls"]]
    assert [c["call_index"] for c in calls] == [0, 1, 2, 3]
    assert calls[0]["tool"] == "exec_command"
    assert calls[0]["command"] == "python3 /workspace/physics/case_brief.py --case DEV006"
    # the harness's chunk framing goes; the exit, the wall time and the tool's own text stay
    assert (calls[0]["exit_code"], calls[0]["running"], calls[0]["wall_s"]) == (0, False, 0.5)
    assert calls[1]["exit_code"] is None and calls[1]["running"] is True and calls[1]["output"] == ""
    assert calls[2]["tool"] == "write_stdin" and calls[2]["command"] == "" and calls[2]["output"] == "H1 rietveld Rwp 4.32%"
    assert calls[3]["output"] == RESIDUAL and calls[3]["wall_s"] == 2.7
    assert all("Chunk ID" not in c["output"] and "Original token count" not in c["output"] for c in calls)


def test_each_call_belongs_to_its_step_and_a_session_poll_to_the_call_that_opened_it() -> None:
    turns = trace(TRAJECTORY, STEPS, IDENTITY)
    assert [c["step"] for t in turns for c in t["calls"]] == [0, 1, 1, 2]
    # a turn sits at its first call's step; a turn with no call at the step before it
    assert [t["step"] for t in turns] == [None, None, None, 0, 1, 1, 2, 2]


def test_the_trace_scrubs_run_names_and_host_paths_and_keeps_container_paths() -> None:
    turns = trace(TRAJECTORY, STEPS, IDENTITY)
    brief = turns[3]["calls"][0]["output"]
    assert brief == "DEV006: Bi Li Ni O, written by [run identity] to [path]"
    text = json.dumps(turns).lower()
    assert all(name.lower() not in text for name in IDENTITY) and "/jobs/" not in text


def test_a_long_output_keeps_its_head_and_tail_and_says_how_much_it_left_out() -> None:
    body = "head line\n" + "x" * (3 * TRACE_OUTPUT_CHARS) + "\nTraceback: the last line"
    trajectory = {"steps": [_step("agent", 0, "", [_call("a", "exec_command", cmd="cat big.log")], {"a": _framed(body)})]}
    [turn] = trace(trajectory, [{"kind": "segment", "call_index": 0}], IDENTITY)
    output = turn["calls"][0]["output"]
    assert len(output) < TRACE_OUTPUT_CHARS + 100
    assert output.startswith("head line") and output.endswith("Traceback: the last line")
    marker = re.search(r"\n\[(\d+) characters left out\]\n", output)
    assert marker and int(marker.group(1)) == len(body) - (len(output) - len(marker.group(0)))


def test_only_the_last_user_turn_before_the_agent_starts_is_the_task_and_a_later_one_is_a_message_to_it() -> None:
    steps = TRAJECTORY["steps"]
    instructions = _step("user", 0, "# AGENTS.md instructions for /workspace\nUse codex conventions.")
    listed = _step("user", 0, "")
    listed["message"] = [{"type": "text", "text": "<environment_context><cwd>/workspace</cwd></environment_context>"}]
    nudge = _step("user", 30, "Please continue until you write the answer.")
    trajectory = {"steps": [steps[0], instructions, listed, steps[2], *steps[3:5], nudge, *steps[5:]]}
    turns = trace(trajectory, STEPS, IDENTITY)
    sources = [t["source"] for t in turns]
    assert sources == ["context", "context", "context", "task", "agent", "agent", "user", "agent", "agent", "agent"]
    assert turns[3]["message"].startswith("Analyze the real powder pattern DEV006")
    assert turns[1]["message"].startswith("# AGENTS.md instructions for /workspace\nUse [run identity] conventions.")
    assert turns[6]["message"] == "Please continue until you write the answer." and turns[6]["step"] == 1


def test_a_path_that_climbs_out_of_the_container_is_scrubbed() -> None:
    trajectory = {"steps": [_step("agent", 0, "", [_call("a", "exec_command", cmd="cat /workspace/../jobs/u7-x/log /workspace/output/H1/result.json")], {"a": ""})]}
    [turn] = trace(trajectory, [{"kind": "segment", "call_index": 0}], IDENTITY)
    assert turn["calls"][0]["command"] == "cat [path] /workspace/output/H1/result.json"


def test_a_tool_whose_arguments_are_text_shows_the_text() -> None:
    patch = "*** Begin Patch\n*** Add File: fit.py\n+print(1)\n*** End Patch"
    trajectory = {"steps": [_step("agent", 0, "", [{"tool_call_id": "a", "function_name": "apply_patch", "arguments": patch}], {"a": "Done"})]}
    [turn] = trace(trajectory, [{"kind": "segment", "call_index": 0}], IDENTITY)
    assert turn["calls"][0]["command"] == patch


def test_a_long_command_and_a_long_message_are_clipped_like_an_output() -> None:
    script = "python3 - <<'EOF'\n" + "x = 1\n" * TRACE_OUTPUT_CHARS + "EOF"
    trajectory = {"steps": [_step("agent", 0, "y" * (2 * TRACE_OUTPUT_CHARS), [_call("a", "exec_command", cmd=script)], {"a": ""})]}
    [turn] = trace(trajectory, [{"kind": "segment", "call_index": 0}], IDENTITY)
    assert len(turn["message"]) < TRACE_OUTPUT_CHARS + 100 and "characters left out" in turn["message"]
    command = turn["calls"][0]["command"]
    assert len(command) < TRACE_OUTPUT_CHARS + 100 and command.startswith("python3 - <<'EOF'") and command.endswith("EOF")


def test_a_function_tool_call_shows_its_arguments() -> None:
    trajectory = {"steps": [_step("agent", 0, "", [_call("a", "refine", candidates=["CIF001779"], run_label="H1")], {"a": "Rwp 4.3%"})]}
    [turn] = trace(trajectory, [{"kind": "refine", "call_index": 0}], IDENTITY)
    [call] = turn["calls"]
    assert call["tool"] == "refine" and json.loads(call["command"]) == {"candidates": ["CIF001779"], "run_label": "H1"}
    assert call["output"] == "Rwp 4.3%" and call["exit_code"] is None and call["wall_s"] is None


def test_function_calls_keep_the_screen_refinement_and_answer_in_order(tmp_path) -> None:
    from phaseeval.workbench.importer import _trajectory
    from phaseeval.workbench.scorecard import _marks
    from phaseeval.workbench.snapshot import Library, _recorded_fit, build_steps

    calls = [
        _call("a", "case_brief"),
        _call("b", "candidates"),
        _call("c", "screen"),
        _call("d", "refine", candidates=["CIF001779"], run_label="H1", pawley_only=False),
        _call("e", "screen", residual="H1"),
        _call("f", "write_analysis", analysis={"phases": [{"candidate_id": "CIF001779"}]}),
    ]
    trajectory = {"steps": [
        _step("agent", i * 5, "", [call], {call["tool_call_id"]: "Recorded output"})
        for i, call in enumerate(calls)
    ] + [_step("agent", 35, "The fit is complete.")]}
    raw = json.dumps(trajectory)
    (tmp_path / "agent").mkdir()
    path = tmp_path / "agent/trajectory.json"
    path.write_text(raw)
    recorded = _trajectory(tmp_path, [])
    fit = _recorded_fit("H1", {"rietveld": {"rwp": 4.3}, "elapsed_seconds": 4}, None)
    steps = build_steps(trajectory, recorded, {"H1": fit}, Library({}, {}), "a" * 64, None)

    assert [s["kind"] for s in steps] == ["segment", "refine", "segment", "answer"]
    assert [s["call_index"] for s in steps] == [0, 3, 4, 5]
    assert steps[0]["title"] == "Read the case brief and the candidate table, then screened candidate phases"
    assert steps[1]["t_start"] == 15 and steps[1]["t_end"] == 19
    assert steps[1]["phases"] == [{"candidate_id": "CIF001779", "formula": None}]
    assert "from_script" not in steps[1]
    assert steps[2]["title"] == "Screened residual peaks"
    turns = trace(trajectory, steps, IDENTITY)
    assert len(turns) == 7
    assert [c["step"] for t in turns for c in t["calls"]] == [0, 0, 0, 1, 2, 3]
    assert [_marks(r.tool, r.summary) for r in recorded] == [(), ("screen",), ("screen",), ("refine",), ("screen",), ("answer",)]
    assert path.read_text() == raw


@pytest.mark.parametrize("arguments", [
    {"run_label": ["bad"], "candidates": ["cod_1010093"]},
    {"run_label": "bad", "candidates": [{}]},
    {"run_label": "bad", "candidates": "cod_1010093"},
    {"run_label": "H1", "candidates": ["cod_1010093"]},
])
def test_rejected_refine_arguments_remain_reviewable(tmp_path: Path, arguments: dict) -> None:
    from phaseeval.workbench.importer import _trajectory
    from phaseeval.workbench.snapshot import Library, _recorded_fit, build_steps

    trajectory = {"steps": [_step("agent", 0, "", [_call("invalid", "refine", **arguments)],
                                  {"invalid": "error: invalid refinement arguments"})]}
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent/trajectory.json").write_text(json.dumps(trajectory))
    fit = _recorded_fit("H1", {"rietveld": {"rwp": 4.3}, "elapsed_seconds": 4}, None)
    steps = build_steps(trajectory, _trajectory(tmp_path, []), {"H1": fit}, Library({}, {}), "a" * 64, None)
    assert steps[0]["kind"] == "segment" and steps[0]["title"] == "Requested a refinement"
    assert [step["call_index"] for step in steps] == [0, None]
    [turn] = trace(trajectory, steps, IDENTITY)
    assert turn["calls"][0]["output"] == "error: invalid refinement arguments"


PEAKS = [
    {"peak_id": "p23.95", "two_theta": 23.95},
    {"peak_id": "p43.40", "two_theta": 43.40},
    {"peak_id": "p50.00", "two_theta": 50.00},
]


def test_a_residual_peak_is_seen_at_the_first_call_that_listed_it_and_never_when_none_did() -> None:
    assert peak_sightings(TRAJECTORY, PEAKS) == {"p23.95": 3, "p43.40": 3, "p50.00": None}


def test_the_screens_unexplained_peak_list_counts_as_a_sighting() -> None:
    screen = "Strong observed peaks the covering set does not explain: 23.94 deg (d 3.714 A, 3%), 61.20 deg (d 1.513 A, 2%)"
    trajectory = {
        "steps": [
            _step("agent", 0, "", [_call("a", "exec_command", cmd="python3 /workspace/physics/screen.py --no-search")], {"a": _framed(screen)}),
            *TRAJECTORY["steps"][3:],
        ]
    }
    assert peak_sightings(trajectory, PEAKS) == {"p23.95": 0, "p43.40": 4, "p50.00": None}


def test_a_number_that_is_not_a_listed_peak_position_is_no_sighting() -> None:
    trajectory = {"steps": [_step("agent", 0, "", [_call("a", "exec_command", cmd="echo")], {"a": _framed("2theta range 23.95 to 80 deg; a=23.95")})]}
    assert peak_sightings(trajectory, PEAKS) == {"p23.95": None, "p43.40": None, "p50.00": None}


def test_a_served_model_path_does_not_scrub_the_word_model() -> None:
    """The Miles pilot served its policy as "/model": the word must stay readable in the contract
    and the agent's prose, while the run's real names are still scrubbed."""
    from types import SimpleNamespace

    from phaseeval.workbench.snapshot import _run_names, _trace_text

    harbor = SimpleNamespace(job_name="agent", trial_id="runner")
    result = SimpleNamespace(agent=SimpleNamespace(model_id="/model", harness="agentxrd-openai-runner"))
    names = _run_names(SimpleNamespace(action=SimpleNamespace(harbor=harbor)), result)
    assert not {"model", "agent", "runner"} & {n.lower() for n in names}
    text = "the agent's runner: a good fit between the model and the data, from agentxrd-openai-runner"
    assert _trace_text(text, names) == "the agent's runner: a good fit between the model and the data, from [run identity]"


def test_model_and_vendor_family_names_in_harness_context_are_scrubbed_and_formulas_kept() -> None:
    """A harness system prompt names its vendor and model family ("You are Codex, based on GPT-5",
    "the openai-docs skill") without naming the run; blind raters must not read either."""
    from phaseeval.workbench.snapshot import _trace_text

    text = "You are Codex, based on GPT-5.1; use the openai-docs skill, not ChatGPT or Claude. Bi2O3 and NiO stay."
    assert _trace_text(text, ()) == (
        "You are [run identity], based on [run identity]; use the [run identity]-docs skill, not [run identity]"
        " or [run identity]. Bi2O3 and NiO stay."
    )
