"""harness/agent/run_agent.py: the model is shown the last SHOWN_CHARS of a tool's output, as
before, and the executor's full output is kept beside the trajectory, named in the observation,
so a reviewer can tell what the tool produced from what the agent was shown (requirements audit
2026-10-27, item 4)."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "harness/agent/run_agent.py"
spec = importlib.util.spec_from_file_location("run_agent", SCRIPT)
run_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_agent)


def test_a_short_output_is_shown_whole_and_names_no_separate_record(tmp_path: Path) -> None:
    result = run_agent.observation(tmp_path, "call_1", "Rwp 3.52%\n[exit 0]")
    assert result == {"source_call_id": "call_1", "content": "Rwp 3.52%\n[exit 0]"}
    assert not (tmp_path / "agent/tool-output").exists()


def test_a_long_output_shows_its_tail_and_keeps_the_whole_beside_the_trajectory(tmp_path: Path) -> None:
    full = "Ranking (top 60 of 60)\n" + "x" * (2 * run_agent.SHOWN_CHARS) + "\n[exit 0]"
    result = run_agent.observation(tmp_path, "call_7", full)
    assert result["content"] == full[-run_agent.SHOWN_CHARS:]
    sha = hashlib.sha256(full.encode()).hexdigest()
    assert result["extra"] == {
        "executor_output": f"agent/tool-output/{sha}.txt",
        "executor_sha256": hashlib.sha256(full.encode()).hexdigest(),
        "executor_chars": len(full),
        "shown_chars": run_agent.SHOWN_CHARS,
    }
    assert (tmp_path / f"agent/tool-output/{sha}.txt").read_text() == full


def test_a_reused_call_id_keeps_both_outputs_and_names_no_outside_path(tmp_path: Path) -> None:
    first = run_agent.observation(tmp_path, "../call_0", "a" * (run_agent.SHOWN_CHARS + 1))
    second = run_agent.observation(tmp_path, "../call_0", "b" * (run_agent.SHOWN_CHARS + 1))
    assert first["extra"]["executor_output"] != second["extra"]["executor_output"]
    assert len(list((tmp_path / "agent/tool-output").iterdir())) == 2
    assert not list(tmp_path.parent.glob("*.txt"))


def test_two_runs_written_to_folders_of_one_name_get_distinct_trial_names(tmp_path: Path) -> None:
    """The Miles pilot wrote three runs to folders named "runner"; PhaseEval keys an import on the
    trial name, so the second was refused as a reused key."""
    first = run_agent.trial_name(tmp_path / "a/runner", "0f3c9a2e-0000-4000-8000-000000000001")
    second = run_agent.trial_name(tmp_path / "b/runner", "77d1e5b4-0000-4000-8000-000000000002")
    assert first == "runner-0f3c9a2e" and second == "runner-77d1e5b4"
