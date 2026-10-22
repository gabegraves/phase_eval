"""snapshot.screenings: the agent's candidate screens, parsed to candidate ids and numbers.

The review page's "How the agent found it" reads these (AXA-43zwk.18). The parser keeps a ranked
line only for a candidate the case library holds and copies no text of the tool's output.
"""

from __future__ import annotations

from types import SimpleNamespace

from phaseeval.workbench.snapshot import screenings

RANKING = """DEV006: 81 observed peaks above 1% of the strongest; match window +-0.15 deg 2theta.
Ranking (top 3 of 60; --top 60 lists all):
1. CIF001779 Bi2O3 P2_1/c (14): fom 0.71 cov 0.92 s +0.1% expl 60% missed 0/4
2. CIF999999 Xx2O3 P1 (1): fom 0.70 cov 0.90 s -0.2% expl 62% missed 0/6
3. CIF012075 NiO Fm-3m (225): fom 0.44 cov 1.00 s +0.1% expl 20% missed 0/3
Greedy covering set (one structure per formula): CIF001779 Bi2O3 P2_1/c (14); CIF999999 Xx (1); CIF012075 NiO Fm-3m (225)
written to /jobs/u6-dev006-qwen-006/task-dev006__abc/screen.json by qwen36-27b
"""


def _call(cid: str, fn: str, **arguments: object) -> dict:
    return {"tool_call_id": cid, "function_name": fn, "arguments": arguments}


def _step(calls: list[dict], outputs: dict[str, str]) -> dict:
    return {"tool_calls": calls, "observation": {"results": [{"source_call_id": k, "content": v} for k, v in outputs.items()]}}


LIBRARY = SimpleNamespace(formula={"CIF001779": "Bi2O3", "CIF012075": "NiO"})


def test_a_screen_read_through_its_session_polls_keeps_library_ids_and_numbers() -> None:
    trajectory = {
        "steps": [
            _step([_call("a", "exec_command", cmd="cat case_brief.json")], {"a": "{}"}),
            _step([_call("b", "exec_command", cmd="python3 physics/screen.py --case DEV006")], {"b": "Process running with session ID 7"}),
            _step([_call("c", "write_stdin", session_id=7)], {"c": "Process exited with code 0\n" + RANKING}),
            _step([_call("d", "exec_command", cmd="python3 physics/screen.py --case DEV006 --residual")], {"d": RANKING}),
        ]
    }
    [screen] = screenings(trajectory, LIBRARY)
    # read at the poll that printed it; the later identical ranking is a re-read and is dropped
    assert screen["call_index"] == 2 and screen["ranked_of"] == 60
    assert [r["candidate_id"] for r in screen["ranking"]] == ["CIF001779", "CIF012075"]
    assert screen["ranking"][1] == {"rank": 3, "candidate_id": "CIF012075", "fom": 0.44, "cov": 1.0, "expl_percent": 20.0, "missed_lines": [0, 3]}
    assert screen["covering_set"] == ["CIF001779", "CIF012075"]
    assert "qwen" not in repr(screen) and "/jobs/" not in repr(screen)


def test_residual_mode_output_is_not_a_candidate_screen() -> None:
    residual = "  23.96 deg (d 3.711 A): 2.7%; lines nearby: CIF001779 Bi2O3 P2_1/c (14) (36%)\n"
    trajectory = {"steps": [_step([_call("a", "exec_command", cmd="python3 physics/screen.py --residual H1")], {"a": residual})]}
    assert screenings(trajectory, LIBRARY) == []


def test_a_ranking_read_back_from_its_log_counts() -> None:
    trajectory = {"steps": [_step([_call("a", "exec_command", cmd="python3 physics/screen.py > s.log"), _call("b", "exec_command", cmd="cat s.log")], {"a": "", "b": RANKING})]}
    [screen] = screenings(trajectory, LIBRARY)
    assert screen["call_index"] == 1 and screen["covering_set"] == ["CIF001779", "CIF012075"]
