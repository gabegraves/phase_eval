"""The failure brief, driven through the real DEV006 Qwen export and the public episode files."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "failure_brief.py"
EXPORT = ROOT / "docs" / "examples" / "failure-brief" / "export-dev006-qwen"
PUBLIC = ROOT / "data" / "dataset-v0-public"


def _run(*args: str) -> str:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=True
    )
    return done.stdout


def test_brief_names_the_failed_step_with_its_evidence() -> None:
    md = _run(str(EXPORT))
    assert "| Quantification | FAIL | deterministic check | K5 | S9 |" in md
    assert "Cited fit run1_all3 (rietveld, wt%): cod1526458 20.92" in md
    assert "Answer (wt%): cod_1010093 35.59, cod_1526458 20.92" in md
    assert "## Where to look" in md
    assert "the harness tool, call 4 (refine)" in md


def test_json_matches_the_records() -> None:
    [brief] = json.loads(_run(str(EXPORT), "--json"))
    assert (brief["case_id"], brief["model"], brief["cited_fit"]) == ("DEV006", "qwen36-27b", "run1_all3")
    failed = [s for s in brief["steps"] if s["outcome"] == "fail"]
    assert [s["step"] for s in failed] == ["quantification"]
    ev = failed[0]["evidence"]
    assert ev["answer_ids_in_fit"] == []
    assert ev["difference_wt_points_after_id_match"] == {
        "cod_1010093": 0.0, "cod_1526458": 0.0, "cod_9008283": 0.0
    }
    assert brief["critiques"] == []
    assert any("reviewer critique" in m for m in brief["not_in_records"])


def test_public_episodes_invent_no_failure() -> None:
    briefs = json.loads(_run(str(PUBLIC), "--json"))
    assert len(briefs) == 6
    assert all(s["outcome"] != "fail" for b in briefs for s in b["steps"])
    assert all(b["where_to_look"] == [] for b in briefs)
