"""GET /api/attempts/snapshots/{digest}/trace/{call_index}: one call of a snapshot's trace in full,
behind the same gate as the blind review. `shown` is everything the agent was shown for the call,
unclipped; `executor` is the tool's full output when the runner kept it beside the trajectory and
the import bound it (by sha256) among the run's receipts. Both pass the run-identity scrub."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_labels import CASE, IMPORTER, OWNER, RATER_A, RATER_B, Bench, _snapshot

from phaseeval.api.app import app
from phaseeval.workbench.model import BlobDigest, ImportedRunResult, ImportRun

TAIL = "tail of the ranking by qwen3-32b\n" * 400
SHOWN = TAIL  # the runner's observation: the last characters of the executor's output
EXECUTOR = ("Ranking (top 60 of 60)\n" + "line written by qwen3-32b in /jobs/u6-dev900/x\n" * 900 + TAIL).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trajectory(executor_sha: str) -> bytes:
    result = {"source_call_id": "c1", "content": SHOWN, "extra": {"executor_sha256": executor_sha, "executor_chars": len(EXECUTOR), "shown_chars": 20000}}
    steps = [
        {"source": "user", "timestamp": "2026-10-21T10:00:00Z", "message": "Solve case DEV900."},
        {"source": "agent", "timestamp": "2026-10-21T10:00:05Z", "message": "Screening.", "tool_calls": [{"tool_call_id": "c1", "function_name": "screen", "arguments": {}}], "observation": {"results": [result]}},
    ]
    return json.dumps({"schema_version": "ATIF-v1.2", "session_id": "s-1", "agent": {"name": "runner"}, "steps": steps}).encode()


def _import(bench: Bench, trial: str, receipts: list[bytes]) -> str:
    action = ImportRun.model_validate(
        {"harbor": {"job_name": "u6-dev900", "trial_id": trial, "task_sha256": "4" * 64, "pattern_sha256": CASE.pattern.sha256, "cif_sha256": {c.candidate_id: c.cif.sha256 for c in CASE.candidates if c.candidate_id == "CIF000001"}}}
    )
    result = ImportedRunResult.model_validate(
        {
            "agent": {"model_id": "qwen3-32b", "model_version": None, "harness": "harbor-local-mcp", "harness_version": "1"},
            "budget_wall_seconds": 1720, "wall_seconds": 900.0, "cost_usd": None,
            "end": {"end": "submitted", "answer": {"verdict": "ACCEPT", "phases": ["CIF000001"]}},
            "basis_fit": None,
            "trajectory": [{"index": 0, "tool": "screen", "arguments_sha256": "6" * 64, "summary": "screen"}],
            "completeness": {"kind": "incomplete", "missing": ["native_lst"]},
        }
    )
    return bench.store.import_run(IMPORTER, bench.root, action, result, {BlobDigest(_sha(r)): r for r in receipts})


@pytest.fixture
def bench(tmp_path: Path):  # noqa: ANN201
    bench = Bench(tmp_path)
    saved = {name: getattr(app.state, name, None) for name in ("workbench_store", "attempt_authenticator")}
    app.state.workbench_store, app.state.attempt_authenticator = bench.store, bench.authenticator
    yield bench
    for name, value in saved.items():
        setattr(app.state, name, value)


def _snap(bench: Bench, receipts: list[bytes], trial: str) -> str:
    run = _import(bench, trial, receipts)
    snap = bench.labels.put_snapshot(_snapshot(bench.root, run, ("CIF000001",)))
    assert bench.assign(["rater-a"], snapshots=[snap]).status_code == 201
    return snap


def test_a_rater_reads_a_calls_full_shown_output_and_the_executors_whole_output_scrubbed(bench: Bench) -> None:
    snap = _snap(bench, [_trajectory(_sha(EXECUTOR)), EXECUTOR], "task-dev900__full")
    response = bench.get(RATER_A, f"/snapshots/{snap}/trace/0")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["call_index"] == 0
    assert body["shown"].startswith("tail of the ranking by [run identity]") and len(body["shown"]) > 6000
    assert body["executor"].startswith("Ranking (top 60 of 60)\nline written by [run identity] in [path]")
    assert "qwen" not in json.dumps(body) and "/jobs/" not in json.dumps(body)


def test_another_receipt_named_as_the_executor_output_is_not_served(bench: Bench) -> None:
    """The executor's output must end with what the agent was shown: a run's config or result file
    named by its sha256 is not served in its place."""
    other = json.dumps({"agent": {"endpoint_label": "lab-gpu-7"}}).encode()
    snap = _snap(bench, [_trajectory(_sha(other)), other], "task-dev900__other")
    body = bench.get(RATER_A, f"/snapshots/{snap}/trace/0").json()
    assert body["executor"] is None and "lab-gpu-7" not in json.dumps(body)


def test_an_executor_record_the_import_did_not_bind_is_not_served(bench: Bench) -> None:
    snap = _snap(bench, [_trajectory(_sha(b"bytes that were never imported"))], "task-dev900__unbound")
    body = bench.get(RATER_A, f"/snapshots/{snap}/trace/0").json()
    assert body["executor"] is None and body["shown"]


def test_the_full_call_has_the_reviews_gate_and_a_missing_call_is_not_found(bench: Bench) -> None:
    snap = _snap(bench, [_trajectory(_sha(EXECUTOR)), EXECUTOR], "task-dev900__gate")
    assert bench.get(RATER_B, f"/snapshots/{snap}/trace/0").status_code == 403
    assert bench.get(None, f"/snapshots/{snap}/trace/0").status_code in (401, 403)
    assert bench.get(RATER_A, f"/snapshots/{snap}/trace/7").status_code == 404
    assert bench.get(OWNER, f"/snapshots/{snap}/trace/0").status_code in (200, 403)
