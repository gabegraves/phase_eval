"""Native evidence is recoverable only through this runner's registered actor handles."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evidence_runner", REPO / "harness/agent/run_agent.py")
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)


@pytest.fixture
def runner(tmp_path):
    inputs = tmp_path / "input"
    shutil.copytree(REPO / "harness/examples/dev006/input", inputs)
    return runner_module.Runner(argparse.Namespace(
        input_dir=inputs, case="DEV006", out=tmp_path / "trial", base_url="https://invalid.example",
        model="scripted", endpoint_label="test", api_key_env="NO_TEST_KEY", max_turns=5, time_limit=30,
    ))


def read(runner, **args):
    result = runner.call("read_evidence", args)
    assert not result.startswith("error:"), result
    return json.loads(result)


def text_pages(runner, handle):
    chunks, start = [], 0
    while True:
        page = read(runner, handle=handle, start=start, limit=127)
        chunks.append(page["text"])
        if page["next_start"] is None:
            return "".join(chunks), page
        start = page["next_start"]


def test_native_scan_pages_reconstruct_every_row_with_local_indexes(runner):
    data = (runner.args.input_dir / "DEV006.xy").read_bytes()
    expected = [[float(x) for x in line.split()] for line in data.decode().splitlines()]
    rows, indexes, start = [], [], 0
    while True:
        page = read(runner, handle="scan", start=start, limit=100)
        assert page["sha256"] == hashlib.sha256(data).hexdigest()
        assert page["units"] == ["degree_2theta", "recorded_intensity"]
        assert len(page["rows"]) <= 100
        rows.extend(page["rows"])
        indexes.extend(page["row_indexes"])
        if page["next_start"] is None:
            break
        start = page["next_start"]
    assert rows == expected
    assert indexes == list(range(len(expected)))
    assert page["total_rows"] == len(expected)


def test_interval_retains_narrow_peak_and_native_indexes(runner):
    whole = read(runner, handle="scan", limit=100)
    low, high = whole["rows"][31][0], whole["rows"][33][0]
    interval = read(runner, handle="scan", two_theta_min=low, two_theta_max=high, limit=2)
    assert interval["row_indexes"] == [31, 32]
    last = read(runner, handle="scan", two_theta_min=low, two_theta_max=high, start=interval["next_start"])
    assert last["row_indexes"] == [33]
    assert interval["rows"] + last["rows"] == whole["rows"][31:34]


@pytest.mark.parametrize("args", [
    {"start": -1}, {"start": True}, {"start": 0.5}, {"start": 999999},
    {"limit": 0}, {"limit": True}, {"limit": 101},
    {"two_theta_min": 20, "two_theta_max": 10}, {"two_theta_min": float("nan")},
    {"two_theta_max": float("inf")}, {"two_theta_min": True}, {"path": "/etc/passwd"},
])
def test_invalid_ranges_are_rejected(runner, args):
    assert runner.call("read_evidence", {"handle": "scan", **args}).startswith("error:")


@pytest.mark.parametrize("handle", [
    "../input/manifest.json", "/etc/passwd", "custody.json", "case_labels.json", ".hidden",
    "fit:another-episode:rietveld", "fit:unused:rietveld", "candidate:unknown",
])
def test_unregistered_paths_are_denied(runner, handle):
    assert runner.call("read_evidence", {"handle": handle}).startswith("error:")


def test_registered_file_cannot_be_replaced_by_symlink_or_changed(runner, tmp_path):
    scan = runner.args.input_dir / "DEV006.xy"
    sibling = tmp_path / "sibling.xy"
    sibling.write_bytes(scan.read_bytes())
    scan.unlink()
    scan.symlink_to(sibling)
    assert runner.call("read_evidence", {"handle": "scan"}).startswith("error:")
    scan.unlink()
    scan.write_text("1 1\n2 2\n")
    assert runner.call("read_evidence", {"handle": "scan"}).startswith("error:")


def test_cif_text_reconstructs_exact_bytes(runner):
    value, page = text_pages(runner, "candidate:cod_1010093")
    raw = (runner.args.input_dir / "candidates/DEV006/cod_1010093.cif").read_bytes()
    assert value.encode() == raw
    assert page["sha256"] == hashlib.sha256(raw).hexdigest()


def test_fit_registration_uses_only_completed_current_episode_stages(runner, monkeypatch):
    csv = b"two_theta,observed,calculated,difference\n10.5,12,11,1\n11.5,22,20,2\n"
    def refine(*args, **kwargs):
        folder = runner.output / "fit1"
        folder.mkdir()
        (folder / "pawley.csv").write_bytes(csv)
        (folder / "rietveld.csv").write_bytes(csv)
        (folder / "result.json").write_text(json.dumps({"pawley": {"stage": "pawley"}, "rietveld": None}))
        return "saved Pawley"
    monkeypatch.setattr(runner, "tool", refine)
    runner.call("refine", {"candidates": ["cod_1010093"], "run_label": "fit1"})
    page = read(runner, handle="fit:fit1:pawley")
    assert page["rows"] == [[10.5, 12, 11, 1], [11.5, 22, 20, 2]]
    assert page["row_indexes"] == [0, 1]
    assert page["sha256"] == hashlib.sha256(csv).hexdigest()
    assert runner.call("read_evidence", {"handle": "fit:fit1:rietveld"}).startswith("error:")
    (runner.output / "fit1/pawley.csv").unlink()
    assert runner.call("read_evidence", {"handle": "fit:fit1:pawley"}).startswith("error:")


@pytest.mark.parametrize("bad", ["1 NaN\n2 5\n", "1 4\n2 inf\n", "2 4\n1 5\n"])
def test_nonfinite_or_unordered_scan_is_rejected(runner, bad):
    scan = runner.args.input_dir / "DEV006.xy"
    scan.write_text(bad)
    manifest = runner.args.input_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    data["cases"][0]["pattern_sha256"] = hashlib.sha256(scan.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    another = runner_module.Runner(argparse.Namespace(**(vars(runner.args) | {"out": runner.out.parent / "new"})))
    assert another.call("read_evidence", {"handle": "scan"}).startswith("error:")


def test_context_and_brief_share_actor_projection(runner):
    manifest = runner.args.input_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    case = data["cases"][0]
    case["chemistry_elements"] = ["HIDDEN", "Ni", "O"]
    case["measurement"] = {"instrument": "recorded device", "private_extension": "HIDDEN"}
    case["material_context"] = {
        "specimen": "powder", "secret_extension": "HIDDEN",
        "precursors": [{"formula": "NiO", "actual_mass_g": 987654, "private": "HIDDEN"}],
        "processing": [{"action": "Heating", "private": "HIDDEN", "temperature_profile": {
            "time_minutes": list(range(300)), "temperature_celsius": list(range(300)), "secret": "HIDDEN"}}],
    }
    case["field_sources"] = {
        "chemistry_elements.0": {"class": "custody", "locator": "HIDDEN"},
        "material_context.precursors.0.actual_mass_g": {"class": "custody", "locator": "HIDDEN"},
        "material_context.processing.0.temperature_profile": {"class": "synthesis_record", "locator": "recipe.json furnace trace"},
    }
    manifest.write_text(json.dumps(data))
    another = runner_module.Runner(argparse.Namespace(**(vars(runner.args) | {"out": runner.out.parent / "context"})))
    raw, _ = text_pages(another, "context")
    context = json.loads(raw)
    assert "HIDDEN" not in raw and "987654" not in raw
    assert context["chemistry_elements"] == [None, "Ni", "O"]
    assert context["material_context"]["processing"][0]["temperature_profile"]["time_minutes"] == list(range(300))
    assert context["field_sources"]["measurement.instrument"]["class"] == "bundled_manifest"
    assert context["field_sources"]["measurement.instrument"]["source_status"] == "unknown"
    assert "recipe.json furnace trace" in raw
    assert context["calibration"] is None
    brief = another.call("case_brief", {})
    assert "HIDDEN" not in brief and "987654" not in brief
    assert "Elements: not recorded Ni O" in brief
    assert "Measurement: recorded device" in brief
    assert "300 points" in brief and "not recorded" in brief


def test_tool_is_advertised_and_actual_recovery_observation_is_retained(runner, monkeypatch):
    assert "read_evidence" in {tool["function"]["name"] for tool in runner_module.TOOLS}
    full = "α\r\n" + "x" * (runner_module.SHOWN_CHARS * 2)
    seen = []
    monkeypatch.setattr(runner, "tool", lambda *a, **kw: full)
    def chat(messages):
        seen.append(messages[-1].copy())
        if len(seen) == 1:
            name, args = "case_brief", {}
        elif len(seen) == 2:
            assert "clipped" in messages[-1]["content"]
            name, args = "read_evidence", {"handle": "output:" + hashlib.sha256(full.encode()).hexdigest(), "limit": 100}
        else:
            return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        return {"choices": [{"message": {"tool_calls": [{"id": str(len(seen)), "function": {"name": name, "arguments": json.dumps(args)}}]}}]}
    monkeypatch.setattr(runner, "chat", chat)
    assert runner.run() == 0
    recovered, _ = text_pages(runner, "output:" + hashlib.sha256(full.encode()).hexdigest())
    assert recovered == full
    trajectory = json.loads((runner.out / "agent/trajectory.json").read_text())
    observations = [r for step in trajectory["steps"] for r in step.get("observation", {}).get("results", [])]
    assert observations[0]["content"] == seen[1]["content"]
    assert json.loads(observations[1]["content"])["text"] == full[:100]


@pytest.mark.parametrize("name", ["../outside.xy", "/etc/passwd", ".hidden.xy", "custody.json"])
def test_manifest_cannot_register_forbidden_scan_path(runner, name):
    manifest = runner.args.input_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    data["cases"][0]["pattern_file"] = name
    manifest.write_text(json.dumps(data))
    try:
        another = runner_module.Runner(argparse.Namespace(**(vars(runner.args) | {"out": runner.out.parent / "bad-path"})))
    except ValueError:
        return
    assert another.call("read_evidence", {"handle": "scan"}).startswith("error:")


def test_candidate_directory_symlink_to_sibling_episode_is_denied(runner, tmp_path):
    directory = runner.args.input_dir / "candidates/DEV006"
    sibling = tmp_path / "sibling-episode"
    directory.rename(sibling)
    directory.symlink_to(sibling, target_is_directory=True)
    assert runner.call("read_evidence", {"handle": "candidate:cod_1010093"}).startswith("error:")


def test_nonfinite_saved_curve_is_rejected(runner, monkeypatch):
    def refine(*args, **kwargs):
        folder = runner.output / "bad"
        folder.mkdir()
        (folder / "pawley.csv").write_text("two_theta,observed,calculated,difference\n10,12,nan,1\n")
        (folder / "result.json").write_text('{"pawley":{"stage":"pawley"}}')
        return "saved incomplete numerical evidence"
    monkeypatch.setattr(runner, "tool", refine)
    runner.call("refine", {"candidates": ["cod_1010093"], "run_label": "bad"})
    assert runner.call("read_evidence", {"handle": "fit:bad:pawley"}).startswith("error:")


def test_unfinished_fit_and_cross_episode_output_are_not_registered(runner, monkeypatch, tmp_path):
    def refine(*args, **kwargs):
        folder = runner.output / "unfinished"
        folder.mkdir()
        (folder / "pawley.csv").write_text("two_theta,observed,calculated,difference\n10,12,11,1\n")
        return "tool stopped before result was saved"
    monkeypatch.setattr(runner, "tool", refine)
    runner.call("refine", {"candidates": ["cod_1010093"], "run_label": "unfinished"})
    assert runner.call("read_evidence", {"handle": "fit:unfinished:pawley"}).startswith("error:")
    other = runner_module.observation(tmp_path / "other", "same-id", "PRIVATE" * runner_module.SHOWN_CHARS)
    assert runner.call("read_evidence", {"handle": other["extra"]["evidence_handle"]}).startswith("error:")


def test_custody_ancestor_cannot_be_overridden_by_safe_child_source(runner):
    manifest = runner.args.input_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    case = data["cases"][0]
    case["material_context"] = {"precursors": [{"formula": "NiO", "actual_mass_g": 987654}]}
    case["field_sources"] = {
        "material_context.precursors": {"class": "operator_note", "locator": "custody.json"},
        "material_context.precursors.0.actual_mass_g": {"class": "packaging", "locator": "manifest.json"},
    }
    manifest.write_text(json.dumps(data))
    another = runner_module.Runner(argparse.Namespace(**(vars(runner.args) | {"out": runner.out.parent / "ancestor"})))
    text, _ = text_pages(another, "context")
    assert "987654" not in text and "custody.json" not in text
    assert "987654" not in another.call("case_brief", {})


def test_prompt_only_advertises_available_actions():
    prompt = (REPO / "harness/physics/README.md").read_text() + (REPO / "harness/agent/task.md").read_text()
    assert "Bash tool" not in prompt
    assert "then runs a tree phase search" not in prompt
    assert "read_evidence" in prompt


def test_runner_cli_loads_outside_repository(tmp_path):
    process = subprocess.run([sys.executable, str(REPO / "harness/agent/run_agent.py"), "--help"],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "--input-dir" in process.stdout
