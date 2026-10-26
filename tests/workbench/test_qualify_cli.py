"""The owner-host qualification CLI and its custody reader (AXA-43zwk.26, Q1 and Q5)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from tests.support.qualification import Bench
from tests.support.store import OWNER

from phaseeval.workbench.qualification import partition_of

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/workbench/qualify.py"
spec = importlib.util.spec_from_file_location("workbench_qualify", SCRIPT)
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


def test_the_custody_reader_returns_candidates_per_reference_phase(tmp_path) -> None:
    case = tmp_path / "DEV006"
    case.mkdir()
    coverage = [
        {"formula": "NiO", "space_group": None, "weight_percent": 40.0, "shipped_candidates": ["CIF012075", "CIF012076"], "covered": True},
        {"formula": "Bi2O3", "space_group": 14, "weight_percent": 35.0, "shipped_candidates": ["CIF005806"], "covered": True},
    ]
    (case / "custody.json").write_text(json.dumps({"reference_phase_coverage": coverage}))
    reference = qualify.CustodyReferences(tmp_path).reference("DEV006")
    assert reference.case_id == "DEV006" and reference.source == "custody reference_phase_coverage"
    assert [(p.formula, p.space_group, p.candidates) for p in reference.phases] == [
        ("NiO", None, ("CIF012075", "CIF012076")),
        ("Bi2O3", 14, ("CIF005806",)),
    ]
    assert qualify.CustodyReferences(tmp_path).reference("DEV102") is None
    assert qualify.CustodyReferences(None).reference("DEV006") is None


def _cli(capsys, store: Path, *argv: str) -> dict:
    qualify.main(["--store", str(store), "--owner", OWNER.principal_id, *argv])
    return json.loads(capsys.readouterr().out)


def test_freeze_run_and_plan_from_the_command_line(tmp_path, capsys) -> None:
    bench = Bench(tmp_path)
    store = bench.store.root
    frozen = _cli(
        capsys, store, "freeze", "--lab", "lab-a", "--name", "cli", "--case", "DEV900", "--rater-kind", "expert",
        "--min-raters", "2", "--identity", "candidate_id", "--policy-version", "v1", "--justified-by", "test",
    )
    assert frozen["roots"] == 1
    report = _cli(capsys, store, "run", "--set", frozen["set"], "--producer", "gpt-6-astra:harness-v2")
    assert report["subject"]["harness_version"] == "2026-10-19"  # the one version the store holds
    assert [c["verdict"] for c in report["cases"]] == ["fail"]
    assert report["claim"] == "agreement_with_lab_judgments"
    assert _cli(capsys, store, "plan", "--set", frozen["set"], "--producer", "qwen36-27b:harness-v2") == {"roots": []}
    expectations = _cli(capsys, store, "expectations", "--set", frozen["set"])
    assert {p["expect"] for p in expectations[bench.root]["phases"]} == {"required", "free"}
    with pytest.raises(SystemExit, match="NoFrozenSplit"):
        qualify.main(["--store", str(store), "--owner", OWNER.principal_id, "run", "--set", frozen["set"], "--judge", "claude-opus-5-5:" + "c" * 64])
    with pytest.raises(SystemExit, match="recorded agents"):
        qualify.main(["--store", str(store), "--owner", OWNER.principal_id, "run", "--set", frozen["set"], "--producer", "nobody:harness-v2"])


def test_a_judge_run_names_its_refusal_and_runs_on_a_tune_partition(tmp_path, capsys) -> None:
    bench = Bench(tmp_path)
    store = bench.store.root
    roots = (bench.root, bench.other)
    seed = next(f"s{i}" for i in range(64) if [partition_of(f"s{i}", r) for r in roots] == ["held_out", "tune"])
    split = bench.freeze("judged", seed)
    judge = ["run", "--set", split, "--judge", "claude-opus-5-5:" + "c" * 64]

    with pytest.raises(SystemExit, match="TunedOnHeldOut"):
        _cli(capsys, store, *judge, "--tuned-on", f"{split}:held_out")
    report = _cli(capsys, store, *judge, "--tuned-on", f"{split}:tune")
    assert report["subject"]["kind"] == "judge"
    assert report["subject"]["tuned_on"] == [{"set_digest": split, "partition": "tune"}]
    for bad in (["--judge", "claude-opus-5-5"], [*judge[3:], "--tuned-on", f"{split}:train"]):
        with pytest.raises(SystemExit, match="--judge MODEL:PROMPT_SHA256"):
            _cli(capsys, store, "run", "--set", split, *bad)
