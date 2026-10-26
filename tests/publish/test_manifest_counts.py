"""Versioned counts over a frozen copy of the published manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseeval.workbench import exports

def reconcile_counts(*args):
    return exports.reconcile_counts(*args)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def v0():
    return (json.loads((FIXTURES / "published-v0-manifest.json").read_bytes()),
            json.loads((FIXTURES / "published-v0-row-counts.json").read_bytes()))


def test_the_published_v0_manifest_counts_accepted_episodes_before_filtering(v0) -> None:
    manifest, rows = v0
    assert manifest["schema_version"] == "axa.workbench.dataset.v1"
    assert manifest["episode_export"]["episodes"] == 25
    assert rows["episodes.jsonl"] == 6
    assert manifest["episode_export"]["episodes"] != rows["episodes.jsonl"]


def test_the_published_v0_manifest_reconciles_without_exception(v0) -> None:
    manifest, rows = v0
    problems = reconcile_counts("axa.workbench.dataset.v1", manifest["episode_export"], manifest["counts"], rows)
    assert problems == ()


def test_a_v1_count_that_disagrees_with_its_file_fails(v0) -> None:
    manifest, rows = v0
    tampered = [dict(row) for row in manifest["counts"]]
    next(r for r in tampered if r["name"] == "snapshots")["count"] = 5
    problems = reconcile_counts("axa.workbench.dataset.v1", manifest["episode_export"], tampered, rows)
    assert [p.code for p in problems] == ["count_mismatch"]
    assert "snapshots" in problems[0].detail


def test_v1_judge_input_count_matches_its_file(v0) -> None:
    manifest, rows = v0
    count = {"name": "judge_inputs", "count": 6, "denominator": 999}
    counts = [*manifest["counts"], count]
    assert rows["judge_inputs.jsonl"] == 6
    assert reconcile_counts(manifest["schema_version"], manifest["episode_export"], counts, rows) == ()

    count["count"] = 777
    problems = reconcile_counts(manifest["schema_version"], manifest["episode_export"], counts, rows)

    assert [(p.code, p.detail) for p in problems] == [
        ("count_mismatch", "judge_inputs: manifest says 777, judge_inputs.jsonl has 6 rows"),
    ]


def test_two_accepted_one_excluded_and_public_filtering():
    export = {"episodes": 2, "excluded": 1, "leaves": ["a", "b", "c"]}
    counts = [{"name": "episodes_exported", "count": 2, "denominator": 3},
              {"name": "episodes_excluded", "count": 1, "denominator": 3}]
    assert reconcile_counts("axa.workbench.dataset.v1", export, counts,
                            {"episodes.jsonl": 2, "excluded.jsonl": 1}) == ()
    counts[0] = {**counts[0], "count": 1}
    assert reconcile_counts("axa.workbench.dataset.v1", export, counts,
                            {"episodes.jsonl": 1, "excluded.jsonl": 1}) == ()


def test_v0_denominator_mismatch_is_rejected(v0):
    manifest, rows = v0
    counts = [dict(row) for row in manifest["counts"]]
    next(r for r in counts if r["name"] == "episodes_exported")["denominator"] = 24
    assert reconcile_counts("axa.workbench.dataset.v1", manifest["episode_export"], counts, rows)


def test_v2_semantics_name_walked_and_exported_separately() -> None:
    rows = {"episodes.jsonl": 6, "snapshots.jsonl": 6}
    counts = [
        {"name": "episodes_exported", "count": 6, "denominator": 25},
        {"name": "snapshots", "count": 6, "denominator": 6},
    ]
    assert reconcile_counts("phaseeval.dataset-manifest.v2", {"episodes_walked": 25, "episodes_exported": 6}, counts, rows) == ()
    assert reconcile_counts("phaseeval.dataset-manifest.v2", {"episodes": 25}, counts, rows)[0].code == "count_mismatch"
    below = reconcile_counts("phaseeval.dataset-manifest.v2", {"episodes_walked": 5, "episodes_exported": 6}, counts, rows)
    assert any("below" in p.detail for p in below)
    assert reconcile_counts("phaseeval.dataset-manifest.v2", {"episodes_walked": 25, "episodes_exported": 7}, counts, rows)


def test_a_count_above_its_denominator_fails_under_both_versions() -> None:
    counts = [{"name": "labels_expert", "count": 3, "denominator": 2}]
    for version, export in (("axa.workbench.dataset.v1", {"episodes": 0}), ("phaseeval.dataset-manifest.v2", {"episodes_walked": 0, "episodes_exported": 0})):
        problems = reconcile_counts(version, export, counts, {})
        assert any("exceeds" in p.detail for p in problems), version
