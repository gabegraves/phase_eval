"""Fixed attempted groups: engineering diagnostics never become scientific rewards."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from phaseeval.reward_profile import profile_group


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(tmp_path, ids=("a", "b", "c", "d")):
    identity = {"case_id": "DEV006", "unknown_pins": ["served_model_revision"]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = save(tmp_path / "group.json", {
        "schema_version": 1, "group_id": "group1", "created_at": "2026-10-01T00:00:00+00:00",
        "comparison_fingerprint": fingerprint, "episode_ids": list(ids),
    })
    return manifest, identity


def receipt(tmp_path, manifest, identity, episode_id, **changes):
    m = json.loads(manifest.read_bytes())
    row = {
        "schema_version": 1, "session_id": f"session-{episode_id}", "trial_id": f"trial-{episode_id}",
        "started_at": "2026-10-02T00:00:00+00:00", "finished_at": "2026-10-02T00:00:10+00:00",
        "group": {"group_id": m["group_id"], "manifest_sha256": sha(manifest), "episode_id": episode_id,
                  "comparison_fingerprint": m["comparison_fingerprint"]},
        "comparison": {"fingerprint": m["comparison_fingerprint"], "identity": identity},
        "setup": {"status": "valid", "detail": None}, "ending": {"kind": "submitted", "detail": None},
        "runtime": {"served_models": ["model"], "system_fingerprints": [], "bgmn_binary_sha256": None},
        "usage": {k: {"known_subtotal": 0, "complete": True, "total": 0}
                  for k in ("prompt_tokens", "completion_tokens")},
        "provider_responses": [], "actions": [], "wall_seconds": 10.0, "cost_usd": None,
        "workbench_import": "refinement_bound",
    }
    row.update(changes)
    return save(tmp_path / f"{episode_id}.json", row)


def attachment(tmp_path, path, *, value=0.0, withheld=("training unpinned",), admitted=False,
               validity="valid", missing=False):
    r = json.loads(path.read_bytes())
    return save(tmp_path / f"{r['trial_id']}-score.json", {
        "schema_version": 1, "runner_receipt_sha256": sha(path), "trial_id": r["trial_id"],
        "episode_score": {
            "episode": hashlib.sha256(r["trial_id"].encode()).hexdigest(), "leaf": "f" * 64,
            "case_id": "DEV006", "model": "model", "harness": "test", "job": "test", "trial": r["trial_id"],
            "answered": True, "end": "submitted", "snapshot": None, "rwp_percent": None,
            "tool_calls": 0, "steps": [], "analyst": None, "analyst_none": False,
            "reward": {"spec_version": "synthetic-test-only", "total": 0.2, "verifiable": 0.2, "judge": 0.0,
                       "terms": [{"id": "answered", "weight": 0.2, "value": 1.0, "missing": missing, "detail": "test"}],
                       "validity": validity, "admitted": admitted, "training": value, "training_withheld": list(withheld)},
        },
    })


def test_all_attempts_visible_without_manufacturing_rewards(tmp_path):
    m, identity = group(tmp_path)
    a = receipt(tmp_path, m, identity, "a")
    b = receipt(tmp_path, m, identity, "b", workbench_import="unsupported_no_refinement")
    c = receipt(tmp_path, m, identity, "c", comparison=None, setup={"status": "invalid", "detail": "missing scan"},
                ending={"kind": "setup_failure", "detail": "missing scan"}, workbench_import="unsupported_setup_invalid")
    report = profile_group(m, [a, b, c], [attachment(tmp_path, a)])
    assert report["counts"] == {"scheduled": 4, "observed": 3, "attempted": 3, "missing": 1,
                                "comparable": 2, "invalid": 1, "unfinished": 0, "excluded": 1,
                                "unscheduled": 0, "independent_episodes": 2}
    assert report["scientific_reward"] == {"known": 0, "missing": 4, "mean": None, "population_std": None}
    assert [r["scientific_status"] for r in report["episodes"]] == [
        "withheld", "unsupported_no_refinement", "execution_invalid", "missing_attempt"]
    assert report["diagnostics"]["answered"]["value"]["known"] == 1
    assert report["diagnostics"]["answered"]["value"]["missing"] == 3


@pytest.mark.parametrize("values,mean,std", [([], None, None), ([0], 0, None), ([0, 0], 0, 0), ([0, 1], 0.5, 0.5)])
def test_population_statistics_only_use_qualified_values(tmp_path, values, mean, std):
    m, identity = group(tmp_path, tuple("ab"))
    receipts = [receipt(tmp_path, m, identity, i) for i in "ab"]
    scores = [attachment(tmp_path, r, value=v, admitted=True, withheld=()) for r, v in zip(receipts, values)]
    stats = profile_group(m, receipts, scores)["scientific_reward"]
    assert stats == {"known": len(values), "missing": 2 - len(values), "mean": mean, "population_std": std}


def test_diagnostic_missingness_preserves_weights(tmp_path):
    m, identity = group(tmp_path, ("a", "b"))
    a, b = [receipt(tmp_path, m, identity, i) for i in "ab"]
    report = profile_group(m, [a, b], [attachment(tmp_path, a), attachment(tmp_path, b, missing=True)])
    term = report["diagnostics"]["answered"]
    assert term["weight"] == 0.2
    assert term["value"] == {"known": 1, "missing": 1, "mean": 1, "population_std": None}
    assert term["weighted_value"]["mean"] == 0.2


@pytest.mark.parametrize("field", ["session_id", "trial_id"])
def test_compact_siblings_are_not_independent_attempts(tmp_path, field):
    m, identity = group(tmp_path, ("a", "b"))
    a = receipt(tmp_path, m, identity, "a")
    b = receipt(tmp_path, m, identity, "b", **{field: json.loads(a.read_bytes())[field]})
    with pytest.raises(ValueError, match="duplicate"):
        profile_group(m, [a, b])


def test_duplicate_input_or_episode_is_rejected(tmp_path):
    m, identity = group(tmp_path, ("a",))
    a = receipt(tmp_path, m, identity, "a")
    with pytest.raises(ValueError, match="duplicate"):
        profile_group(m, [a, a])
    other = save(tmp_path / "other.json", json.loads(a.read_bytes()) | {"session_id": "different", "trial_id": "different"})
    with pytest.raises(ValueError, match="duplicate"):
        profile_group(m, [a, other])


def test_changed_identity_is_explicitly_excluded(tmp_path):
    m, identity = group(tmp_path, ("a",))
    changed = identity | {"case_id": "OTHER"}
    fingerprint = hashlib.sha256(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    a = receipt(tmp_path, m, identity, "a", comparison={"fingerprint": fingerprint, "identity": changed})
    report = profile_group(m, [a])
    assert report["counts"]["excluded"] == 1
    assert "comparison_mismatch" in report["episodes"][0]["exclusions"]


def test_unscheduled_receipts_and_unfinished_attempts_remain_visible(tmp_path):
    m, identity = group(tmp_path, ("a", "b"))
    a = receipt(tmp_path, m, identity, "a", finished_at=None, ending=None, wall_seconds=None)
    x = receipt(tmp_path, m, identity, "x")
    report = profile_group(m, [x, a])
    assert report["counts"]["unfinished"] == 1
    assert report["counts"]["missing"] == 1
    assert report["counts"]["unscheduled"] == 1
    assert report["unscheduled"][0]["episode_id"] == "x"
    assert report["episodes"][0]["scientific_status"] == "unfinished"


def test_failures_do_not_become_scientific_zero(tmp_path):
    m, identity = group(tmp_path, ("a", "b"))
    a = receipt(tmp_path, m, identity, "a", ending={"kind": "provider_failure", "detail": "HTTP failure"})
    b = receipt(tmp_path, m, identity, "b", ending={"kind": "no_submission", "detail": None})
    report = profile_group(m, [a, b], [attachment(tmp_path, a, admitted=True, withheld=())])
    assert report["counts"]["invalid"] == 1
    assert report["episodes"][0]["scientific_status"] == "execution_invalid"
    assert report["episodes"][1]["scientific_status"] == "pending_qualification"
    assert report["scientific_reward"]["known"] == 0


def test_original_costs_keep_zero_and_unknown_separate(tmp_path):
    m, identity = group(tmp_path, ("a", "b", "c"))
    usage = {k: {"known_subtotal": 7, "complete": False, "total": None} for k in ("prompt_tokens", "completion_tokens")}
    a = receipt(tmp_path, m, identity, "a", cost_usd=0)
    b = receipt(tmp_path, m, identity, "b", usage=usage, actions=[{
        "tool_call_id": "call1", "function_name": "refine", "started_at": "2026-10-02T00:00:01+00:00",
        "duration_seconds": 3.0, "status": "completed"}])
    report = profile_group(m, [a, b])
    assert report["costs"]["prompt_tokens"] == {"known_subtotal": 7, "complete": False, "total": None,
                                                 "known_episodes": 1, "missing_episodes": 2}
    assert report["costs"]["cost_usd"]["known_subtotal"] == 0
    assert report["costs"]["cost_usd"]["total"] is None
    assert report["episodes"][1]["costs"]["action_seconds"]["total"] == 3


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"wall_seconds": -1}, {"cost_usd": True},
    {"usage": {"prompt_tokens": {"known_subtotal": 2, "complete": True, "total": 3}}},
])
def test_malformed_receipts_fail_closed(tmp_path, change):
    m, identity = group(tmp_path, ("a",))
    a = receipt(tmp_path, m, identity, "a", **change)
    with pytest.raises(ValueError):
        profile_group(m, [a])


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":1}', '{"value":NaN}', '{"value":1e999}'])
def test_non_strict_json_rejected(tmp_path, text):
    m, _ = group(tmp_path, ("a",))
    bad = tmp_path / "bad.json"
    bad.write_text(text)
    with pytest.raises(ValueError):
        profile_group(m, [bad])


@pytest.mark.parametrize("mutation", ["receipt_hash", "trial", "score_trial", "boolean_reward", "nonfinite_reward", "duplicate_terms", "admission_type"])
def test_bad_scorecard_binding_or_metrics_rejected(tmp_path, mutation):
    m, identity = group(tmp_path, ("a",))
    a = receipt(tmp_path, m, identity, "a")
    path = attachment(tmp_path, a)
    row = json.loads(path.read_bytes())
    if mutation == "receipt_hash": row["runner_receipt_sha256"] = "a" * 64
    if mutation == "trial": row["trial_id"] = "another"
    if mutation == "score_trial": row["episode_score"]["trial"] = "another"
    if mutation == "boolean_reward": row["episode_score"]["reward"]["training"] = True
    if mutation == "nonfinite_reward": row["episode_score"]["reward"]["training"] = float("nan")
    if mutation == "duplicate_terms": row["episode_score"]["reward"]["terms"] *= 2
    if mutation == "admission_type": row["episode_score"]["reward"]["admitted"] = "true"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        profile_group(m, [a], [path])


def test_manifest_binding_and_chronology_fail_closed(tmp_path):
    m, identity = group(tmp_path, ("a",))
    a = receipt(tmp_path, m, identity, "a", started_at="2026-09-01T00:00:00+00:00")
    assert "manifest_postdates_attempt" in profile_group(m, [a])["episodes"][0]["exclusions"]
    m.write_text(m.read_text() + "\n")
    assert "manifest_digest_mismatch" in profile_group(m, [a])["episodes"][0]["exclusions"]


def test_runtime_variation_and_unknown_pins_are_reported_without_changing_identity(tmp_path):
    m, identity = group(tmp_path, ("a", "b"))
    a = receipt(tmp_path, m, identity, "a")
    b = receipt(tmp_path, m, identity, "b", runtime={"served_models": ["different"]})
    report = profile_group(m, [a, b])
    assert report["counts"]["comparable"] == 2
    assert len(report["runtime_variants"]) == 2
    assert report["unknown_pins"] == ["served_model_revision"]


def test_input_order_invariant_and_cli_runs_outside_repo(tmp_path):
    m, identity = group(tmp_path, ("b", "a"))
    a, b = [receipt(tmp_path, m, identity, i) for i in "ab"]
    scores = [attachment(tmp_path, a), attachment(tmp_path, b)]
    expected = profile_group(m, [a, b], scores)
    assert expected == profile_group(m, [b, a], scores[::-1])
    script = Path(__file__).resolve().parents[2] / "scripts/profile_rewards.py"
    result = subprocess.run([sys.executable, str(script), "--manifest", str(m), "--receipt", str(b),
                             "--receipt", str(a), "--scorecard", str(scores[0]), "--scorecard", str(scores[1])],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_constructor_pending_is_attempted_but_not_an_independent_policy_episode(tmp_path):
    m, identity = group(tmp_path, ("a", "b"))
    a = receipt(tmp_path, m, identity, "a", comparison=None, setup={"status": "pending", "detail": None},
                finished_at=None, ending=None, wall_seconds=None)
    report = profile_group(m, [a])
    assert report["counts"]["attempted"] == 1
    assert report["counts"]["missing"] == 1
    assert report["counts"]["unfinished"] == 1
    assert report["counts"]["independent_episodes"] == 0


def test_current_scorecard_placeholder_remains_unavailable(tmp_path):
    from phaseeval.workbench.scorecard import _reward
    m, identity = group(tmp_path, ("a",))
    a = receipt(tmp_path, m, identity, "a")
    path = attachment(tmp_path, a)
    row = json.loads(path.read_bytes())
    row["episode_score"]["reward"] = _reward((), False, [], invalid=False, admission=(True, "")).model_dump(mode="json")
    save(path, row)
    report = profile_group(m, [a], [path])
    assert report["scientific_reward"]["known"] == 0
    assert report["scientific_reward"]["mean"] is None
    assert report["episodes"][0]["reward_gates"]["training_withheld"]


@pytest.mark.parametrize("mutation", ["duplicate_scorecard", "different_spec", "different_weight"])
def test_ambiguous_scorecards_fail_closed(tmp_path, mutation):
    m, identity = group(tmp_path, ("a", "b"))
    a, b = [receipt(tmp_path, m, identity, i) for i in "ab"]
    first, second = attachment(tmp_path, a), attachment(tmp_path, b)
    if mutation == "duplicate_scorecard":
        second = first
    else:
        row = json.loads(second.read_bytes())
        if mutation == "different_spec": row["episode_score"]["reward"]["spec_version"] = "other"
        if mutation == "different_weight": row["episode_score"]["reward"]["terms"][0]["weight"] = 0.5
        save(second, row)
    with pytest.raises(ValueError):
        profile_group(m, [a, b], [first, second])


def test_duplicate_scheduled_ids_rejected(tmp_path):
    m, _ = group(tmp_path, ("a", "a"))
    with pytest.raises(ValueError, match="duplicate"):
        profile_group(m, [])
