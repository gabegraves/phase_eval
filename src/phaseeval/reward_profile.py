"""Read-only profiling of fixed attempted groups; never assigns or authorizes rewards."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from phaseeval.workbench.scorecard import EpisodeScore

SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
SHA256 = re.compile(r"[a-f0-9]{64}")
INFRASTRUCTURE = {"setup_failure", "provider_failure", "tool_failure"}
ENDINGS = INFRASTRUCTURE | {"submitted", "no_submission", "budget_exhausted"}


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _number(value, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value), "metric must be finite and numeric, not boolean")
    _require(not nonnegative or value >= 0, "metric must be nonnegative")
    return value


def _time(value):
    parsed = datetime.fromisoformat(value)
    _require(parsed.tzinfo is not None, "timestamps must be timezone-aware")
    return parsed


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _stats(values, denominator):
    return {"known": len(values), "missing": denominator - len(values),
            "mean": statistics.fmean(values) if values else None,
            "population_std": statistics.pstdev(values) if len(values) >= 2 else None}


def _cost(known, complete):
    return {"known_subtotal": known, "complete": complete, "total": known if complete else None}


def _receipt(record, sha, manifest, manifest_sha):
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1, "unsupported runner receipt version")
    for key in ("session_id", "trial_id"):
        _require(isinstance(record[key], str) and record[key], f"missing {key}")
    started = _time(record["started_at"])
    finished = record["finished_at"]
    if finished is not None:
        _require(_time(finished) >= started, "finish precedes start")
    ending = record["ending"]
    _require(ending is None or isinstance(ending, dict) and ending["kind"] in ENDINGS, "unknown episode ending")
    _require((finished is None) == (ending is None), "finish and ending must be recorded together")
    status = record["setup"]["status"]
    _require(status in {"pending", "valid", "invalid"}, "unknown setup status")
    _require(record["workbench_import"] in {"refinement_bound", "unsupported_no_refinement", "unsupported_setup_invalid"}, "unknown workbench import status")
    group = record["group"]
    _require(group is None or isinstance(group, dict), "group must be an object or null")
    comparison = record["comparison"]
    _require(comparison is None or isinstance(comparison, dict), "comparison must be an object or null")
    _require(isinstance(record["runtime"], dict), "runtime must be an object")
    exclusions = []
    if group is None:
        exclusions.append("unbound_attempt")
    else:
        for field, expected, reason in (
            ("group_id", manifest["group_id"], "group_mismatch"),
            ("manifest_sha256", manifest_sha, "manifest_digest_mismatch"),
            ("comparison_fingerprint", manifest["comparison_fingerprint"], "scheduled_comparison_mismatch"),
        ):
            if group.get(field) != expected:
                exclusions.append(reason)
    if started < _time(manifest["created_at"]):
        exclusions.append("manifest_postdates_attempt")
    if comparison is not None:
        _require(isinstance(comparison["identity"], dict), "comparison identity must be an object")
        _require(_hash(comparison["identity"]) == comparison["fingerprint"], "comparison identity digest mismatch")
        if comparison["fingerprint"] != manifest["comparison_fingerprint"]:
            exclusions.append("comparison_mismatch")
    elif status == "valid":
        exclusions.append("comparison_missing")
    invalid = status == "invalid" or ending is not None and ending["kind"] in INFRASTRUCTURE
    unfinished = ending is None or status == "pending"
    if invalid:
        exclusions.append("execution_invalid")
    if unfinished:
        exclusions.append("unfinished")
    costs = {}
    for key in ("prompt_tokens", "completion_tokens"):
        usage = record["usage"][key]
        known, complete, total = usage["known_subtotal"], usage["complete"], usage["total"]
        _require(type(known) is int and known >= 0 and type(complete) is bool, "invalid token usage")
        _require((type(total) is int and total == known) if complete else total is None, "inconsistent token total")
        costs[key] = _cost(known, complete and not unfinished)
    for key in ("wall_seconds", "cost_usd"):
        value = record[key]
        if value is not None:
            _number(value, nonnegative=True)
        costs[key] = _cost(value if value is not None else 0, value is not None and not unfinished)
    durations = []
    _require(isinstance(record["actions"], list), "actions must be an array")
    for action in record["actions"]:
        _require(action["status"] in {"started", "completed", "failed"}, "unknown action status")
        duration = action["duration_seconds"]
        if duration is not None:
            durations.append(_number(duration, nonnegative=True))
    costs["action_seconds"] = _cost(sum(durations), len(durations) == len(record["actions"]) and not unfinished)
    return {"episode_id": group.get("episode_id") if group else None, "receipt_sha256": sha,
            "session_id": record["session_id"], "trial_id": record["trial_id"], "setup": record["setup"],
            "ending": ending, "workbench_import": record["workbench_import"], "costs": costs,
            "exclusions": exclusions, "invalid": invalid, "unfinished": unfinished,
            "runtime": record["runtime"], "comparison": comparison, "scientific_reward": None,
            "scientific_status": "pending_qualification", "scorecard_sha256": None}


def _scorecard(record):
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1, "unsupported scorecard attachment version")
    raw = record["episode_score"]
    reward = raw["reward"]
    # Existing models own scorecard semantics; these checks prevent numeric coercion.
    for key in ("training", "total", "verifiable", "judge"):
        _number(reward[key])
    for term in reward["terms"]:
        _number(term["value"])
        _number(term["weight"], nonnegative=True)
        _require(type(term["missing"]) is bool, "missing must be boolean")
    _require(type(reward["admitted"]) is bool, "admitted must be boolean")
    if raw["rwp_percent"] is not None:
        _number(raw["rwp_percent"], nonnegative=True)
    score = EpisodeScore.model_validate_json(json.dumps(raw, allow_nan=False), strict=True)
    ids = [t.id for t in score.reward.terms]
    _require(len(set(ids)) == len(ids), "duplicate reward term")
    _require(score.trial == record["trial_id"], "scorecard trial binding mismatch")
    return score


def profile_group(manifest_path, receipt_paths, scorecard_attachment_paths=()):
    """Left-join every scheduled slot. Ambiguous/invalid files raise ValueError.

    Well-formed incompatible attempts remain excluded rows; missing slots remain
    missing. Attachments wrap an existing EpisodeScore and bind its trial and exact
    runner-receipt bytes. These owner-host files are provenance, not signatures.
    """
    try:
        return _profile_group(manifest_path, receipt_paths, scorecard_attachment_paths)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed profile input: {exc}") from exc


def _profile_group(manifest_path, receipt_paths, scorecard_attachment_paths):
    manifest, manifest_sha = _read(manifest_path)
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1, "unsupported manifest version")
    _require(isinstance(manifest["group_id"], str) and SAFE_ID.fullmatch(manifest["group_id"]), "invalid group ID")
    _time(manifest["created_at"])
    _require(isinstance(manifest["comparison_fingerprint"], str) and SHA256.fullmatch(manifest["comparison_fingerprint"]), "invalid comparison fingerprint")
    ids = manifest["episode_ids"]
    _require(isinstance(ids, list) and ids and all(isinstance(i, str) and SAFE_ID.fullmatch(i) for i in ids), "invalid episode IDs")
    _require(len(set(ids)) == len(ids), "duplicate scheduled episode ID")
    rows = [_receipt(*_read(path), manifest, manifest_sha) for path in receipt_paths]
    for field in ("session_id", "trial_id", "episode_id"):
        values = [r[field] for r in rows if r[field] is not None]
        _require(len(values) == len(set(values)), f"duplicate {field}; fragments/retries are not independent episodes")
    rows.sort(key=lambda r: (r["episode_id"] or "", r["receipt_sha256"]))
    by_sha = {r["receipt_sha256"]: r for r in rows}
    scores = {}
    score_episodes = set()
    for path in scorecard_attachment_paths:
        record, sha = _read(path)
        score = _scorecard(record)
        bound = record["runner_receipt_sha256"]
        _require(bound in by_sha, "scorecard has no matching receipt hash")
        row = by_sha[bound]
        _require(row["trial_id"] == record["trial_id"], "scorecard trial binding mismatch")
        _require(bound not in scores and score.episode not in score_episodes, "duplicate scorecard episode")
        _require(row["workbench_import"] == "refinement_bound", "scorecard attached to unsupported workbench import")
        if row["comparison"] is not None:
            _require(score.case_id == row["comparison"]["identity"]["case_id"], "scorecard case mismatch")
        scores[bound] = score
        score_episodes.add(score.episode)
        row["scorecard_sha256"] = sha
    specs = {s.reward.spec_version for s in scores.values()}
    _require(len(specs) <= 1, "mixed reward specifications must be profiled separately")
    by_id = {r["episode_id"]: r for r in rows}
    episodes = []
    for episode_id in sorted(ids):
        if episode_id not in by_id:
            episodes.append({"episode_id": episode_id, "scientific_status": "missing_attempt", "scientific_reward": None,
                             "receipt_sha256": None, "exclusions": ["missing_attempt"], "costs": None})
            continue
        row = by_id[episode_id]
        score = scores.get(row["receipt_sha256"])
        if row["invalid"]:
            row["scientific_status"] = "execution_invalid"
        elif row["unfinished"]:
            row["scientific_status"] = "unfinished"
        elif row["exclusions"]:
            row["scientific_status"] = "incompatible_attempt"
        elif row["workbench_import"] != "refinement_bound":
            row["scientific_status"] = row["workbench_import"]
        elif score is not None:
            reward = score.reward
            if reward.validity == "valid" and reward.admitted and not reward.training_withheld:
                row["scientific_status"], row["scientific_reward"] = "available", reward.training
            else:
                row["scientific_status"] = "withheld"
            row["reward_gates"] = {"validity": reward.validity, "admitted": reward.admitted,
                                   "training_withheld": list(reward.training_withheld)}
        episodes.append(row)
    observed = [r for r in episodes if r["receipt_sha256"] is not None]
    compatible = [r for r in observed if not r["exclusions"]]
    diagnostics = {}
    for score in scores.values():
        for term in score.reward.terms:
            if term.id in diagnostics:
                _require(diagnostics[term.id]["weight"] == term.weight, "inconsistent diagnostic weights")
            diagnostics[term.id] = {"weight": term.weight}
    for name, term in sorted(diagnostics.items()):
        values = [t.value for r in compatible if r["receipt_sha256"] in scores
                  and scores[r["receipt_sha256"]].reward.validity == "valid"
                  for t in scores[r["receipt_sha256"]].reward.terms if t.id == name and not t.missing]
        term.update(value=_stats(values, len(ids)), weighted_value=_stats([v * term["weight"] for v in values], len(ids)))
    costs = {}
    for name in ("prompt_tokens", "completion_tokens", "wall_seconds", "cost_usd", "action_seconds"):
        values = [r["costs"][name] for r in observed]
        known = sum(v["complete"] for v in values)
        costs[name] = _cost(sum(v["known_subtotal"] for v in values), known == len(ids)) | {
            "known_episodes": known, "missing_episodes": len(ids) - known}
    variants = defaultdict(list)
    for row in compatible:
        variants[_hash(row["runtime"])].append(row)
    runtime_variants = [{"fingerprint": fingerprint, "runtime": members[0]["runtime"],
                         "episode_ids": [r["episode_id"] for r in members],
                         "scientific_reward": _stats([r["scientific_reward"] for r in members if r["scientific_reward"] is not None], len(members))}
                        for fingerprint, members in sorted(variants.items())]
    report = {
        "schema_version": 1, "group_id": manifest["group_id"], "manifest_sha256": manifest_sha,
        "comparison_fingerprint": manifest["comparison_fingerprint"], "reward_spec_version": next(iter(specs), None),
        "counts": {"scheduled": len(ids), "observed": len(observed), "attempted": len(observed),
                   "missing": len(ids) - len(observed), "comparable": len(compatible),
                   "invalid": sum(r["invalid"] for r in observed), "unfinished": sum(r["unfinished"] for r in observed),
                   "excluded": sum(bool(r["exclusions"]) for r in observed), "unscheduled": len(rows) - len(observed),
                   "independent_episodes": sum(r["setup"]["status"] == "valid" for r in observed)},
        "episodes": episodes, "unscheduled": [r for r in rows if r["episode_id"] not in ids],
        "scientific_reward": _stats([r["scientific_reward"] for r in episodes if r["scientific_reward"] is not None], len(ids)),
        "diagnostics": dict(sorted(diagnostics.items())), "costs": costs, "runtime_variants": runtime_variants,
        "unknown_pins": sorted({pin for r in compatible for pin in r["comparison"]["identity"].get("unknown_pins", [])}),
        "limitations": ["Owner-host bindings are not signatures or proof against a malicious operator.",
                        "Overall statistics describe available observations; exclusions and missing rewards can bias them.",
                        "Runtime variants have separate statistics; a mixed overall distribution does not compare one realized policy.",
                        "This report does not qualify rewards, validate scientific judgments, or authorize training."],
    }
    json.dumps(report, allow_nan=False)
    return report
