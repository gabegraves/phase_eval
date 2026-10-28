"""Replay saved evidence through the runner; scripted plumbing proof, never a learning result."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from harness.agent.run_agent import Runner, comparison_fingerprint, git_commit, now
from phaseeval.reward_profile import profile_group
from phaseeval.workbench import cli, importer
from phaseeval.workbench.evidence import Curve
from phaseeval.workbench.importers.external import parse_xy


def read(runner, handle, **kwargs):
    result = runner.call("read_evidence", {"handle": handle, **kwargs})
    assert not result.startswith("error:"), result
    return json.loads(result)


def reconstruct(runner, handle, expected, original_bytes):
    values, indexes, start, pages = [], [], 0, 0
    while True:
        page = read(runner, handle, start=start)
        assert page["sha256"] == hashlib.sha256(original_bytes).hexdigest()
        values.extend(page.get("rows", page.get("text", "")))
        indexes.extend(page.get("row_indexes", []))
        pages += 1
        if page["next_start"] is None:
            break
        assert page["next_start"] > start
        start = page["next_start"]
    if "text" in page:
        assert "".join(values) == expected
        assert "".join(values).encode() == original_bytes
    else:
        assert values == expected
        assert indexes == list(range(len(expected)))
    return {"sha256": page["sha256"], "pages": pages,
            "native_rows" if "rows" in page else "characters": len(expected), "exact": True}


def response(*calls):
    return {"choices": [{"message": {"role": "assistant", "content": "SCRIPTED replay",
            "tool_calls": [{"id": f"call-{i}", "function": {"name": name, "arguments": json.dumps(args)}}
                           for i, (name, args) in enumerate(calls)]}}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--saved-fit", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", type=Path, required=True, help="new proof directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    common = dict(input_dir=args.input_dir.resolve(), case=args.case, base_url="https://scripted.invalid/v1",
                  model="SCRIPTED-no-model", endpoint_label="SCRIPTED-saved-fit-replay",
                  api_key_env="PHASEEVAL_UNUSED_PROOF_KEY", max_turns=4, time_limit=600,
                  decoding={}, model_revision=None, out=args.out / "a")
    fingerprint = comparison_fingerprint(argparse.Namespace(**common))
    manifest_path = args.out / "group.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "group_id": "scripted-proof",
        "created_at": now(), "comparison_fingerprint": fingerprint, "episode_ids": ["a", "b", "c", "d"]}, indent=2) + "\n")

    # ponytail: this proof replaces only provider/solver execution. Commission real sampling separately.
    def runner(slot, inputs=args.input_dir.resolve()):
        return Runner(argparse.Namespace(**(common | {"input_dir": inputs, "out": args.out / slot,
                      "group_manifest": manifest_path, "episode_id": slot})))

    a = runner("a")
    request = json.loads((args.saved_fit / "request.json").read_bytes())
    candidates = [row["candidate_id"] for row in request["candidates"]]
    replies = iter([response(("read_evidence", {"handle": "scan"}),
                            ("refine", {"candidates": candidates, "run_label": "saved"}),
                            ("read_evidence", {"handle": "fit:saved:rietveld_profile"}))])

    def interrupted_provider(_messages):
        try:
            return next(replies)
        except StopIteration:
            raise ConnectionError("SCRIPTED provider interruption after saved-fit replay") from None

    def replay_fit(name, *_args, **_kwargs):
        assert name == "refine.py"
        target = a.output / "saved"
        target.mkdir()
        for name in ("request.json", "result.json", "pawley.csv", "rietveld_lattice.csv", "rietveld_profile.csv", "rietveld.csv"):
            if (args.saved_fit / name).is_file():
                shutil.copyfile(args.saved_fit / name, target / name)
        return "SCRIPTED replay of saved fit; no solver invoked"

    a.chat, a.tool = interrupted_provider, replay_fit
    assert a.run() == 1
    assert a.receipt["ending"]["kind"] == "provider_failure"
    trace = json.loads((a.out / "agent/trajectory.json").read_bytes())
    assert len(trace["steps"][2]["observation"]["results"]) == 3
    package = cli.harness_case_package(args.input_dir, args.case, args.out / "case-package")
    labels, custody = cli.user_case_records(package, args.case, args.out)
    case, _ = importer.read_case_package(package, labels, custody=custody)
    action, imported, _ = importer.read_harbor_run(a.out, case)
    assert imported.end.end == "invalid" and imported.end.cause == "provider_error"
    assert action.harbor.trial_id == a.receipt["trial_id"]

    b = runner("b")
    replies_b = iter([response(("write_analysis", {"analysis": {"case_id": args.case, "phases": [],
        "weight_fractions": {}, "fit_artifacts": [], "reasoning": "SCRIPTED unresolved answer; no scientific judgment."}})), response()])
    b.chat = lambda _messages: next(replies_b)
    assert b.run() == 0 and b.receipt["ending"]["kind"] == "submitted"
    assert b.receipt["workbench_import"] == "unsupported_no_refinement"
    broken = args.out / "missing-scan-input"
    shutil.copytree(args.input_dir, broken)
    (broken / a.evidence.case["pattern_file"]).unlink()
    c = runner("c", broken)
    assert c.run() == 1 and c.receipt["ending"]["kind"] == "setup_failure"

    scan_bytes = (args.input_dir / a.evidence.case["pattern_file"]).read_bytes()
    scan_rows = parse_xy(scan_bytes).tolist()
    curve_bytes = (args.saved_fit / "rietveld_profile.csv").read_bytes()
    curve = Curve.parse_csv(curve_bytes)
    columns = [curve.two_theta, curve.observed, curve.calculated, curve.difference]
    if curve.background is not None:
        columns.append(curve.background)
    evidence = {"scan": reconstruct(a, "scan", scan_rows, scan_bytes),
                "saved_rietveld_profile": reconstruct(a, "fit:saved:rietveld_profile", list(map(list, zip(*columns))), curve_bytes)}
    for row in a.evidence.candidates:
        raw = (args.input_dir / "candidates" / args.case / row["file"]).read_bytes()
        evidence["candidate:" + row["candidate_id"]] = reconstruct(a, "candidate:" + row["candidate_id"], raw.decode(), raw)
    # Context has a safe projection, not the bytes of a custody/source document.
    evidence["context"] = reconstruct(a, "context", a.evidence.context.decode(), a.evidence.context)
    context = json.loads(a.evidence.context)
    assert context["calibration"] is None
    midpoint = len(scan_rows) // 2
    lower, upper = scan_rows[midpoint - 1][0], scan_rows[midpoint + 1][0]
    region = read(a, "scan", two_theta_min=lower, two_theta_max=upper)
    assert region["rows"] == scan_rows[midpoint - 1:midpoint + 2]
    assert region["row_indexes"] == list(range(midpoint - 1, midpoint + 2))
    forbidden = ["../case_labels.json", "/etc/passwd", "custody.json", "fit:sibling:rietveld"]
    assert all(a.call("read_evidence", {"handle": handle}).startswith("error:") for handle in forbidden)

    receipts = [r.out / "runner_receipt.json" for r in (a, b, c)]
    profile = profile_group(manifest_path, receipts)
    assert profile["counts"] == {"scheduled": 4, "observed": 3, "attempted": 3, "missing": 1,
        "comparable": 1, "invalid": 2, "unfinished": 0, "excluded": 2, "unscheduled": 0, "independent_episodes": 2}
    assert profile["scientific_reward"] == {"known": 0, "missing": 4, "mean": None, "population_std": None}
    (args.out / "profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    proof = {"schema_version": 1, "recorded_at": now(), "case_id": args.case,
        "proof_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "git_head": git_commit(),
        "comparison_fingerprint": fingerprint, "mode": "SCRIPTED provider and SAVED fit replay; no model, solver, or optimizer invoked",
        "evidence": evidence, "targeted_region": {"native_row_indexes": region["row_indexes"], "exact": True},
        "forbidden_handles_rejected": len(forbidden), "calibration": context["calibration"],
        "persisted_results_before_provider_failure": 3, "imported_end": imported.end.model_dump(mode="json"),
        "counts": profile["counts"], "scientific_reward": profile["scientific_reward"],
        "endings": {r.receipt["group"]["episode_id"]: r.receipt["ending"]["kind"] for r in (a, b, c)},
        "no_refinement_workbench_import": b.receipt["workbench_import"], "unknown_pins": profile["unknown_pins"],
        "limitations": ["Development case; not held-out evaluation.", "Direct caller reconstruction is not learned policy behavior.",
            "Context check verifies projected-byte preservation, not scientific completeness.",
            "Zero qualified rewards; no evaluator accuracy or scientific learning measured."]}
    (args.out / "proof.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"proof": str(args.out / "proof.json"), "counts": profile["counts"], "scientific_reward": profile["scientific_reward"]}))


if __name__ == "__main__":
    main()
