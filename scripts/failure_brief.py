"""Failure brief: how one agent run failed, step by step, from the records alone.

    PYTHONPATH=src uv run --project launcher python scripts/failure_brief.py EXPORT_DIR \
        [--run ID] [--out brief.md] [--json]

EXPORT_DIR is a workbench export (`python -m phaseeval.workbench.cli export`) or the public
dataset directory. Step outcomes come from the scorecard (`scorecard_view`), never recomputed
here. Evidence is copied from snapshots.jsonl, judge_inputs.jsonl, labels.*.jsonl and
step_feedback.jsonl. A field the records lack is listed in the footer, never filled in.
The published-reference source is off: this script reads only the export.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phaseeval.workbench.scorecard import (  # noqa: E402
    DEFINITION,
    FAILURE_CODES,
    SOURCE_TEXT,
    scorecard_view,
)

SOURCE_NAME = {
    "deterministic_check": "deterministic check",
    "model_judge": "model judge (can only fail a step)",
    "expert_label": "human label",
    "reference_match": "published reference",
}
# The step's trace anchor is a harness tool call, or the model's final answer.
ANSWER_STEPS = {"quantification", "final_answer"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _norm(cid: str) -> str:
    return cid.replace("_", "").lower()


def _k5_evidence(snap: dict[str, Any], judge: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    answer = (snap.get("reported_answer") or {}).get("weight_fractions") or {}
    ev: dict[str, Any] = {
        "answer_weight_percent": {k: round(v * 100, 2) for k, v in answer.items()},
    }
    fit = (judge or {}).get("fit") or {}
    detail = fit.get("phase_detail")
    if not detail:
        missing.append("cited fit phase amounts (judge_inputs fit.phase_detail)")
        return ev, missing
    fit_wt = {p["candidate_id"]: p.get("weight_percent") for p in detail}
    ev["cited_fit"] = fit.get("run_label")
    ev["cited_fit_stage"] = fit.get("stage")
    ev["cited_fit_weight_percent"] = fit_wt
    exact = [k for k in answer if k in fit_wt]
    by_norm = {_norm(k): k for k in fit_wt}
    renamed = {k: by_norm[_norm(k)] for k in answer if k not in fit_wt and _norm(k) in by_norm}
    ev["answer_ids_in_fit"] = exact
    ev["answer_ids_matching_fit_without_underscore"] = renamed
    if renamed:
        ev["difference_wt_points_after_id_match"] = {
            k: round(ev["answer_weight_percent"][k] - (fit_wt[v] or 0.0), 2)
            for k, v in renamed.items()
        }
    return ev, missing


def _evidence(step: Any, snap: dict[str, Any] | None, judge: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Fields behind a failed step, taken only from the records."""
    if step.check is None:
        return {}, []
    ev: dict[str, Any] = {"check": step.check, "detail": step.detail}
    missing: list[str] = []
    if snap is None:
        return ev, ["frozen answer snapshot"]
    full = next((c for c in snap.get("checks") or [] if c.get("code") == step.check), None)
    if full:  # the scorecard truncates check text; the snapshot holds it whole
        ev["detail"] = f"{full['status']} ({full['policy_version']}): {full['detail']}"
    if step.check == "K2":
        ev["residual_peaks"] = [
            {
                "two_theta_deg": p.get("two_theta_deg"),
                "height_percent_of_strongest_net_peak": p.get("height_percent"),
                "residual_counts": p.get("residual_counts"),
            }
            for p in snap.get("residual_peaks") or []
        ]
    elif step.check == "K1":
        ev["rwp_percent"] = snap.get("rwp_percent")
        ev["stage"] = snap.get("stage")
    elif step.check == "K5":
        k5, missing = _k5_evidence(snap, judge)
        ev |= k5
        ev["rwp_percent"] = snap.get("rwp_percent")
    elif step.check == "submitted":
        ev["answered"] = snap.get("answered")
    return ev, missing


def _where(step: Any, ev: dict[str, Any], trace: dict[int, Any]) -> str:
    call = step.call_index
    row = trace.get(call) if call is not None else None
    if row is None and step.step in ANSWER_STEPS and trace:
        row = trace[max(trace)]  # no answer-marked call: the run's last recorded call
    at = f"call {row.index} ({row.tool})" if row else "no recorded call"
    target = "the model's final answer" if step.step in ANSWER_STEPS else "the harness tool"
    head = f"{DEFINITION[step.step].title} ({step.check}, {step.failure_code}): "
    line = f"{head}{target}, {at}."
    renamed = ev.get("answer_ids_matching_fit_without_underscore") or {}
    if step.check == "K5" and renamed and not ev.get("answer_ids_in_fit"):
        refine = [r for r in trace.values() if "refine" in r.marks]
        cands = None
        if refine:
            try:
                cands = json.loads(refine[-1].summary).get("candidates")
            except (ValueError, AttributeError):
                cands = None
        diffs = ev.get("difference_wt_points_after_id_match") or {}
        same = bool(diffs) and all(abs(d) <= 0.01 for d in diffs.values())
        if cands and set(cands) == set(renamed) and same:
            # Amounts agree and the ids changed spelling between the refine call and its fit.
            line = (
                f"{head}the harness tool, call {refine[-1].index} (refine). It was passed"
                f" {', '.join(cands)}; its fit {ev.get('cited_fit')} reports"
                f" {', '.join(renamed.values())}. The final answer uses the ids passed to refine,"
                " and its amounts equal the fit's within 0.01 wt% points once the underscore is dropped."
            )
        else:
            line += (
                f" The answer ids ({', '.join(renamed)}) are not in the cited fit {ev.get('cited_fit')},"
                f" which lists {', '.join(renamed.values())}."
            )
    elif step.check == "K2" and ev.get("residual_peaks"):
        p = ev["residual_peaks"][0]
        line += (
            f" Unexplained residual peak at {p['two_theta_deg']} deg,"
            f" {p['height_percent_of_strongest_net_peak']}% of the strongest net peak."
        )
    return line


def build(export: Path, run: str | None = None) -> list[dict[str, Any]]:
    view = scorecard_view(export, reference=lambda _case: None)
    snaps = {s["origin"]: s for s in _jsonl(export / "snapshots.jsonl")}
    judges = {j["snapshot"].get("origin"): j for j in _jsonl(export / "judge_inputs.jsonl")}
    labels = _jsonl(export / "labels.expert.jsonl") + _jsonl(export / "labels.model_judge.jsonl")
    feedback = _jsonl(export / "step_feedback.jsonl")
    title = {c.code: c.title for c in FAILURE_CODES}
    briefs = []
    for e in view.episodes:
        if run and not e.episode.startswith(run):
            continue
        snap, judge = snaps.get(e.episode), judges.get(e.episode)
        trace = {r.index: r for r in view.traces.get(e.episode, ())}
        footer: list[str] = []
        steps = []
        for s in e.steps:
            ev, missing = _evidence(s, snap, judge) if s.outcome == "fail" else ({}, [])
            footer += [f"{s.step}: {m}" for m in missing]
            steps.append(
                {
                    "step": s.step,
                    "title": DEFINITION[s.step].title,
                    "question": DEFINITION[s.step].question,
                    "outcome": "unassessed" if s.outcome == "unknown" else s.outcome,
                    "source": SOURCE_NAME.get(s.source) if s.source else None,
                    "check": s.check,
                    "failure_code": s.failure_code,
                    "failure_title": title.get(s.failure_code) if s.failure_code else None,
                    "call_index": s.call_index,
                    "detail": s.detail,
                    "readings": [
                        {"source": SOURCE_TEXT[r.source], "check": r.check, "outcome": r.outcome}
                        for r in s.readings
                    ],
                    "evidence": ev,
                    "where_to_look": _where(s, ev, trace) if s.outcome == "fail" else None,
                }
            )
        critiques = [
            {
                "kind": lb["rater"].get("kind"),
                "item": lb.get("item"),
                "answer": lb.get("answer"),
                "text": " ".join(t for t in (lb.get("note"), lb.get("reason")) if t),
            }
            for lb in labels
            if e.snapshot and lb.get("snapshot") == e.snapshot and (lb.get("note") or lb.get("reason"))
        ] + [
            {
                "kind": f["rater"].get("kind"),
                "item": f"step feedback, call {f['step'].get('call_index')}",
                "answer": f.get("mark"),
                "text": f.get("note", ""),
            }
            for f in feedback
            if f["step"].get("attempt") == e.episode and f.get("note")
        ]
        if not critiques:
            footer.append(
                "reviewer critique: no label or step-feedback text for this run in labels.*.jsonl or step_feedback.jsonl"
            )
        footer.append("published reference: not read (export only), so no reference-match source")
        if snap is None:
            footer.append("frozen answer snapshot: absent")
        if judge is None:
            footer.append("judge input (cited fit curves and amounts): absent")
        briefs.append(
            {
                "schema": "failure-brief.v0",
                "run": e.episode,
                "case_id": e.case_id,
                "model": e.model,
                "harness": e.harness,
                "end": e.end,
                "tool_calls": e.tool_calls,
                "rwp_percent": e.rwp_percent,
                "cited_fit": ((judge or {}).get("fit") or {}).get("run_label"),
                "steps": steps,
                "critiques": critiques,
                "where_to_look": [s["where_to_look"] for s in steps if s["where_to_look"]],
                "not_in_records": footer,
            }
        )
    return briefs


def render(b: dict[str, Any]) -> str:
    out = [
        f"# Failure brief: {b['case_id']}, {b['model']}",
        "",
        f"Run `{b['run'][:12]}` · harness {b['harness']} · end: {b['end']} · {b['tool_calls']} tool calls"
        + (f" · cited fit {b['cited_fit']}" if b["cited_fit"] else "")
        + (f" · Rwp {b['rwp_percent']}%" if b["rwp_percent"] is not None else ""),
        "",
        "## Steps",
        "",
        "| Step | Outcome | Source | Check | Code | Call |",
        "|---|---|---|---|---|---|",
    ]
    for s in b["steps"]:
        out.append(
            f"| {s['title']} | {s['outcome'].upper() if s['outcome'] == 'fail' else s['outcome']} |"
            f" {s['source'] or '-'} | {s['check'] or '-'} | {s['failure_code'] or '-'} |"
            f" {'-' if s['call_index'] is None else s['call_index']} |"
        )
    failed = [s for s in b["steps"] if s["outcome"] == "fail"]
    out += ["", "## Failures", ""]
    if not failed:
        out += ["No step failed. Steps without a pass or fail reading are unassessed.", ""]
    for s in failed:
        out += [f"### {s['title']}: {s['failure_code']} {s['failure_title'] or ''}".rstrip(), ""]
        out.append(f"- Decided by: {s['source']}, {s['check']}")
        ev = s["evidence"]
        for p in ev.get("residual_peaks", []):
            out.append(
                f"- Residual peak at {p['two_theta_deg']} deg: {p['height_percent_of_strongest_net_peak']}%"
                f" of the strongest net peak ({p['residual_counts']} counts)"
            )
        if "rwp_percent" in ev:
            out.append(f"- Rwp: {ev['rwp_percent']}%")
        if "answer_weight_percent" in ev:
            out.append(
                "- Answer (wt%): "
                + ", ".join(f"{k} {v}" for k, v in ev["answer_weight_percent"].items())
            )
        if "cited_fit_weight_percent" in ev:
            out.append(
                f"- Cited fit {ev['cited_fit']} ({ev['cited_fit_stage']}, wt%): "
                + ", ".join(f"{k} {v}" for k, v in ev["cited_fit_weight_percent"].items())
            )
            out.append(f"- Answer ids found in the fit: {len(ev['answer_ids_in_fit'])}")
        out.append(f"- Check text: {ev.get('detail', s['detail'])}")
        out.append("")
    out += ["## Reviewer critique", ""]
    if b["critiques"]:
        out += [f"- {c['kind']}, {c['item']} = {c['answer']}: {c['text']}" for c in b["critiques"]]
    else:
        out.append("None recorded.")
    out += ["", "## Where to look", ""]
    out += [f"- {w}" for w in b["where_to_look"]] or ["- No failed step."]
    out += ["", "---", "", "Not in the records, left out: " + "; ".join(b["not_in_records"]) + "."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", type=Path)
    ap.add_argument("--run", help="episode digest or prefix")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = ap.parse_args(argv)
    briefs = build(args.export, args.run)
    if not briefs:
        print(f"no agent run matched in {args.export}", file=sys.stderr)
        return 1
    text = (
        json.dumps(briefs, indent=2) + "\n"
        if args.json
        else "\n".join(render(b) for b in briefs)
    )
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
