"""Method review page chronology, validation, and retry behavior."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def run_js(tmp_path: Path, expression: str) -> object:
    harness = tmp_path / "method.js"
    harness.write_text(
        """
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const ctx = vm.createContext({console, URL, URLSearchParams, setTimeout, clearTimeout});
vm.runInContext(src.slice(0, src.indexOf('\\n$("auth").addEventListener')), ctx);
Promise.resolve(vm.runInContext(process.argv[3], ctx)).then(x => process.stdout.write(JSON.stringify(x)));
"""
    )
    result = subprocess.run(
        ["node", str(harness), str(SCRIPT), expression],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


TRACE = json.dumps(
    [
        {"source": "task", "message": "Initial task", "calls": []},
        {
            "source": "agent",
            "message": "Try two checks",
            "calls": [
                {"call_index": 7, "tool": "exec", "command": "first", "output": "first result"},
                {"call_index": 8, "tool": "exec", "command": "second", "output": "sibling result"},
            ],
        },
        {"source": "context", "message": "Late context", "calls": []},
        {"source": "agent", "message": "Final prose with no calls", "calls": []},
    ]
)


@needs_node
def test_events_keep_late_context_and_final_prose_in_recorded_order(tmp_path: Path) -> None:
    events = run_js(tmp_path, f"methodEvents({TRACE}).map(x => [refKey(x.ref), x.kind, x.text])")
    assert events == [
        ["0:message", "message", "Initial task"],
        ["1:message", "message", "Try two checks"],
        ["1:7", "call", "first"],
        ["1:8", "call", "second"],
        ["2:message", "message", "Late context"],
        ["3:message", "message", "Final prose with no calls"],
    ]


@needs_node
def test_selected_call_prefix_contains_only_earlier_turns_not_sibling_results(tmp_path: Path) -> None:
    html = run_js(
        tmp_path,
        f"(() => {{ const t={TRACE}, e=methodEvents(t).find(x => refKey(x.ref)==='1:7'); return methodPrefix(t,e); }})()",
    )
    assert "Initial task" in html
    assert "first result" not in html
    assert "sibling result" not in html
    assert "Late context" not in html
    assert "Final prose" not in html


@needs_node
def test_validation_separates_evidence_error_and_actual_repair(tmp_path: Path) -> None:
    errors = run_js(
        tmp_path,
        """(() => {
          const M={draft:blankMethodDraft()};
          M.draft.verdict='fail'; M.draft.note='wrong turn';
          const a=methodDraftError(M);
          M.draft.earliest_error={turn_index:1,call_index:7};
          const b=methodDraftError(M);
          M.draft.evidence=[{turn_index:0,call_index:null}]; M.draft.repair_status='repaired';
          const c=methodDraftError(M);
          M.draft.observed_repair={turn_index:3,call_index:null};
          return [a,b,c,methodDraftError(M)];
        })()""",
    )
    assert errors == [
        "Fail needs an earliest error.",
        "Fail needs evidence.",
        "Repaired requires an observed later repair, and other statuses cannot cite one.",
        "",
    ]


@needs_node
def test_pass_requires_linked_evidence(tmp_path: Path) -> None:
    errors = run_js(
        tmp_path,
        """(() => { const M={draft:blankMethodDraft()}; M.draft.verdict='pass'; M.draft.note='supported'; const a=methodDraftError(M); M.draft.evidence=[{turn_index:0,call_index:null}]; return [a,methodDraftError(M)]; })()""",
    )
    assert errors == ["Pass needs evidence.", ""]


@needs_node
def test_failed_save_retry_reuses_submission_and_saved_edit_supersedes_it(tmp_path: Path) -> None:
    result = run_js(
        tmp_path,
        """(() => {
          const M={draft:blankMethodDraft(),saved:null,submissionId:null};
          M.draft.verdict='defer'; M.draft.note='trace missing';
          const first=methodRequest(M,'d');
          const retry=methodRequest(M,'d');
          M.saved={submission_id:first.submission_id}; M.submissionId=null; M.draft.note='new note';
          const correction=methodRequest(M,'d');
          return [first.submission_id,retry.submission_id,correction.submission_id,correction.supersedes];
        })()""",
    )
    assert result[0] == result[1]
    assert result[2] != result[0]
    assert result[3] == result[0]


@needs_node
def test_select_events_keep_refs_and_restore_repair_status_enum(tmp_path: Path) -> None:
    result = run_js(
        tmp_path,
        f"""(() => {{
          const M={{data:{{trace:{TRACE}}},draft:blankMethodDraft()}};
          updateMethodSelect(M,'earliest_error','1:7');
          updateMethodSelect(M,'repair_status','repaired');
          updateMethodSelect(M,'observed_repair','3:message');
          return [M.draft.earliest_error,M.draft.observed_repair,M.draft.repair_status,
            methodControlValue('earliest_error',M.draft.earliest_error),
            methodControlValue('observed_repair',M.draft.observed_repair),
            methodControlValue('repair_status',M.draft.repair_status)];
        }})()""",
    )
    assert result == [
        {"turn_index": 1, "call_index": 7},
        {"turn_index": 3, "call_index": None},
        "repaired",
        "1:7",
        "3:message",
        "repaired",
    ]


@needs_node
def test_method_first_visit_restores_outcome_once(tmp_path: Path) -> None:
    result = run_js(
        tmp_path,
        """(async () => {
          S.queue=[{digest:'d'}]; S.mode='method';
          const restored=[];
          status=()=>{}; render=()=>view();
          window={scrollTo:()=>{}}; sessionStorage={getItem:()=>null};
          call=async(name)=> {
            if(name==='trajectoryReview') return {trace:[],reviews:[]};
            if(name==='review') return {digest:'d',steps:[]};
            if(name==='labels') return [{item:'scientific',answer:'yes'}];
            return [];
          };
          restoreRefits=async()=>restored.push('refits');
          loadComposition=async()=>restored.push('composition');
          loadReveal=async()=>restored.push('reveal');
          await openSnapshot(0);
          const existed=Boolean(S.views.d);
          S.mode='outcome'; await openSnapshot(0); await openSnapshot(0);
          return [existed,S.views.d.labels['L|scientific|'].answer,restored,S.views.d.outcomeLoaded];
        })()""",
    )
    assert result == [True, "yes", ["refits", "composition", "reveal"], True]
