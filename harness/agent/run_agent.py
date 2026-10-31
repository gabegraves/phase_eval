"""Run an AgentXRD-style agent with any OpenAI-compatible model on one case folder.

Writes a Harbor-shaped trial folder that phaseeval.workbench.importer.read_harbor_run reads:
config.json, result.json, agent/trajectory.json, artifacts/workspace/output/.
The API key is read from the environment and never written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PHYSICS = HERE.parent / "physics"
PYTHON = REPO / "launcher" / ".venv" / "bin" / "python"
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
HARNESS = "agentxrd-openai-runner"
SHOWN_CHARS = 20000  # the tail of a tool's output the model is shown
TEXTURE_ORDERS = (0, 2, 4, 6, 8, 10)
MAX_GRAIN_DIAMETER_UM = 1000.0

TOOLS = [
    {"type": "function", "function": {"name": "case_brief", "description": "Chemistry, measurement, precursors, processing and declared target of the case.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "candidates", "description": "One line per candidate CIF: id, formula, space group, crystal system, cell.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "screen", "description": "Rank candidates against observed peaks (no tree search). With residual=<run_label>, list the largest residual peaks of that saved fit.", "parameters": {"type": "object", "properties": {"residual": {"type": ["string", "null"]}}}}},
    {"type": "function", "function": {"name": "refine", "description": "Staged BGMN refinement (Pawley then Rietveld) of the named candidates. Writes /workspace/output/<run_label>/.", "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "items": {"type": "string"}, "minItems": 1}, "run_label": {"type": "string"}, "pawley_only": {"type": "boolean"}, "texture_order": {"type": "integer", "enum": list(TEXTURE_ORDERS), "description": "Spherical harmonic order of the final stage's texture model. Default 4. 0 means no texture model."}, "refine_zero": {"type": "boolean", "description": "Refine the zero point (EPS1). Default false."}, "refine_transparency": {"type": "boolean", "description": "Refine sample transparency (EPS3). Correlates with EPS1 and EPS2. Default false."}, "grain_diameter_um": {"type": ["number", "null"], "exclusiveMinimum": 0, "maximum": MAX_GRAIN_DIAMETER_UM, "description": "Measured mean grain diameter in micrometres for the microabsorption factor, at most 1000. Default unset."}}, "required": ["candidates", "run_label"]}}},
    {"type": "function", "function": {"name": "write_analysis", "description": "Validate and write the answer to /workspace/output/analysis.json.", "parameters": {"type": "object", "properties": {"analysis": {"type": "object"}}, "required": ["analysis"]}}},
]


def fit_flags(a: dict) -> list[str]:
    """refine.py's flags for the fit options in a refine call. ValueError names a bad one, before
    the call can spend its run label."""
    flags = []
    order = a.get("texture_order")
    if order is not None:
        if type(order) is not int or order not in TEXTURE_ORDERS:
            raise ValueError(f"texture_order must be one of {list(TEXTURE_ORDERS)}")
        flags += ["--texture-order", str(order)]
    for key, flag in (("refine_zero", "--refine-zero"), ("refine_transparency", "--refine-transparency")):
        value = a.get(key)
        if value is not None and type(value) is not bool:
            raise ValueError(f"{key} must be true or false")
        if value:
            flags.append(flag)
    diameter = a.get("grain_diameter_um")
    if diameter is not None:
        if type(diameter) not in (int, float) or not 0 < diameter <= MAX_GRAIN_DIAMETER_UM:
            raise ValueError(f"grain_diameter_um must be a number above 0 and at most {MAX_GRAIN_DIAMETER_UM:g}")
        flags += ["--grain-diameter-um", repr(float(diameter))]
    return flags


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = args.out.resolve()
        self.output = self.out / "artifacts" / "workspace" / "output"
        self.output.mkdir(parents=True, exist_ok=True)
        (self.out / "agent").mkdir(exist_ok=True)
        manifest = json.loads((args.input_dir / "candidates" / args.case / "manifest.json").read_text())
        self.library = {row["candidate_id"] for row in manifest["candidates"]}
        self.labels: set[str] = set()
        self.sid = str(uuid.uuid4())

    def tool(self, script: str, *extra: str, output: bool = False) -> str:
        cmd = [str(PYTHON if PYTHON.exists() else sys.executable), str(PHYSICS / script), "--input-dir", str(self.args.input_dir.resolve()), "--case", self.args.case]
        if output:
            cmd += ["--output-dir", str(self.output)]
        cmd += list(extra)
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        env.pop(self.args.api_key_env, None)
        try:
            p = subprocess.run(cmd, cwd=PHYSICS, env=env, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return "error: tool timed out after 1800 s"
        text = p.stdout + (f"\n[stderr]\n{p.stderr}" if p.stderr.strip() else "") + f"\n[exit {p.returncode}]"
        return self.scrub(text)

    def scrub(self, text: str) -> str:
        """Container paths only: no host paths, base URL or key reach the model or the record."""
        for real, shown in ((str(self.output), "/workspace/output"), (str(self.args.input_dir.resolve()), "/workspace/input"), (str(REPO), "."), (self.args.base_url, "<endpoint>"), (str(Path.home()), "~")):
            text = text.replace(real, shown)
        key = os.environ.get(self.args.api_key_env)
        return text.replace(key, "<key>") if key else text

    def call(self, name: str, a: dict) -> str:
        if name == "case_brief":
            return self.tool("case_brief.py")
        if name == "candidates":
            return self.tool("candidates.py")
        if name == "screen":
            r = a.get("residual")
            if r:
                if not isinstance(r, str) or not LABEL.match(r):
                    return "error: residual must be a run label"
                return self.tool("screen.py", "--residual", r, output=True)
            return self.tool("screen.py", "--no-search", output=True)
        if name == "refine":
            c, label = a.get("candidates"), a.get("run_label")
            if not isinstance(c, list) or not c or not all(isinstance(x, str) for x in c):
                return "error: candidates must be a non-empty list of candidate ids"
            if bad := sorted(set(c) - self.library):
                return f"error: not in the candidate library: {bad}"
            if not isinstance(label, str) or not LABEL.match(label):
                return "error: run_label must match [A-Za-z0-9][A-Za-z0-9_.-]*"
            if label in self.labels or (self.output / label).exists():
                return f"error: run label {label} already used"
            try:
                options = fit_flags(a)
            except ValueError as e:
                return f"error: {e}"
            self.labels.add(label)
            extra = ["--candidates", *c, "--run-label", label] + (["--pawley-only"] if a.get("pawley_only") else []) + options
            out = self.tool("refine.py", *extra, output=True)
            for f in (self.output / label).rglob("*.json"):
                f.write_text(self.scrub(f.read_text()))  # refine.py records host paths; keep container paths
            return out
        if name == "write_analysis":
            return self.write_analysis(a.get("analysis"))
        return f"error: unknown tool {name}"

    def write_analysis(self, x: object) -> str:
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except ValueError as e:
                return f"error: analysis is not JSON: {e}"
        if not isinstance(x, dict):
            return "error: analysis must be an object"
        if x.get("case_id") != self.args.case:
            return f"error: case_id must be {self.args.case}"
        phases = x.get("phases")
        if not isinstance(phases, list) or not all(isinstance(p, dict) and isinstance(p.get("candidate_id"), str) for p in phases):
            return "error: phases must be a list of objects with candidate_id"
        wf = x.get("weight_fractions") or {}
        if not isinstance(wf, dict):
            return "error: weight_fractions must be an object"
        ids = {p["candidate_id"] for p in phases} | set(wf)
        if bad := sorted(ids - self.library):
            return f"error: not in the candidate library: {bad}"
        for k, v in wf.items():
            if v is not None and (not isinstance(v, (int, float)) or not 0 <= v <= 1):
                return f"error: weight fraction for {k} must be within 0..1"
        fa = x.get("fit_artifacts") or []
        if not isinstance(fa, list) or not all(isinstance(p, str) for p in fa):
            return "error: fit_artifacts must be a list of paths"
        (self.output / "analysis.json").write_text(json.dumps(x, indent=2))
        return "wrote /workspace/output/analysis.json"

    def chat(self, messages: list) -> dict:
        body = json.dumps({"model": self.args.model, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.args.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.args.base_url.rstrip("/") + "/chat/completions", body, headers)
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def run(self) -> int:
        a = self.args
        system = (PHYSICS / "README.md").read_text() + "\n\n" + (HERE / "task.md").read_text().replace("{case_id}", a.case)
        task_sha = hashlib.sha256(system.encode()).hexdigest()
        messages: list = [{"role": "system", "content": system}, {"role": "user", "content": f"Solve case {a.case}."}]
        steps: list = [{"step_id": 1, "source": "system", "message": system}, {"step_id": 2, "source": "user", "message": messages[1]["content"]}]
        started, t0 = now(), time.monotonic()
        exception, turns, usage = None, 0, {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            while True:
                if turns >= a.max_turns or time.monotonic() - t0 > a.time_limit:
                    exception = {"exception_type": "AgentTimeoutError", "exception_message": f"stopped after {turns} turns, {time.monotonic() - t0:.0f} s"}
                    break
                turns += 1
                resp = self.chat(messages)
                for k in usage:
                    usage[k] += (resp.get("usage") or {}).get(k) or 0
                msg = resp["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None} | {"role": "assistant"})
                step = {"step_id": len(steps) + 1, "source": "agent", "timestamp": now(), "message": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content"), "tool_calls": [], "observation": {"results": []}}
                for c in calls:
                    fn = c["function"]
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {"_unparsed": fn.get("arguments")}
                    full = self.call(fn["name"], args) if isinstance(args, dict) and "_unparsed" not in args else "error: arguments are not a JSON object"
                    result = observation(self.out, c["id"], full)
                    step["tool_calls"].append({"tool_call_id": c["id"], "function_name": fn["name"], "arguments": args})
                    step["observation"]["results"].append(result)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": result["content"]})
                steps.append(step)
                self.save_trajectory(steps)
                if not calls:
                    break
        except urllib.error.HTTPError as e:
            kind = "ApiRateLimitError" if e.code == 429 else "ProviderError"
            exception = {"exception_type": kind, "exception_message": self.scrub(f"HTTP {e.code}: {e.read()[:300]!r}")}
        except Exception as e:
            exception = {"exception_type": type(e).__name__, "exception_message": self.scrub(str(e))[:500]}
        finished = now()
        self.save_trajectory(steps)
        trial = trial_name(a.out, self.sid)
        label = a.endpoint_label
        (self.out / "config.json").write_text(json.dumps({
            "trials_dir": self.out.parent.name, "task": {"case_id": a.case, "input_dir": "/workspace/input"},
            "agent": {"name": HARNESS, "model_name": a.model, "endpoint_label": label, "override_timeout_sec": a.time_limit, "max_turns": a.max_turns},
            "tool_versions": {"harness_git_commit": git_commit()},
        }, indent=2))
        (self.out / "result.json").write_text(json.dumps({
            "trial_name": trial, "task_name": a.case, "task_checksum": task_sha,
            "agent_info": {"name": HARNESS, "version": git_commit(), "model_info": {"name": a.model, "provider": label}},
            "agent_result": {"n_input_tokens": usage["prompt_tokens"], "n_output_tokens": usage["completion_tokens"], "cost_usd": None},
            "agent_execution": {"started_at": started, "finished_at": finished},
            "exception_info": exception, "turns": turns, "wall_seconds": round(time.monotonic() - t0, 1),
            "status": "error" if exception else ("submitted" if (self.output / "analysis.json").is_file() else "no_submission"),
        }, indent=2))
        print(json.dumps({"out": a.out.name, "turns": turns, "exception": exception, "answer": (self.output / "analysis.json").is_file()}))
        return 0 if exception is None else 1

    def save_trajectory(self, steps: list) -> None:
        (self.out / "agent" / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.2", "session_id": self.sid, "agent": {"name": HARNESS, "model_name": self.args.model}, "steps": steps}, indent=2))


def trial_name(out: Path, session_id: str) -> str:
    """The output folder's name and the run's session id: unique per run, so runs written to
    folders of one name do not share the trial name an import is keyed on."""
    return f"{out.resolve().name}-{session_id[:8]}"


def observation(out: Path, call_id: str, full: str) -> dict:
    """The trajectory's record of one call's output: the tail the model is shown, and, when that
    is a truncation, the executor's full output kept at agent/tool-output/<its sha256>.txt (by
    content, so a server that reuses call ids cannot overwrite an earlier call's output)."""
    result = {"source_call_id": call_id, "content": full[-SHOWN_CHARS:]}
    if len(full) > SHOWN_CHARS:
        sha = hashlib.sha256(full.encode()).hexdigest()
        name = f"agent/tool-output/{sha}.txt"
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(full)
        result["extra"] = {
            "executor_output": name,
            "executor_sha256": sha,
            "executor_chars": len(full),
            "shown_chars": SHOWN_CHARS,
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint-label", default=None, help="optional name for the endpoint, recorded instead of the URL")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--time-limit", type=int, default=3440)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error(f"{a.out} is not empty")
    return Runner(a).run()


if __name__ == "__main__":
    sys.exit(main())
