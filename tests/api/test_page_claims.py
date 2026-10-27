"""The browser checks legacy claims without letting their length stall the page."""

import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"


@pytest.mark.parametrize("text", ["no other " * 12000, "1" * 100000, "Rwp 3.50% " + " " * 2000],
                         ids=["repeated-words", "digits", "matching-prefix"])
def test_a_long_claim_is_marked_in_linear_time(text):
    run = subprocess.run(["node", "-e", r'''
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[1], "utf8");
const ctx = vm.createContext({ text: JSON.parse(fs.readFileSync(0, "utf8")) });
vm.runInContext(src.slice(0, src.indexOf('\n$("auth").addEventListener')), ctx);
process.stdout.write(JSON.stringify(vm.runInContext(`(() => {
  const e = { digest: "snapshot", answer: { claims: [text], rwp: 3.5 } };
  S.ep = e; view().unlocked = true; S.fullClaims = true;
  return { mark: claimMark({ text }, e), html: writeUp(e) };
})()`, ctx, { timeout: 1000 })));
''', str(SCRIPT)], input=json.dumps(text), capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
    shown = json.loads(run.stdout)
    assert shown["mark"][:2] == ["none", "Not checked"]
    assert "long" in shown["mark"][2].lower()
    assert text in shown["html"]


def test_claims_at_the_limit_keep_their_existing_checks():
    run = subprocess.run(["node", "-e", r'''
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[1], "utf8");
const ctx = vm.createContext({});
vm.runInContext(src.slice(0, src.indexOf('\n$("auth").addEventListener')), ctx);
process.stdout.write(JSON.stringify(vm.runInContext(`[
  claimMark({ text: "Rwp 3.50%".padEnd(2000) }, { answer: { rwp: 3.5 } }),
  claimMark({ text: "No additional unexplained peaks." }, {}),
  claimMark({ text: "Peaks at 23.9–24.0 deg belong to the fitted phase." },
    { residual_peaks: [{ two_theta_deg: 23.95, height_percent: 6 }] }),
  claimMark({ text: "CIF000001 was never tested." },
    { steps: [{ kind: "refine", phases: ["CIF000001"] }] }),
]`, ctx, { timeout: 1000 })));
''', str(SCRIPT)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
    assert [mark[1] for mark in json.loads(run.stdout)] == [
        "Rwp = cited fit", "Not checked", "Disputed by a check", "Contradicted",
    ]
