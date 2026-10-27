"""Exercise the review page and its report with hostile response and local state text."""

import json
import subprocess
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/phaseeval/api/assets/workbench.js"
ATTACK = '\"><img src=x onerror=alert(1)><svg onload=alert(2)>&\''


class Markup(HTMLParser):
    def handle_starttag(self, tag, attrs):
        assert tag != "img", "response text became an HTML element"
        assert not any(name.startswith("on") for name, _ in attrs), attrs


@pytest.mark.parametrize("sink", ["screen", "refit", "plot", "text"])
def test_adversarial_values_stay_literal_in_the_page_and_report(sink):
    run = subprocess.run(["node", "-e", r'''
const fs = require("node:fs"), vm = require("node:vm");
const src = fs.readFileSync(process.argv[1], "utf8");
const { sink, attack } = JSON.parse(fs.readFileSync(0, "utf8"));
const nodes = {};
const ctx = vm.createContext({ sink, attack, URL, CSS: { escape: String }, location: { href: "http://localhost/workbench" },
  history: { replaceState() {} }, document: {
    getElementById(id) { return nodes[id] ||= {}; },
    querySelectorAll() { return []; }, querySelector() { return null; },
  },
});
vm.runInContext(src.slice(0, src.indexOf('\n$("auth").addEventListener')), ctx);
const report = vm.runInContext(`(() => {
  const e = { digest: "snapshot", origin: "attempt", case_id: "case", answer: {
    phases: [{ candidate_id: "p1", formula: "NiO" }], claims: ["An ordinary claim."] },
    fit: { two_theta: [10, 11], observed: [1, 2], calculated: [1, 2], difference: [0, 0] } };
  S.ep = e; S.queue = [e]; const V = view(); V.unlocked = true;
  V.signoff = { verdict: "unresolved" };
  if (sink === "screen") {
    e.candidates = [{ candidate_id: "p2", formula: "NiO" }, { candidate_id: "p3", formula: "LiF" }];
    e.screenings = [{ ranked_of: attack, covering_set: ["p3"], ranking: [
      { candidate_id: "p1", rank: attack }, { candidate_id: "p2", rank: attack },
      { candidate_id: "p3", rank: attack } ] }];
  } else if (sink === "refit") {
    V.refits = [{ status: "running", started: attack, edit: { candidates: ["p1"] } }];
  } else if (sink === "plot") {
    e.fit.two_theta[0] = attack;
  } else {
    e.case_id = attack; e.answer.claims = [{ kind: attack, text: attack }];
    e.answer.phases[0].formula = attack;
    e.answer.submitted = { [attack]: [attack] };
    e.answer.quantity_basis = { [attack]: attack };
    e.brief = { specimen: attack, processing: [{ action: attack, atmosphere: attack }] };
    e.checks = [{ code: attack, name: attack, detail: attack }];
    V.labels[labelKey("phase_supported", "p1")] = { answer: "no", note: attack };
    V.comment = attack; V.revealed = { identity: { model: attack },
      reference: { phases: [{ formula: attack }], source: attack } };
  }
  render(); return analysisReport(e);
})()`, ctx, { timeout: 1000 });
process.stdout.write(JSON.stringify([nodes.main.innerHTML, report]));
''', str(SCRIPT)], input=json.dumps({"sink": sink, "attack": ATTACK}), capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
    page, report = json.loads(run.stdout)
    for html in (page, report):
        Markup().feed(html)
        assert ATTACK not in html
    assert escape(ATTACK, quote=True).replace("&#x27;", "&#39;") in page
