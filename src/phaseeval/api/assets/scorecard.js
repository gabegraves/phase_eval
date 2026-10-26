const KEY = "phaseeval.admin_token";
try {
  const saved = sessionStorage.getItem(KEY) || sessionStorage.getItem("axa.dashboard.token");
  if (saved) sessionStorage.setItem(KEY, saved);
  for (const key of ["axa.dashboard.token", "axa.owner_token", "phaseeval.rater_token"]) sessionStorage.removeItem(key);
} catch (e) {}
const tokenInput = document.getElementById("token");
const error = document.getElementById("error");
const panel = document.getElementById("panel");
let VIEW = null;
let FILTER = null;  // {model, step} from a heatmap cell
// ?model=<model> scopes every table on the server (Q6: the video shows one model only).
const SCOPE = new URLSearchParams(location.search).get("model");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SOURCE = { expert_label: "expert label", reference_match: "published reference", deterministic_check: "deterministic check", model_judge: "model judge" };
const SHORT = { expert_label: "expert", reference_match: "reference", deterministic_check: "check", model_judge: "judge" };
const TAGCLASS = { reference_match: "ref", model_judge: "judge" };
const GLYPH = { pass: "✓ pass", fail: "✕ fail", unknown: "? unknown", disputed: "! disputed", noanswer: "– no answer" };
// Disputed and no-answer are subsets of fail: every count still holds them as fails; the page shows them apart.
const kindOf = (s) => (s.outcome !== "fail" ? s.outcome : s.disputed ? "disputed" : s.no_answer ? "noanswer" : "fail");
const agentFail = (s) => s.outcome === "fail" && !s.disputed && !s.no_answer;
function token() { try { return sessionStorage.getItem(KEY); } catch (e) { return null; } }
// state: "token" (this tab holds the admin token), "local" (the server runs in local mode and
// knows this machine's owner), "none" (show the token box), "pending" (asking the server).
function signedIn(state) {
  document.getElementById("auth").hidden = state !== "none";
  document.getElementById("session").hidden = state !== "token" && state !== "local";
  document.getElementById("session-text").textContent = state === "local" ? "Local mode: this machine's owner" : "Signed in with this tab's admin token";
  document.getElementById("signout").hidden = state !== "token";
}
const bearer = () => { const t = token(); return t ? { Authorization: "Bearer " + t } : {}; };
async function load() {
  const t = token();
  signedIn("pending");
  error.textContent = "";
  // Ask without a token first: a server in local mode answers its own machine directly.
  const response = await fetch("/api/attempts/scorecard" + (SCOPE ? "?model=" + encodeURIComponent(SCOPE) : ""), { headers: bearer(), cache: "no-store" });
  if (!response.ok) {
    const refused = response.status === 401 || response.status === 403;
    if (refused) signedIn("none");
    if (t || !refused) error.textContent = "Scorecard request failed: " + response.status + (refused ? ". This page needs the admin owner token." : ".");
    return;
  }
  const body = await response.json();
  signedIn(body.local_mode ? "local" : "token");
  VIEW = body.view;
  document.getElementById("tables").innerHTML = body.html;  // escaped server-side (scorecard.render)
  drawLead();
  const dl = document.getElementById("pairs-download");
  if (dl) dl.addEventListener("click", () => downloadPairs("pairs.jsonl", "preference-pairs.jsonl"));
  const demo = document.getElementById("demo-download");
  if (demo) demo.addEventListener("click", () => downloadPairs("demo-pairs.jsonl", "demo-pairs-revealed.jsonl"));
}
// Every number below is read from VIEW (the same view the server's tables render); nothing is recomputed differently.
function sourcesOf(model, step) {
  const counts = {};
  for (const e of VIEW.episodes) {
    if (e.model !== model) continue;
    const s = e.steps.find((x) => x.step === step);
    if (s && s.source) counts[s.source] = (counts[s.source] || 0) + 1;
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1]);
}
function tagFor(model, step) {
  const src = sourcesOf(model, step);
  if (!src.length) return '<span class="tag none">no source decides</span>';
  return src.map(([s, n]) => `<span class="tag ${TAGCLASS[s] || ""}">${esc(SOURCE[s])}${src.length > 1 ? " " + n : ""}</span>`).join(" ");
}
function drawLead() {
  const lead = document.getElementById("lead");
  if (!lead) return;
  const steps = VIEW.steps;
  const decided = VIEW.matrix.filter((r) => r.source === "decided");
  const cases = [...new Set(VIEW.episodes.map((e) => e.case_id))].sort();
  const expert = VIEW.expert_labels;
  const scope = SCOPE
    ? `<p class="scope">Only ${esc(SCOPE)} runs on this page. <a class="qbtn" href="/scorecard">Show all models</a></p>`
    : `<p class="scope">Models: <a class="qbtn" href="/scorecard" aria-current="true">All</a>${decided.map((r) => `<a class="qbtn" href="/scorecard?model=${encodeURIComponent(r.model)}">${esc(r.model)} only</a>`).join("")}</p>`;
  let html = scope + `<p class="summary"><b>${VIEW.episodes.length} agent runs</b> on ${esc(cases.join(", "))} by ${decided.length} ${decided.length === 1 ? "model" : "models"}. Each number names its source: <span class="tag ref">published reference</span> <span class="tag">deterministic check</span> <span class="tag judge">model judge</span>. ${expert ? `<b>${expert} expert labels.</b>` : `<span class="tag none">no expert labels yet</span>`}</p>`;
  html += `<section class="panel" id="heat-panel"><header><h2>Model by step</h2><span class="hint">Each cell: runs that pass that step, over all of the model's runs (n). Click a cell to list those runs.</span></header><div class="heat" role="grid">`;
  html += `<div class="colh"></div>` + steps.map((d) => `<div class="colh">${esc(d.title)}</div>`).join("");
  for (const row of decided) {
    html += `<div class="rowh"><b>${esc(row.model)}</b><span>n = ${row.episodes} runs</span></div>`;
    for (const d of steps) {
      const c = row.cells[d.step];
      const agent = c.failed - c.disputed - c.no_answer;
      const p = c.passed / c.denominator, f = agent / c.denominator, dd = c.disputed / c.denominator, na = c.no_answer / c.denominator, u = c.unknown / c.denominator;
      const tint = p >= f ? `rgba(34,197,94,${(0.06 + 0.34 * p).toFixed(3)})` : `rgba(239,68,68,${(0.06 + 0.34 * f).toFixed(3)})`;
      const parts = [`${agent} fail`, c.disputed ? `${c.disputed} disputed` : "", c.no_answer ? `${c.no_answer} no answer` : "", `${c.unknown} unknown`].filter(Boolean).join(", ");
      const on = FILTER && FILTER.model === row.model && FILTER.step === d.step;
      html += `<button type="button" class="hm" data-model="${esc(row.model)}" data-hstep="${esc(d.step)}" aria-pressed="${on}" data-tint="${tint}">
        <span class="big">${c.passed}<small> / ${c.denominator} pass</small></span>
        <span class="bar" aria-hidden="true"><i class="p" data-width="${(p * 100).toFixed(1)}"></i><i class="f" data-width="${(f * 100).toFixed(1)}"></i><i class="d" data-width="${(dd * 100).toFixed(1)}"></i><i class="n" data-width="${(na * 100).toFixed(1)}"></i><i class="u" data-width="${(u * 100).toFixed(1)}"></i></span>
        <span class="sub">${parts}, n = ${c.denominator}</span>
        <span>${tagFor(row.model, d.step)}</span></button>`;
    }
  }
  html += `</div><div class="legend"><span><i class="sw pass"></i>pass</span><span><i class="sw fail"></i>fail</span><span><i class="sw disputed"></i>disputed: misses the published reference only on a phase its source authors report</span><span><i class="sw noanswer"></i>no answer: the run ended on its budget or time limit</span><span><i class="sw unknown"></i>unknown: no source reads pass or fail</span><span>Disputed and no answer still count as fail in the tables below.</span><span>Per-source counts are in "Model by step, every source" below.</span></div></section>`;
  html += runsHtml();
  lead.innerHTML = html;
  lead.querySelectorAll("[data-tint]").forEach((cell) => { cell.style.backgroundColor = cell.dataset.tint; });
  lead.querySelectorAll("[data-width]").forEach((bar) => { bar.style.width = bar.dataset.width + "%"; });
}
function runsHtml() {
  const steps = VIEW.steps;
  let runs = VIEW.episodes.slice();
  const rank = { fail: 0, unknown: 1, pass: 2 };
  if (FILTER) {
    runs = runs.filter((e) => e.model === FILTER.model);
    runs.sort((a, b) => rank[a.steps.find((s) => s.step === FILTER.step).outcome] - rank[b.steps.find((s) => s.step === FILTER.step).outcome]);
  }
  const title = (k) => (steps.find((d) => d.step === k) || {}).title || k;
  let html = `<div class="runs-bar" id="runs"><h2>Runs</h2>`;
  html += FILTER
    ? `<span class="filter">${runs.length} runs of <b>${esc(FILTER.model)}</b>, ordered by ${esc(title(FILTER.step))}: fail, unknown, pass</span> <button type="button" class="qbtn" id="clear-filter">Show all ${VIEW.episodes.length} runs</button>`
    : `<span class="filter">All ${runs.length} runs. Click a step to open its evidence and the trace at that step; the ringed step is the first one that fails.</span>`;
  html += `</div><div class="runs" role="table">`;
  html += `<div class="hd">Run</div><div class="hd">Ending, Rwp</div>` + steps.map((d) => `<div class="hd ${FILTER && FILTER.step === d.step ? "on" : ""}">${esc(d.title)}</div>`).join("") + `<div class="hd">First failing step</div><div class="hd">Reward v0</div><div class="hd">Analyst coding</div>`;
  for (const e of runs) {
    const first = e.steps.find(agentFail);
    const ended = e.steps.find((s) => s.no_answer);
    html += `<div class="row" id="ep-${esc(e.episode)}">`;
    html += `<div class="run"><b>${esc(e.case_id)} ${esc(e.model)}</b><span class="mono">${esc(e.job)}</span></div>`;
    html += `<div class="end">${esc(e.end)}<span class="rw">${e.rwp_percent == null ? "no fit" : "Rwp " + e.rwp_percent.toFixed(2) + "%"}</span></div>`;
    for (const s of e.steps) {
      const isFirst = first && first.step === s.step;
      const kind = kindOf(s);
      const sub = kind === "noanswer" ? "budget or timeout" : (s.source ? SHORT[s.source] : "no source") + (s.failure_code ? " " + s.failure_code : "");
      const hint = `${title(s.step)}: ${kind === "noanswer" ? "no answer, the run ended on its budget or time limit" : kind === "disputed" ? "misses the published reference only on a disputed phase" : s.outcome}${s.source ? " (" + SOURCE[s.source] + ")" : ""}${s.failure_code ? ". " + s.failure_code + ": " + codeTitle(s.failure_code) : ""}`;
      html += `<div class="${FILTER && FILTER.step === s.step ? "col-on" : ""}"><button type="button" class="chip ${kind} ${isFirst ? "first" : ""}" data-ep="${esc(e.episode)}" data-step="${esc(s.step)}" title="${esc(hint)}"><b>${GLYPH[kind]}</b><small>${esc(sub)}</small></button></div>`;
    }
    html += `<div class="firstfail">${first ? `<b>${esc(title(first.step))}</b>&nbsp;${esc(SHORT[first.source] || "")} ${esc(first.failure_code || "")}` : ended ? "no answer (budget or timeout)" : "none"}</div>`;
    html += `<div class="rw" title="Reward v0, 0 to 1: ${esc(e.reward.terms.map((t) => t.id + " " + (t.weight * t.value).toFixed(2)).join(", "))}">${e.reward.total.toFixed(2)}</div>`;
    html += `<div>${e.analyst ? `<button type="button" class="chip analyst" data-ep="${esc(e.episode)}" data-step="analyst" title="${esc(e.analyst.code + ": " + codeTitle(e.analyst.code))}"><b>${esc(e.analyst.code)}</b><small>${e.analyst.disputed ? "disputed" : "first divergence"}</small></button>` : `<span class="na">${e.analyst_none ? "none found" : "not coded"}</span>`}</div>`;
    html += `</div>`;
  }
  html += `</div><dl class="codes">${VIEW.failure_codes.map((c) => `<dt>${esc(c.code)}</dt><dd>${esc(codeTitle(c.code))}</dd>`).join("")}<dt>Reward v0</dt><dd>0 to 1 per run: 0.80 from the checks and the published reference, 0.20 from model judges; a run with no answer scores 0 (spec in the Reward section).</dd><dt>Analyst coding</dt><dd>where an agent's reading of the trace against custody puts the first divergence; not an expert label.</dd></dl>`;
  return html;
}
function codeTitle(code) {
  const c = (VIEW.failure_codes || []).find((x) => x.code === code);
  if (!c) return "";
  const plain = { S3: "misread the material context before searching", S4: "the search missed a phase in the published reference", S5: "the reported phase list is off", S6: "the wrong polymorph, or a polymorph mislabeled", S8: "the refinement degraded or did not converge", S9: "the amounts are off or not reconciled", S10: "the fit does not support the conclusion", S11: "the run ended without an answer (budget, context or tool misuse)" };
  return plain[code] || c.title;
}
async function downloadPairs(path, name) {
  const response = await fetch("/api/attempts/scorecard/" + path, { headers: bearer() });
  if (!response.ok) { error.textContent = "Pairs request failed: " + response.status; return; }
  const url = URL.createObjectURL(await response.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}
function open(epId, stepName) {
  const e = VIEW.episodes.find((x) => x.episode === epId);
  if (!e) return;
  const trace = VIEW.traces[epId] || [];
  let at = null, html = "";
  if (stepName === "analyst") {
    const a = e.analyst;
    at = a.call_index;
    document.getElementById("p-title").textContent = `Analyst coding ${a.code}: ${e.job}`;
    html += `<p class="shortfall">An agent's reading of this trajectory against custody (failures-trajectories.md), not an expert label.</p>`;
    html += `<p><b>${esc(a.code)}</b>${a.disputed ? " (disputed call)" : ""}: ${esc(a.note)}</p>`;
    html += at == null ? `<p class="meta">The doc locates this divergence in the agent's reasoning or answer, not at one call.</p>` : "";
  } else {
    const s = e.steps.find((x) => x.step === stepName);
    const d = VIEW.steps.find((x) => x.step === stepName);
    at = s.call_index;
    document.getElementById("p-title").textContent = `${d.title}: ${s.outcome}${s.source ? " (" + SOURCE[s.source] + ")" : ""}`;
    html += `<p>${esc(d.question)} ${s.failure_code ? `Failure code <b>${esc(s.failure_code)}</b>.` : ""}</p>`;
    html += `<table class="readings"><caption>Every source on this step, in precedence order</caption><thead><tr><th>source</th><th>check</th><th>reads</th><th>evidence</th></tr></thead><tbody>`;
    for (const dec of d.deciders) {
      const r = s.readings.find((x) => x.source === dec.source && x.check === dec.check);
      if (!r) continue;
      const decided = s.source === r.source && s.check === r.check;
      html += `<tr><td>${decided ? "<b>" : ""}${esc(SOURCE[r.source])}${decided ? " (decides)</b>" : ""}</td><td><code>${esc(r.check)}</code></td><td><span class="cell reading ${r.outcome}">${esc(r.outcome)}</span></td><td>${esc(r.detail)}</td></tr>`;
    }
    html += `</tbody></table>`;
  }
  document.getElementById("p-sub").innerHTML = `${esc(e.case_id)} · ${esc(e.model)} · <span class="mono">${esc(e.job)}</span> · ${esc(e.end)}${e.rwp_percent != null ? ` · Rwp ${e.rwp_percent.toFixed(2)}%` : ""}`;
  html += `<p class="links">${e.snapshot ? `<a href="/workbench?snapshot=${encodeURIComponent(e.snapshot)}" target="_blank" rel="noopener">Open this answer's review page</a>` : `<span class="na">No answer snapshot, so no review page.</span>`}</p>`;
  html += `<h4>Trace: ${trace.length} recorded calls${at != null ? `, opened at call ${at}` : ""}</h4><ol class="trace" id="trace">`;
  for (const row of trace) {
    html += `<li class="${row.index === at ? "at" : ""}" data-i="${row.index}"><span>${row.index}</span><span class="tag">${esc(row.tool)}${row.marks.length ? "<br>" + esc(row.marks.join(", ")) : ""}</span><span class="s">${esc(row.summary)}</span></li>`;
  }
  html += `</ol>`;
  document.getElementById("p-body").innerHTML = html;
  panel.classList.add("open");
  const hit = document.querySelector("#trace li.at");
  if (hit) hit.scrollIntoView({ block: "center" });
}
document.addEventListener("click", (event) => {
  const hm = event.target.closest("button.hm[data-model]");
  if (hm) {
    const same = FILTER && FILTER.model === hm.dataset.model && FILTER.step === hm.dataset.hstep;
    FILTER = same ? null : { model: hm.dataset.model, step: hm.dataset.hstep };
    drawLead();
    if (FILTER) document.getElementById("runs").scrollIntoView({ block: "start" });
    return;
  }
  if (event.target.closest("#clear-filter")) { FILTER = null; drawLead(); return; }
  const cell = event.target.closest("button[data-ep][data-step]");
  if (cell) { open(cell.dataset.ep, cell.dataset.step); return; }
  const jump = event.target.closest("a[data-jump]");
  if (jump) {
    if (FILTER) { FILTER = null; drawLead(); }
    const row = document.getElementById("ep-" + jump.dataset.jump);
    if (row) {
      event.preventDefault();
      row.firstElementChild.scrollIntoView({ block: "center" });
      row.classList.add("flash"); setTimeout(() => row.classList.remove("flash"), 1500);
    }
  }
});
document.getElementById("close").addEventListener("click", () => panel.classList.remove("open"));
document.addEventListener("keydown", (event) => { if (event.key === "Escape") panel.classList.remove("open"); });
document.getElementById("auth").addEventListener("submit", (event) => {
  event.preventDefault();
  try { sessionStorage.setItem(KEY, tokenInput.value.trim()); } catch (e) { /* storage unavailable */ }
  tokenInput.value = "";
  load();
});
document.getElementById("signout").addEventListener("click", () => {
  try { sessionStorage.removeItem(KEY); } catch (e) { /* storage unavailable */ }
  VIEW = null; FILTER = null;
  document.getElementById("tables").innerHTML = "";
  panel.classList.remove("open");
  signedIn("none");
});
load();
