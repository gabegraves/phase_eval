// The workbench defaults to retrospective method review. Its independent Outcome view keeps the
// evidence-first order and reveals reference-bearing fields only after scientific sign-off.
"use strict";

const TOKEN_KEY = "axa.owner_token";
const REFITS_KEY = "axa.workbench.refits.";
const PEAKS_KEY = "axa.workbench.peaks.";

// Every request the page makes. tests/workbench/test_page_contract.py matches each entry against
// the app's router table and the blind-route whitelist (Build manager D10): before sign-off the
// page reads only blind projections, never the raw attempt, path, tree or compare routes.
const ROUTES = {
  queue: "GET /api/attempts/queue",
  review: "GET /api/attempts/snapshots/{digest}/review",
  revealed: "GET /api/attempts/snapshots/{digest}/revealed",
  files: "GET /api/attempts/snapshots/{digest}/files.zip", // 403 until this rater's own sign-off
  labels: "GET /api/attempts/snapshots/{digest}/labels",
  postLabel: "POST /api/attempts/snapshots/{digest}/labels",
  postPeakCall: "POST /api/attempts/snapshots/{digest}/peak-calls",
  postSignoff: "POST /api/attempts/snapshots/{digest}/signoff",
  postSnapshot: "POST /api/attempts/snapshots",
  postPermissions: "POST /api/attempts/permissions",
  stepFeedback: "GET /api/attempts/step-feedback",
  postStepFeedback: "POST /api/attempts/step-feedback",
  postActions: "POST /api/attempts/{digest}/actions",
  attemptReview: "GET /api/attempts/{digest}/review",
  traceCall: "GET /api/attempts/snapshots/{digest}/trace/{call}", // one call in full, the review's gate
  composition: "GET /api/attempts/roots/{root}/composition",
  postComposition: "POST /api/attempts/roots/{root}/composition",
  impliedComposition: "POST /api/attempts/implied-composition",
  capabilities: "GET /api/capabilities",
  trajectoryReview: "GET /api/attempts/snapshots/{digest}/trajectory-review",
  postTrajectoryReview: "POST /api/attempts/snapshots/{digest}/trajectory-review",
};

// The frozen rubric (labels.RUBRIC_TEXT, agentxrd-rubric-v1). The contract test pins this wording.
const RUBRIC = {
  phase_supported: "Is this phase supported by the pattern?",
  phase_plausible: "Is this phase chemically plausible for this sample?",
  amounts_defensible: "Are the reported amounts defensible on their basis?",
  fit_supports_conclusion: "Does the fit support the conclusion?",
};
const PHASE_ITEMS = ["phase_supported", "phase_plausible"];
const ANALYSIS_ITEMS = ["amounts_defensible", "fit_supports_conclusion"];
const ANSWERS = [["yes", "Yes"], ["no", "No"], ["cannot_assess", "Can't tell"]];
const PEAK_CAUSES = [["missing_phase", "Missing phase"], ["already_fitted", "Already fitted"], ["noise_or_profile", "Noise or profile"], ["cannot_assess", "Can't tell"]];
const VERDICTS = [["would_sign", "I would sign this analysis"], ["would_not_sign", "I would not sign it"], ["unresolved", "Unresolved"]];
const MARKS = [["wrong_move", "Wrong move"], ["went_wrong_here", "This is where it went wrong"], ["good_move", "Good move"], ["note", "Note"]];
const MARK_CHIP = { wrong_move: "WRONG MOVE", went_wrong_here: "WENT WRONG HERE", good_move: "GOOD MOVE", note: "NOTE" };
const PERMISSIONS = [
  ["participation_and_storage", "Store my labels and take part in this review. Required before any label is stored."],
  ["internal_evaluation_and_calibration", "Use my labels for internal evaluation and to calibrate model judges."],
  ["public_release", "Include my labels in a public dataset release."],
  ["training", "Use my labels to train models."],
  ["public_identity_disclosure", "Name me publicly alongside released labels."],
];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (x, d = 2) => (x == null || x === "" || Number.isNaN(Number(x)) ? "n/a" : Number(x).toFixed(d));
const clock = (s) => { const t = Math.max(0, Math.round(Number(s) || 0)); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`; };
const SUB = { "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆" };
function sgShort(sg) {
  let s = String(sg || "").replace(/\s*\((?:no\.?\s*)?\d+\)$/i, "");
  s = s.replace(/^([PCIFAR])1(.+)1$/, "$1$2").replace(/_?([2346])_?([1-5])/g, (m, a, b) => a + (SUB[b] || b));
  return s;
}

const S = {
  queue: [], i: 0, ep: null, me: null, error: "",
  mode: "method", methods: {},
  permission: undefined, // undefined: unknown (no read route); null: refused for want of one; else the record this tab stored
  consentOpen: false,
  peakSel: null, phaseSel: null, showSteps: false, traceTime: false, traceFull: {}, markOpen: null, markDraft: null, refitDraft: null,
  views: {}, steps: {}, openClaims: new Set(),
  refit: { available: true, note: "" }, // ROUTES.capabilities: refit needs a BGMN this server can run
};

const METHOD_PROTOCOL = "xrd-method-v1";
function blankMethodDraft() {
  return { verdict: "", note: "", evidence: [], earliest_error: null, observed_repair: null, repair_status: "not_applicable", proposed_repair: "" };
}
function refKey(ref) { return ref ? `${ref.turn_index}:${ref.call_index == null ? "message" : ref.call_index}` : ""; }
function methodControlValue(name, value) { return value == null ? "" : name === "repair_status" ? value : refKey(value); }
function sameRef(a, b) { return refKey(a) === refKey(b); }
function methodActionDescription(call) {
  let args = call.command ?? call.input ?? call.arguments;
  if (typeof args === "string") { try { args = JSON.parse(args); } catch { args = null; } }
  const validArgs = args && typeof args === "object" && !Array.isArray(args);
  switch (call.tool) {
    case "case_brief": return "Requested the assigned case’s chemistry, measurement metadata, precursors and processing history.";
    case "candidates": return "Requested the available candidate structures, including formulas, symmetry and unit cells.";
    case "screen":
      if (!validArgs) return "Requested a candidate or residual peak screen; inspect the recorded inputs for its scope.";
      return typeof args.residual === "string" && args.residual ? `Requested unexplained residual peaks from fit ${args.residual} and nearby candidate lines.` : "Requested a ranking of candidate structures against the observed diffraction peaks.";
    case "refine":
      if (!validArgs) return "Requested a refinement; inspect the recorded inputs for its configuration.";
      return `Requested ${args.pawley_only === true ? "a Pawley profile fit" : "staged Pawley/Rietveld refinement"}${Array.isArray(args.candidates) && args.candidates.length ? ` with ${args.candidates.join(", ")}` : ""}${typeof args.run_label === "string" ? `, saved under ${args.run_label}` : ""}.`;
    case "write_analysis": return "Requested validation and saving of the submitted phase analysis and its cited fit artifacts.";
    default: return call.command || call.tool || `Call ${call.call_index}`;
  }
}
function methodEvents(trace) {
  return (trace || []).flatMap((turn, turn_index) => {
    const message = turn.message ? [{ ref: { turn_index, call_index: null }, turn, kind: "message", text: turn.message }] : [];
    return message.concat((turn.calls || []).map((call) => ({ ref: { turn_index, call_index: call.call_index }, turn, call, kind: "call", text: methodActionDescription(call) })));
  });
}
function methodRefFor(M, key) { const event = methodEvents((M.data || {}).trace).find((x) => refKey(x.ref) === key); return event ? event.ref : null; }
function updateMethodSelect(M, field, value) {
  M.draft[field] = field === "repair_status" ? value : methodRefFor(M, value);
  if (field === "earliest_error") {
    M.draft.evidence = M.draft.evidence.filter((x) => !M.draft.earliest_error || x.turn_index < M.draft.earliest_error.turn_index);
    M.draft.observed_repair = null;
    M.draft.repair_status = M.draft.earliest_error ? "unknown" : "not_applicable";
  } else if (field === "repair_status" && value !== "repaired") M.draft.observed_repair = null;
}
function methodState(digest) {
  if (!S.methods[digest]) S.methods[digest] = { data: null, selected: null, draft: blankMethodDraft(), saved: null, baseline: JSON.stringify(blankMethodDraft()), submissionId: null, saving: false, error: "" };
  return S.methods[digest];
}
function methodDirty(M) { return JSON.stringify(M.draft) !== M.baseline; }
function methodBlocked() { const M = S.ep && methodState(S.ep.digest); return Boolean(M && (M.saving || methodDirty(M))); }
function reviewDraft(record) {
  const d = blankMethodDraft();
  if (!record) return d;
  Object.keys(d).forEach((k) => { if (record[k] !== undefined) d[k] = record[k]; });
  return d;
}
function restoreMethod(M, record) {
  M.saved = record || null;
  M.draft = reviewDraft(record);
  M.baseline = JSON.stringify(M.draft);
  M.submissionId = null;
  M.error = "";
}
function newSubmissionId() {
  if (globalThis.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `review-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
function methodRequest(M, digest) {
  M.submissionId = M.submissionId || newSubmissionId();
  return { snapshot: digest, submission_id: M.submissionId, supersedes: M.saved && M.saved.submission_id || null, protocol_version: METHOD_PROTOCOL, ...M.draft };
}

// ---------------------------------------------------------------------------------------------
// auth and api
// ---------------------------------------------------------------------------------------------
function token() { try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; } }
function setToken(value) {
  try { value ? sessionStorage.setItem(TOKEN_KEY, value) : sessionStorage.removeItem(TOKEN_KEY); } catch {}
  $("auth").classList.toggle("hidden", Boolean(value));
}
function url(name, params = {}, query = null) {
  const path = ROUTES[name].split(" ")[1].replace(/\{(\w+)\}/g, (_, k) => encodeURIComponent(params[k]));
  return query ? `${path}?${new URLSearchParams(query)}` : path;
}
async function call(name, params = {}, { body, query, raw = false } = {}) {
  const method = ROUTES[name].split(" ")[0];
  const headers = { Authorization: `Bearer ${token()}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(url(name, params, query), { method, headers, cache: "no-store", body: body === undefined ? undefined : JSON.stringify(body) });
  if (response.status === 401) {
    setToken("");
    throw httpError("The server refused this token (401). Paste a current one.", 401);
  }
  if (!response.ok) {
    let detail = "";
    try { const j = await response.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j.error ?? ""); } catch {}
    throw httpError(detail || `HTTP ${response.status}`, response.status);
  }
  return response.status === 204 ? null : raw ? response.blob() : response.json();
}
function httpError(message, status) { const error = new Error(message); error.status = status; return error; }
const listOf = (x, ...keys) => (Array.isArray(x) ? x : keys.map((k) => x && x[k]).find(Array.isArray) || []);
function status(message, isError = false) { const el = $("status"); el.textContent = message || ""; el.classList.toggle("err", isError); }

// ---------------------------------------------------------------------------------------------
// the blind payload, normalized
// ---------------------------------------------------------------------------------------------
// A phase as the answer named it (formula, space group, crystal system, name: verbatim, may be
// absent) and, apart, the candidate CIF's metadata. A payload without "cif" names only the CIF.
function phasesOf(e) {
  const src = (e.answer && e.answer.phases) || e.phases || (e.fit && e.fit.phase_detail) || [];
  return src.map((p) => {
    const cif = p.cif ? { formula: p.cif.formula ?? null, sg: p.cif.space_group ?? null } : { formula: p.formula ?? null, sg: p.space_group ?? null };
    const reported = p.cif ? { formula: p.formula ?? null, sg: p.space_group ?? null, system: p.crystal_system ?? null, name: p.name ?? null } : null;
    return {
      id: p.phase_id ?? p.candidate_id, cif, reported,
      formula: (reported && reported.formula) || cif.formula, sg: (reported && reported.sg) || cif.sg,
      wt: p.weight_percent ?? null, esd: p.esd ?? p.weight_percent_error ?? null,
      shift: p.lattice_shift_pct ?? p.lattice_shift_percent ?? null,
    };
  });
}
const normFormula = (f) => String(f || "").replace(/[\s_()·]/g, "").replace(/[₀-₉]/g, (c) => String(c.charCodeAt(0) - 8320)).toLowerCase();
const normSg = (g) => String(g || "").replace(/\s*\((?:no\.?\s*)?\d+\)$/i, "").replace(/[\s_]/g, "").replace(/[₀-₉]/g, (c) => String(c.charCodeAt(0) - 8320)).toLowerCase();
function phaseNames(p) {
  if (!p.reported) return `<b>${esc(p.cif.formula || p.id)}</b> <span class="mono muted">${esc(sgShort(p.cif.sg))}</span><div class="mono muted" style="font-size:11px">${esc(p.id)}</div>`;
  const r = p.reported, said = [r.name, r.formula, r.sg && sgShort(r.sg), r.system].filter(Boolean);
  const differs = (r.formula && normFormula(r.formula) !== normFormula(p.cif.formula)) || (r.sg && normSg(r.sg) !== normSg(p.cif.sg));
  return `<div><span class="lbl">Answer</span> ${said.length ? `<b>${esc(said.join(" · "))}</b>` : '<span class="muted">not recorded</span>'}</div>
    <div class="mono muted" style="font-size:11px"><span class="lbl">CIF</span> ${esc(p.cif.formula || "")} ${esc(sgShort(p.cif.sg))} · ${esc(p.id)}</div>${differs ? '<div class="req">the answer\'s words differ from the CIF</div>' : ""}`;
}
function residualCheck(e) { return (e.checks || []).find((c) => c.code === "K2" || /residual/i.test(c.name || "")) || null; }
function peaksOf(e) {
  if (Array.isArray(e.residual_peaks)) return e.residual_peaks.map((p) => ({ id: p.peak_id, tt: p.two_theta_deg, pct: p.height_percent, counts: p.residual_counts }));
  const r = residualCheck(e);
  return (r && r.peaks || []).map((p) => ({ id: `p${fmt(p.two_theta, 2)}`, tt: p.two_theta, pct: p.pct_of_strongest_peak ?? p.percent, counts: null }));
}
const refines = (e) => (e.steps || []).map((s, i) => ({ ...s, i })).filter((s) => s.kind === "refine");
const claimsOf = (e) => ((e.answer && e.answer.claims) || e.claims || []).map((c) => (typeof c === "string" ? { kind: "", text: c } : c));
const isAnswered = (e) => (e.answered != null ? Boolean(e.answered) : Boolean(e.answer));
function citedOf(e) {
  const label = (e.answer && e.answer.cited_fit) || null;
  const rwp = (e.answer && e.answer.rwp) ?? (e.fit && e.fit.rwp) ?? e.rwp_percent ?? null;
  return { label, rwp };
}
function formulaOf(e, cid) {
  const p = phasesOf(e).find((x) => x.id === cid) || (e.candidates || []).find((x) => x.candidate_id === cid);
  return (p && p.formula) || cid;
}

// ---------------------------------------------------------------------------------------------
// per-snapshot review state: answers mirror the latest stored record per item
// ---------------------------------------------------------------------------------------------
function view(d = S.ep && S.ep.digest) {
  if (!S.views[d]) S.views[d] = { opened: Date.now(), seconds: 0, labels: {}, peaks: {}, drafts: {}, crit: {}, pendingNo: {}, saving: {}, errors: {}, unlocked: false, verdict: null, comment: "", signoff: null, revealed: null, revealError: "", others: [], refits: [], marks: {} };
  return S.views[d];
}
const labelKey = (item, phase) => `L|${item}|${phase || ""}`;
const peakKey = (id) => `P|${id}`;
const stepKey = (attempt, callIndex) => `${attempt}:${callIndex ?? "-"}`;

function itemKeys(e) {
  const keys = [];
  phasesOf(e).forEach((p) => PHASE_ITEMS.forEach((item) => keys.push(labelKey(item, p.id))));
  return keys;
}
function evidenceDone(e, V) {
  return itemKeys(e).every((k) => V.labels[k]) && peaksOf(e).every((p) => V.peaks[p.id]);
}
function signoffItems(e) { return [...itemKeys(e), ...ANALYSIS_ITEMS.map((item) => labelKey(item, null))]; }
function refreshUnlock(e, V) { if (!V.unlocked && evidenceDone(e, V)) V.unlocked = true; }
const mine = (r) => !S.me || !r.rater || r.rater.principal_id === S.me;
const byTime = (a, b) => String(a.created_at_utc || "").localeCompare(String(b.created_at_utc || ""));

function restoreLabels(V, labels) {
  labels.filter(mine).sort(byTime).forEach((l) => { V.labels[labelKey(l.item, l.phase_id)] = { answer: l.answer, reason: l.reason || "", note: l.note || "", exposure: l.exposure }; });
  V.others = labels.filter((l) => !mine(l));
}
function restoreMarks(V, records) {
  records.filter(mine).sort(byTime).forEach((r) => {
    const k = stepKey(r.step.attempt, r.step.call_index);
    (V.marks[k] = V.marks[k] || []).push({ mark: r.mark, note: r.note || "", exposure: r.exposure });
  });
}

// ---------------------------------------------------------------------------------------------
// writes. Labels are immutable records: every change posts a new one; the server stamps the
// rater, its kind and the exposure from the bearer token, never from this page (R32, KTD3).
// ---------------------------------------------------------------------------------------------
function noteRefusal(err) {
  if (err.status === 403 && /permission|participat|consent/i.test(err.message)) { S.permission = null; status("Your labels are not stored until you record your participation choice below.", true); }
  else status(`Not stored: ${err.message}`, true);
}
// note is the rater's free-text critique of the item; a new answer keeps the stored critique.
async function saveLabel(item, phase, answer, reason = "", note = null) {
  const e = S.ep, V = view(), k = labelKey(item, phase);
  const critique = note ?? ((V.labels[k] && V.labels[k].note) || "");
  const body = { snapshot: e.digest, item, phase_id: phase || null, answer, reason, note: critique };
  V.saving[k] = true; delete V.errors[k]; render();
  try {
    const rec = await call("postLabel", { digest: e.digest }, { body });
    V.labels[k] = { answer, reason, note: critique, exposure: rec && rec.exposure };
    delete V.drafts[k];
    if (note != null) { delete V.crit[k]; delete V.pendingNo[k]; }
    if (rec && rec.rater) S.me = rec.rater.principal_id;
    refreshUnlock(e, V);
    // The cursor moves on once the record is stored (and after an unlock adds the sign-off items).
    if (S.cur === k && !(k in V.crit)) S.cur = nextKey(e, k);
  } catch (err) { V.errors[k] = err.message; noteRefusal(err); }
  delete V.saving[k]; render();
}
async function savePeak(id, cause, note = "") {
  const e = S.ep, V = view(), k = peakKey(id);
  V.saving[k] = true; delete V.errors[k]; render();
  try {
    const rec = await call("postPeakCall", { digest: e.digest }, { body: { snapshot: e.digest, peak_id: id, cause, note } });
    V.peaks[id] = { cause, note };
    try { sessionStorage.setItem(PEAKS_KEY + e.digest, JSON.stringify(V.peaks)); } catch {}
    delete V.drafts[k];
    if (rec && rec.rater) S.me = rec.rater.principal_id;
    refreshUnlock(e, V);
    if (S.cur === k) S.cur = nextKey(e, k);
  } catch (err) { V.errors[k] = err.message; noteRefusal(err); }
  delete V.saving[k]; render();
}
async function saveSignoff() {
  const e = S.ep, V = view();
  V.saving.signoff = true; delete V.errors.signoff; render();
  try {
    await call("postSignoff", { digest: e.digest }, { body: { snapshot: e.digest, verdict: V.verdict, comment: V.comment.trim() } });
    V.signoff = { verdict: V.verdict };
    status("");
    await loadReveal(e.digest, true);
  } catch (err) { V.errors.signoff = err.message; noteRefusal(err); }
  delete V.saving.signoff; render();
}
async function loadReveal(digest, afterSignoff = false) {
  const V = view(digest);
  try {
    V.revealed = await call("revealed", { digest });
    V.revealError = "";
    // The server reveals per pattern; this snapshot counts as signed only with its own sign-off.
    const own = V.revealed && V.revealed.signoff;
    if (own && (!own.snapshot || own.snapshot === digest)) V.signoff = V.signoff || { verdict: own.verdict };
    const labels = V.revealed && Array.isArray(V.revealed.labels) ? V.revealed.labels : await call("labels", { digest }).catch(() => null);
    if (labels) V.others = listOf(labels, "labels", "items").filter((l) => !mine(l));
  } catch (err) {
    if ((err.status === 403 || err.status === 404) && !afterSignoff) { V.revealed = null; return false; }
    V.revealError = err.message;
  }
  return true;
}
async function saveMark(ref, mark, note) {
  const V = view(), k = stepKey(ref.attempt, ref.call_index);
  V.saving[k] = true; delete V.errors[k]; render();
  try {
    const rec = await call("postStepFeedback", {}, { body: { step: ref, mark, note } });
    (V.marks[k] = V.marks[k] || []).push({ mark, note, exposure: rec && rec.exposure });
    S.markOpen = null; S.markDraft = null;
  } catch (err) { V.errors[k] = err.message; noteRefusal(err); }
  delete V.saving[k]; render();
}
async function savePermission(form) {
  const body = Object.fromEntries(PERMISSIONS.map(([k]) => [k, Boolean(form.elements[k] && form.elements[k].checked)]));
  try {
    const rec = await call("postPermissions", {}, { body });
    S.permission = rec || body;
    S.consentOpen = false;
    if (rec && rec.principal_id) S.me = rec.principal_id;
    status(body.participation_and_storage ? "Participation recorded. Your labels are stored from now on." : "Recorded. Without participation no label is stored.");
  } catch (err) { status(`Not stored: ${err.message}`, true); }
  render();
}

// ---------------------------------------------------------------------------------------------
// refits through act(): one gesture, a phase edit then a refine (R13). The child attempt shows
// beside its parent in the refinement strip with its own Rwp; the parent is never changed.
// ---------------------------------------------------------------------------------------------
// One step of the blind gesture result (review.GestureStep: digest, parent, verb, status, fit,
// failure_reason; D10). Nothing of the parent, actor or annotations comes back.
function attemptSummary(v) {
  const step = v || {};
  const fit = step.fit || null;
  const failed = step.status !== "completed" || (!fit && Boolean(step.failure_reason));
  const reason = step.failure_reason || (step.status === "withheld" ? "withheld by the custody scan" : step.status ? `outcome ${step.status}` : "no outcome returned");
  return {
    digest: step.digest, failed, reason,
    rwp: fit ? fit.rwp_percent : null, gof: fit ? fit.gof : null, stage: fit ? fit.stage : null,
    phases: fit && fit.phases ? fit.phases.map((p) => p.candidate_id) : [],
    elapsed: null,
  };
}
function reviewSummary(digest, payload) {
  const f = payload.fit || {};
  return { digest, failed: false, reason: "", rwp: f.rwp ?? payload.rwp_percent ?? null, gof: f.gof ?? payload.gof ?? null, stage: f.stage || payload.stage || null, phases: phasesOf(payload).map((p) => p.id), elapsed: null, review: payload };
}
async function childReview(child) {
  try { child.review = await call("attemptReview", { digest: child.digest }); } catch (err) { child.reviewError = err.message; return child; }
  const sum = reviewSummary(child.digest, child.review);
  return { ...child, rwp: child.rwp ?? sum.rwp, gof: child.gof ?? sum.gof, stage: child.stage || sum.stage, phases: child.phases.length ? child.phases : sum.phases, review: child.review };
}
function rememberRefits(origin, refits) {
  const keep = refits.filter((r) => r.status === "done" && r.child).map((r) => ({ digest: r.child.digest, edit: r.edit, rationale: r.rationale, elapsed: r.elapsed, snapshot: r.snapshot || null }));
  try { sessionStorage.setItem(REFITS_KEY + origin, JSON.stringify(keep)); } catch {}
}
async function restoreRefits(e, V) {
  let saved = [];
  try { saved = JSON.parse(sessionStorage.getItem(REFITS_KEY + e.origin) || "[]"); } catch {}
  for (const r of saved) {
    if (V.refits.some((x) => x.child && x.child.digest === r.digest)) continue;
    try { V.refits.push({ ...r, status: "done", child: reviewSummary(r.digest, await call("attemptReview", { digest: r.digest })) }); }
    catch { /* an attempt this principal can no longer read is not shown */ }
  }
}
async function startRefit(draft) {
  const e = S.ep, V = view();
  if (V.refits.some((r) => r.status === "running")) return;
  const key = `w4-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const r = { id: key, edit: { verb: draft.verb, candidates: draft.candidates }, rationale: draft.rationale.trim(), status: "running", started: Date.now(), child: null, error: "" };
  V.refits.push(r); S.refitDraft = null; render();
  try {
    const out = await call("postActions", { digest: e.origin }, { body: { steps: [
      { idempotency_key: `${key}-edit`, rationale: r.rationale, action: { verb: draft.verb, candidates: draft.candidates } },
      { idempotency_key: `${key}-refine`, rationale: "", action: { verb: "refine", recipe: "staged_rietveld" } },
    ] } });
    const views = listOf(out, "views", "attempts", "created");
    const child = attemptSummary(views.length ? views[views.length - 1] : out);
    if (child.failed) { r.status = "failed"; r.error = child.reason; } else { r.child = await childReview(child); r.status = "done"; }
  } catch (err) { r.status = "failed"; r.error = err.message; }
  r.elapsed = (Date.now() - r.started) / 1000;
  rememberRefits(e.origin, V.refits);
  render();
}
async function labelCorrection(r) {
  const e = S.ep, V = view();
  r.saving = true; render();
  try {
    const out = await call("postSnapshot", {}, { body: { origin: r.child.digest, kind: "expert_correction" } });
    r.snapshot = out.digest;
    rememberRefits(e.origin, V.refits);
    if (!S.queue.some((q) => q.digest === out.digest)) S.queue.splice(S.i + 1, 0, { digest: out.digest, kind: "expert_correction", case_id: e.case_id });
    r.saving = false;
    await openSnapshot(S.queue.findIndex((q) => q.digest === out.digest));
    return;
  } catch (err) { r.error = `Could not freeze the correction: ${err.message}`; }
  r.saving = false; render();
}
const editText = (e, edit) => `${edit.verb === "remove_phases" ? "remove" : "add"} ${edit.candidates.map((c) => formulaOf(e, c)).join(", ")}`;

// ---------------------------------------------------------------------------------------------
// claim checks. A check names the artifact it compared against; a residual peak challenges a
// completeness claim but is not proof of another phase (science review, AXA-43zwk.8).
// ---------------------------------------------------------------------------------------------
function claimMark(c, e) {
  const t = c.text || "";
  const peaks = peaksOf(e).slice().sort((a, b) => b.pct - a.pct);
  const steps = refines(e);
  const cited = citedOf(e);
  const near = (a, b) => a != null && b != null && Math.abs(Number(a) - Number(b)) < 0.006;
  // Dogfood A3: P2_1/c Bi2O3 is the monoclinic alpha form; delta-Bi2O3 is the cubic Fm-3m form.
  const monoclinicBi2O3 = phasesOf(e).some((p) => normFormula(p.formula) === "bi2o3" && normSg(p.sg) === "p21/c");
  if (monoclinicBi2O3 && /\bdelta[- ](?:bi2o3|bi₂o₃|form|type|phase|polymorph)\b|δ-?bi/i.test(t))
    return ["hold", "Polymorph misnamed", "The answer's Bi2O3 is the P2₁/c CIF, the monoclinic alpha form; delta-Bi2O3 is cubic (Fm-3m)."];
  if (/gsas/i.test(t))
    return ["none", "Engine note", "Provenance only: this run's refinements ran on BGMN."];
  if (/atomic coordinates?|atom positions?|coordinates? (were|was) (not )?refined/i.test(t))
    return ["none", "Not checked", "The record does not show whether atomic coordinates varied."];
  if (/\bno\b[^.]*\b(additional|other|extra|unexplained|further)\b[^.]*\b(phases?|peaks?|reflections?)\b/i.test(t))
    return ["none", "Not checked", peaks.length
      ? `The residual check (a heuristic) lists an unexplained peak at ${fmt(peaks[0].tt)}° (${fmt(peaks[0].pct, 1)}%); a peak height cannot test a wt% limit, and a residual is not proof of a phase.`
      : "The residual check (a heuristic) lists no unexplained peak; a peak height cannot test a wt% limit."];
  const angles = [...t.matchAll(/(\d+(?:\.\d+)?)\s*(?:(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*)?(?:deg\b|°)/gi)].map((m) => [Number(m[1]), Number(m[2] || m[1])]);
  if (angles.length && /phases? already in the fit|belong(s|ing)? to|accounted for by|from (the )?(existing|fitted)|overlap/i.test(t)) {
    const hit = peaks.find((p) => angles.some(([lo, hi]) => p.tt >= lo - 0.15 && p.tt <= hi + 0.15));
    if (hit) return ["hold", "Disputed by a check", `The residual check (a heuristic) calls ${fmt(hit.tt)}° unexplained: net observed intensity there is over 5 times the fitted phases' net calculated intensity.`];
  }
  const ids = [...t.matchAll(/\b(CIF\d{6}|COD\d{7})\b/g)].map((m) => m[1]);
  if (ids.length && /not tested|never tested|untested/i.test(t)) {
    const tested = ids.filter((id) => steps.some((s) => (s.phases || []).some((p) => (p.candidate_id || p) === id)));
    return tested.length
      ? ["stop", "Contradicted", `${tested.join(", ")} was refined in this episode.`]
      : ["none", "No test found", `No test found in the available recorded steps for ${ids.join(", ")}; they may be incomplete.`];
  }
  const nums = [...t.matchAll(/Rwp\s*(?:is\s+|of\s+|from\s+|=\s*)?(\d+\.\d+)\s*%?(?:\s*(?:vs\.?|versus|to|and)\s*(\d+\.\d+)\s*%?)?/gi)].flatMap((m) => [m[1], m[2]]).filter(Boolean).map(Number);
  if (nums.length) {
    const where = nums.map((n) => (near(n, cited.rwp) ? "the cited fit" : (steps.find((s) => near(s.rwp, n)) || {}).run_label || null));
    if (where.every((w) => w === "the cited fit")) return ["pass", "Rwp = cited fit", `Rwp ${nums.map((n) => fmt(n)).join(", ")} equals the cited fit's${cited.label ? ` (${cited.label})` : ""}; this checks the number only, not the rest of the statement.`];
    if (where.every(Boolean)) return ["pass", "Rwp in a recorded run", `Each quoted Rwp equals a recorded refinement (${[...new Set(where)].join(", ")}); this checks the numbers only, not the rest of the statement.`];
    return ["hold", "Rwp not found", "A quoted Rwp equals no recorded refinement; the steps may be incomplete."];
  }
  const run = steps.find((s) => s.run_label && t.includes(s.run_label));
  if (run) return ["none", "Run found", `${run.run_label} ran (Rwp ${fmt(run.rwp)}); that confirms the run, not the statement.`];
  return ["none", "Not checked", ""];
}

// ---------------------------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------------------------
function seg(attrs, current, opts, { draft = null, saving = false, disabled = false } = {}) {
  return `<span class="seg ${current || draft ? "" : "missing"} ${saving ? "saving" : ""}" role="group">${opts.map(([v, l]) => `<button type="button" ${attrs} data-val="${v}" aria-pressed="${current === v}" class="${draft === v && current !== v ? "draft" : ""}" ${disabled || saving ? "disabled" : ""}>${l}</button>`).join("")}</span>`;
}
function reasonBox(V, k, what) {
  if (!(k in V.drafts)) return "";
  return `<div class="reasonbox"><input type="text" data-reason="${esc(k)}" data-fk="r${esc(k)}" value="${esc(V.drafts[k])}" placeholder="Why can't you tell? (required)" aria-label="Reason ${esc(what)}"><button type="button" class="qbtn" data-act="save-reason" data-key="${esc(k)}" ${String(V.drafts[k]).trim() ? "" : "disabled"}>Save</button></div>`;
}
// The critique: free text on why the answer is what it is, stored as the label's note. "No" opens it.
function critiqueButton(V, k) {
  const l = V.labels[k];
  return l && !(k in V.crit) ? `<button type="button" class="linkbtn crit-btn" data-act="critique" data-key="${esc(k)}" title="Critique (c)" aria-label="Critique">✎</button>` : "";
}
const critiqueRequired = (V, k) => Boolean(V.pendingNo[k] || (V.labels[k] && V.labels[k].answer === "no"));
function critiqueBox(V, k) {
  const need = critiqueRequired(V, k);
  if (k in V.crit) return `<div class="reasonbox"><input type="text" data-crit="${esc(k)}" data-fk="c${esc(k)}" value="${esc(V.crit[k])}" placeholder="${need ? "Critique: what is wrong (required for No)" : "Critique (optional)"}" aria-label="Critique"><button type="button" class="qbtn" data-act="save-critique" data-key="${esc(k)}" ${need && !String(V.crit[k]).trim() ? "disabled" : ""}>${V.pendingNo[k] ? "Save No" : "Save"}</button></div>`;
  const note = V.labels[k] && V.labels[k].note;
  return note ? `<div class="reasonnote">✎ ${esc(note)}</div>` : "";
}
function openCritique(k) {
  const V = view();
  if (!V.labels[k]) return;
  if (!(k in V.crit)) V.crit[k] = V.labels[k].note || "";
  render();
  const inp = document.querySelector(`[data-crit="${CSS.escape(k)}"]`);
  if (inp) inp.focus();
}
// Keyboard review: a cursor walks the rubric (each phase's two items, the peaks, then the analysis
// items once the write-up is open); the number keys answer the item under it.
function cursorKeys(e) {
  const V = view();
  return [...itemKeys(e), ...peaksOf(e).sort((a, b) => b.pct - a.pct).map((p) => peakKey(p.id)), ...(V.unlocked ? ANALYSIS_ITEMS.map((item) => labelKey(item, null)) : [])];
}
const answeredKey = (V, k) => (k.startsWith("P|") ? Boolean(V.peaks[k.slice(2)]) : Boolean(V.labels[k]));
function nextKey(e, from) {
  const ks = cursorKeys(e), V = view(), i = ks.indexOf(from);
  return [...ks.slice(i + 1), ...ks.slice(0, Math.max(i, 0))].find((k) => !answeredKey(V, k)) || ks[Math.min(i + 1, ks.length - 1)];
}
function showCursor() {
  document.querySelectorAll(".cur").forEach((el) => el.classList.remove("cur"));
  const el = S.cur && document.querySelector(`[data-ck="${CSS.escape(S.cur)}"]`);
  if (el) el.classList.add("cur");
  return el;
}
function rubricKey(ev) {
  const e = S.ep;
  if (!e || ev.metaKey || ev.ctrlKey || ev.altKey) return false;
  const V = view(), ks = cursorKeys(e);
  if (!ks.length) return false;
  if (!ks.includes(S.cur)) S.cur = ks.find((k) => !answeredKey(V, k)) || ks[0];
  const k = S.cur;
  if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
    S.cur = ks[(ks.indexOf(k) + (ev.key === "ArrowDown" ? 1 : -1) + ks.length) % ks.length];
    const el = showCursor(); if (el) el.scrollIntoView({ block: "nearest" });
    return true;
  }
  const opts = k.startsWith("P|") ? PEAK_CAUSES : ANSWERS, n = Number(ev.key);
  if (Number.isInteger(n) && n >= 1 && n <= opts.length) {
    const b = document.querySelector(`[data-ck="${CSS.escape(k)}"] button[data-val="${opts[n - 1][0]}"]`);
    if (b && !b.disabled) b.click();
    return true;
  }
  if (ev.key === "c" && !k.startsWith("P|")) { openCritique(k); return true; }
  return false;
}
function errLine(V, k) { return V.errors[k] ? `<div class="err">${esc(V.errors[k])}</div>` : ""; }
function storedReason(V, k) { const l = V.labels[k] || V.peaks[k.slice(2)]; const r = l && (l.reason || l.note); return l && (l.answer === "cannot_assess" || l.cause === "cannot_assess") && r ? `<div class="reasonnote">Reason: ${esc(r)}</div>` : ""; }
const recordingBlocked = () => Boolean(S.permission && !S.permission.participation_and_storage);

// One fixed color per phase, everywhere the phase appears. Okabe-Ito (colorblind-safe) in the order
// of the research brief's phase color spec (overnight/research/scientific-interface.md): close pairs
// apart. Its blue is 3.6:1 on the dark page, enough for lines and swatches but not for text, so phase
// names stay in the page ink beside a swatch. The answer's phases take the first colors in answer
// order; a phase outside the answer (a refit's added phase) takes the next free one.
const PHASE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"];
const extraPhases = [];
const phaseOrder = (e) => (e ? [...new Set([...phasesOf(e).map((p) => p.id), ...Object.keys((e.fit && e.fit.phase_curves) || {})])] : []);
function phaseColor(cid, e = S.ep) {
  const order = phaseOrder(e);
  let i = order.indexOf(cid);
  if (i < 0) { if (!extraPhases.includes(cid)) extraPhases.push(cid); i = order.length + extraPhases.indexOf(cid); }
  // Past the palette a phase is neutral grey: a wrapped color would make two phases look like one.
  return i < PHASE_COLORS.length ? PHASE_COLORS[i] : "#8a8a8a";
}
const swatch = (cid, e) => `<i class="sw" style="background:${phaseColor(cid, e)}"></i>`;
// Peak positions for a phase's tick row: the local maxima of its calculated contribution, as the
// server picks phase windows, above 0.5% of that phase's strongest point.
function phasePeaks(xs, ys) {
  const v = ys.map((y) => Number(y) || 0), top = Math.max(...v);
  const out = [];
  for (let i = 1; i < v.length - 1; i++) if (v[i] > 0.005 * top && v[i] > v[i - 1] && v[i] >= v[i + 1]) out.push(xs[i]);
  return out;
}
// The residual-peak marker, a shape no phase uses: filled at or above the 5% level, hollow below.
const peakMark = (x, y, p, size = 6) => `<path d="M${(x - size).toFixed(1)},${(y - size * 1.6).toFixed(1)} L${(x + size).toFixed(1)},${(y - size * 1.6).toFixed(1)} L${x.toFixed(1)},${y.toFixed(1)} Z" fill="${p.pct >= 5 ? "var(--obs)" : "none"}" stroke="var(--obs)" stroke-width="1.4"/>`;
// Intensity scales GSAS-II and WinPLOTR both offer. The difference strip stays linear.
const YSCALES = [["lin", "lin"], ["sqrt", "√"], ["log", "log"]];
const yScaled = (v) => (S.yscale === "sqrt" ? Math.sign(v) * Math.sqrt(Math.abs(v)) : S.yscale === "log" ? Math.log10(Math.max(v, 1)) : v);
function gridStep(span) { return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20].find((s) => span / s <= 10) || 20; }

function plot(e, interactive = true) {
  const f = e.fit;
  if (!f || !f.two_theta || !f.two_theta.length) return `<p class="muted" style="padding:14px">No saved fit curve.</p>`;
  const order = phaseOrder(S.ep || e), rank = (cid) => (order.includes(cid) ? order.indexOf(cid) : order.length);
  const curves = Object.entries(f.phase_curves || {}).filter(([, ys]) => Array.isArray(ys) && ys.length === f.two_theta.length).sort(([a], [b]) => rank(a) - rank(b));
  const full = [f.two_theta[0], f.two_theta[f.two_theta.length - 1]];
  const [x0, x1] = S.range && S.range[1] - S.range[0] > 0.2 ? [Math.max(full[0], S.range[0]), Math.min(full[1], S.range[1])] : full;
  const idx = f.two_theta.map((t, i) => i).filter((i) => f.two_theta[i] >= x0 && f.two_theta[i] <= x1);
  const tt = idx.map((i) => f.two_theta[i]);
  const rows = curves.map(([cid, ys]) => [cid, phasePeaks(f.two_theta, ys).filter((t) => t >= x0 && t <= x1)]);
  // The left margin fits the longest tick-row formula (12.5 px mono, about 7.6 px a character).
  const L = Math.max(76, 14 + 7.6 * Math.max(0, ...rows.map(([cid]) => String(formulaOf(e, cid)).length)));
  const W = 1400, R = 12, T = 22, mainB = 300, tTop = mainB + 12, rowH = 13, dTop = tTop + rows.length * rowH + 10, dB = dTop + 70, H = dB + 24;
  // calculated = background + the sum of the phase contributions, so each phase sits on the background.
  const bg = curves.length ? idx.map((i) => f.calculated[i] - curves.reduce((sum, [, ys]) => sum + (Number(ys[i]) || 0), 0)) : null;
  const stacked = curves.map(([cid, ys]) => [cid, idx.map((i, k) => (Number(ys[i]) || 0) + bg[k])]);
  const obs = idx.map((i) => f.observed[i]), calc = idx.map((i) => f.calculated[i]), diff = idx.map((i) => f.difference[i]);
  const ys = [...obs, ...calc, ...(bg || [])].map(yScaled);
  const ymin = Math.min(...ys), ymax = Math.max(...ys), dmax = Math.max(...diff.map(Math.abs)) || 1;
  const X = (t) => L + (t - x0) / (x1 - x0) * (W - L - R);
  const Y = (v) => T + (1 - (yScaled(v) - ymin) / (ymax - ymin || 1)) * (mainB - T);
  const DY = (v) => (dTop + dB) / 2 - v / dmax * (dB - dTop) / 2;
  const line = (vs, F) => "M" + tt.map((t, k) => `${X(t).toFixed(1)},${F(vs[k]).toFixed(1)}`).join(" L");
  const dots = tt.map((t, k) => `<rect x="${(X(t) - 0.9).toFixed(1)}" y="${(Y(obs[k]) - 0.9).toFixed(1)}" width="1.8" height="1.8" fill="var(--obs)"/>`).join("");
  const V = view(), col = (cid) => phaseColor(cid, S.ep || e), dim = (cid) => S.phaseSel && S.phaseSel !== cid;
  const phaseLines = stacked.map(([cid, vs]) => `<path d="${line(vs, Y)}" fill="none" stroke="${col(cid)}" stroke-width="${S.phaseSel === cid ? 2 : 1.1}" stroke-opacity="${dim(cid) ? 0.3 : 0.9}" vector-effect="non-scaling-stroke"/>`).join("");
  const tickRows = rows.map(([cid, at], j) => {
    const y = tTop + j * rowH;
    return `<text class="axis tickname" x="${L - 8}" y="${y + 10}" text-anchor="end">${esc(formulaOf(e, cid))}</text>` +
      (at.length ? `<path d="${at.map((t) => `M${X(t).toFixed(1)},${y} V${y + 10}`).join(" ")}" stroke="${col(cid)}" stroke-width="2" stroke-opacity="${dim(cid) ? 0.3 : 1}"/>` : "");
  }).join("");
  const marks = peaksOf(e).filter((p) => p.tt >= x0 && p.tt <= x1).sort((a, b) => a.tt - b.tt).map((p) => {
    const sel = S.peakSel === p.id, x = X(p.tt);
    const k = tt.reduce((best, t, j) => (Math.abs(t - p.tt) < Math.abs(tt[best] - p.tt) ? j : best), 0);
    const y = Math.max(T + 12, Y(Math.max(obs[k], calc[k])) - 6);
    const w = Math.max(16, X(p.tt + 0.35) - X(p.tt - 0.35));
    const body = `<rect x="${(x - w / 2).toFixed(1)}" y="${T - 8}" width="${w.toFixed(1)}" height="${dB - T + 8}" fill="${sel ? "rgba(255,255,255,0.08)" : "transparent"}"/>
      <line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${dTop}" y2="${dB}" stroke="var(--obs)" stroke-opacity=".35" stroke-dasharray="2 3"/>
      ${peakMark(x, y, p, sel ? 7 : 6)}<text class="axis" x="${(x + 9).toFixed(1)}" y="${(y - 4).toFixed(1)}" style="fill:var(--obs)">${fmt(p.tt, 2)}°</text>`;
    return interactive ? `<g class="peak-box" data-peak="${esc(p.id)}" role="button" tabindex="0" aria-label="Unexplained peak at ${fmt(p.tt)} degrees${V.peaks[p.id] ? ", called" : ""}"><title>Unexplained peak, ${fmt(p.pct, 1)}% of the strongest net peak. Click to call it.</title>${body}</g>` : `<g>${body}</g>`;
  }).join("");
  const pc = S.phaseSel && f.phase_curves && f.phase_curves[S.phaseSel];
  const contrib = pc ? `<clipPath id="band"><rect x="${L}" y="${dTop}" width="${W - L - R}" height="${dB - dTop}"/></clipPath><path class="phase-curve" d="${line(idx.map((i) => pc[i]), DY)}" fill="none" stroke="${col(S.phaseSel)}" stroke-width="1.4" clip-path="url(#band)" vector-effect="non-scaling-stroke"/>` : "";
  const step = gridStep(x1 - x0), ticks = [];
  for (let t = Math.ceil(x0 / step) * step; t <= x1 + 1e-9; t += step) ticks.push(t);
  const ylab = S.yscale === "sqrt" ? "√I" : S.yscale === "log" ? "log₁₀ I" : "I (counts)";
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" data-zoom="${x0},${x1},${L},${W - R},${W}" aria-label="Observed points, calculated pattern, phase contributions and peak positions, and the difference curve">
    <text class="axis" x="${L}" y="${T - 8}">${ylab}</text><text class="axis" x="${L - 8}" y="${(dTop + dB) / 2 + 4}" text-anchor="end">Δ</text>
    ${phaseLines}
    <path d="${line(calc, Y)}" fill="none" stroke="var(--calc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
    ${dots}${tickRows}
    <line x1="${L}" x2="${W - R}" y1="${DY(0).toFixed(1)}" y2="${DY(0).toFixed(1)}" stroke="var(--line-strong)"/>
    <path d="${line(diff, DY)}" fill="none" stroke="var(--calc)" stroke-width="1" vector-effect="non-scaling-stroke"/>
    ${contrib}${marks}
    ${ticks.map((t) => `<text class="axis" x="${X(t).toFixed(1)}" y="${H - 6}" text-anchor="middle">${+t.toFixed(1)}</text>`).join("")}
    <text class="axis" x="${L - 8}" y="${H - 6}" text-anchor="end">2θ (°)</text>
    <rect class="zoom-sel" x="0" y="${T - 8}" width="0" height="${dB - T + 8}"/>
  </svg>`;
}
// The legend and view controls under a plot: what each mark is, so no sentence has to say it.
function plotLegend(e, extra = "") {
  const f = e.fit || {}, hasPhases = Object.keys(f.phase_curves || {}).length;
  const scale = YSCALES.map(([v, l]) => `<button type="button" data-act="yscale" data-val="${v}" aria-pressed="${(S.yscale || "lin") === v}">${l}</button>`).join("");
  return `<div class="legend"><span><i class="sw obs"></i>observed</span><span><i class="sw calc"></i>calculated</span>${hasPhases ? "<span><i class=\"sw ticks\"></i>phase peaks</span>" : ""}<span><i class="sw diff"></i>difference</span>${peaksOf(e).length ? '<span title="Filled: at or above 5% of the strongest net peak. Hollow: below.">▼ unexplained peak</span>' : ""}${extra}
    <span class="ctl"><span class="seg mini" role="group" aria-label="Intensity scale">${scale}</span>${S.range ? '<button type="button" class="qbtn small" data-act="zoom-reset">Full 2θ range</button>' : '<span class="muted">drag across the plot to zoom</span>'}</span></div>`;
}

// The fit at native resolution around the selected residual peak (W3 residual_windows: every
// measured point within 0.5 deg, background and per-phase contributions from the .dia). The
// strided overview can flatten a narrow feature; this window does not. calculated = background +
// the sum of the phase contributions (measured on DEV006 within 0.02 counts), so each phase is
// drawn on top of the background.
function phaseWindows(e, cid) {
  return [...(e.phase_windows || []), ...(e.residual_windows || [])].filter((x) => x.phase_id === cid);
}
function zoomPlot(e) {
  // A selected residual peak shows its window; otherwise "Show its peaks" shows the phase's own
  // windows (phase_windows: native rows around that phase's strongest calculated reflections, the
  // rule named by the server), stepped one at a time.
  const pw = !S.peakSel && S.phaseSel ? phaseWindows(e, S.phaseSel) : [];
  const w = S.peakSel ? (e.residual_windows || []).find((x) => x.peak_id === S.peakSel) : pw[Math.min(S.phaseWin || 0, pw.length - 1)];
  if (!w || !w.two_theta_deg || w.two_theta_deg.length < 2) return "";
  const W = 1400, H = 280, L = 76, R = 12, T = 14, mainB = 196, dTop = 210, dB = 262;
  const t = w.two_theta_deg, x0 = t[0], x1 = t[t.length - 1];
  const curves = Object.entries(w.phase_curves || {});
  const bg = w.background || null;
  const stacked = curves.map(([cid, ys]) => [cid, ys.map((v, i) => v + (bg ? bg[i] : 0))]);
  const all = [...w.observed, ...w.calculated, ...(bg || []), ...stacked.flatMap(([, ys]) => ys)];
  const ymin = Math.min(...all), ymax = Math.max(...all);
  const dmax = Math.max(...w.difference.map(Math.abs)) || 1;
  const X = (v) => L + (v - x0) / (x1 - x0 || 1) * (W - L - R);
  const Y = (v) => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (mainB - T);
  const DY = (v) => (dTop + dB) / 2 - v / dmax * (dB - dTop) / 2;
  const path = (ys, F) => "M" + t.map((v, i) => `${X(v).toFixed(1)},${F(ys[i]).toFixed(1)}`).join(" L");
  const col = (cid) => phaseColor(cid, S.ep || e);
  const dots = t.map((v, i) => `<circle cx="${X(v).toFixed(1)}" cy="${Y(w.observed[i]).toFixed(1)}" r="2" fill="var(--obs)"/>`).join("");
  const lines = stacked.map(([cid, ys]) => `<path d="${path(ys, Y)}" fill="none" stroke="${col(cid)}" stroke-width="${S.phaseSel === cid ? 2 : 1.1}" stroke-opacity="${S.phaseSel && S.phaseSel !== cid ? 0.45 : 0.95}" vector-effect="non-scaling-stroke"/>`).join("");
  const legend = stacked.map(([cid]) => `<span>${swatch(cid, S.ep || e)}${esc(formulaOf(e, cid))}</span>`).join("");
  const ticks = []; for (let v = Math.ceil(x0 * 10) / 10; v <= x1 + 1e-9; v += 0.1) ticks.push(v);
  const center = w.center_deg ?? (S.peakSel ? Number(S.peakSel.replace(/^p/, "")) : null) ?? (x0 + x1) / 2;
  const what = S.peakSel ? "Unexplained peak" : `${swatch(S.phaseSel, S.ep || e)}${esc(formulaOf(e, S.phaseSel))} peak ${Math.min(S.phaseWin || 0, pw.length - 1) + 1} of ${pw.length}`;
  const step = !S.peakSel && pw.length > 1 ? ` <button type="button" class="qbtn small" data-act="phase-win" data-d="-1">Previous</button> <button type="button" class="qbtn small" data-act="phase-win" data-d="1">Next</button>` : "";
  const rule = !S.peakSel ? (w.selection || e.phase_window_rule || "") : "";
  return `<div class="zoom" id="zoom"><div class="zoom-h"><b>${what}</b> at ${fmt(center, 2)}° · every measured point within ±${fmt((x1 - x0) / 2, 1)}° (${t.length})${bg ? "" : " · no background or phase curves stored"}${rule ? ` <span class="info" tabindex="0" title="Window rule: ${esc(rule)}">window rule</span>` : ""}${step}</div>
    <svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="Native-resolution fit around ${S.peakSel ? "the selected residual peak" : "a calculated peak of the selected phase"}">
      ${bg ? `<path d="${path(bg, Y)}" fill="none" stroke="var(--bgline)" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"/>` : ""}
      ${lines}
      <path d="${path(w.calculated, Y)}" fill="none" stroke="var(--calc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
      ${dots}
      <path d="${path(w.difference, DY)}" fill="none" stroke="var(--calc)" vector-effect="non-scaling-stroke"/>
      <line x1="${L}" x2="${W - R}" y1="${DY(0).toFixed(1)}" y2="${DY(0).toFixed(1)}" stroke="var(--line-strong)"/>
      ${ticks.map((v) => `<text class="axis" x="${X(v).toFixed(1)}" y="${H - 2}" text-anchor="middle">${v.toFixed(1)}°</text>`).join("")}
    </svg>
    <div class="legend"><span><i class="sw obs"></i>observed</span><span><i class="sw calc"></i>calculated</span>${bg ? `<span><i class="sw bg"></i>background</span>${legend}` : ""}<span><i class="sw diff"></i>difference</span></div></div>`;
}

function markControl(ref) {
  if (!ref) return "";
  const V = view(), k = stepKey(ref.attempt, ref.call_index);
  S.steps[k] = ref;
  const chips = (V.marks[k] || []).map((m) => `<span class="chip mark" title="${esc(m.note)}">${MARK_CHIP[m.mark] || esc(m.mark)}${m.note ? " ✎" : ""}</span>`).join("");
  const open = S.markOpen === k;
  const d = S.markDraft || { mark: null, note: "" };
  const box = open ? `<div class="markbox" role="group" aria-label="Feedback on this step">
      <div class="row">${seg(`data-act="mark-pick"`, null, MARKS, { draft: d.mark })}</div>
      <textarea data-mark-note="1" data-fk="m${esc(k)}" placeholder="${d.mark === "note" ? "Your note (required)" : "Note (optional)"}">${esc(d.note)}</textarea>
      <div class="row"><button type="button" class="qbtn" data-act="mark-save" data-step="${esc(k)}" ${d.mark && (d.mark !== "note" || d.note.trim()) && !V.saving[k] ? "" : "disabled"}>Save mark</button><button type="button" class="qbtn" data-act="mark-cancel">Cancel</button>${errLine(V, k)}</div>
    </div>` : "";
  return `<div class="marks">${chips}<button type="button" class="qbtn small" data-act="mark-open" data-step="${esc(k)}" aria-expanded="${open}">${open ? "Marking" : "Mark step"}</button></div>${box}`;
}
function stepRef(e, s) {
  const kind = s.step_kind || (s.kind === "refine" ? "refine" : s.kind === "segment" ? "segment" : s.kind === "answer" ? "tool_call" : null);
  if (!kind || ("attempt" in s && !s.attempt)) return null;
  // labels.StepRef: an imported step is keyed by its first call index; only an attempt step is the attempt itself.
  if (kind !== "attempt" && s.call_index == null) return null;
  const label = s.kind === "refine" ? s.run_label : s.kind === "answer" ? "Wrote an answer" : s.title;
  return { attempt: s.attempt || e.origin, call_index: kind === "attempt" ? null : s.call_index, step_kind: kind, label: String(label || "").slice(0, 300) };
}

// What the agent did, in one line, from the blind payload's own steps: the candidate screen, the
// refinements, the fit it cited and whether it submitted. The payload names no model, harness or
// episode, so neither does this line (PI 19:47 UTC: viewers have never seen the agent).
function agentLine(e) {
  const steps = e.steps || [];
  if (!steps.length) return `<div class="agentline" id="agent-line"><span class="lbl">What the agent did</span> <span class="muted">its steps are not recorded in this snapshot.</span></div>`;
  const screens = steps.filter((s) => s.kind === "segment" && /screened candidate phases/i.test(s.title || "")).length;
  const runs = refines(e), cited = citedOf(e);
  const rwps = runs.map((r) => r.rwp).filter((x) => x != null);
  const answers = steps.filter((s) => s.kind === "answer").length;
  const end = Math.max(...steps.map((s) => Number(s.t_end ?? s.t ?? 0)));
  const parts = [
    screens ? `screened the candidate library${screens > 1 ? ` in ${screens} steps` : ""}` : "no candidate screen identified in the recorded steps",
    `ran ${runs.length} refinement${runs.length === 1 ? "" : "s"}${rwps.length > 1 ? ` (Rwp ${fmt(Math.min(...rwps))} to ${fmt(Math.max(...rwps))}%)` : ""}`,
    cited.label ? `cited <span class="mono">${esc(cited.label)}</span>${cited.rwp != null ? ` (Rwp ${fmt(cited.rwp)}%)` : ""}` : "cited no fit",
    isAnswered(e) ? `submitted an answer${answers > 1 ? ` (written ${answers} times)` : ""}` : "submitted no answer",
  ];
  return `<div class="agentline" id="agent-line"><span class="lbl">What the agent did</span> ${parts.join(", ")} · ${steps.length} recorded steps over ${clock(end)}.${info("agent", "An AI agent analyzed this pattern with search and refinement tools. You judge its answer; which agent ran is hidden until you sign off.")}</div>`;
}
// One plain line per check and rubric term, behind a small toggle that stays open across renders.
const CHECK_PLAIN = {
  K1: "Compares the structure-based Rietveld fit with the simpler Pawley fit of the same phases; a much worse Rietveld fit hints at a wrong structure or phase.",
  K2: "Finds peaks in the measured pattern that no fitted phase produces; a large one can mean a phase is missing, but it is a prompt to look, not proof.",
  K3: "Measures how far each phase's refined cell edges moved from its CIF; a big move can mean a different compound. Advisory only.",
  K5: "Confirms the weight fractions in the answer are the ones the cited refinement produced.",
};
const RUBRIC_PLAIN = {
  phase_supported: "Yes if the measured peaks show this phase's pattern clearly enough to say it is in the sample.",
  phase_plausible: "Yes if this phase could exist in this sample, given the precursors and the processing.",
  amounts_defensible: "Yes if the weight fractions are believable for how they were computed: crystalline phases only, normalized to 100.",
  fit_supports_conclusion: "Yes if the refinement, its residual and the checks back the agent's final phase list.",
};
const FIT_PLAIN = "Rwp is the weighted mismatch between the measured and calculated patterns; lower is closer. GoF divides it by the lowest Rwp the data allow; near 1 is ideal.";
const PEAKS_PLAIN = "Call each listed peak: a missing phase, already fitted (it belongs to a phase in the fit), or noise or profile.";
function info(key, text) {
  return `<details class="info" data-dk="info-${esc(key)}"><summary aria-label="What this means" title="What this means">i</summary><div role="note">${esc(text)}</div></details>`;
}
function evidence(e) {
  const f = e.fit || {}, V = view(), ph = phasesOf(e), cited = citedOf(e);
  const legend = ph.map((p) => `<span>${swatch(p.id)}${esc(p.formula || p.id)}${p.wt != null ? ` ${fmt(p.wt, 1)}` : ""}</span>`).join("");
  const strip = refines(e).map((s) => `<div class="card ${cited.label && s.run_label === cited.label ? "cited" : ""}" data-step-card="${s.i}">
      ${cited.label && s.run_label === cited.label ? '<span class="tag">cited by the answer</span>' : ""}
      <div class="rwp">Rwp ${fmt(s.rwp)}</div>
      <div class="ph">${(s.phases || []).map((p) => `${swatch(p.candidate_id || p)}${esc(p.formula || p.candidate_id || p)} <span class="muted">${esc(sgShort(p.space_group))}</span>`).join(" + ")}</div>
      <div class="lab">${esc(s.run_label || "")} · ${clock(s.t_start)}${s.from_script ? " · from a script" : ""}</div>
      ${markControl(stepRef(e, s))}</div>`).join("");
  const mineCards = V.refits.map((r) => {
    const head = `<span class="tag">your refit · ${esc(editText(e, r.edit))}</span>`;
    if (r.status === "running") return `<div class="card mine running">${head}<div class="rwp">Running <span data-elapsed="${r.started}">${clock((Date.now() - r.started) / 1000)}</span></div><div class="lab">BGMN${e.case_id === "DEV006" ? ", usually 1 to 2.5 min here" : ""}. Keep labeling.</div></div>`;
    if (r.status === "failed") return `<div class="card mine failed">${head}<div class="rwp">Refit failed</div><div class="lab">${esc(r.error)}</div><div class="lab">The answer is unchanged.</div></div>`;
    const c = r.child, ref = { attempt: c.digest, call_index: null, step_kind: "attempt", label: `Expert refit: ${editText(e, r.edit)}`.slice(0, 300) };
    const delta = c.rwp != null && cited.rwp != null ? ` <span class="muted">(${c.rwp - cited.rwp >= 0 ? "+" : ""}${fmt(c.rwp - cited.rwp)})</span>` : "";
    return `<div class="card mine" data-child="${esc(c.digest)}">${head}<div class="rwp">Rwp ${fmt(c.rwp)}${delta}</div>
      <div class="ph">${c.phases.map((p) => `${swatch(p)}${esc(formulaOf(e, p))}`).join(" + ")}</div>
      <div class="lab">${esc(c.stage || "")} · ${clock(r.elapsed)} · child of this answer</div>
      ${c.review && c.review.fit ? `<button type="button" class="qbtn small" data-act="child-fit" data-child="${esc(c.digest)}" aria-pressed="${S.childFit === c.digest}">${S.childFit === c.digest ? "Back to the answer's fit" : "Show its fit"}</button>` : ""}
      ${r.snapshot ? `<button type="button" class="qbtn small" data-act="open-snapshot" data-digest="${esc(r.snapshot)}">Open its labels</button>` : `<button type="button" class="qbtn small" data-act="label-correction" data-refit="${esc(r.id)}" ${r.saving ? "disabled" : ""}>Label this correction</button>`}
      ${r.error ? `<div class="err">${esc(r.error)}</div>` : ""}
      ${markControl(ref)}</div>`;
  }).join("");
  const shown = S.childFit && (V.refits.find((r) => r.child && r.child.digest === S.childFit) || {}).child;
  if (shown && shown.review) {
    const cf = shown.review.fit || {};
    return `<section class="term"><div class="term-bar"><span class="term-t" style="color:var(--fg)">Your refit (child attempt) · ${esc(cf.stage || "")} · Rwp ${fmt(cf.rwp)}% · GoF ${fmt(cf.gof)}</span><span class="term-legend">${phasesOf(shown.review).map((p) => `<span>${swatch(p.id)}${esc(p.formula || p.id)}${p.wt != null ? ` ${fmt(p.wt, 1)}` : ""}</span>`).join("")}</span></div>
      <div class="term-body">${plot(shown.review, false)}${plotLegend(shown.review)}</div>
      <div class="term-foot">Markers here are this refit's residual check. Your peak calls stay on the answer. <button type="button" class="qbtn small" data-act="child-fit" data-child="${esc(shown.digest)}">Back to the answer's fit</button>
      ${mineCards ? `<div style="margin-top:10px" id="refits"><div class="strip"><div class="card cited"><span class="tag">parent · this answer</span><div class="rwp">Rwp ${fmt(cited.rwp)}</div><div class="lab">${esc(cited.label || "cited fit")} · unchanged</div></div><span class="sep">→</span>${mineCards}</div></div>` : ""}</div></section>`;
  }
  const phaseNote = S.phaseSel && f.phase_curves && f.phase_curves[S.phaseSel] ? `<span title="Removing that phase without a refit would add this much to the residual.">${swatch(S.phaseSel)}${esc(formulaOf(e, S.phaseSel))} on the difference scale</span>` : "";
  return `<section class="term">
    <div class="term-bar"><span class="term-t">${isAnswered(e) ? "Cited fit" : "Best saved fit, no answer"} · ${esc(f.stage || e.stage || "")} · Rwp ${fmt(f.rwp ?? e.rwp_percent)}% · GoF ${fmt(f.gof ?? e.gof)}</span>${info("fit", FIT_PLAIN)}<span class="term-legend">${legend}${ph.some((p) => p.wt != null) ? " wt%" : ""}</span></div>
    <div class="term-body">${plot(e)}${plotLegend(e, phaseNote)}${zoomPlot(e)}</div>
    <div class="term-foot">
      <div><span class="muted">Agent refinements, in order</span>
      <div class="strip">${strip || '<span class="muted">No refinement finished.</span>'}</div></div>
      ${mineCards ? `<div style="margin-top:10px" id="refits"><span class="muted">Your refits (child attempts)</span>
      <div class="strip"><div class="card cited"><span class="tag">parent · this answer</span><div class="rwp">Rwp ${fmt(cited.rwp)}</div><div class="ph">${ph.map((p) => esc(p.formula || p.id)).join(" + ")}</div><div class="lab">${esc(cited.label || "cited fit")} · unchanged</div></div><span class="sep">→</span>${mineCards}</div></div>` : ""}
    </div>
  </section>`;
}

// How a check reads on the page and in the report (dogfood A4, F2): a passing K1 names what passed,
// and K2 never reads "pass" while it lists a weak peak for review.
function checkView(c, e) {
  if (c.code === "K1" && c.status === "pass") return { name: "Rietveld did not degrade after Pawley", label: "no degradation", cls: "pass" };
  if (c.code === "K2" && c.status !== "flag" && peaksOf(e).length) return { name: c.name, label: "weak peak listed for review (below the 5% flag)", cls: "hold" };
  const label = { pass: "pass", flag: "check", info: "note" }[c.status] || "n/a";
  return { name: c.name, label, cls: { pass: "pass", flag: "stop", info: "hold" }[c.status] || "none" };
}
function peakPanel(e) {
  const V = view(), peaks = peaksOf(e).sort((a, b) => b.pct - a.pct);
  const blocked = recordingBlocked();
  const rows = peaks.map((p) => {
    const k = peakKey(p.id), cur = V.peaks[p.id] && V.peaks[p.id].cause;
    return `<div class="peakrow ${S.peakSel === p.id ? "sel" : ""}" data-peakrow="${esc(p.id)}" data-ck="${esc(k)}">
      <span class="chip peak">${p.pct >= 5 ? "▼" : "▽"} ${fmt(p.tt, 2)}°</span><span title="${p.pct >= 5 ? "At or above the 5% level the checkpoint study flagged." : "Below the 5% level the checkpoint study flagged: a minor phase or a profile mismatch."}">${fmt(p.pct, 1)}% of the strongest net peak${p.counts != null ? ` · ${fmt(p.counts, 0)} counts` : ""}</span>
      ${seg(`data-act="peak" data-peak="${esc(p.id)}"`, cur, PEAK_CAUSES, { draft: k in V.drafts ? "cannot_assess" : null, saving: V.saving[k], disabled: blocked })}
      ${reasonBox(V, k, `for the peak at ${fmt(p.tt, 2)} degrees`)}${storedReason(V, k)}${errLine(V, k)}
    </div>`;
  }).join("");
  const r = residualCheck(e);
  const others = (e.checks || []).filter((c) => c !== r);
  return `<section class="panel">
    <h3>Unexplained peaks${info("peaks", PEAKS_PLAIN)} <span class="hint">${peaks.length ? `${peaks.filter((p) => V.peaks[p.id]).length} of ${peaks.length} called` : "none listed"}</span></h3>
    <div class="body">${rows || `<p class="muted" style="margin:0">The residual check (K2) lists none.</p>`}
      ${r ? `<details data-dk="k2"><summary>How the residual check works</summary><div class="plain">${esc(CHECK_PLAIN.K2)}</div><span class="muted" style="font-size:12px">${esc(r.detail)} A residual peak is a prompt to look, not proof of a missing phase.</span></details>` : ""}</div>
    <h3>Other checks</h3>
    <div class="body">${others.map((c) => `<details class="check" data-dk="check-${esc(c.code || c.name)}" ${c.status === "flag" ? "open" : ""}><summary>${((v) => `<span class="chip ${v.cls}">${esc(v.label.toUpperCase())}</span><span>${esc(v.name)}</span>`)(checkView(c, e))}</summary>${CHECK_PLAIN[c.code] ? `<div class="plain">${esc(CHECK_PLAIN[c.code])}</div>` : ""}<div class="why">${esc(c.detail)}</div></details>`).join("") || '<p class="muted" style="margin:0">No other checks.</p>'}</div>
  </section>`;
}

function chemistryPanel(e) {
  const c = e.chemistry;
  const head = `<h3>Chemistry <span class="hint">evidence for "plausible", never a gate</span></h3>`;
  if (!c) return `<section class="panel" id="chemistry">${head}<div class="body muted">Not available for this snapshot.</div>${compositionPanel(e)}</section>`;
  const states = (a) => Object.entries(a || {}).map(([el, v]) => `${el} ${v > 0 ? "+" : ""}${v}`).join(", ");
  const eb = c.element_balance;
  // Missing or unparseable precursor context is not a comparison, and a snapshot frozen before
  // element_balance existed has no recorded check: neither is inferred (Science, AXA-43zwk.20).
  const recorded = eb != null;
  const comparable = recorded && eb.status === "compared";
  const CB = { balanced: "formal assignment found", no_assignment_found: "no formal assignment under the states tried", unbalanced: "no formal assignment under the states tried", not_computable: "not computable" };
  const tip = (parts) => esc(parts.filter(Boolean).join("\n"));
  const rows = (c.phases || []).map((p) => {
    const cb = p.charge_balance || {}, h = p.hull || {};
    const said = CB[cb.status] || String(cb.status || "").replace(/_/g, " ");
    const assigned = cb.assignment && Object.keys(cb.assignment).length ? states(cb.assignment) : "";
    const matches = (h.formula_matches || []).map((m) => `${m.entry_id} · ${m.formula} · SG ${m.space_group_number} · ${m.e_above_hull_ev_per_atom != null ? `${fmt(m.e_above_hull_ev_per_atom, 3)} eV/atom` : "no value"}`);
    const hullTip = tip([h.status === "found" ? "" : "No value for this candidate's code.", [h.entry_id, h.space_group_number != null ? `SG ${h.space_group_number}` : "", h.source].filter(Boolean).join(" · "), h.detail, matches.length ? "Same formula in the table (context, not this phase's value):" : "", ...matches]);
    const outside = !recorded ? '<span class="muted" title="check not recorded">not recorded</span>' : !comparable || p.elements_not_in_precursors == null ? '<span class="muted">not computable</span>' : p.elements_not_in_precursors.length ? esc(p.elements_not_in_precursors.join(", ")) : "none";
    return `<tr><td>${swatch(p.phase_id)}<b>${esc(formulaOf(e, p.phase_id) || p.formula)}</b></td>
      <td title="${tip([said, cb.policy])}">${assigned && cb.status === "balanced" ? esc(assigned) : esc(said)}</td>
      <td class="num" title="${hullTip}">${h.status === "found" ? fmt(h.e_above_hull_ev_per_atom, 3) : '<span class="muted">n/a</span>'}</td>
      <td>${outside}</td><td>${identityText(p.identity)}</td></tr>`;
  }).join("");
  const un = c.precursor_elements_unaccounted;
  const balanceLine = !recorded ? "element comparison not recorded for this snapshot."
    : !comparable || un == null
    ? `element comparison not computable${eb.reason ? ` (${esc(eb.reason)})` : ""}.`
    : un.length ? `no answered phase contains ${esc(un.join(", "))}.` : "every precursor element is in an answered phase.";
  const basis = ((c.phases || []).find((p) => p.hull && p.hull.basis) || {}).hull;
  // The hull value is metadata associated by formula and space group, not this structure's computed
  // stability: the default view says so and names its source behind a focusable disclosure (CR13).
  const sources = (c.phases || []).map((p) => { const h = p.hull || {}; return `<li>${swatch(p.phase_id)}<b>${esc(formulaOf(e, p.phase_id) || p.formula)}</b> ${h.status === "found" ? `${fmt(h.e_above_hull_ev_per_atom, 3)} eV/atom` : "no value for this candidate's code"}${h.entry_id ? ` · ${esc(h.entry_id)}` : ""}${h.space_group_number != null ? ` · SG ${esc(h.space_group_number)}` : ""}${h.source ? ` · ${esc(h.source)}` : ""}${h.detail ? ` · ${esc(h.detail)}` : ""}${(h.formula_matches || []).length ? `<div class="muted">Same formula in the table (context, not this phase's value): ${(h.formula_matches || []).map((m) => `${esc(m.entry_id)} ${esc(m.formula)} SG ${esc(m.space_group_number)} ${m.e_above_hull_ev_per_atom != null ? `${fmt(m.e_above_hull_ev_per_atom, 3)} eV/atom` : "no value"}`).join("; ")}</div>` : ""}</li>`; }).join("");
  return `<section class="panel" id="chemistry">${head}${rows ? `<div style="overflow-x:auto"><table class="t chemt"><thead><tr><th>Phase</th><th>Charge balance</th><th class="num">Approx. hull (eV/atom)</th><th>Outside precursors</th><th>Formula vs CIF</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}<div class="body">${rows ? "" : '<p class="muted" style="margin:0">No answered phase to assess.</p>'}
    ${rows ? `<p class="muted hullq" id="hull-qualifier" style="font-size:12px;margin:0 0 6px">Approx. hull: a Materials Project value matched by formula and space group, not this structure's computed stability.</p>
    <details data-dk="hull-sources"><summary>Hull sources</summary><ul class="srcs">${sources}</ul></details>` : ""}
    <p class="muted" style="font-size:12px;margin:0">Precursors ${esc((c.precursors || []).join(", ") || "none declared")}: ${balanceLine}</p>
    <details data-dk="chem-notes"><summary>About these values</summary><p class="muted" style="font-size:12px;margin:4px 0 0">Hull distance: ${basis ? esc(basis.basis) : "an approximate Materials Project value that Dara's metadata associates with this formula and space group under the candidate's database code, not a computed stability of this structure"}. A charge balance is a formal oxidation-state assignment, not a measured valence. An element outside the precursors is a question for you (C and O exchange with air), not a violation. Hover a cell for its source.</p></details></div>${compositionPanel(e)}</section>`;
}

// Typed composition (XRF, ICP or EDS) beside the chemistry evidence (plan U14, N9). The page never
// prefills or invents a measurement; with none stored it shows only the composition the answer
// implies, which the server computes from the phase formulas and weight fractions.
function compositionPanel(e) {
  if (!ROUTES.impliedComposition) return "";
  const V = view(), c = V.composition;
  if (!c) return "";
  const pct = (x) => (x == null ? "n/a" : fmt(Number(x) * 100, 1));
  const im = c.implied;
  // W5 ImpliedComposition: percentages only when status is computed; otherwise the reason and
  // the named phases that lacked a weight fraction or a readable formula (Science 10:27 UTC).
  const missingInputs = im ? [
    (im.phases_without_weight || []).length ? `No weight fraction: ${esc(im.phases_without_weight.join(", "))}.` : "",
    (im.phases_without_formula || []).length ? `No readable formula: ${esc(im.phases_without_formula.join(", "))}.` : "",
    (im.excluded || []).length ? `Left out for want of a formula or weight fraction: ${esc(im.excluded.join(", "))}.` : "",
  ].filter(Boolean).join(" ") : "";
  const computed = im && (im.status ? im.status === "computed" : Boolean(im.weight_fractions || im.atomic_fractions));
  const implied = !im
    ? `<p class="muted" style="margin:0">${esc(c.impliedError || "The server returned no implied composition for this answer.")}</p>`
    : !computed
      ? `<p class="muted" style="margin:0" id="implied-unavailable">Unavailable: ${esc(im.reason || "not computable")}. ${missingInputs}</p>`
      : (() => {
        const w = im.weight_fractions || {}, a = im.atomic_fractions || {};
        const els = [...new Set([...Object.keys(w), ...Object.keys(a)])];
        return `<table class="t"><thead><tr><th>Element</th><th class="num">wt%</th><th class="num">at%</th></tr></thead><tbody>${els.map((el) => `<tr><td>${esc(el)}</td><td class="num">${pct(w[el])}</td><td class="num">${pct(a[el])}</td></tr>`).join("")}</tbody></table>
      <p class="muted" style="font-size:11.5px;margin:6px 0 0">${esc(im.basis || "")}${missingInputs ? ` ${missingInputs}` : ""}</p>`;
      })();
  const records = (c.records || []).map((r) => `<div class="chem"><b>${esc(r.method)}</b> <span class="muted">${esc(r.basis)} basis · ${esc(r.source)}</span><div class="mono" style="font-size:11.5px">${Object.entries(r.fractions || {}).map(([el, v]) => `${esc(el)} ${fmt(v, 3)}`).join(" · ")}</div></div>`).join("");
  const d = c.draft;
  const ready = d.method && d.basis && d.source.trim() && d.rows.length && d.rows.every((r) => /^[A-Z][a-z]?$/.test(r.el.trim()) && r.v.trim() !== "" && Number.isFinite(Number(r.v)));
  return `<details class="comp" data-dk="composition"><summary><h3>Composition <span class="hint">implied by the answer · ${(c.records || []).length ? `${c.records.length} measured` : "none measured"}</span></h3></summary><div class="body">
    <p class="muted" style="font-size:12px;margin:0 0 6px">Implied by this answer</p>${implied}
    <p class="muted" style="font-size:12px;margin:12px 0 6px">Measured (XRF, ICP or EDS)</p>${records || '<p class="muted" style="margin:0">None entered.</p>'}
    <details data-dk="comp-form"><summary>Enter a measurement</summary><form id="composition-form" class="refit" style="padding:10px 0 0;border:0">
      <div class="row"><select data-comp="method" aria-label="Method"><option value="">Method</option>${["XRF", "ICP", "EDS"].map((m) => `<option ${d.method === m ? "selected" : ""}>${m}</option>`).join("")}</select>
      <select data-comp="basis" aria-label="Basis"><option value="">Basis</option>${[["weight", "weight fraction"], ["atomic", "atomic fraction"]].map(([v, l]) => `<option value="${v}" ${d.basis === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      <input type="text" data-comp="source" data-fk="comp-source" value="${esc(d.source)}" placeholder="Source (required)" aria-label="Source"></div>
      ${d.rows.map((r, i) => `<div class="row"><input type="text" data-comp-el="${i}" data-fk="comp-el-${i}" value="${esc(r.el)}" placeholder="Element" style="flex:0 0 90px" aria-label="Element ${i + 1}"><input type="text" data-comp-v="${i}" data-fk="comp-v-${i}" value="${esc(r.v)}" placeholder="Fraction" style="flex:0 0 120px" aria-label="Fraction ${i + 1}"></div>`).join("")}
      <div class="row"><button type="button" class="qbtn" data-act="comp-row">Add an element</button><button type="submit" class="qbtn" ${ready && !c.saving ? "" : "disabled"}>Store measurement</button>${c.error ? `<span class="err">${esc(c.error)}</span>` : ""}</div>
    </form></details></div></details>`;
}
async function loadComposition(e, V) {
  if (!ROUTES.impliedComposition) return;
  const c = V.composition = { implied: null, impliedError: "", records: [], draft: { method: "", basis: "", source: "", rows: [{ el: "", v: "" }] }, saving: false, error: "" };
  const phases = phasesOf(e).map((p) => ({ phase_id: p.id, formula: p.formula, weight_percent: p.wt }));
  if (phases.length) {
    try { c.implied = await call("impliedComposition", {}, { body: { phases } }); } catch (err) { c.impliedError = `No implied composition: ${err.message}`; }
  } else c.impliedError = "This snapshot names no phase.";
  try { c.records = listOf(await call("composition", { root: e.root }), "records", "items", "measurements"); } catch {}
}
async function saveComposition() {
  const e = S.ep, c = view().composition, d = c.draft;
  c.saving = true; c.error = ""; render();
  try {
    const body = { basis: d.basis, method: d.method, source: d.source.trim(), fractions: Object.fromEntries(d.rows.map((r) => [r.el.trim(), Number(r.v)])) };
    const rec = await call("postComposition", { root: e.root }, { body });
    c.records.push(rec || body);
    c.draft = { method: "", basis: "", source: "", rows: [{ el: "", v: "" }] };
  } catch (err) { c.error = `Not stored: ${err.message}`; }
  c.saving = false; render();
}

function phaseTable(e) {
  const V = view(), curves = (e.fit && e.fit.phase_curves) || {}, blocked = recordingBlocked();
  const running = V.refits.some((r) => r.status === "running") || !S.refit.available;
  const cell = (item, p) => {
    const k = labelKey(item, p.id), cur = V.labels[k] && V.labels[k].answer;
    return `<td data-ck="${esc(k)}">${seg(`data-act="answer" data-item="${item}" data-phase="${esc(p.id)}"`, cur, ANSWERS, { draft: k in V.drafts ? "cannot_assess" : V.pendingNo[k] ? "no" : null, saving: V.saving[k], disabled: blocked })}${critiqueButton(V, k)}${reasonBox(V, k, `${item} for ${p.formula || p.id}`)}${storedReason(V, k)}${critiqueBox(V, k)}${errLine(V, k)}</td>`;
  };
  const rows = phasesOf(e).map((p) => `<tr data-phase-row="${esc(p.id)}" style="--pc:${phaseColor(p.id)}">
      <td class="phcell">${phaseNames(p)}<div class="phase-acts">${curves[p.id] ? `<button type="button" class="qbtn phase" data-act="phase-curve" data-phase="${esc(p.id)}" aria-pressed="${S.phaseSel === p.id}">${S.phaseSel === p.id ? "Hide its peaks" : "Show its peaks"}</button>` : ""}<button type="button" class="qbtn" data-act="remove" data-phase="${esc(p.id)}" ${running || !e.origin ? "disabled" : ""}>Remove and refit</button></div></td>
      <td class="num">${p.wt != null ? fmt(p.wt) : "n/a"}${p.esd != null ? ` ± ${fmt(p.esd)}` : ""}</td>
      <td class="num">${p.shift != null ? fmt(p.shift) + "%" : "n/a"}</td>
      ${cell("phase_supported", p)}${cell("phase_plausible", p)}
    </tr>${e.screenings || (e.steps || []).length ? `<tr class="origin-row" style="--pc:${phaseColor(p.id)}"><td colspan="5"><details data-dk="origin-${esc(p.id)}"><summary>How the agent found it</summary>${phaseOrigin(e, p)}</details></td></tr>` : ""}`).join("");
  const missing = itemKeys(e).filter((k) => !V.labels[k]).length;
  return `<section class="panel" id="phases"><h3>Phases <span class="keys" title="Keys move a cursor through the rubric: each phase's two questions, then the peaks, then the sign-off questions. Peaks take 1 to 4. A No is stored with its critique. Arrow left and right or j and k change snapshot. Ctrl or Cmd+Enter stores the sign-off."><kbd>1</kbd> yes <kbd>2</kbd> no + critique <kbd>3</kbd> can't tell <kbd>c</kbd> critique <kbd>↑↓</kbd> item <kbd>n</kbd> next</span><span class="hint">${missing ? `<span class="req">${missing} answers needed</span>` : "every phase answered"}</span></h3>
    <div style="overflow-x:auto"><table class="t"><thead><tr><th>Phase</th><th class="num">wt%<div class="basis">modeled crystalline, ± formal esd</div></th><th class="num">Cell shift</th><th class="q">${esc(RUBRIC.phase_supported)}${info("phase_supported", RUBRIC_PLAIN.phase_supported)}</th><th class="q">${esc(RUBRIC.phase_plausible)}${info("phase_plausible", RUBRIC_PLAIN.phase_plausible)}</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${screenExtras(e)}${answerBasis(e)}
    <details class="body basis-note" id="wt-basis" data-dk="wt-basis"><summary>What wt% and ± mean here</summary>wt% are shares of the crystalline phases this fit models, normalized to 100; amorphous or unmodeled content is not counted. ± is the refinement's formal esd${e.answer && e.answer.esd_source ? ` (${esc(e.answer.esd_source)})` : ""}, not a measurement uncertainty.${e.brief && e.brief.profile_caveat ? " The stand-in instrument profile further restricts these amounts." : ""}</details>
    <div class="body muted" style="font-size:12.5px">Nothing is pre-filled. The write-up opens when every phase and peak has an answer. <button type="button" class="qbtn" data-act="add" ${running || !e.origin ? "disabled" : ""}>Add a phase and refit</button></div>
    ${S.refit.available ? "" : `<div class="body muted" id="refit-off">${esc(S.refit.note)}</div>`}
    ${refitComposer(e)}</section>`;
}

// How the agent found a phase, from its own record: the candidate screen that ranked it, the
// refinement that first tried it, and the same-formula candidates it set aside. The server parses
// the screen to candidate ids and numbers (snapshot.screenings); no tool text reaches the page.
const FOM_TIP = "fom = coverage x square root of the matched peak-height share, as the screen tool computes it. A high fom says the lines are there, not that the phase is.";
function candLabel(e, cid) {
  const c = (e.candidates || []).find((x) => x.candidate_id === cid) || {};
  // Only a phase of this answer or its fit owns a color; a set-aside candidate gets a hollow mark.
  const mark = phaseOrder(S.ep || e).includes(cid) ? swatch(cid, S.ep || e) : '<i class="sw other"></i>';
  return `${mark}${esc(formulaOf(e, cid) || c.formula || cid)} ${esc(sgShort(c.space_group))} <span class="mono muted">${esc(cid)}</span>`;
}
function screenAt(e, index) {
  const seg = (e.steps || []).find((x) => x.call_index != null && x.call_index <= index && index < x.call_index + (x.calls || 1));
  return seg ? clock(seg.t_start ?? seg.t) : null;
}
function screenRank(e, cid) {
  for (const sc of e.screenings || []) {
    const r = (sc.ranking || []).find((x) => x.candidate_id === cid);
    if (r) return { sc, r };
  }
  return null;
}
const refinesWith = (e, cid) => refines(e).filter((x) => (x.phases || []).some((q) => (q.candidate_id || q) === cid));
function phaseOrigin(e, p) {
  const lines = [];
  const hit = screenRank(e, p.id);
  if (hit) {
    const when = screenAt(e, hit.sc.call_index);
    lines.push(`Candidate screen${when ? ` at ${when}` : ""} ranked it <b>#${hit.r.rank}</b>${hit.sc.ranked_of ? ` of ${hit.sc.ranked_of}` : ""}: <abbr title="${esc(FOM_TIP)}">fom</abbr> ${fmt(hit.r.fom)}; ${fmt(hit.r.cov * 100, 0)}% of its simulated strong-line intensity (lines at least 5% of its strongest) matches observed peaks; its lines match ${fmt(hit.r.expl_percent, 0)}% of the selected observed peak height${(hit.sc.covering_set || []).includes(p.id) ? ", in the screen's covering set" : ""}.`);
  } else if ((e.screenings || []).length) {
    lines.push("No candidate screen in this record lists it among the candidates it printed.");
  } else lines.push('<span class="muted">No candidate screen in this record.</span>');
  const first = refinesWith(e, p.id)[0];
  lines.push(first ? `First refined in <span class="mono">${esc(first.run_label || "a refinement")}</span> at ${clock(first.t_start)} (Rwp ${fmt(first.rwp)}).` : "No recorded refinement includes it.");
  // Same-formula candidates the agent screened or refined and left out of the answer.
  const answered = new Set(phasesOf(e).map((q) => q.id));
  const seen = new Set([...(e.screenings || []).flatMap((sc) => (sc.ranking || []).map((x) => x.candidate_id)), ...refines(e).flatMap((x) => (x.phases || []).map((q) => q.candidate_id || q))]);
  const aside = [...seen].filter((cid) => !answered.has(cid) && normFormula(formulaOf(e, cid)) === normFormula(formulaOf(e, p.id)));
  lines.push(aside.length ? `Set aside: ${aside.slice(0, 5).map((cid) => { const h = screenRank(e, cid), runs = refinesWith(e, cid); return `${candLabel(e, cid)}${h ? ` ranked #${h.r.rank} (fom ${fmt(h.r.fom)})` : ""}${(e.screenings || []).some((sc) => (sc.covering_set || []).includes(cid)) ? ", in the covering set" : ""}${runs.length ? `, refined in ${runs.map((x) => `<span class="mono">${esc(x.run_label || "")}</span> (Rwp ${fmt(x.rwp)})`).join(", ")}` : ", never refined"}`; }).join("; ")}.` : "No other candidate of this formula was screened or refined.");
  return `<ul class="origin">${lines.map((l) => `<li>${l}</li>`).join("")}</ul>`;
}
// Covering-set picks of the agent's screen that the answer does not hold, with what happened to them.
function screenExtras(e) {
  const answered = new Set(phasesOf(e).map((q) => q.id));
  const forms = new Set(phasesOf(e).map((q) => normFormula(q.formula)));
  const extras = [...new Set((e.screenings || []).flatMap((sc) => sc.covering_set || []))].filter((cid) => !answered.has(cid) && !forms.has(normFormula(formulaOf(e, cid))));
  if (!extras.length) return "";
  return `<details class="body screen-extras" data-dk="screen-extras"><summary>Screen picks not in the answer (${extras.length})</summary><ul class="origin">${extras.map((cid) => { const h = screenRank(e, cid), runs = refinesWith(e, cid); return `<li>${candLabel(e, cid)}: in the covering set${h ? `, ranked #${h.r.rank} (fom ${fmt(h.r.fom)})` : ""}; ${runs.length ? `refined in ${runs.map((x) => `<span class="mono">${esc(x.run_label || "")}</span> (Rwp ${fmt(x.rwp)})`).join(", ")}, then left out` : "never refined"}.</li>`; }).join("")}</ul></details>`;
}

// W5 IdentityCheck: does the phase's formula agree with its own CIF's formula. Every state reads
// differently; a snapshot frozen before the check existed says so (Science 10:27 UTC).
function identityText(id) {
  if (!id) return '<span class="muted">not recorded</span>';
  if (id.status === "consistent") return `<span title="${esc(id.detail || "")}">consistent</span>`;
  if (id.status === "mismatch") return `mismatch: ${esc(id.detail)}; nothing is computed on this phase`;
  return `<span title="${esc(id.detail || "")}">not checked</span>`;
}

// The answer's quantity_basis, verbatim: text, or an object shown key by key (W3 6924a759f).
function answerBasis(e) {
  const a = e.answer || {};
  const has = "quantity_basis" in a || "reported_quantity_basis" in e;
  if (!has) return "";
  const qb = a.quantity_basis ?? e.reported_quantity_basis ?? null;
  const show = (v) => (v != null && typeof v === "object" ? JSON.stringify(v) : String(v));
  const body = qb == null ? '<span class="muted">None recorded in this snapshot.</span>'
    : typeof qb === "object" && !Array.isArray(qb) ? `<dl class="kvl">${Object.entries(qb).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(show(v))}</dd>`).join("")}</dl>`
    : esc(show(qb));
  return `<details class="body answer-basis" id="answer-basis" data-dk="answer-basis"><summary>The answer's stated basis for its amounts (verbatim)</summary><div>${body}</div></details>`;
}

// The submitted answer file (AXA-43zwk.21): keys, numbers, units and nulls as written. The
// server has already left out fit_artifacts and replaced absolute paths and run names with markers.
function answerFile(e) {
  const doc = e.answer && e.answer.submitted;
  if (doc == null || typeof doc !== "object") return "";
  const node = (v) => v !== null && typeof v === "object"
    ? Array.isArray(v) ? `<ol class="doc">${v.map((x) => `<li>${node(x)}</li>`).join("")}</ol>`
      : `<dl class="kvl">${Object.entries(v).map(([k, x]) => `<dt>${esc(k)}</dt><dd>${node(x)}</dd>`).join("")}</dl>`
    : esc(v === null ? "null" : String(v));
  return `<details class="answer-file" id="answer-file" data-dk="answer-file"><summary>Submitted answer file (paths and run identity redacted)</summary>${node(doc)}</details>`;
}

function refitComposer(e) {
  const d = S.refitDraft;
  if (!d || d.snapshot !== e.digest) return "";
  const have = new Set(phasesOf(e).map((p) => p.id));
  const options = (e.candidates || []).filter((c) => !have.has(c.candidate_id));
  const pick = d.verb === "add_phases"
    ? (options.length
      ? `<select data-refit-candidate="1" data-fk="refit-cand" aria-label="Candidate to add"><option value="">Choose a candidate from the case library</option>${options.map((c) => `<option value="${esc(c.candidate_id)}" ${d.candidates[0] === c.candidate_id ? "selected" : ""}>${esc(c.formula || "")} ${esc(sgShort(c.space_group))} · ${esc(c.candidate_id)}</option>`).join("")}</select>`
      : `<input type="text" data-refit-candidate="1" data-fk="refit-cand" value="${esc(d.candidates[0] || "")}" placeholder="Candidate id from the case library, e.g. CIF001772" aria-label="Candidate id to add">`)
    : `<b>Remove ${esc(formulaOf(e, d.candidates[0]))}</b> <span class="mono muted">${esc(d.candidates[0])}</span>`;
  return `<div class="refit" id="refit-composer"><div class="row">${pick} <span class="muted">then staged Rietveld on BGMN${e.case_id === "DEV006" ? ", usually 1 to 2.5 min here" : ""}</span></div>
    <div class="row"><input type="text" data-refit-rationale="1" data-fk="refit-why" value="${esc(d.rationale)}" placeholder="Why (optional, stored with the attempt)" aria-label="Why this refit"></div>
    <div class="row"><button type="button" class="btn" data-act="run-refit" ${d.candidates.length && d.candidates[0] ? "" : "disabled"}>Run refit</button><button type="button" class="btn ghost" data-act="cancel-refit">Cancel</button>
    <span class="muted">Creates a child attempt. The answer stays unchanged.</span></div></div>`;
}

const firstSentence = (t) => { const m = String(t).match(/^[\s\S]{40,}?[.;](?=\s+[A-Z(])/); return m ? m[0] : String(t); };
function writeUp(e) {
  const V = view(), claims = claimsOf(e);
  if (!V.unlocked) return `<section class="panel" id="writeup"><h3>The agent's write-up</h3><div class="lock" title="Evidence comes before the analysis to reduce anchoring.">Opens after every phase and peak has an answer.</div></section>`;
  const steps = e.steps || [];
  const body = e.kind === "expert_correction"
    ? `<p class="muted" style="margin:0">An expert correction has no agent write-up.</p>`
    : claims.length ? claims.map((c, i) => {
      const [cls, label, why] = claimMark(c, e);
      // Each claim opens on its first sentence, the agent's full words one click away. A claim a
      // check disputes or contradicts opens in full, so the rater reads what the check refers to.
      const text = String(c.text || ""), head = firstSentence(text), full = S.fullClaims || S.openClaims.has(i) !== (cls === "hold" || cls === "stop");
      const more = head !== text ? ` <button type="button" class="linkbtn" data-act="claim-more" data-i="${i}" aria-expanded="${full}">${full ? "less" : "more"}</button>` : "";
      return `<div class="claim"><div><span class="k">${esc(c.kind || "")}</span> ${esc(full ? text : head)}${more}</div><span class="chip ${cls}">${esc(label.toUpperCase())}</span>${why ? `<div class="why">${esc(why)}</div>` : ""}</div>`;
    }).join("")
    : isAnswered(e) && (e.missing || []).includes("answer text") ? `<p class="muted" style="margin:0">The answer's own words are not in this snapshot.</p>`
    : `<p class="muted" style="margin:0">This episode wrote no answer and no claims.</p>`;
  return `<section class="panel" id="writeup"><h3>The agent's write-up ${claims.length ? `<button type="button" class="qbtn small" data-act="claims-all" aria-pressed="${Boolean(S.fullClaims)}">${S.fullClaims ? "First sentences" : "Full text"}</button>` : ""}<span class="hint" title="Some claims are checked against the run record. The default is Not checked.">claim checks use the run record</span></h3>
    <div class="body">${body}${answerFile(e)}
    <p class="muted" style="font-size:12px;margin:10px 0 0">You can still change any answer; each change is a new record.</p></div>
    ${steps.length ? `<h3><button type="button" class="qbtn" data-act="steps">${S.showSteps ? "Hide" : "Show"} ${(e.trace || []).length ? "the trace" : "every step"} (${steps.length} steps)</button><span class="hint">${(e.trace || []).length ? "what the agent said, ran and was shown · " : ""}mark any step</span></h3>` : ""}
    ${S.showSteps ? traceView(e) : ""}
  </section>`;
}

const stepText = (x) => x.kind === "refine" ? `Refine ${esc((x.phases || []).map((p) => `${p.formula || p.candidate_id || p} ${sgShort(p.space_group)}`).join(", "))} · Rwp ${fmt(x.rwp)}` : x.kind === "compaction" ? "Context compacted" : x.kind === "answer" ? "Wrote an answer" : esc(x.title);

// The agent's trace: the context the harness gave it, its task, then under each step its mark
// control, the agent's own words and every call it made there, with the command and what the
// environment returned. The fit the answer cites and the answer itself are tied to the steps that
// made them, and each residual peak to the call whose screen.py listing first held it (a text
// match, peak_sightings). "As it happened" lists every turn in recorded order. Without a recorded
// trace, the steps alone.
function traceView(e) {
  const steps = e.steps || [], turns = e.trace || [], seen = e.peak_sightings || {};
  const firstShows = {}, callAt = {};
  for (const p of peaksOf(e)) if (seen[p.id] != null) (firstShows[seen[p.id]] = firstShows[seen[p.id]] || []).push(p);
  for (const t of turns) for (const c of t.calls) callAt[c.call_index] = t.t;
  const stepAt = (i) => (steps[i] ? `step ${i + 1} · ${stepText(steps[i])}` : "");
  // only a fit the answer cites; the payload shows the lowest-Rwp fit when none is cited
  const cited = e.fit && e.fit.role === "cited by the answer" ? e.fit.run_label : null;
  const citedStep = cited == null ? -1 : steps.findIndex((x) => x.kind === "refine" && x.run_label === cited);
  const answerStep = steps.map((x) => x.kind).lastIndexOf("answer");
  const status = (code) => { const c = (e.checks || []).find((k) => k.code === code); return c ? `${code} ${c.status === "flag" ? "flag" : c.status === "pass" ? "pass" : "n/a"}` : ""; };
  const badges = (i) => [i === citedStep ? ["the answer cites this fit", status("K1"), status("K5")].filter(Boolean).join(" · ") : "", i === answerStep ? "wrote the submitted answer" : ""]
    .filter(Boolean).map((b) => `<span class="chip none">${esc(b)}</span>`).join("");
  const say = (t) => {
    const text = t.message.trim();
    if (!text) return "";
    if (t.source === "context") return `<details class="call" data-dk="ctx-${turns.indexOf(t)}"><summary>Context the harness added</summary><pre class="out">${esc(text)}</pre></details>`;
    return `<p class="say${t.source === "user" ? " told" : ""}">${t.source === "user" ? "Told the agent: " : ""}${esc(text)}</p>`;
  };
  const exit = (c) => [c.running ? "still running" : c.exit_code != null ? `exit ${c.exit_code}` : "", c.wall_s ? `${fmt(c.wall_s, 1)} s` : ""].filter(Boolean).join(" · ");
  const callView = (c, t, where = "") => {
    const head = c.tool === "write_stdin" ? "More output of the running command" : c.command.split("\n")[0].slice(0, 160) || c.tool;
    const shows = (firstShows[c.call_index] || []).map((p) => `<span class="chip peak">first shows ${fmt(p.tt, 2)}°</span>`).join("");
    const full = S.traceFull[`${e.digest}:${c.call_index}`];
    const more = c.executor_chars != null ? `<p class="muted">the tool produced ${c.executor_chars} characters; the agent was shown less of it</p>` : "";
    const body = full && !full.error
      ? `<pre class="out">${esc(full.shown)}</pre>${full.executor != null ? `<p class="muted">The tool's own output, as the runner kept it:</p><pre class="out">${esc(full.executor)}</pre>` : ""}`
      : `<pre class="out">${c.output ? esc(c.output) : '<span class="muted">(no output)</span>'}</pre>${c.clipped || c.executor_chars != null ? `<button type="button" class="qbtn small" data-act="trace-full" data-call="${c.call_index}">Full record</button>${full ? ` <span class="err">${esc(full.error)}</span>` : ""}` : ""}`;
    return `<details class="call" data-dk="call-${c.call_index}"><summary><span class="t">${clock(t)}</span>${where ? `<span class="muted">${where}</span>` : ""}<code>${esc(head)}</code><span class="muted">${esc(exit(c))}</span>${shows}</summary>
      ${c.command.includes("\n") || c.command.length > 160 ? `<pre class="cmd">${esc(c.command)}</pre>` : ""}${more}${body}</details>`;
  };
  const items = (i) => turns.flatMap((t) => [t.step === i && t.source !== "task" ? say(t) : "", ...t.calls.filter((c) => c.step === i).map((c) => callView(c, t.t))]).join("");
  const links = [];
  if (turns.length) {
    if (citedStep >= 0) links.push(`The answer cites fit <span class="mono">${esc(cited)}</span>, made at ${clock(steps[citedStep].t_start)} (step ${citedStep + 1})${answerStep >= 0 ? `; the submitted answer was written at ${clock(steps[answerStep].t)} (step ${answerStep + 1})` : ""}.`);
    for (const p of peaksOf(e)) links.push(`${fmt(p.tt, 2)}° residual peak: ${seen[p.id] != null ? `first found in the recorded text${callAt[seen[p.id]] != null ? ` at ${clock(callAt[seen[p.id]])}` : ""}` : "not found in the recorded text (only screen.py peak listings are searched)"}`);
  }
  const context = turns.filter((t) => t.source === "context" && t.step == null && t.message.trim());
  const given = context.length ? `<details class="call" data-dk="trace-context"><summary>What the harness gave the agent (${context.length})</summary>${context.map((t) => `<pre class="out">${esc(t.message.trim())}</pre>`).join("")}</details>` : "";
  const task = turns.filter((t) => t.source === "task").map((t) => `<details class="call" data-dk="trace-task"><summary>The task the agent was given</summary><pre class="out">${esc(t.message)}</pre></details>`).join("");
  const before = turns.filter((t) => t.source === "agent" && t.step == null).map(say).join("");
  const order = turns.length ? `<div class="trace-order">${[["steps", "By step"], ["time", "As it happened"]].map(([k, l]) => `<button type="button" class="qbtn small" data-act="trace-order" data-order="${k}" aria-pressed="${(k === "time") === Boolean(S.traceTime)}">${l}</button>`).join("")}</div>` : "";
  const head = links.length || given || task || before || order ? `<div class="trace-head">${order}${links.length ? `<ul class="sightings">${links.map((l) => `<li>${l}</li>`).join("")}</ul>` : ""}${given}${task}${S.traceTime ? "" : before}</div>` : "";
  if (S.traceTime && turns.length) {
    const flat = turns.filter((t) => t.source !== "task" && !(t.source === "context" && t.step == null)).map((t) => [say(t), ...t.calls.map((c) => callView(c, t.t, stepAt(c.step)))].join("")).join("");
    return `${head}<div class="turns trace-flat">${flat}</div>`;
  }
  return `${head}
    <ol class="steps trace">${steps.map((x, i) => `<li><span class="t">${clock(x.kind === "compaction" || x.kind === "answer" ? x.t : x.t_start)}</span><span class="${x.kind === "refine" ? "r" : ""}">${stepText(x)}${badges(i)}</span><span></span>${stepRef(e, x) ? `<div class="markbox-host" style="grid-column:2 / -1">${markControl(stepRef(e, x))}</div>` : ""}${turns.length ? `<div class="turns">${items(i)}</div>` : ""}</li>`).join("")}</ol>`;
}

// The first step where the analysis went wrong, as one pick (errors compound downstream). It is
// stored as step feedback "went_wrong_here" on that step, the record the step marks already use.
function firstFailingStep(e) {
  const V = view();
  const steps = (e.steps || []).map((x) => [x, stepRef(e, x)]).filter(([, ref]) => ref);
  if (!steps.length) return "";
  const marked = steps.findIndex(([, ref]) => (V.marks[stepKey(ref.attempt, ref.call_index)] || []).some((m) => m.mark === "went_wrong_here"));
  const label = (x) => `${clock(x.kind === "compaction" || x.kind === "answer" ? x.t : x.t_start)} · ${x.kind === "refine" ? `${x.run_label || "refine"}${x.rwp != null ? ` · Rwp ${fmt(x.rwp)}` : ""}` : x.kind === "answer" ? "Wrote an answer" : x.title || x.kind}`;
  return `<label class="ffs">First failing step <select data-first-fail="1" aria-label="First failing step"><option value="">not chosen</option>${steps.map(([x], i) => `<option value="${i}" ${marked === i ? "selected" : ""}>${esc(label(x))}</option>`).join("")}</select></label>`;
}
function signOff(e) {
  const V = view(), blocked = recordingBlocked();
  const items = signoffItems(e), answered = items.filter((k) => V.labels[k]).length;
  const peaks = peaksOf(e), called = peaks.filter((p) => V.peaks[p.id]).length;
  const unsupported = phasesOf(e).some((p) => (V.labels[labelKey("phase_supported", p.id)] || {}).answer === "no");
  const complete = answered === items.length && called === peaks.length;
  const signed = Boolean(V.signoff);
  const critiqueMissing = V.verdict === "would_not_sign" && !V.comment.trim();
  const canSign = V.unlocked && complete && !signed && !blocked && V.verdict && !critiqueMissing && !(V.verdict === "would_sign" && (unsupported || !isAnswered(e)));
  const analysis = ANALYSIS_ITEMS.map((item) => {
    const k = labelKey(item, null), cur = V.labels[k] && V.labels[k].answer;
    return `<div class="item" data-item-row="${item}" data-ck="${esc(k)}"><span>${esc(RUBRIC[item])}${info(item, RUBRIC_PLAIN[item])}</span><span class="ans">${seg(`data-act="answer" data-item="${item}" data-phase=""`, cur, ANSWERS, { draft: k in V.drafts ? "cannot_assess" : V.pendingNo[k] ? "no" : null, saving: V.saving[k], disabled: !V.unlocked || blocked })}${critiqueButton(V, k)}</span>${reasonBox(V, k, item)}${storedReason(V, k)}${critiqueBox(V, k)}${errLine(V, k)}</div>`;
  }).join("");
  const reason = !V.unlocked ? "Opens after the evidence stage."
    : signed ? "Your sign-off is stored."
    : !complete ? `Answer every item first: ${items.length - answered} rubric answer${items.length - answered === 1 ? "" : "s"} and ${peaks.length - called} peak call${peaks.length - called === 1 ? "" : "s"} left.`
    : V.verdict === "would_sign" && !isAnswered(e) ? "No answer to sign. Choose \"would not sign\" or \"unresolved\"."
    : V.verdict === "would_sign" && unsupported ? "A phase is marked unsupported. Choose \"would not sign\" or \"unresolved\", or label a refit."
    : !V.verdict ? "Choose a verdict."
    : critiqueMissing ? "Write a critique: what is wrong with this analysis." : "";
  const opt = ([v, l]) => `<button type="button" class="btn ghost" data-act="verdict" data-val="${v}" aria-pressed="${(signed ? V.signoff.verdict : V.verdict) === v}" ${!V.unlocked || signed || blocked ? "disabled" : ""}>${l}</button>`;
  return `<section class="panel" id="signoff"><h3>Sign-off <span class="hint"><span class="count">${answered} of ${items.length} rubric answers · ${called} of ${peaks.length} peak calls</span></span></h3>
    <div class="items">${analysis}</div>
    <div class="sign">${VERDICTS.map(opt).join("")}
      ${V.verdict && V.verdict !== "would_sign" && !signed ? firstFailingStep(e) : ""}
      <textarea data-comment="1" data-fk="comment" placeholder="${V.verdict === "would_not_sign" ? "Critique: what is wrong with this analysis (required)" : "Critique (optional)"}" ${!V.unlocked || signed ? "disabled" : ""}>${esc(V.comment)}</textarea>
      <button type="button" class="btn" data-act="sign" ${canSign && !V.saving.signoff ? "" : "disabled"}>Store sign-off</button>
      ${signed ? '<button type="button" class="btn" data-act="report" id="report-btn">Analysis report</button><button type="button" class="btn ghost" data-act="files" id="files-btn">Raw files (.zip)</button>' : ""}
      ${reason ? `<span class="reason">${esc(reason)}</span>` : ""}${errLine(V, "signoff")}</div></section>`;
}

// reveal:begin. The only code that reads run identity, sibling episodes, the reference comparison
// or other raters' labels. Its input is the /revealed response (and labels the server returns
// after the reveal), fetched only after the sign-off POST succeeded (KTD12).
// Your answers beside every other rater's latest answer, item by item, with n. A model judge is
// never shown as an expert, and an owner label is a revealed demo label, not expert calibration.
function agreementTable(e, V) {
  const who = (r) => (!r ? "rater" : r.kind === "model_judge" ? `Judge ${(r.judge && r.judge.model) || r.principal_id || ""} (model, not an expert)` : r.kind === "owner" ? `${r.principal_id || "owner"} (owner, revealed demo label)` : `${r.principal_id || "expert"} (expert)`);
  const raters = new Map();
  V.others.slice().sort(byTime).forEach((l) => {
    const id = (l.rater && (l.rater.principal_id || l.rater.kind)) || "rater";
    if (!raters.has(id)) raters.set(id, { name: who(l.rater), answers: {} });
    raters.get(id).answers[labelKey(l.item, l.phase_id)] = l.answer;
  });
  const items = signoffItems(e).filter((k) => V.labels[k] || [...raters.values()].some((r) => r.answers[k]));
  if (!items.length || !raters.size) return "";
  const name = (k) => { const [, item, phase] = k.split("|"); return phase ? `${formulaOf(e, phase)} ${item === "phase_supported" ? "supported" : "plausible"}` : item === "amounts_defensible" ? "amounts defensible" : "fit supports conclusion"; };
  const said = (a) => (a ? (ANSWERS.find(([v]) => v === a) || [a, a])[1] : "");
  const cols = [...raters.values()];
  const agree = cols.map((r) => { const both = items.filter((k) => V.labels[k] && r.answers[k] && V.labels[k].answer !== "cannot_assess" && r.answers[k] !== "cannot_assess"); return `${both.filter((k) => V.labels[k].answer === r.answers[k]).length} of ${both.length}`; });
  return `<div style="overflow-x:auto;margin-top:8px"><table class="t agree"><thead><tr><th>Item</th><th>You</th>${cols.map((r) => `<th>${esc(r.name)}</th>`).join("")}</tr></thead><tbody>
    ${items.map((k) => { const mine = V.labels[k] && V.labels[k].answer; return `<tr><td>${esc(name(k))}</td><td>${esc(said(mine))}</td>${cols.map((r) => `<td class="${mine && r.answers[k] && mine !== "cannot_assess" && r.answers[k] !== "cannot_assess" && r.answers[k] !== mine ? "differs" : ""}">${esc(said(r.answers[k]))}</td>`).join("")}</tr>`; }).join("")}
    <tr><td class="muted">Agrees with you (Can't tell excluded)</td><td></td>${agree.map((a) => `<td class="num">${a}</td>`).join("")}</tr></tbody></table></div>`;
}
// Dogfood F4: an exact match to a reference that lists only the declared precursors does not explain
// a residual peak the check still lists.
function exactNote(e, rv) {
  const m = rv.reference_match, peak = peaksOf(e).sort((a, b) => b.pct - a.pct)[0];
  if (!m || !m.phase_list_exact || !peak) return "";
  const declared = new Set(((e.brief && e.brief.precursors) || []).map((p) => normFormula(p.formula)));
  const onlyPrecursors = rv.reference && Array.isArray(rv.reference.phases) && rv.reference.phases.length && rv.reference.phases.every((p) => declared.has(normFormula(p.formula)));
  return `<p style="margin:0 0 8px" id="exact-note">${onlyPrecursors ? "The published reference lists the declared precursors only; an" : "An"} exact match does not explain the unexplained peak at ${fmt(peak.tt, 2)}° (${fmt(peak.pct, 1)}% of the strongest net peak).</p>`;
}
function revealPanel(e) {
  const V = view();
  if (!V.signoff) return "";
  const rv = V.revealed;
  if (!rv) return `<section class="panel reveal" id="reveal"><h3>After sign-off</h3><div class="body muted">${V.revealError ? `Sign-off stored. Run identity opens when you have signed every answer in this pattern's cohort, and the reference when every assigned rater has. <span class="mono" style="font-size:11.5px">(${esc(V.revealError)})</span>` : "Loading the reveal."}</div>
    <div class="sign"><button type="button" class="btn" data-act="next">Next snapshot</button></div></section>`;
  const raw = rv.identity || rv.run || null;
  const id = raw && raw.corrected_episode ? raw.corrected_episode : raw;
  const by = raw && raw.corrected_episode ? raw.by : null;
  const sibs = listOf(rv.siblings || rv.episodes || []);
  const refShown = Boolean(rv.reference_revealed) || sibs.some((r) => r.reference_match);
  const match = (m) => (m.phase_list_exact ? '<span class="chip pass">EXACT</span>' : `<span class="chip stop">DIFFERS</span> <span class="muted" style="font-size:11.5px">${esc([...(m.missed || []).map((x) => "missed " + x), ...(m.extra || []).map((x) => "extra " + x)].join(", "))}</span>`);
  const pending = rv.reveal && (rv.reveal.pending_raters || []).length;
  const ref = rv.reference && Array.isArray(rv.reference.phases) ? rv.reference : null;
  const refLine = ref ? `<p style="margin:0 0 8px" id="reference-list">Reference phase list: <b>${ref.phases.map((p) => esc(`${p.formula}${p.space_group ? ` ${sgShort(p.space_group)}` : ""}${p.weight_percent != null ? ` ${fmt(p.weight_percent, 1)} wt%` : ""}`)).join(", ")}</b>${ref.source ? ` <span class="muted" style="font-size:12px">(${ref.source === "custody reference_phase_coverage" ? "published reference refinement" : esc(ref.source)})</span>` : ""}</p>` : "";
  const phs = (r) => (r.phases || r.answer_phases || []).map((p) => (typeof p === "string" ? p : `${p.formula || p.candidate_id} ${sgShort(p.space_group)}`.trim())).join(" + ");
  const idLine = id ? `<p style="margin:0 0 8px">${by ? `A correction by <b>${esc(by.principal_id || by.kind || "a human")}</b> of the run: <b>` : "This run: <b>"}${esc(id.model || id.model_name || "model not recorded")}</b>${id.harness ? `, harness ${esc(id.harness)}` : ""}${id.series ? `, series ${esc(id.series)}` : ""}${id.job ? ` <span class="mono muted">${esc(id.job)}</span>` : ""}${id.ending ? `, ${esc(id.ending)}` : ""}${id.wall_s != null ? ` in <span class="num">${clock(id.wall_s)}</span>` : ""}${id.tool_calls != null ? `, ${esc(id.tool_calls)} tool calls` : ""}${id.refine_runs != null ? `, ${esc(id.refine_runs)} refinements` : ""}${id.cost_usd !== undefined ? `, cost ${id.cost_usd != null ? "$" + fmt(id.cost_usd) : "not recorded"}` : ""}.</p>` : "";
  // The triage route, present only once the server holds this rater's own sign-off on this snapshot.
  const rt = rv.route;
  const causes = rt ? (rt.checks || []).filter((c) => c.routes && c.status !== "pass") : [];
  const units = { K1: "Rwp percentage points of rise after a good Pawley fit", K2: "% of the strongest net observed peak", K3: "largest absolute a/b/c shift from the CIF (% of that CIF axis; phases at least 2 wt%)", K5: "wt% points, answer minus cited fit" };
  const why = causes.map((c) => { const n = [c.value != null ? fmt(c.value, 2) : "", c.threshold != null ? `threshold ${fmt(c.threshold, 2)}` : ""].filter(Boolean).join(", "); return `${c.code} ${String(c.status).replace("_", " ")}${n ? ` ${n}${units[c.code] ? ` (${units[c.code]})` : ""}` : ""}`; }).join("; ");
  const routeLine = rt ? `<p style="margin:0 0 8px" id="route-line">Triage ${rt.route === "expert_flagged" ? `routed this answer to an expert: ${esc(why)}` : rt.route === "expert_audit" ? "routed this answer to an expert as an audit sample; no routing check fired" : "cleared this answer: no routing check fired and it was not drawn for audit"}. <span class="muted" style="font-size:12px">Policy ${esc(rt.policy_version)}${rt.provisional ? ", provisional thresholds" : ""}.</span></p>` : "";
  const groups = new Map();
  sibs.filter((r) => phs(r)).forEach((r) => { const k = phs(r).split(" + ").sort().join(" + "); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); });
  const split = groups.size >= 2 ? `<p style="margin:0 0 6px">Answers on ${esc(e.case_id)} split into ${groups.size} phase lists.</p><ul class="split">${[...groups].map(([k, rs]) => {
    const models = [...new Set(rs.map((r) => r.model))].map((m) => `${m} ×${rs.filter((r) => r.model === m).length}`).join(", ");
    const res = [...new Set(rs.map((r) => (r.residual_max_pct != null ? fmt(r.residual_max_pct, 1) + "%" : "n/a")))].join(" or ");
    return `<li><b>${esc(k)}</b>${rs.some((r) => r.this) ? ' <span style="color:var(--accent)">this run</span>' : ""}<br>${rs.length} answers (${esc(models)}), largest unexplained peak ${esc(res)}</li>`;
  }).join("")}</ul>` : "";
  const table = sibs.length ? `<div style="overflow-x:auto"><table class="t"><thead><tr><th>Episode</th><th>Model</th><th>Harness</th><th>Ending</th><th>Phases</th><th class="num">Residual</th><th class="num">Wall</th>${refShown ? "<th>Reference phase list</th>" : ""}</tr></thead><tbody>
    ${sibs.map((r) => `<tr><td class="mono" style="font-size:11.5px">${esc(r.job || String(r.episode_id || "").slice(0, 12))}${r.this ? ' <span style="color:var(--accent)">this</span>' : ""}</td><td>${esc(r.model)}</td><td>${esc(r.harness)}</td><td>${esc(r.ending)}</td><td style="font-size:12px">${esc(phs(r))}</td><td class="num">${r.residual_max_pct != null ? fmt(r.residual_max_pct, 1) + "%" : "n/a"}</td><td class="num">${r.wall_s ? clock(r.wall_s) : ""}</td>${refShown ? `<td>${r.reference_match ? match(r.reference_match) : "n/a"}</td>` : ""}</tr>`).join("")}
    </tbody></table></div>` : "";
  const known = new Set(["identity", "run", "siblings", "episodes", "reference_revealed", "signoff", "digest", "root", "schema_version", "kind", "reveal", "labels", "reference_match", "reference", "route"]);
  const rest = Object.entries(rv).filter(([k]) => !known.has(k));
  const restTable = rest.length ? `<details style="padding:0 16px 10px"><summary>Everything else the server revealed</summary><table class="t">${rest.map(([k, v]) => `<tr><td class="mono" style="font-size:11.5px">${esc(k)}</td><td class="mono" style="font-size:11.5px;white-space:pre-wrap">${esc(typeof v === "string" ? v : JSON.stringify(v, null, 1))}</td></tr>`).join("")}</table></details>` : "";
  const others = V.others.length ? `<p class="muted" style="margin:8px 0 0">Other raters' labels on this snapshot: ${V.others.length} (${esc([...new Set(V.others.map((l) => (l.rater && l.rater.kind) || "rater"))].join(", "))}).</p>${agreementTable(e, V)}` : "";
  return `<section class="panel reveal" id="reveal"><h3>After sign-off <span class="hint">Shown only because the server stored your verdict</span></h3>
    <div class="body">${idLine}${routeLine}${refLine}${rv.reference_match ? `<p style="margin:0 0 8px">This answer against it: ${match(rv.reference_match)} <span class="muted" style="font-size:12px">formula level; every polymorph of a formula counts, so this is not a polymorph check</span></p>${exactNote(e, rv)}` : ""}${others}</div>
    ${sibs.length ? `<details class="others" style="padding:0 16px 10px"><summary>Other episodes on ${esc(e.case_id)} (${sibs.length}) and how their answers split</summary><div style="padding-top:8px">${split}</div>${table}</details>` : ""}${restTable}
    <div class="sign"><span class="muted">${refShown ? "Reference shown. Later labels here are stored as revealed." : `The reference opens when every assigned rater has signed off${pending ? ` (${pending} to go)` : ""}. Later labels are stored as revealed.`}</span>
      <button type="button" class="btn" data-act="next">Next snapshot</button></div>
  </section>`;
}
// reveal:end

// ---------------------------------------------------------------------------------------------
// the analysis report: after this rater's own sign-off, one button writes the whole analysis as a
// print-ready page (the browser saves the PDF). It is built only from what the server already
// returned to this rater: the blind payload, this rater's own records and refits, and the
// /revealed response, rendered by revealPanel. Nothing else is fetched (owner request, 18:47 UTC).
// ---------------------------------------------------------------------------------------------
const REPORT_CSS = `:root { --fg: #111; --ink-2: #333; --ink-3: #555; --ink-4: #666; --line: #ddd; --line-soft: #eee; --line-strong: #bbb;
  --obs: #111; --calc: #6b6b6b; --bgline: #8a8a8a; --accent: #0072B2; --pass: #1a7f37; --hold: #7a5c00; --stop: #b33a00; --no: #b00020; --phase: #6b4fa0; }
* { box-sizing: border-box; } body { margin: 0; background: #fff; color: var(--fg); font: 400 12.5px/1.5 "IBM Plex Sans", system-ui, sans-serif; }
.page { max-width: 1100px; margin: 0 auto; padding: 28px 32px; } h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 14px; margin: 22px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.meta { color: var(--ink-3); } .mono, .num { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; } .muted { color: var(--ink-3); }
table { width: 100%; border-collapse: collapse; } th, td { text-align: left; vertical-align: top; padding: 5px 8px; border-bottom: 1px solid var(--line-soft); font-size: 12px; } th { color: var(--ink-4); font-weight: 500; }
svg.plot { width: 100%; display: block; } .axis { font: 400 11px "IBM Plex Mono", monospace; fill: #555; } .axis.tickname { font-size: 12.5px; font-weight: 500; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; } .sw.obs { width: 4px; height: 4px; background: var(--obs); margin: 0 6px 0 3px; vertical-align: 2px; }
.sw.calc, .sw.diff, .sw.bg { width: 14px; height: 0; border-top: 1.5px solid var(--calc); vertical-align: 3px; } .sw.bg { border-top-style: dashed; border-top-color: var(--bgline); }
.sw.ticks { width: 12px; height: 9px; background: repeating-linear-gradient(90deg, #444 0 1.5px, transparent 1.5px 4px); } .sw.other { box-shadow: inset 0 0 0 1.5px #777; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font: 400 11px "IBM Plex Mono", monospace; color: var(--ink-3); margin: 4px 0 8px; } .legend .ctl, button { display: none !important; }
.brief { display: flex; flex-wrap: wrap; gap: 2px 18px; } .brief b { font-weight: 600; } .chip { font: 700 10px/1 "IBM Plex Mono", monospace; padding: 3px 6px; border: 1px solid #999; border-radius: 10px; white-space: nowrap; }
.chip.stop { background: #111; color: #fff; } .claim { padding: 5px 0; border-bottom: 1px solid var(--line-soft); } .claim .k { font: 500 11px "IBM Plex Mono", monospace; color: var(--ink-4); } .why { color: var(--ink-3); font-size: 11.5px; }
.panel h3 { font-size: 13px; margin: 6px 0; } .panel .body { padding: 4px 0; } .hint { font-weight: 400; color: var(--ink-4); margin-left: 8px; } details > summary { list-style: none; color: var(--ink-3); } details[open] > summary { font-weight: 500; }
.keep { break-inside: avoid; } .tools { position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
.tools .print { display: inline-block !important; padding: 7px 14px; border: 1px solid #111; border-radius: 8px; background: #111; color: #fff; font-weight: 600; cursor: pointer; }
@media print { .tools { display: none; } .page { padding: 0; max-width: none; } h2 { break-after: avoid; } @page { margin: 14mm; } }`;
// The page's own renderers, captured with every view toggle at its default and every disclosure open.
function reportPlot(e, review) {
  const keep = { phaseSel: S.phaseSel, peakSel: S.peakSel, range: S.range, yscale: S.yscale };
  S.phaseSel = null; S.peakSel = null; S.range = null; S.yscale = "lin";
  try { return openAll(plot(review || e, false) + plotLegend(review || e)); } finally { Object.assign(S, keep); }
}
const openAll = (html) => html.replace(/<details(?![^>]*\bopen\b)/g, "<details open").replace(/<button[\s\S]*?<\/button>/g, "").replace(/<form[\s\S]*?<\/form>/g, "");
// On paper the dark-page palette's orange, sky blue and yellow fall under 3:1; the research brief's
// light-background variants of the same hues replace them (scientific-interface.md, phase color spec).
const PAPER = { "#E69F00": "#C38700", "#56B4E9": "#4A9BC8", "#F0E442": "#9E962C" };
function answerText(V, k) {
  const l = V.labels[k];
  if (!l) return '<span class="muted">not answered</span>';
  const said = (ANSWERS.find(([v]) => v === l.answer) || [l.answer, l.answer])[1];
  return `<b>${esc(said)}</b>${l.reason ? `<div class="why">Reason: ${esc(l.reason)}</div>` : ""}${l.note ? `<div class="why">Critique: ${esc(l.note)}</div>` : ""}`;
}
// How the server recorded this rater's sign-off (CR21): kind and exposure, so a printed owner
// record never reads as an expert's blind judgment.
function provenance(rec) {
  if (!rec || !rec.rater) return "kind and exposure not returned";
  const kind = rec.rater.kind, exposure = rec.exposure || "exposure not returned";
  if (kind === "owner") return `owner, ${exposure}: a ${exposure === "revealed" ? "revealed " : ""}demonstration label, not expert calibration`;
  if (kind === "expert") return `expert, ${exposure}${exposure === "revealed" ? ": stored as revealed, not a blind label" : ""}`;
  if (kind === "model_judge") return "model judge, not an expert";
  return `${kind}, ${exposure}`;
}
function analysisReport(e) {
  const V = view();
  if (!V.signoff) return "";
  const own = V.revealed && V.revealed.signoff;
  const who = (own && own.rater && own.rater.principal_id) || S.me || "this rater";
  const f = e.fit || {}, ph = phasesOf(e), cited = citedOf(e);
  const verdict = (VERDICTS.find(([v]) => v === V.signoff.verdict) || [V.signoff.verdict, V.signoff.verdict])[1];
  const comment = (V.revealed && V.revealed.signoff && V.revealed.signoff.comment) || V.comment || "";
  const phases = ph.map((p) => `<tr><td>${swatch(p.id)}<b>${esc(p.formula || p.id)}</b> <span class="mono">${esc(sgShort(p.sg))}</span><div class="mono muted">${esc(p.id)}${p.reported && p.cif.formula && p.cif.formula !== p.formula ? ` · CIF ${esc(p.cif.formula)}` : ""}</div></td>
    <td class="num">${p.wt != null ? fmt(p.wt) : "n/a"}${p.esd != null ? ` ± ${fmt(p.esd)}` : ""}</td><td class="num">${p.shift != null ? fmt(p.shift) + "%" : "n/a"}</td>
    <td>${answerText(V, labelKey("phase_supported", p.id))}</td><td>${answerText(V, labelKey("phase_plausible", p.id))}</td></tr>`).join("");
  const causes = Object.fromEntries(PEAK_CAUSES);
  const peaks = peaksOf(e).map((p) => `<tr><td class="num">${fmt(p.tt, 2)}°</td><td class="num">${fmt(p.pct, 1)}%</td><td>${V.peaks[p.id] ? `<b>${esc(causes[V.peaks[p.id].cause] || V.peaks[p.id].cause)}</b>${V.peaks[p.id].note ? `<div class="why">${esc(V.peaks[p.id].note)}</div>` : ""}` : '<span class="muted">not called</span>'}</td></tr>`).join("");
  const checks = (e.checks || []).map((c) => checkView(c, e)).map((v, i) => [e.checks[i], v]).map(([c, v]) => `<tr><td>${esc(c.code || "")}</td><td>${esc(v.name)}</td><td>${esc(v.label)}</td><td class="why">${esc(c.detail)}</td></tr>`).join("");
  const refits = V.refits.filter((r) => r.child).map((r) => {
    const c = r.child;
    return `<div class="keep"><h3>${esc(editText(e, r.edit))}: child attempt <span class="mono">${esc(String(c.digest).slice(0, 12))}</span></h3>
      <p>Rwp ${fmt(c.rwp)}% against the answer's ${fmt(cited.rwp)}%${c.stage ? `, stage ${esc(c.stage)}` : ""}${r.elapsed != null ? `, ${clock(r.elapsed)} on BGMN` : ""}. Phases: ${c.phases.map((x) => `${swatch(x)}${esc(formulaOf(e, x))} <span class="mono muted">${esc(x)}</span>`).join(", ")}.${r.rationale ? ` Why: ${esc(r.rationale)}.` : ""}${r.snapshot ? ` Frozen as expert correction <span class="mono">${esc(String(r.snapshot).slice(0, 12))}</span>.` : ""}</p>
      ${c.review && c.review.fit ? reportPlot(e, c.review) : ""}</div>`;
  }).join("") || '<p class="muted">No refit in this session.</p>';
  const claims = claimsOf(e).map((c) => { const [, label, why] = claimMark(c, e); return `<div class="claim"><span class="k">${esc(c.kind || "")}</span> ${esc(c.text)} <span class="chip">${esc(label.toUpperCase())}</span>${why ? `<div class="why">${esc(why)}</div>` : ""}</div>`; }).join("") || '<p class="muted">No claims in this snapshot.</p>';
  const firstFail = (e.steps || []).map((x) => [x, stepRef(e, x)]).find(([, ref]) => ref && (V.marks[stepKey(ref.attempt, ref.call_index)] || []).some((m) => m.mark === "went_wrong_here"));
  const analysis = ANALYSIS_ITEMS.map((item) => `<tr><td>${esc(RUBRIC[item])}</td><td>${answerText(V, labelKey(item, null))}</td></tr>`).join("");
  const when = new Date().toISOString().replace("T", " ").slice(0, 16) + " UTC";
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhaseEval report · ${esc(e.case_id)} · ${esc(String(e.digest).slice(0, 12))}</title><style>${REPORT_CSS}</style></head><body><div class="page">
<div class="tools"><button type="button" class="print" onclick="window.print()">Print or save as PDF</button></div>
<h1>PhaseEval analysis report</h1>
<div class="meta">${esc(e.case_id)} · ${esc(e.kind === "expert_correction" ? "expert correction" : "agent answer")} · snapshot <span class="mono">${esc(String(e.digest).slice(0, 12))}</span> · rater ${esc(who)} (${esc(provenance(own))}) · generated ${when}</div>
<h2>Sample</h2>${openAll(agentLine(e))}<div class="brief">${briefBar(e)}</div>
<h2>Cited fit</h2><div class="keep"><p>${esc(f.stage || e.stage || "")} · Rwp ${fmt(f.rwp ?? e.rwp_percent)}% · GoF ${fmt(f.gof ?? e.gof)}${cited.label ? ` · cited run <span class="mono">${esc(cited.label)}</span>` : ""} · ${ph.map((p) => `${swatch(p.id)}${esc(p.formula || p.id)}${p.wt != null ? ` ${fmt(p.wt, 1)} wt%` : ""}`).join(" · ")}</p>${reportPlot(e)}</div>
<h2>Phases and your answers</h2><table><thead><tr><th>Phase · CIF id</th><th class="num">wt% <span class="muted">modeled crystalline, ± formal esd</span></th><th class="num">Cell shift</th><th>${esc(RUBRIC.phase_supported)}</th><th>${esc(RUBRIC.phase_plausible)}</th></tr></thead><tbody>${phases}</tbody></table>
<h2>Unexplained peaks and your calls</h2>${peaks ? `<table><thead><tr><th>2θ</th><th>Height, % of strongest net peak</th><th>Your call</th></tr></thead><tbody>${peaks}</tbody></table>` : '<p class="muted">The residual check lists none.</p>'}
<h2>Checks</h2><table><thead><tr><th>Code</th><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>${checks}</tbody></table>
<h2>Chemistry</h2>${openAll(chemistryPanel(e))}
<h2>Refits</h2>${refits}
<h2>The agent's write-up</h2>${claims}
<h2>Sign-off</h2><table><tbody>${analysis}<tr><td>Verdict</td><td><b>${esc(verdict)}</b></td></tr><tr><td>Recorded as</td><td>${esc(who)}, ${esc(provenance(own))}${own && own.created_at_utc ? ` · ${esc(String(own.created_at_utc).slice(0, 16).replace("T", " "))} UTC` : ""}</td></tr><tr><td>Critique</td><td>${comment ? esc(comment) : '<span class="muted">none written</span>'}</td></tr>${firstFail ? `<tr><td>First failing step</td><td>${clock(firstFail[0].t_start ?? firstFail[0].t)} · ${esc(firstFail[0].run_label || firstFail[0].title || firstFail[0].kind)}</td></tr>` : ""}</tbody></table>
<h2>After sign-off</h2>${openAll(revealPanel(e)) || '<p class="muted">Nothing revealed.</p>'}
<p class="muted" style="margin-top:18px">Built in the browser from the records the server returned to this rater after their sign-off. It shows nothing the review page does not.</p>
</div></body></html>`.replace(/#E69F00|#56B4E9|#F0E442/g, (c) => PAPER[c]);
}
// The pattern, CIFs and the cited fit's BGMN files as a zip, from the gated files route.
async function downloadFiles() {
  const e = S.ep;
  try {
    const blob = await call("files", { digest: e.digest }, { raw: true });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `phaseeval-${String(e.digest).slice(0, 12)}.zip`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  } catch (err) { status(`No files: ${err.message}`, true); }
}
function openReport() {
  const html = analysisReport(S.ep);
  if (!html) return;
  const w = window.open("", "_blank");
  if (!w) { status("The browser blocked the report window. Allow pop-ups for this page.", true); return; }
  w.document.open(); w.document.write(html); w.document.close();
  // Dogfood F7: the same page as a file for the rater's records.
  const tools = w.document.querySelector(".tools");
  if (tools) {
    const a = w.document.createElement("a");
    a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    a.download = `phaseeval-report-${String(S.ep.digest).slice(0, 12)}.html`;
    a.textContent = "Save as HTML"; a.className = "print"; a.style.marginLeft = "8px"; a.style.textDecoration = "none";
    tools.appendChild(a);
  }
}

function consentPanel() {
  if (!S.consentOpen && (S.permission === undefined || (S.permission && S.permission.participation_and_storage))) return "";
  return `<section class="panel consent" id="consent" style="margin-top:14px"><h3>Before your labels are stored</h3>
    <form class="body" id="consent-form">
      <p class="muted" style="margin:0 0 8px">Each choice is separate and starts unchecked. Withdrawing later removes your records from every later export.</p>
      ${PERMISSIONS.map(([k, text]) => `<label><input type="checkbox" name="${k}"> <span>${esc(text)}</span></label>`).join("")}
      <button type="submit" class="btn" style="margin-top:8px">Record my choices</button>${S.consentOpen ? ' <button type="button" class="btn ghost" data-act="consent-close" style="margin-top:8px">Close</button>' : ""}
    </form></section>`;
}

function briefBar(e) {
  const b = e.brief || {}, acq = b.acquisition || {};
  const proc = (b.processing || []).map((p) => {
    const bits = [p.temperature_c != null ? `${fmt(p.temperature_c, 0)} °C` : "", p.duration_min != null ? `for ${fmt(p.duration_min, 0)} min` : "", p.atmosphere ? `in ${p.atmosphere}` : "", p.furnace ? `furnace ${p.furnace}` : ""].filter(Boolean);
    const other = Object.entries(p.other || {}).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`);
    return `<b>${esc(p.action || p.operation || "step")}</b>${bits.length || other.length ? ` (${esc([...bits, ...other].join(", "))})` : ""}`;
  });
  const precursors = (b.precursors || []).map((p) => p.formula + (p.mass_g != null ? ` ${fmt(p.mass_g, 3)} g` : "")).join(", ");
  return `<span><b>${esc(e.case_id || b.case_id)}</b> ${esc(b.specimen || "")}</span>
    <span>Precursors <b>${esc(precursors || "none declared")}</b>${b.precursor_amounts_note ? ` <span class="muted">(amounts ${esc(b.precursor_amounts_note)})</span>` : ""}</span>
    ${b.declared_target && b.declared_target.formula ? `<span>Target <b>${esc(b.declared_target.formula)}</b></span>` : ""}
    <span>λ <b class="num">${esc(acq.wavelength_angstrom ?? "")} Å</b>${acq.instrument ? ` ${esc(acq.instrument)}` : ""}</span>
    <span>Profile <b>${esc(b.instrument_profile || "")}</b>${b.profile_caveat ? ` <span class="caveat" title="No calibrated instrument profile for this scan.${b.quantitative_claims && /restricted/.test(b.quantitative_claims) ? " Weight fractions carry this caveat." : ""}">stand-in${b.quantitative_claims && /restricted/.test(b.quantitative_claims) ? ", wt% approximate" : ""}</span>` : ""}</span>
    <span>Processing ${proc.length ? proc.join(" → ") : '<span class="muted">none recorded</span>'}</span>
    ${(e.missing || []).length ? `<span class="muted" title="${esc(e.missing.join(", "))}">${e.missing.length} field${e.missing.length === 1 ? "" : "s"} not recorded</span>` : ""}`;
}

function eventLabel(event) {
  const n = event.ref.turn_index + 1;
  if (event.kind === "message") return `Turn ${n} · ${event.turn.source || "message"}`;
  return `Turn ${n} · call ${event.ref.call_index} · ${event.call.tool || "tool"}`;
}
function callResult(call) {
  return call.output ?? call.result ?? call.shown ?? call.content ?? "";
}
function methodEventBody(event) {
  if (!event) return '<p class="muted">Select a recorded event.</p>';
  const rationale = event.kind === "call" && event.turn.message ? `<div class="method-block"><span>Recorded rationale</span><pre>${esc(event.turn.message)}</pre></div>` : "";
  if (event.kind === "message") return `<div class="method-block"><span>Selected message</span><pre>${esc(event.text)}</pre></div>`;
  const action = event.call.command || event.call.input || event.call.arguments || event.call.tool || "(action not recorded)";
  const result = callResult(event.call);
  return `${rationale}<div class="method-block"><span>Selected action · ${esc(event.call.tool || "tool")}</span><p>${esc(methodActionDescription(event.call))}</p><details><summary>Recorded inputs</summary><pre>${esc(typeof action === "string" ? action : JSON.stringify(action, null, 2))}</pre></details></div><div class="method-block"><span>Result shown</span><pre>${result === "" ? '<span class="muted">(no shown result recorded)</span>' : esc(typeof result === "string" ? result : JSON.stringify(result, null, 2))}</pre></div>`;
}
function methodPrefix(trace, selected) {
  if (!selected) return "";
  const turns = (trace || []).slice(0, selected.ref.turn_index);
  if (!turns.length) return '<p class="muted">No earlier turns.</p>';
  return turns.map((turn, i) => `<details class="prefix-turn" ${i === turns.length - 1 ? "open" : ""}><summary>Turn ${i + 1} · ${esc(turn.source || "message")} ${(turn.calls || []).map(c => esc(c.tool)).join(", ")}</summary>${turn.message ? `<pre>${esc(turn.message)}</pre>` : ""}${(turn.calls || []).map((call) => `<div class="prefix-call"><span>${esc(call.tool || "tool")} · call ${esc(call.call_index)}</span><pre>${esc(callResult(call) || "(no shown result recorded)")}</pre></div>`).join("")}</details>`).join("");
}
function methodDraftError(M) {
  const d = M.draft;
  if (!d.verdict) return "Choose Pass, Fail, or Defer.";
  if (!d.note.trim()) return "Explain the verdict in the note.";
  if (d.verdict === "fail" && !d.earliest_error) return "Fail needs an earliest error.";
  if ((d.verdict === "pass" || d.verdict === "fail") && !d.evidence.length) return `${d.verdict === "pass" ? "Pass" : "Fail"} needs evidence.`;
  if (d.verdict === "pass" && d.earliest_error && d.repair_status !== "repaired") return "Pass with an earlier error requires an observed repair.";
  if (!d.earliest_error && (d.observed_repair || d.repair_status !== "not_applicable")) return "Repair fields require an earliest error.";
  if (d.earliest_error && d.repair_status === "not_applicable") return "Choose the repair status.";
  if ((d.repair_status === "repaired") !== Boolean(d.observed_repair)) return "Repaired requires an observed later repair, and other statuses cannot cite one.";
  return "";
}
function methodReview(e) {
  const M = methodState(e.digest), data = M.data || {}, trace = data.trace || [], events = methodEvents(trace);
  if (M.loadError) return `<section class="panel method-intro"><h3>Method review unavailable</h3><div class="body"><p class="err">Could not load the recorded trajectory: ${esc(M.loadError)}</p><button type="button" class="qbtn" data-act="method-reload">Retry</button></div></section>`;
  const selected = events.find((x) => sameRef(x.ref, M.selected)) || events[0] || null;
  if (selected && !M.selected) M.selected = selected.ref;
  const d = M.draft, earliestTurn = d.earliest_error && d.earliest_error.turn_index;
  const evidenceAllowed = (event) => earliestTurn == null || event.ref.turn_index < earliestTurn;
  const errorEvents = events.filter((x) => x.turn.source === "agent");
  const repairEvents = errorEvents.filter((x) => d.earliest_error && x.ref.turn_index > d.earliest_error.turn_index);
  const option = (event) => `<option value="${esc(refKey(event.ref))}">${esc(eventLabel(event))} · ${esc(String(event.text || "").slice(0, 100))}</option>`;
  const limitations = (data.limitations || []).map((x) => `<li>${esc(x)}</li>`).join("");
  const verdicts = [["pass", "Pass"], ["fail", "Fail"], ["defer", "Defer"]].map(([v, l]) => `<button type="button" class="btn ghost" data-act="method-verdict" data-val="${v}" aria-pressed="${d.verdict === v}">${l}</button>`).join("");
  const saved = M.saved ? `Saved ${esc(M.saved.created_at_utc || "record")}${M.saved.submission_id ? ` · ${esc(M.saved.submission_id)}` : ""}` : "Not yet saved";
  return `<div class="method-shell">
    <section class="panel method-intro"><h3>Method review v1 <span class="hint">independent of scientific outcome review</span></h3><div class="body"><p>Review the recorded sequence using only information available before the selected turn. This is a retrospective record and may not reconstruct model context after compaction.</p>${limitations ? `<ul class="limitations">${limitations}</ul>` : ""}</div></section>
    <div class="method-grid"><section class="panel method-events"><h3>Recorded chronology <span class="hint">messages and calls</span></h3><div class="event-list">${events.length ? events.map((event) => `<button type="button" class="event-row" data-act="method-event" data-ref="${esc(refKey(event.ref))}" aria-pressed="${sameRef(event.ref, selected && selected.ref)}"><span>${esc(eventLabel(event))}</span><small>${esc(String(event.text || "(no message)").slice(0, 160))}</small></button>`).join("") : '<p class="muted">No trajectory was recorded for this snapshot.</p>'}</div></section>
      <div class="method-detail"><section class="panel"><h3>Earlier turns only</h3><div class="body method-prefix">${methodPrefix(trace, selected)}</div></section><section class="panel" id="method-selected-event"><h3>Selected event</h3><div class="body">${methodEventBody(selected)}</div></section></div></div>
    <section class="panel method-form"><h3>Method judgment <span class="hint">explain your judgment and cite the evidence</span></h3><div class="body">
      <div class="method-rubric"><span>Was the action justified by information and budget?</span><span>Did it usefully discriminate possibilities?</span><span>Did interpretation or revision distinguish measurement from model uncertainty?</span><span>Was verification or recovery adequate?</span><span>Was stopping justified, and what uncertainty remained?</span></div>
      <div class="sign">${verdicts}</div><label>Verdict note<textarea aria-label="Verdict note" data-method="note" data-fk="method-note" maxlength="4000">${esc(d.note)}</textarea></label>
      <fieldset><legend>Evidence</legend><div class="evidence-list">${events.map((event) => `<div class="evidence-row"><label class="evidence-ref"><input type="checkbox" data-method-evidence="${esc(refKey(event.ref))}" ${d.evidence.some((x) => sameRef(x, event.ref)) ? "checked" : ""} ${evidenceAllowed(event) ? "" : "disabled"}><span>${esc(eventLabel(event))} · ${esc(String(event.text || "").slice(0, 120))}</span></label><button type="button" class="qbtn small" data-act="method-event" data-ref="${esc(refKey(event.ref))}" data-jump="true" aria-label="View ${esc(eventLabel(event))}">View</button></div>`).join("") || '<span class="muted">No recorded events.</span>'}</div></fieldset>
      <div class="method-fields"><label>Earliest error<select aria-label="Earliest error" data-method="earliest_error"><option value="">No observed error</option>${errorEvents.map(option).join("")}</select></label><label>Actual repair status<select aria-label="Actual repair status" data-method="repair_status" ${d.earliest_error ? "" : "disabled"}><option value="not_applicable">Not applicable</option><option value="repaired">Repaired</option><option value="unrepaired">Unrepaired</option><option value="unknown">Unknown</option></select></label><label>Observed later repair<select aria-label="Observed later repair" data-method="observed_repair" ${d.earliest_error ? "" : "disabled"}><option value="">None recorded</option>${repairEvents.map(option).join("")}</select></label></div>
      <label>Optional proposed repair<textarea aria-label="Optional proposed repair" data-method="proposed_repair" data-fk="method-proposed" maxlength="4000">${esc(d.proposed_repair)}</textarea></label>
      <div class="method-save"><button type="button" class="btn" data-act="method-save" ${methodDraftError(M) || !methodDirty(M) || M.saving ? "disabled" : ""}>${M.saving ? "Saving…" : M.submissionId ? "Retry save" : M.saved ? "Save correction" : "Save review"}</button><button type="button" class="btn ghost" data-act="method-discard" ${methodDirty(M) && !M.saving ? "" : "disabled"}>Discard edits</button><span class="muted">${saved}</span>${methodDraftError(M) && methodDirty(M) ? `<span class="req">${esc(methodDraftError(M))}</span>` : ""}${M.error ? `<span class="err">Not saved: ${esc(M.error)}</span>` : ""}</div>
    </div></section></div>`;
}

async function saveMethodReview() {
  const e = S.ep, M = methodState(e.digest), problem = methodDraftError(M);
  if (problem || M.saving || !methodDirty(M)) return;
  const body = methodRequest(M, e.digest);
  M.saving = true; M.error = ""; render();
  try {
    const record = await call("postTrajectoryReview", { digest: e.digest }, { body });
    restoreMethod(M, record);
    status("Method review saved.");
  } catch (err) { M.error = err.message; }
  M.saving = false; render();
}

function render() {
  const active = document.activeElement, fk = active && active.dataset && active.dataset.fk, caret = active && active.selectionStart;
  const e = S.ep;
  const n = S.queue.length;
  if (!e) {
    $("crumb").innerHTML = token() ? (n ? "Loading" : "Review queue · empty") : "Review queue";
    $("stages").innerHTML = ""; $("brief").innerHTML = "";
    $("main").innerHTML = S.error ? `<p class="muted">${esc(S.error)}</p>` : "";
    return;
  }
  const V = view();
  S.steps = {};
  const M = methodState(e.digest);
  document.querySelectorAll('.qnav button').forEach((button) => { button.disabled = methodBlocked(); });
  const kind = e.kind === "expert_correction" ? "expert correction" : e.kind === "agent_answer" ? "agent answer" : esc(e.kind || "");
  $("crumb").innerHTML = `Queue <b>${S.i + 1} of ${n}</b> · ${esc(e.case_id || e.digest.slice(0, 12))} · ${kind}`;
  const modeButtons = `<span class="mode-toggle" role="group" aria-label="Review mode"><button type="button" class="qbtn" data-act="mode" data-mode="method" aria-pressed="${S.mode === "method"}">Method</button><button type="button" class="qbtn" data-act="mode" data-mode="outcome" aria-pressed="${S.mode === "outcome"}" ${S.mode === "method" && methodDirty(M) ? "disabled" : ""}>Outcome</button></span>`;
  if (S.mode === "method") {
    $("stages").innerHTML = `${modeButtons}<span class="stage ${M.saved ? "done" : "now"}"><span class="dot"></span>${M.saved ? "Method review saved" : "Review the recorded method"}</span><button type="button" class="qbtn small" data-act="consent-open" style="margin-left:auto">Participation choices</button>`;
    $("brief").innerHTML = "";
    $("main").innerHTML = `${consentPanel()}${methodReview(e)}`;
    if (M.saving || (M.error && M.submissionId)) document.querySelectorAll('[data-method], [data-method-evidence], [data-act="method-verdict"]').forEach((control) => { control.disabled = true; });
    const set = (name, value) => { const el = document.querySelector(`[data-method="${name}"]`); if (el) el.value = methodControlValue(name, value); };
    set("earliest_error", M.draft.earliest_error); set("repair_status", M.draft.repair_status); set("observed_repair", M.draft.observed_repair);
    const u = new URL(location.href); u.searchParams.set("snapshot", e.digest); u.searchParams.delete("view"); history.replaceState(null, "", u);
    if (fk) { const el = document.querySelector(`[data-fk="${CSS.escape(fk)}"]`); if (el) { el.focus(); if (caret != null && el.setSelectionRange) el.setSelectionRange(caret, caret); } }
    return;
  }
  const evDone = evidenceDone(e, V);
  $("stages").innerHTML = modeButtons + [["Judge the evidence", V.unlocked, !V.unlocked], ["Read the write-up", Boolean(V.signoff), V.unlocked && !V.signoff], ["Sign off", Boolean(V.signoff), false], ["Reveal", Boolean(V.signoff && V.revealed), Boolean(V.signoff) && !V.revealed]]
    .map(([l, done, cur]) => `<span class="stage ${done ? "done" : cur ? "now" : ""}"><span class="dot"></span>${l}</span>`).join("") + (V.unlocked && !evDone ? `<span class="stage"><span class="req">an evidence answer was cleared</span></span>` : "") + (V.revealed && !V.signoff ? `<span class="stage"><span class="req">Reveal already seen for this pattern: labels here are stored as revealed.</span></span>` : "") + `<button type="button" class="qbtn small" data-act="consent-open" style="margin-left:auto">Participation choices</button>`;
  $("brief").innerHTML = agentLine(e) + briefBar(e);
  // A re-render keeps every disclosure the rater opened.
  const opened = new Set([...document.querySelectorAll("details[data-dk][open]")].map((d) => d.dataset.dk));
  $("main").innerHTML = `${consentPanel()}<div class="wide">${evidence(e)}</div><div class="grid"><div class="stack">${phaseTable(e)}</div><div class="stack">${peakPanel(e)}${chemistryPanel(e)}</div></div><div class="stack" style="margin-top:16px">${writeUp(e)}${signOff(e)}${revealPanel(e)}</div>`;
  document.querySelectorAll("details[data-dk]").forEach((d) => { if (opened.has(d.dataset.dk)) d.open = true; });
  if (!S.cur || !cursorKeys(e).includes(S.cur)) S.cur = cursorKeys(e).find((k) => !answeredKey(V, k)) || null;
  showCursor();
  const u = new URL(location.href); u.searchParams.set("snapshot", e.digest); history.replaceState(null, "", u);
  if (fk) {
    const el = document.querySelector(`[data-fk="${CSS.escape(fk)}"]`);
    if (el) { el.focus(); if (caret != null && el.setSelectionRange) try { el.setSelectionRange(caret, caret); } catch {} }
  }
}

// ---------------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------------
async function openSnapshot(i) {
  if (i < 0 || i >= S.queue.length) return;
  if (methodBlocked()) { status("Save or discard method edits before leaving this snapshot.", true); return; }
  if (S.ep) { const old = view(); old.seconds += Math.round((Date.now() - old.opened) / 1000); }
  S.i = i; S.peakSel = null; S.phaseSel = null; S.range = null; S.openClaims = new Set(); S.cur = null; S.childFit = null; S.showSteps = false; S.markOpen = null; S.markDraft = null; S.refitDraft = null;
  const digest = S.queue[i].digest;
  status("Loading snapshot.");
  try {
    const M = methodState(digest);
    if (!M.data) {
      try { M.data = await call("trajectoryReview", { digest }); M.loadError = ""; }
      catch (err) { M.loadError = err.message; throw err; }
      const reviews = listOf(M.data.reviews, "reviews", "items");
      restoreMethod(M, reviews.length ? reviews[reviews.length - 1] : null);
    }
    if (S.mode === "method") {
      S.ep = { ...S.queue[i], digest };
      status(""); render(); window.scrollTo(0, 0); return;
    }
    const [payload, labels] = await Promise.all([call("review", { digest }), call("labels", { digest }).catch(() => [])]);
    S.ep = payload;
    S.ep.digest = S.ep.digest || digest;
    const V = view(digest);
    V.opened = Date.now();
    if (!V.outcomeLoaded) {
      restoreLabels(V, listOf(labels, "labels", "items"));
      // No route reads peak calls back; this tab remembers the calls the server accepted.
      try { Object.assign(V.peaks, JSON.parse(sessionStorage.getItem(PEAKS_KEY + digest) || "{}")); } catch {}
      refreshUnlock(payload, V);
      const attempts = [...new Set((payload.steps || []).map((s) => s.attempt).filter(Boolean))];
      await restoreRefits(payload, V);
      V.refits.filter((r) => r.child).forEach((r) => attempts.push(r.child.digest));
      const marks = await Promise.all(attempts.map((a) => call("stepFeedback", {}, { query: { attempt: a } }).catch(() => [])));
      marks.forEach((m) => restoreMarks(V, listOf(m, "step_feedback", "items")));
      await loadComposition(payload, V);
      await loadReveal(digest);
      V.outcomeLoaded = true;
    }
    status("");
  } catch (err) { S.ep = { ...S.queue[i], digest }; S.error = `Could not load snapshot ${digest.slice(0, 12)}: ${err.message}`; status(S.error, true); }
  render();
  window.scrollTo(0, 0);
}
async function boot() {
  if (!token()) {
    try {
      const response = await fetch("/api/launcher/review-session", { method: "POST", credentials: "same-origin", cache: "no-store" });
      if (response.ok) setToken((await response.json()).token || "");
    } catch {}
  }
  if (!token()) {
    $("auth").classList.remove("hidden");
    S.error = "Open PhaseEval with its launcher to sign in. Review links then work in new tabs in this browser for 12 hours. You can also use a review token below.";
    render(); return;
  }
  S.error = "";
  $("auth").classList.add("hidden");
  try { const c = await call("capabilities"); S.refit = { available: c.refit.available !== false, note: c.refit.note || "" }; } catch {}
  try {
    const q = await call("queue");
    S.queue = listOf(q, "items", "snapshots", "queue").map((x) => (typeof x === "string" ? { digest: x } : x)).filter((x) => x && x.digest);
    if (!S.queue.length) { S.error = "The review queue is empty."; render(); return; }
    const wanted = new URLSearchParams(location.search).get("snapshot");
    await openSnapshot(Math.max(0, S.queue.findIndex((x) => x.digest === wanted)));
  } catch (err) { S.error = err.message; status(err.message, true); render(); }
}

// ---------------------------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------------------------
async function loadTraceCall(digest, index) {
  const k = `${digest}:${index}`;
  try { S.traceFull[k] = await call("traceCall", { digest, call: index }); } catch (err) { S.traceFull[k] = { error: err.message }; }
  render();
}
function go(d) { if (methodBlocked()) { status("Save or discard method edits before navigating.", true); return; } if (S.queue.length) openSnapshot((S.i + d + S.queue.length) % S.queue.length); }
$("auth").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const value = $("token-input").value.trim();
  if (!value) return;
  setToken(value); $("token-input").value = "";
  boot();
});
document.addEventListener("submit", (ev) => {
  if (ev.target.id === "consent-form") { ev.preventDefault(); savePermission(ev.target); }
  else if (ev.target.id === "composition-form") { ev.preventDefault(); saveComposition(); }
});
document.addEventListener("input", (ev) => {
  const t = ev.target, V = S.ep && view();
  if (!V) return;
  if ((t.dataset.method === "note" || t.dataset.method === "proposed_repair") && S.ep) { methodState(S.ep.digest).draft[t.dataset.method] = t.value; render(); }
  else if (t.dataset.reason) { V.drafts[t.dataset.reason] = t.value; const b = t.parentElement.querySelector('[data-act="save-reason"]'); if (b) b.disabled = !t.value.trim(); }
  else if (t.dataset.crit) { V.crit[t.dataset.crit] = t.value; const b = t.parentElement.querySelector('[data-act="save-critique"]'); if (b) b.disabled = critiqueRequired(V, t.dataset.crit) && !t.value.trim(); }
  else if (t.dataset.comment) {
    // "would not sign" needs its critique; the sign button follows the text without a re-render.
    const was = Boolean(V.comment.trim()); V.comment = t.value;
    if (V.verdict === "would_not_sign" && was !== Boolean(t.value.trim())) render();
  }
  else if (t.dataset.markNote) { S.markDraft = { ...(S.markDraft || { mark: null }), note: t.value }; const b = t.closest(".markbox").querySelector('[data-act="mark-save"]'); if (b) b.disabled = !(S.markDraft.mark && (S.markDraft.mark !== "note" || t.value.trim())); }
  else if (t.dataset.refitRationale && S.refitDraft) S.refitDraft.rationale = t.value;
  else if ((t.dataset.compEl || t.dataset.compV || t.dataset.comp === "source") && V.composition) {
    const d = V.composition.draft;
    if (t.dataset.comp === "source") d.source = t.value; else d.rows[Number(t.dataset.compEl ?? t.dataset.compV)][t.dataset.compEl != null ? "el" : "v"] = t.value;
    const b = document.querySelector('#composition-form button[type="submit"]');
    if (b) b.disabled = !(d.method && d.basis && d.source.trim() && d.rows.every((r) => /^[A-Z][a-z]?$/.test(r.el.trim()) && r.v.trim() !== "" && Number.isFinite(Number(r.v))));
  }
  else if (t.dataset.refitCandidate && S.refitDraft) { S.refitDraft.candidates = t.value.trim() ? [t.value.trim()] : []; const b = document.querySelector('[data-act="run-refit"]'); if (b) b.disabled = !S.refitDraft.candidates.length; }
});
document.addEventListener("change", (ev) => {
  const t = ev.target;
  if (t.dataset.methodEvidence && S.ep) {
    const M = methodState(S.ep.digest), ref = methodRefFor(M, t.dataset.methodEvidence);
    if (ref) M.draft.evidence = t.checked ? [...M.draft.evidence, ref] : M.draft.evidence.filter((x) => !sameRef(x, ref));
    render();
  } else if (t.dataset.method && S.ep) {
    const M = methodState(S.ep.digest), field = t.dataset.method;
    updateMethodSelect(M, field, t.value);
    render();
  } else if (t.dataset.refitCandidate && S.refitDraft) { S.refitDraft.candidates = t.value ? [t.value] : []; render(); }
  else if (t.dataset.firstFail && S.ep && t.value !== "") {
    const ref = (S.ep.steps || []).map((x) => stepRef(S.ep, x)).filter(Boolean)[Number(t.value)];
    if (ref) saveMark(ref, "went_wrong_here", "first failing step");
  }
  else if (t.dataset.comp && t.tagName === "SELECT" && S.ep && view().composition) { view().composition.draft[t.dataset.comp] = t.value; render(); }
});
function saveCritique(k) {
  const V = view(), l = V.labels[k], text = String(V.crit[k] ?? "").trim();
  if (!(k in V.crit) || (!l && !V.pendingNo[k]) || (critiqueRequired(V, k) && !text)) return;
  const [, item, phase] = k.split("|");
  saveLabel(item, phase || null, V.pendingNo[k] ? "no" : l.answer, V.pendingNo[k] ? "" : l.reason || "", text);
}
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && ev.target.dataset && ev.target.dataset.reason) { ev.preventDefault(); const b = ev.target.parentElement.querySelector('[data-act="save-reason"]'); if (b && !b.disabled) b.click(); return; }
  if (ev.target.dataset && ev.target.dataset.crit) {
    if (ev.key === "Enter") { ev.preventDefault(); saveCritique(ev.target.dataset.crit); }
    else if (ev.key === "Escape") { const k = ev.target.dataset.crit; delete view().crit[k]; delete view().pendingNo[k]; render(); }
    return;
  }
  if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") { const b = document.querySelector('[data-act="sign"]'); if (b && !b.disabled) { ev.preventDefault(); b.click(); } return; }
  if (ev.target.closest("input, textarea, select, [contenteditable]")) return;
  if (rubricKey(ev)) { ev.preventDefault(); return; }
  if (ev.key === "j" || ev.key === "n" || ev.key === "ArrowRight") go(1);
  if (ev.key === "k" || ev.key === "ArrowLeft") go(-1);
  if ((ev.key === "Enter" || ev.key === " ") && ev.target.closest(".peak-box")) { ev.preventDefault(); S.peakSel = ev.target.closest(".peak-box").dataset.peak; render(); }
});
// 2θ zoom: drag across a plot; its tick rows, markers and difference strip follow the range.
let drag = null;
document.addEventListener("mousedown", (ev) => {
  const svg = ev.button === 0 && ev.target.closest && ev.target.closest("svg.plot[data-zoom]");
  if (!svg) return;
  const [x0, x1, l, r, w] = svg.dataset.zoom.split(",").map(Number), box = svg.getBoundingClientRect();
  const vb = (cx) => Math.min(r, Math.max(l, (cx - box.left) / box.width * w));
  drag = { svg, vb, at: vb(ev.clientX), t: (v) => x0 + (v - l) / (r - l) * (x1 - x0) };
});
document.addEventListener("mousemove", (ev) => {
  if (!drag) return;
  const v = drag.vb(ev.clientX), sel = drag.svg.querySelector(".zoom-sel");
  if (sel) { sel.setAttribute("x", Math.min(v, drag.at).toFixed(1)); sel.setAttribute("width", Math.abs(v - drag.at).toFixed(1)); }
});
document.addEventListener("mouseup", (ev) => {
  if (!drag) return;
  const d = drag, v = d.vb(ev.clientX);
  drag = null;
  if (Math.abs(v - d.at) < 8) { const sel = d.svg.querySelector(".zoom-sel"); if (sel) sel.setAttribute("width", 0); return; }
  S.range = [d.t(Math.min(v, d.at)), d.t(Math.max(v, d.at))]; S.zoomedAt = Date.now();
  render();
});
document.addEventListener("click", (ev) => {
  const box = ev.target.closest(".peak-box");
  if (box && S.zoomedAt && Date.now() - S.zoomedAt < 400) return;
  if (box) { S.peakSel = box.dataset.peak; render(); const row = document.querySelector(`[data-peakrow="${CSS.escape(S.peakSel)}"]`); if (row) row.scrollIntoView({ block: "nearest" }); return; }
  const b = ev.target.closest("button[data-act]");
  if (!b || b.disabled) return;
  const e = S.ep, V = e && view(), act = b.dataset.act;
  if (act === "prev") return go(-1);
  if (act === "method-reload") { methodState(e.digest).loadError = ""; openSnapshot(S.i); return; }
  if (act === "mode") { S.mode = b.dataset.mode; openSnapshot(S.i); return; }
  if (act === "consent-open" || act === "consent-close") { S.consentOpen = act === "consent-open"; render(); if (S.consentOpen) $("consent").scrollIntoView({ block: "nearest" }); return; }
  if (act === "next") return go(1);
  if (!e) return;
  if (act === "method-event") { const M = methodState(e.digest), ref = methodRefFor(M, b.dataset.ref); if (ref) M.selected = ref; render(); if (b.dataset.jump) $("method-selected-event").scrollIntoView({block:"nearest"}); }
  else if (act === "method-verdict") { methodState(e.digest).draft.verdict = b.dataset.val; render(); }
  else if (act === "method-save") saveMethodReview();
  else if (act === "method-discard") { restoreMethod(methodState(e.digest), methodState(e.digest).saved); status("Method edits discarded."); render(); }
  else if (act === "answer") {
    const item = b.dataset.item, phase = b.dataset.phase || null, k = labelKey(item, phase), val = b.dataset.val;
    S.cur = k;
    if (val === "cannot_assess") { delete V.pendingNo[k]; delete V.crit[k]; if (!(k in V.drafts)) V.drafts[k] = (V.labels[k] && V.labels[k].reason) || ""; render(); const inp = document.querySelector(`[data-reason="${CSS.escape(k)}"]`); if (inp) inp.focus(); return; }
    delete V.drafts[k];
    if (val === "no") {
      // Fail needs a critique: the answer is stored with it, from the critique box.
      if (!(k in V.crit)) V.crit[k] = (V.labels[k] && V.labels[k].note) || "";
      if ((V.labels[k] || {}).answer !== "no") V.pendingNo[k] = true;
      render();
      const inp = document.querySelector(`[data-crit="${CSS.escape(k)}"]`); if (inp) inp.focus();
      return;
    }
    delete V.pendingNo[k]; delete V.crit[k];
    if ((V.labels[k] || {}).answer !== val) saveLabel(item, phase, val); else { S.cur = nextKey(e, k); render(); }
  } else if (act === "peak") {
    const id = b.dataset.peak, k = peakKey(id), val = b.dataset.val;
    S.peakSel = id; S.cur = k;
    if (val === "cannot_assess") { if (!(k in V.drafts)) V.drafts[k] = (V.peaks[id] && V.peaks[id].note) || ""; render(); const inp = document.querySelector(`[data-reason="${CSS.escape(k)}"]`); if (inp) inp.focus(); return; }
    delete V.drafts[k];
    if ((V.peaks[id] || {}).cause !== val) savePeak(id, val); else render();
  } else if (act === "save-reason") {
    const k = b.dataset.key, why = String(V.drafts[k] || "").trim();
    if (!why) return;
    if (k.startsWith("P|")) savePeak(k.slice(2), "cannot_assess", why);
    else { const [, item, phase] = k.split("|"); saveLabel(item, phase || null, "cannot_assess", why); }
  } else if (act === "critique") { S.cur = b.dataset.key; openCritique(b.dataset.key); }
  else if (act === "save-critique") saveCritique(b.dataset.key);
  else if (act === "verdict") { V.verdict = b.dataset.val; render(); }
  else if (act === "sign") saveSignoff();
  else if (act === "report") openReport();
  else if (act === "files") downloadFiles();
  else if (act === "phase-curve") { S.phaseSel = S.phaseSel === b.dataset.phase ? null : b.dataset.phase; S.phaseWin = 0; if (S.phaseSel && phaseWindows(e, S.phaseSel).length) S.peakSel = null; render(); }
  else if (act === "phase-win") { const n = phaseWindows(e, S.phaseSel).length; S.phaseWin = ((S.phaseWin || 0) + Number(b.dataset.d) + n) % n; render(); }
  else if (act === "remove") { S.refitDraft = { snapshot: e.digest, verb: "remove_phases", candidates: [b.dataset.phase], rationale: "" }; render(); const c = $("refit-composer"); if (c) c.scrollIntoView({ block: "nearest" }); }
  else if (act === "add") { S.refitDraft = { snapshot: e.digest, verb: "add_phases", candidates: [], rationale: "" }; render(); const c = $("refit-composer"); if (c) c.scrollIntoView({ block: "nearest" }); }
  else if (act === "cancel-refit") { S.refitDraft = null; render(); }
  else if (act === "run-refit" && S.refitDraft) startRefit(S.refitDraft);
  else if (act === "label-correction") { const r = V.refits.find((x) => x.id === b.dataset.refit); if (r) labelCorrection(r); }
  else if (act === "open-snapshot") { const i = S.queue.findIndex((q) => q.digest === b.dataset.digest); if (i >= 0) openSnapshot(i); }
  else if (act === "steps") { S.showSteps = !S.showSteps; render(); }
  else if (act === "trace-order") { S.traceTime = b.dataset.order === "time"; render(); }
  else if (act === "trace-full") loadTraceCall(e.digest, b.dataset.call);
  else if (act === "yscale") { S.yscale = b.dataset.val; render(); }
  else if (act === "claim-more") { const i = Number(b.dataset.i); S.openClaims.has(i) ? S.openClaims.delete(i) : S.openClaims.add(i); render(); }
  else if (act === "claims-all") { S.fullClaims = !S.fullClaims; S.openClaims.clear(); render(); }
  else if (act === "zoom-reset") { S.range = null; render(); }
  else if (act === "child-fit") { S.childFit = S.childFit === b.dataset.child ? null : b.dataset.child; S.phaseSel = null; render(); }
  else if (act === "comp-row" && V.composition) { V.composition.draft.rows.push({ el: "", v: "" }); render(); }
  else if (act === "mark-open") { const k = b.dataset.step; S.markOpen = S.markOpen === k ? null : k; S.markDraft = { mark: null, note: "" }; render(); }
  else if (act === "mark-pick") { S.markDraft = { ...(S.markDraft || { note: "" }), mark: b.dataset.val }; render(); }
  else if (act === "mark-cancel") { S.markOpen = null; S.markDraft = null; render(); }
  else if (act === "mark-save") { const ref = S.steps[b.dataset.step]; if (ref && S.markDraft && S.markDraft.mark) saveMark(ref, S.markDraft.mark, S.markDraft.note.trim()); }
});
window.addEventListener("beforeunload", (ev) => { if (methodBlocked()) { ev.preventDefault(); ev.returnValue = ""; } });
setInterval(() => {
  if (!S.ep) return;
  const V = view();
  $("timer").textContent = `Review ${clock((Date.now() - V.opened) / 1000 + V.seconds)}`;
  document.querySelectorAll("[data-elapsed]").forEach((el) => { el.textContent = clock((Date.now() - Number(el.dataset.elapsed)) / 1000); });
}, 1000);
boot();
