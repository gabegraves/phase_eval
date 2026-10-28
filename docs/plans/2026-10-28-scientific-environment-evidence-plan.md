---
title: Scientific Environment Evidence - Plan
type: feat
date: 2026-10-28
deepened: 2026-10-28
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
---
# Scientific Environment Evidence - Plan

## Goal Capsule

Make the recorded-XRD environment inspectable and its attempts auditable before scientific RL training. Extend the existing runner, preserve failed trials, and profile fixed attempted groups without treating withheld rewards as zeros. Implement U1–U4 in order. Completion means the engineering checks and actual-data demonstrations below pass; it does not mean scientific reward qualification or learning has occurred.

## Product Contract

### Summary

Expose native scan and fit evidence through bounded tools. Record the task boundary and every episode ending. Add a read-only attempt profiler. Preserve the existing separation of scientific admission, objective diagnostics, method review, and training reward.

### Problem Frame

The runner saves full diffraction curves but shows only summaries. Its prompt advertises unavailable shell access. Setup and provider failures can disappear before import, and budget checks occur only between turns. The current scorecard has withheld numeric placeholders that cannot establish a useful policy-gradient signal.

### Actors and Flows

- A1. Scientific agent inspects a frozen recorded case, compares supplied candidates, runs fits, and submits supported conclusions or unresolved alternatives.
- A2. Environment operator packages development cases and inspects complete attempted groups.
- A3. Scientific reviewer adjudicates outcomes and method through the existing workbench and the separately developed assisted-review system.
- F1. Frozen case → inspect evidence → choose analysis → receive results → persist observable trajectory → submit or stop. Covers R1–R6.
- F2. Planned attempt group → fresh episodes → profile all attempts and missing judgments → qualify future training separately. Covers R6–R8.

### Requirements

Evidence access:
- R1. The agent can reconstruct the full native observed scan and saved fit curves, or inspect a selected angle interval, without hidden downsampling.
- R2. Reads expose artifact identity, digest, range and continuation, and accept only registered actor-visible evidence from this case or episode. Hidden labels, custody/reference files, arbitrary host files and sibling episodes remain inaccessible.
- R3. Available context retains units and source references, and absent preparation or calibration facts remain unknown. Advertised actions match the function interface.

Episode integrity:
- R4. Validate contained, hashed case inputs before tools use them; reject malformed data and preserve the reason. A new episode cannot overwrite or reuse a prior episode directory.
- R5. Persist completed calls before later calls can fail, enforce remaining wall budget at external calls, and distinguish setup/provider failure, budget exhaustion and ordinary policy endings. Earlier provisional evidence survives termination.
- R6. Validate answer structure and evidence ownership without requiring confident phase identification, a fit, or quantification. Scientific defensibility remains reviewable rather than mechanically inferred.

Profiling:
- R7. A fixed attempted-group manifest names scheduled episode IDs and a common task/policy/budget identity before any episode starts; its digest is bound into each runner receipt before setup. Each slot permits one attempt; retrying creates a new separately reported group and never replaces a failed slot. Missing, invalid, duplicate and incompatible attempts are visible; compact fragments or multiple fits never count as independent episodes.
- R8. Report per-episode endings, known costs, diagnostic availability, and qualified scientific-reward availability. Unknown/withheld values remain missing; no new reward, admission, judge authority or optimizer is enabled.

### Acceptance Examples

- AE1. A narrow peak between overview samples remains present when the agent requests its native interval. Covers R1–R2.
- AE2. A request for a label file or a symlink to another episode fails without returning its contents. Covers R2.
- AE3. Missing calibration is still unknown after inspection; declared profile assumptions remain distinguishable from measured calibration. Covers R3.
- AE4. A provider failure after one successful tool call retains that observation and an infrastructure ending. Covers R4–R5.
- AE5. Empty phases with an unresolved explanation and no claimed quantities remains structurally representable; fabricated fit paths and boolean fractions fail. Covers R6.
- AE6. Four scheduled attempts with no qualified rewards yield zero observed scientific rewards and null mean/spread, not four scientific failures. Covers R7–R8.

### Scope and Key Decisions

Use the existing recorded-XRD task and single-agent runner. Preserve Miles as the commissioned training integration; no framework migration. Keep outcome reward qualification separate from method assessment (session-settled: user-approved — chosen over immediate dense method rewards because uncalibrated method signals invite exploitation). Governs R6–R8.

Nodeterm owns assisted review, judge escalation, triage and label semantics. This work does not edit those modules. Full cohort source-ledger integration, new measurement acquisition, dynamic candidate search, per-scan calibration, expert labeling and a scientific optimizer update remain subsequent milestones with real data/authority dependencies. The engineering slice must not claim those milestones complete.

## Planning Contract

### Context and Research

The accepted C01–C18 inventory is preserved in `docs/research/scientific-environment-concerns.md`.

Current code anchors: `harness/agent/run_agent.py`, `harness/physics/refine.py`, `harness/physics/case_brief.py`, `src/phaseeval/workbench/importer.py`, `evidence.py`, `qualification.py`, `scorecard.py`, and `docs/xrd-method-review.md`. Saved stage CSVs already contain native observed/calculated/difference values. Existing blinded projections and immutable receipts should be reused. Public replay data are development examples, not independent holdouts.

The inspected development review stores contain zero expert labels/sign-offs. Their absence prevents expert ordering and judge calibration now; it does not prevent evidence-access or objective-integrity work. Historical numerical agreement is not exact-current-parser replay.

External design guidance: [NVIDIA scientific agents](https://developer.nvidia.com/blog/how-to-train-scientific-agents-with-reinforcement-learning/) motivates simple outcome signals, repeated-attempt profiling and stability monitoring. [Snorkel environment validation](https://snorkel.ai/blog/rl-environments-for-llm-agents-design-rewards-and-validation/) motivates explicit task, tool, grader and adversarial checks. Their advice supplies design criteria, not evidence that PhaseEval is scientifically validated.

### Key Technical Decisions

- KTD1. Add one bounded evidence action to the existing Runner; use registered logical handles, native numeric pages and bounded text pages. Reuse curve parsers and containment/hash checks. A general filesystem reader adds unnecessary exposure. Governs R1–R3.
- KTD2. Keep comparison identity in a versioned runner receipt: actor inputs, prompt/tool schemas, relevant source digests, declared policy/decoding identity and budget. Keep session IDs and observed runtime provenance outside the comparison fingerprint; report runtime variation and unknowns separately. Known software/profile pins belong in the stable boundary, and missing pins prevent claims of complete reproducibility. Do not alter exported expert-label schemas. Governs R4–R7.
- KTD3. Keep no-refine and setup-failed attempts in runner receipts and the profiler. The legacy workbench requires refinement-read CIF hashes; do not fabricate these from available candidates. Import refinement-bound runs with normalized terminal classifications; record unsupported no-refine import explicitly. A later versioned workbench migration can represent initial-bundle binding. Governs R5–R6.
- KTD4. Profile existing episode/scorecard records under a predeclared attempt manifest. Left-join scheduled IDs to versioned runner receipts, then optional hashed scorecard attachments; do not rely on scorecards for attempts the workbench cannot import. Use eligibility and missingness masks; keep diagnostic and scientific statistics separate. Never derive a scientific reward from fit quality or model agreement. Governs R7–R8.

### System-Wide Impact

Runner tools feed both the local CLI and the retained Miles subclass. Additive tool changes require prompt/schema fingerprints and tests through Runner.call. Importer compatibility must retain refinement-bound failed runs without weakening case binding; no-refine attempts remain preserved outside that legacy model. Native evidence becomes actor-visible only after the tool result is recorded; reviewer-only full output remains distinct from evidence actually read. The private Miles subclass overrides run/chat/call; public-runner proof does not commission its budgets or sampling. That adapter needs a separate compatibility run before training. No changes to `assisted_review.py`, `triage.py`, `labels.py`, or review routes are planned.

### Risks and Dependencies

Containment and hash validation protect the input boundary. Budget cancellation must terminate spawned process groups and retain completed evidence. Current Cu/AERIS assumptions limit generalization; they are disclosed rather than generalized without calibration. The actual-data proof uses existing development material and cannot establish scientific accuracy. Real repeated policy runs require a declared available endpoint; no resource-heavy launch is needed to prove the environment plumbing.

### Open Questions

No product blocker for U1–U4. Expert scientific labels, a qualified reward version, frozen independent specimen groups, and available training resources are required before a bounded scientific update. Actual local counts and missing prerequisites will be recorded in the completion evidence, not invented. The future policy-gradient choice must be revisited using measured reward variation, usable group sizes, clipping/KL behavior and held-out changes.

## Implementation Units

### U1. Expose native evidence

- Goal: satisfy R1–R3 and AE1–AE3 through the existing agent interface.
- Dependencies: none. Decisions: KTD1. Flow: F1.
- Files: `harness/agent/run_agent.py`, `harness/agent/task.md`, `harness/physics/README.md`, `harness/physics/case_brief.py`, `harness/physics/casedata.py`, a small evidence helper under `harness/agent/`, `tests/harness/test_run_agent_output.py`, focused evidence tests; `CONTEXT.md` only for necessary domain terms.
- Approach: register the case's actor-safe metadata, scan, supplied CIFs, this episode's saved fit artifacts and overflow outputs. Row indexes are local to each artifact; raw-scan and fitted grids may differ. Reuse `Curve.parse_csv` where applicable. Preserve native row indexes and full permitted context arrays. Use a shared allowlisted actor-context projection for the reader and case_brief; exclude operator/custody sources and unknown extension fields. Preserve safe source locators without exposing whole custody documents. Legacy bare fields retain bundled-manifest provenance with missing richer provenance explicit. Make clipping and its recovery handle visible in the actual observation. Correct prompt claims about shell and tree search.
- Execution note: add focused failing coverage before production edits; reuse current fixtures.
- Test scenarios: exact page reconstruction; selected narrow region; raw scan before fit; malformed ranges; missing/unfinished stage; nonfinite data; traversal/absolute path/symlink/cross-episode/hidden-label denial; clipped output recovery; actual tool result retained in trajectory; hidden context extensions and custody-sourced amounts do not leak through either reader or case_brief.
- Verification: focused tests pass and the tool reconstructs a real development scan and saved fit without numerical alteration.

### U2. Preserve reproducible episode boundaries

- Goal: satisfy R4–R6 and AE4–AE5.
- Dependencies: U1. Decisions: KTD2–KTD3. Flow: F1.
- Files: `harness/agent/run_agent.py`, small helper if needed, `src/phaseeval/workbench/importer.py`, focused harness/import tests. Avoid changing label schemas.
- Approach: validate/fingerprint case inputs before tool access; record scheduled identity before fallible setup. Require clean output at the reusable entry point. Preserve steps as actions complete. Pass remaining budget to requests/subprocesses and stop process descendants. Preserve per-response usage availability and per-action timing, with incomplete totals labeled as known subtotals. Normalize runner errors and minimally extend importer endings; preserve no-refine attempts in runner records with an explicit legacy-import limitation. Tighten shared answer/action validators while allowing unresolved/no-fit conclusions.
- Execution note: characterization and failure-first tests, then the smallest shared-path changes.
- Test scenarios: two clean episodes have distinct IDs and the same comparison fingerprint; changed inputs/prompt/tools/budget change identity; invalid/missing input records setup failure; provider failure after successful call retains evidence; expired budget stops later batch calls; process timeout is bounded; duplicate candidate/label or boolean/nonfinite quantity fails; fabricated or foreign fit fails; no-fit unresolved submission is retained with valid initial identity and explicit unsupported legacy import; stale identity fails closed.
- Verification: focused lifecycle/import tests and existing workbench compatibility tests pass; a scripted-provider run traverses runner → persisted files → importer.

### U3. Profile fixed attempted groups

- Goal: satisfy R7–R8 and AE6.
- Dependencies: U2. Decision: KTD4. Flow: F2.
- Files: a small module under `src/phaseeval/` and command under `scripts/`, `tests/rl/test_reward_profile.py`, `rl/README.md`.
- Approach: define a versioned fixed-attempt manifest and report from runner receipts with optional existing scorecard attachments bound by trial identity and receipt hash. Left-join the scheduled manifest to runner records and optional hashed scorecards. Report unsupported no-refine workbench import separately from execution validity. Consume all episode rows rather than latest-per-root summaries. Expose attempted/observed/invalid/missing counts, exclusion reasons, independent episode count, diagnostic statistics with missingness, scientific availability and costs. Unknown provider usage stays unknown; never substitute workbench executor cost for original tool time. Diagnostic terms retain known/missing counts without redistributing missing weights. No scheduling service or trainer adapter is added.
- Execution note: table-driven adversarial examples precede implementation.
- Test scenarios: all withheld produces null scientific statistics; equal known observations have zero spread; mixed known observations compute documented population spread; infrastructure failure stays separate from policy failure; missing episodes remain counted; duplicate IDs/compact siblings and changed comparison identities fail; malformed/nonfinite metrics fail; scorecards with placeholder training zero remain unavailable; scientific spread is null below two known observations; structurally unsupported no-refine review is distinct from pending qualification; input order does not change results; one refined, one no-refine, one setup-failed and one missing scheduled attempt all remain visible; mixed reported zero/missing/known usage yields an incomplete labeled subtotal.
- Verification: focused tests and CLI report over actual development records; lack of a real comparable group is reported rather than synthesized.

### U4. Prove the milestone and preserve the roadmap

- Goal: verify R1–R8 and retain each accepted scientific-environment concern in a durable artifact.
- Dependencies: U1–U3.
- Files: `docs/research/scientific-environment-readiness.md`, `docs/research/scientific-environment-concerns.md`, `docs/examples/` for sanitized proof only, relevant READMEs, completion evidence outside this immutable plan.
- Approach: run integrated checks through actual Runner tools and original local scan/fit data. Record source identities, commands, outcomes and limitations. Use `docs/research/scientific-environment-concerns.md` to retain C01–C18 status and explicit dependencies on Nodeterm, expert labels and future training. Include first-principles gradient rationale, current Miles defaults and conditional experiment hypotheses; no estimator is declared selected by unmeasured data.
- Test scenarios: actual scan and curve native rows/values reconstruct with original artifact digests; text pages reconstruct bytes where text is exposed; scripted mixed-success/failure episodes import and profile; protected evidence never enters the actor response; no scientific training reward is enabled. Re-run appropriate existing blindness, importer, scorecard and qualification regression tests.
- Verification: proof artifact reports observed outcomes and unavailable scientific checks separately; no invented evaluator sensitivity/specificity or learning improvement.

## Verification Contract

- V1. U1 focused evidence tests plus real native scan/curve reconstruction through the tool interface.
- V2. U2 focused failure, budget, clean-reset, binding and answer-validation tests, with an integrated persisted-trial import.
- V3. U3 deterministic profiling checks and an actual-data CLI run that retains missingness.
- V4. Existing relevant workbench blindness, import/export, scorecard and qualification tests; inspect actual diff for cross-chat ownership conflicts.
- V5. Final code review and simplification; record failures and fixes. Expensive numerical refitting is only required if numerical execution/parsing behavior changes. Existing saved native evidence suffices for the evidence-reader demonstration, with that limit stated.

## Definition of Done

| Unit | Required observed result |
|---|---|
| U1 | Agent can reconstruct native evidence through permitted bounded actions; prohibited reads fail. |
| U2 | Fresh immutable trial identity, bounded external calls, retained partial traces, failure import and explicit no-refine legacy limitation. |
| U3 | Fixed-attempt report distinguishes missing scientific signal from observed zero and preserves all planned attempts. |
| U4 | Durable proof and C01–C18 roadmap accurately separate engineering completion, review underway and scientific prerequisites. |

No optimizer launch, reward promotion, expert labeling or claims of scientific learning follow automatically from this milestone. Changes remain on a separate branch for integration with the concurrent assisted-review work.
