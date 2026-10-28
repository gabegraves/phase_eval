# Scientific environment evidence and readiness

2026-10-28. Completion evidence for [U1–U4](../plans/2026-10-28-scientific-environment-evidence-plan.md). The public recorded-XRD runner is now inspectable and its attempted groups auditable. Scientific reward calibration and a scientific training update remain outstanding.

## What the proof establishes

The [runnable proof](../../scripts/prove_scientific_environment.py) uses the real `Runner.call`, persisted trial files, `read_harbor_run` and `profile_group`. Provider responses are **scripted**. The refinement action copies an **existing saved fit**; no provider request, new refinement or optimizer runs. Full reconstruction calls are made by the proof's caller, not a learned policy. These distinctions are also in the [sanitized measured output](../examples/scientific-environment-proof.json).

| Check | Observed result | Limit |
|---|---|---|
| Native raw scan | All 8,282 rows reconstructed exactly in 83 pages; original row indexes retained. SHA256 `1d38258dbd2c135df41fd40fc8fe1a3746a1d21e0ed191f08f3fb45b2a33cc7e`. | One exposed DEV006 development scan. |
| Saved Rietveld profile | All 8,282 observed/calculated/difference rows, plus background where present, reconstructed exactly in 83 pages. SHA256 `0146adad4df7c5bf0f87ed2e4dde5f04b8e6f48a7726649ece72bb59601da71c`. | Saved evidence access, not a new fit or proof of phase correctness. |
| CIF/context pages and targeted region | Three CIF texts and projected context reconstruct exactly; three native rows in a selected interval match; calibration remains null. Four forbidden handles return errors. | Context comparison proves preservation of the safe projection, not completeness of experimental history. Broader adversarial cases are in focused tests. |
| Interrupted episode | Three tool results survive a later provider interruption; real importer classifies it `invalid` / `provider_error` and retains trial identity. | Scripted failure, not a reliability estimate. |
| No-fit conclusion | Empty phases and unquantified unresolved answer are retained as `submitted`. | Legacy workbench import remains `unsupported_no_refinement`; no fake refinement binding is invented. |
| Fixed attempt group | Four scheduled slots, three receipts, two validly initialized episodes, two invalid executions, one missing slot, one comparable completed attempt. | Two initialized episodes are not two independent specimens or two usable training rollouts. |
| Scientific signal | Zero known rewards; mean and population standard deviation are null. | This demonstrates missingness accounting, not a flat measured reward distribution. |

The manifest is written before any runner is constructed. The proof checks both native values against independently invoked original parsers (`parse_xy`, `Curve.parse_csv`) and artifact digests against original bytes. The profiler left-joins all scheduled slots. Its optional scorecard binding and withheld-placeholder behavior are covered by deterministic counterexamples, not invented expert labels.

Run from the repository root, substituting real paths. `--out` must be a new directory with no symlink parents; on macOS use `/private/tmp/...` instead of `/tmp/...`.

```sh
uv run --project launcher python scripts/prove_scientific_environment.py \
  --input-dir /path/to/full-library \
  --saved-fit /path/to/refine_001 \
  --case DEV006 --out /path/to/new-proof

uv run --project launcher python scripts/profile_rewards.py \
  --manifest /path/to/new-proof/group.json \
  --receipt /path/to/new-proof/a/runner_receipt.json \
  --receipt /path/to/new-proof/b/runner_receipt.json \
  --receipt /path/to/new-proof/c/runner_receipt.json \
  --out /path/to/new-proof/profile-cli.json

uv run --project launcher --with pytest --with hypothesis --with jsonschema python -m pytest
```

Final validation on the implementation host recorded **560 passed, 46 skipped, 11 warnings** for the full suite after integrating main `430534e` and all review fixes. The real-data proof was regenerated from clean source head `e63b50f`; parsed CLI and API profiles agreed. The simplification pass reused existing ID/hash helpers and bounded page-index allocation while retaining validation. Skips include unavailable private custody/smoke stores, GSAS/bubblewrap/Linux integration and BGMN slices; they are not validated by that result. Warnings concern existing FastAPI/Starlette deprecations and pytest integration/slow markers. Focused checks cover evidence containment, input drift, answer validation, persisted failures, request/process timeouts, reward accounting and existing workbench compatibility. Windows process cancellation has not been exercised on this macOS host. No separate lint/typecheck command is configured in the project manifests.

Completed code review `20261028-101246-3196c3cb` covered correctness, project rules, tests, maintainability, agent access, security, contracts and reliability, plus an independent Anthropic review. Three fix batches resolved all four retained findings: accepted leading-hyphen values reach the physics parsers; redacted context renders as unknown; refused retries cannot occupy scheduled slots; and scientific validity, admission and withholding gates have independent regression checks. The gate tests also fail when their corresponding condition is removed. No actionable finding remains unapplied. The review is engineering evidence, not scientific approval.

## Boundaries that remain

The actor sees allowed scientific metadata with source locators. Missing preparation, withheld amounts and unestablished scan calibration stay unknown. The bundled Cu/AERIS profile is an assumption, not measured individual-scan calibration. Three supplied candidates define a bounded comparison problem; arbitrary candidate discovery and new measurement acquisition are outside this interface.

Runner receipts bind inputs, prompt/tools, relevant source bytes, requested policy/decoding and budgets. They record observed runtime versions separately. Served model revision, unspecified decoding defaults, BGMN binary and installed-environment agreement with the lock remain explicit unknown pins. These owner-host records detect mismatches; they are not signatures, a malicious-operator sandbox or a complete reproducible runtime image.

No-refine and setup-failed attempts survive in receipts and profiles. The legacy workbench still requires recorded refinement input hashes, so supporting initial-bundle-only episodes there needs a versioned model/schema change. The retained private Miles `PilotRunner` overrides `run`, `chat` and `call`; this public-runner proof does not commission its sampling, budgets, token masks or trainer connection.

Timeout checks cover subprocess cancellation on this host; real BGMN cancellation and operator interruption remain integration gaps. The review also identified unverified edge behavior around partial refinement output during timeout and a provider response arriving at the wall deadline. These limits must be exercised before unattended production collection. Receipts without explicit successful slot-claim provenance remain unclaimed; the profiler does not infer a claim or recover missing receipts from the claim ledger.

The checked `review/dataset`, `review-v2/dataset` and public `data/dataset-v0-public` exports each contain **zero expert-label records and zero sign-offs**. This is scoped evidence about these stores. `validate-evaluator` requires approximately 100 domain-expert binary-labeled traces per failure mode before its calibration workflow; deterministic code checks do not satisfy that prerequisite. No sensitivity, specificity, judge agreement or learning gain is reported. Nodeterm's [assisted review](../architecture/assisted-method-review.md) supplies proposals and exception/audit routing; proposals remain model output rather than expert truth or training reward.

## Status of the accepted concerns

The [C01–C18 inventory](scientific-environment-concerns.md) remains the source of the concerns. Status here describes this milestone, not closure of the whole scientific program.

| Concern | Status and next dependency |
|---|---|
| C01 — Valid scientific loop | Engineering advanced; admitted tasks, qualified reward, actual update and independent evaluation remain. |
| C02 — Atom Scale architecture | External question remains: request a failed-run decision history and the resulting evaluated system change. |
| C03 — Two histories | Partial: frozen input/context and ordered observations/actions are joined. Rich specimen/preparation history needs source-ledger integration. |
| C04 — Unknown facts | Closed for the bounded interface: allowlisted provenance and explicit unknown calibration. Better facts require real measurements/records. |
| C05 — Available context | Projection preserves supplied safe context. The wider eligible scan inventory is not yet an admitted, joined cohort. |
| C06 — Inspectable evidence | Closed for registered native scan, saved fit, CIF, context and overflow pages; actual-data reconstruction passes. |
| C07 — Candidate coverage | Three-candidate boundary is explicit. Wider frozen library, alternatives and incomplete-library cases remain. |
| C08 — Recorded environment | Bounded analysis actions, budget and endings exist. Real multi-attempt policy sampling and the Miles adapter need commissioning. |
| C09 — Partial conclusions | Structurally supported without fitting/quantifying. Scientific merit and excessive-abstention judgments require experts; workbench no-fit import remains limited. |
| C10 — Method assessment | Preserved separately from reward. Nodeterm owns assisted method review; no dense process reward was enabled. |
| C11 — Reward exploitation | Objective malformed-input and missingness attacks tested. Scientific checklist/verbosity/abstention/recovery attacks need contrasting labeled attempts. |
| C12 — Expert validation | Dependency: actual expert labels, reviewer consistency, ordering checks and judge calibration. |
| C13 — Learning evidence | Zero scientific optimizer updates in this milestone. Private trainer qualification and monitoring remain. |
| C14 — Simple baselines | Seven nonempty subsets/fixed procedure remain planned comparisons under matched settings and costs. |
| C15 — Value of method signal | Outcome-only versus outcome-plus-method ablation remains after reliable outcome evaluation. |
| C16 — Holdouts | Dependency: specimen-group splits, exposure ledger and protected independent cases. DEV006 is development material. |
| C17 — Public reproducibility | Public evidence reader, lifecycle receipts, profiler and proof added. Private trainer adapter, runtime pins and licensed/source dependencies remain explicit. |
| C18 — Borrowing Periodic | Context, tools and evaluation boundaries inform the design. No equivalence to Periodic's private data, laboratory loop or training scale is claimed. |

## Conditional policy-gradient choice

The objective is expected **qualified episode outcome**, `J(θ) = E[R(τ)]`, over admitted cases and trajectories sampled from policy `πθ`. With environment transitions independent of θ, its score-function gradient is `E[(R(τ) − b) × Σ_t ∇θ log πθ(a_t | h_t)]` for a baseline independent of the sampled action. Here `h_t` includes available observations and previous actions; tool-response tokens are not sampled policy actions. An episode reward can train this policy without pretending to identify the causal value of every intermediate decision.

For G independent attempts on one case, mean centering uses `A_i = R_i − mean(R)`. At G=1 this is zero. Including the current reward in the mean scales the ideal expected score-function gradient by `(G−1)/G` under the independent-sample assumptions. A leave-one-out baseline gives `A_i^LOO = G/(G−1) × A_i` for fixed G>1. Thus, changing this baseline alone is a fixed scaling distinction; clipping, reuse, masks, regularization and length reduction still matter. [RLOO](https://arxiv.org/abs/2402.14740) and [PPO](https://arxiv.org/abs/1707.06347) describe related mechanisms, not measured XRD superiority.

The first online candidate is **synchronous, critic-free, mean-centered, PPO-clipped GRPO**, after a qualified outcome scorer and useful starting policy exist. Verified-demonstration SFT is a sensible starting baseline when reliable behavior is rare; it is optional if the initial policy already operates reliably. Disable group-standard-deviation scaling initially: small noisy differences should not receive automatically amplified influence. This is an experimental choice; mean centering cannot fix systematic judge error. Original [GRPO](https://arxiv.org/html/2402.03300v3) standardizes group rewards, so the proposed mean-only variant must be named explicitly.

Keep token aggregation explicit. Per-response length averaging changes the expected episode objective's weighting; prefer a tested fixed declared denominator if the actual backend supports it. [DrGRPO](https://arxiv.org/html/2503.20783v2) motivates this choice in mathematical reasoning; transfer to XRD is a hypothesis. The inspected Miles revision `a22fd824191c64ba7c5d6c531f125406e4ce9cc6` inherits GRPO defaults and response/rollout length averaging. Its singleton rollout-only diagnostic does not establish an informative update. Its experimental constant-denominator reducer is **not commissioned for the current FSDP path**. No optimizer setting is changed here.

Before choosing between variants, measure trustworthy repeated-attempt reward variation, scorer repeatability, usable group sizes, exclusions and total cost. Compare unchanged policy and qualified SFT, then a small outcome-only RL arm under identical tools/budgets and independent evaluation. Commission actual backward/update/checkpoint behavior for the intended model and hardware: 4B rollout memory is not 4B optimizer memory. Check token masks, compact-fragment invariance and scalar loss/gradient on a known batch through the real backend. Only then test a targeted normalization alternative and whether method supervision adds scientific value. Historical July trajectories can support qualified demonstrations and preferences; they are not automatically fresh on-policy samples.
