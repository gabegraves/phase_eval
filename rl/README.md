# PhaseEval RL: environment and reward profiling

The public recorded-XRD runner now exposes native evidence, saves episode receipts, and supports fixed attempted-group profiling. These are environment engineering capabilities. They do not establish a qualified scientific reward or a scientific learning result. The old replay scorer remains a compatibility smoke test. Miles remains the retained training integration; the public runner changes need a separate compatibility run through its private adapter before training.

## What exists today

| Piece | Where | State |
| --- | --- | --- |
| Recorded-XRD runner | `harness/agent/run_agent.py` | Six functions, including bounded native scan/fit/context inspection. Fresh trial directories, input fingerprints, budgets and durable failure receipts. |
| Fixed-attempt profiler | `scripts/profile_rewards.py` | Reads all scheduled slots and supplied receipts, optionally joined to existing scorecards. Missing/withheld rewards remain unavailable. Never scores, schedules or trains. |
| Replay scorer (`replay/`) | This folder | Runs from public inputs only. Scores a phase-list answer against the six public Qwen episodes with reward v0. Targets `verifiers==0.1.14`. |
| Verifiers wrapper, v0 | Private AgentXRD modules | Targets `verifiers==0.1.14`. 30 of 30 tests passed on CPU, measured 2026-10-22. Does not import under verifiers 0.4. |
| Verifiers wrapper, v1 port | Private AgentXRD modules | Targets `verifiers>=0.4`. 65 of 65 tests passed on CPU, measured 2026-10-22. Not published. Runs on synthetic fixture rows, not on PhaseEval episodes. |
| Prime Intellect Environments Hub entry | None | Nothing is published on the Hub. |

The two historical wrappers depend on private AgentXRD modules and private data, so they cannot be installed from this repository. Their recorded test counts are not validation of the current runner or a reason to migrate the retained Miles integration.

## Profile a fixed attempted group

First compute `harness.agent.run_agent.comparison_fingerprint(args)` using the same case, input directory, endpoint/model, decoding and budgets planned for every episode. This read-only function performs input validation without creating a trial or invoking a model or physics tool. Then save a manifest **before constructing any runner**:

```json
{
  "schema_version": 1,
  "group_id": "dev006-development-1",
  "created_at": "2026-10-28T12:00:00+00:00",
  "comparison_fingerprint": "<64-character fingerprint returned by preflight>",
  "episode_ids": ["a", "b", "c", "d"]
}
```

Pass `--group-manifest /path/group.json --episode-id a` to the existing runner, with a fresh `--out` directory, for each scheduled slot. The runner binds the exact manifest bytes before fallible setup and consumes each slot once. A retry needs a new separately reported group; do not delete slot claims or replace failed attempts. Do not edit the manifest after execution begins.

Profile every receipt, including failed or unfinished ones. A scheduled slot with no receipt remains missing:

```sh
uv run --project launcher python scripts/profile_rewards.py \
  --manifest /path/group.json \
  --receipt /path/a/runner_receipt.json \
  --receipt /path/b/runner_receipt.json \
  --receipt /path/c/runner_receipt.json \
  --out /path/new-profile.json
```

The output file must be new. Historical runs without receipts cannot be retroactively declared comparable attempted groups. Missing receipts cannot distinguish an unstarted slot from lost records. Attempted counts include setup failures and pending construction; `independent_episodes` counts distinct trials that reached valid setup, not independent specimens or usable training rollouts.

An optional `--scorecard /path/attachment.json` accepts an owner-produced wrapper around one **existing full `EpisodeScore`**, not a latest-per-root summary:

```json
{
  "schema_version": 1,
  "runner_receipt_sha256": "<SHA256 of the exact runner_receipt.json bytes>",
  "trial_id": "<the receipt trial_id>",
  "episode_score": "<replace this string with the complete existing EpisodeScore object>"
}
```

The score's `trial` must match, and the report records the attachment's own hash. These local bindings detect mismatches; they are not signatures or protection against an operator fabricating data. Use all episode rows from `scorecard_view(...).episodes`; do not collapse independent trials or count compact fragments as new episodes. Duplicate episode/session/trial IDs, ambiguous scorecard attachments, malformed/nonfinite metrics and mixed reward versions fail closed. Well-formed incompatible receipts remain excluded rows with reasons; unscheduled receipts remain visible.

Scientific statistics include only completed, compatible, execution-valid, refinement-bound attempts whose existing reward says `validity="valid"`, `admitted=true`, and `training_withheld=[]`. The current scorecard always withholds its training reward. Its placeholder `training=0.0` therefore yields **zero known rewards and null mean/spread**, not a measured zero reward. Below two known values spread is null; otherwise it is population standard deviation. Diagnostic terms retain their original weights and separate known/missing counts; missing weights are never redistributed.

No-refine attempts retain `unsupported_no_refinement`, distinct from pending qualification: the legacy workbench cannot import them without a future versioned binding change. Costs come only from the original receipt, including action durations and incomplete usage subtotals. Missing usage is distinct from measured zero. The report separates runtime variants and unknown pins from the stable comparison fingerprint. Overall available-observation statistics may be biased by missing/excluded trials and do not establish a single realized policy when runtimes differ.

One input the next step needs was measured on 2026-10-22: the public staged BGMN refinement (`harness/physics/refine.py`, the last command of `harness/examples/dev006/run.sh`) ran on DEV006 with the three COD precursor structures on a laptop CPU (BGMN at one thread). It took about 90 s (result.json elapsed_seconds 89.2). All four stages converged (Pawley Rwp 3.31%, final Rietveld Rwp 3.52%), and the Rietveld fit did not degrade after the good Pawley fit, so the fit-quality check (K1, Rietveld did not degrade after a good Pawley fit) passes. That is one measurement of the per-rollout cost of a refinement call in a tool-using environment.

## Historical replay smoke test

`replay/phaseeval_replay` is a `verifiers` `SingleTurnEnv` with one task per public episode (six tasks over two benchmark cases: DEV006, the NiO, Li2CO3 and Bi2O3 mix, and DEV102, a bismuth germanium oxide mix). The prompt shows the case id and the declared elements and asks for `{"phases": [...]}`. It never shows the recorded answer or the recorded run's tool trace. The smoke test checks that no prompt contains a recorded phase formula.

The reward applies `REWARD_SPEC_V0` from `phaseeval.workbench.scorecard`:

- `answered` (0.20) for a parseable phase list.
- two physics checks, K1 (fit did not degrade, 0.10) and K2 (no unexplained residual peak at or above the flag, 0.15), only when the answer's reduced-formula set equals the recorded run's answer. Those checks were measured on that run's saved fit, and a different answer has no fit, so they score 0.
- K5 (answer equals the cited fit, 0.15) scores 0: the completion is a phase list with no amounts, so it has nothing to compare with the fit's.
- The reference and model-judge terms (0.40) score 0, as the public scorecard scores them.

Keep this scorer as a compatibility smoke test for reward v0. It is not a scientific training environment: the policy cannot see the scan, so its reward measures recall of a recorded answer, not inference from evidence.

Commands, from the repository root, with `uv` and Python 3.12:

```sh
cd rl/replay
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "verifiers==0.1.14" ../..
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -I smoke.py ../../data/dataset-v0-public
```

Output measured on 2026-10-27 (exit 0):

```
env: SingleTurnEnv rows: 6
DEV006 234720087f62 phases ["Bi2O3", "Li2CO3", "NiO"] | scorecard total 0.6 | replay recorded/wrong/none [0.45, 0.2, 0.0]
DEV006 9571407be9c0 phases ["Bi2O3", "Li2CO3", "NiO"] | scorecard total 0.6 | replay recorded/wrong/none [0.45, 0.2, 0.0]
DEV006 926870d70846 phases ["Bi2O3", "Li2CO3", "NiO"] | scorecard total 0.6 | replay recorded/wrong/none [0.45, 0.2, 0.0]
DEV102 ff13046c52ad phases ["Bi2O3", "Ge(Bi3O5)4", "GeBi2O5", "GeO2"] | scorecard total 0.6 | replay recorded/wrong/none [0.45, 0.2, 0.0]
DEV102 0dcec46c2dd2 phases ["Bi2O3", "Ge(Bi3O5)4", "GeBi2O5", "GeO2"] | scorecard total 0.6 | replay recorded/wrong/none [0.45, 0.2, 0.0]
DEV102 548ca616c24c phases ["Bi2O3", "Ge(Bi3O5)4", "GeBi2O5", "GeO2"] | scorecard total 0.6 | replay recorded/wrong/none [0.45, 0.2, 0.0]
prompts that contain a recorded phase formula: none
```

The smoke test scores three completions per task with the environment's own reward function: the recorded answer (0.45), a wrong answer (0.2) and no answer (0.0). "scorecard total" is the public scorecard's total for the recorded run, which leaves out the uncalibrated judge share. It calls no model.

## What the replay does not do

- It is not an RL environment. The policy has no tools and cannot refine, so it cannot earn the K-check terms for any answer other than the recorded one.
- It targets `verifiers==0.1.14`. The Prime Intellect Environments Hub and prime-rl need `verifiers>=0.4`, whose API is different (`verifiers.v1` tasksets, toolsets and harnesses). This scorer does not load there.
- No model was trained or evaluated with it.
- It covers two public cases and six episodes, and every recorded episode scores 0.45, so the reward has no spread across the recorded runs.
- No expert labels exist yet. Reward v0 is provisional.

## Before scientific training

The next dependencies are expert outcome/method judgments, a calibrated and versioned reward, independent specimen-level holdouts, a measured comparable repeated-attempt distribution, and a commissioned Miles adapter. Better fit statistics or more recorded trajectories alone do not meet those requirements. The durable [environment concerns](../docs/research/scientific-environment-concerns.md) and [implementation plan](../docs/plans/2026-10-28-scientific-environment-evidence-plan.md) preserve the milestones.
