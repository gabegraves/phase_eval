# PhaseEval RL: status and plan

This folder holds the public part of the reinforcement learning work around PhaseEval. Today it holds one thing: a replay scorer. It is not an RL environment. No model has been trained on it.

## What exists today

| Piece | Where | State |
| --- | --- | --- |
| Replay scorer (`replay/`) | This folder | Runs from public inputs only. Scores a phase-list answer against the six public Qwen episodes with reward v0. Targets `verifiers==0.1.14`. |
| Verifiers wrapper, v0 | Private AgentXRD modules | Targets `verifiers==0.1.14`. 30 of 30 tests passed on CPU, measured 2026-10-22. Does not import under verifiers 0.4. |
| Verifiers wrapper, v1 port | Private AgentXRD modules | Targets `verifiers>=0.4`. 65 of 65 tests passed on CPU, measured 2026-10-22. Not published. Runs on synthetic fixture rows, not on PhaseEval episodes. |
| Prime Intellect Environments Hub entry | None | Nothing is published on the Hub. |

The two wrappers depend on private AgentXRD modules and private data, so they cannot be installed from this repository.

One input the next step needs was measured on 2026-10-22: the public staged BGMN refinement (`harness/physics/refine.py`, the last command of `harness/examples/dev006/run.sh`) ran on DEV006 with the three COD precursor structures on a laptop CPU (BGMN at one thread). It took about 90 s (result.json elapsed_seconds 89.2). All four stages converged (Pawley Rwp 3.31%, final Rietveld Rwp 3.52%), and the Rietveld fit did not degrade after the good Pawley fit, so the fit-quality check (K1, Rietveld did not degrade after a good Pawley fit) passes. That is one measurement of the per-rollout cost of a refinement call in a tool-using environment.

## What this folder runs

`replay/phaseeval_replay` is a `verifiers` `SingleTurnEnv` with one task per public episode (six tasks over two benchmark cases: DEV006, the NiO, Li2CO3 and Bi2O3 mix, and DEV102, a bismuth germanium oxide mix). The prompt shows the case id and the declared elements and asks for `{"phases": [...]}`. It never shows the recorded answer or the recorded run's tool trace. The smoke test checks that no prompt contains a recorded phase formula.

The reward applies `REWARD_SPEC_V0` from `phaseeval.workbench.scorecard`:

- `answered` (0.20) for a parseable phase list.
- the three physics checks, K1 (fit did not degrade), K2 (no unexplained residual peak at or above the flag) and K5 (answer equals the cited fit), 0.40 together, only when the answer's reduced-formula set equals the recorded run's answer. Those checks were measured on that run's saved fit, and a different answer has no fit, so they score 0.
- The reference and model-judge terms (0.40) score 0, as the public scorecard scores them.

Commands, from the repository root, with `uv` and Python 3.12:

```sh
cd rl/replay
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "verifiers==0.1.14" ../..
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -I smoke.py ../../data/dataset-v0-public
```

Output measured on 2026-10-22 (exit 0):

```
env: SingleTurnEnv rows: 6
DEV006 234720087f62 phases ["Bi2O3", "Li2CO3", "NiO"] | scorecard total 0.6 | replay recorded/wrong/none [0.6, 0.2, 0.0]
DEV006 9571407be9c0 phases ["Bi2O3", "Li2CO3", "NiO"] | scorecard total 0.6 | replay recorded/wrong/none [0.6, 0.2, 0.0]
DEV006 926870d70846 phases ["Bi2O3", "Li2CO3", "NiO"] | scorecard total 0.6 | replay recorded/wrong/none [0.6, 0.2, 0.0]
DEV102 ff13046c52ad phases ["Bi2O3", "Ge(Bi3O5)4", "GeBi2O5", "GeO2"] | scorecard total 0.6 | replay recorded/wrong/none [0.6, 0.2, 0.0]
DEV102 0dcec46c2dd2 phases ["Bi2O3", "Ge(Bi3O5)4", "GeBi2O5", "GeO2"] | scorecard total 0.6 | replay recorded/wrong/none [0.6, 0.2, 0.0]
DEV102 548ca616c24c phases ["Bi2O3", "Ge(Bi3O5)4", "GeBi2O5", "GeO2"] | scorecard total 0.6 | replay recorded/wrong/none [0.6, 0.2, 0.0]
prompts that contain a recorded phase formula: none
```

The smoke test scores three completions per task with the environment's own reward function: the recorded answer (0.6), a wrong answer (0.2) and no answer (0.0). It calls no model.

## What it does not do

- It is not an RL environment. The policy has no tools and cannot refine, so it cannot earn the K-check terms for any answer other than the recorded one.
- It targets `verifiers==0.1.14`. The Prime Intellect Environments Hub and prime-rl need `verifiers>=0.4`, whose API is different (`verifiers.v1` tasksets, toolsets and harnesses). This scorer does not load there.
- No model was trained or evaluated with it.
- It covers two public cases and six episodes, and every recorded episode scores 0.6, so the reward has no spread across the recorded runs.
- No expert labels exist yet. Reward v0 is provisional.

## Coming very soon

This is a plan, and none of it is done yet. It is the next thing we are building.

1. **A tool-using environment on `verifiers>=0.4`.** One task per case, starting with DEV006 from `harness/examples/dev006` and its COD structures. The model calls the four public harness tools (case brief, candidate table, peak screen, staged BGMN refinement) and submits a phase list. The reward is deterministic and public: the refinement passes K1, no unexplained residual peak at or above the K2 flag, and the phase list matches the case's published reference (for DEV006, NiO, Li2CO3 and Bi2O3 from the DARA benchmark). The prompt must never show the reference. BGMN would download on first use, as `harness/examples/dev006/run.sh` does.
2. **One-click setup on Prime Intellect.** Publish the environment on the Prime Intellect Environments Hub, so a lab can set it up in one step and run evaluations and RL training runs of its own agents directly on the platform. The cases that grade an agent can then also train it.
3. **More public cases** with openly licensed structures, so the reward has spread and the environment is not a single-case test.
4. **A reward calibrated against expert labels** once they exist, replacing the provisional weights of reward v0.
