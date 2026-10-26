# PhaseEval public dataset v0

Dataset version `agentxrd-workbench-eval-v0`, public scope. `scripts/workbench/build_eval_dataset.py --scope public` built it from the AgentXRD run store. The release then removed every model-judge label and restated the label counts in `manifest.json`, so the manifest describes the files that ship. Check the files with:

```sh
PYTHONPATH=src uv run --locked python scripts/workbench/validate_eval_dataset.py data/dataset-v0-public
```

## What it is for

Evaluating AgentXRD analyses of real powder XRD patterns, and calibrating judges against expert labels once expert labels exist. It holds agent episodes and frozen analysis snapshots. It holds no training data and supports no claim of learning gains.

## Contents

Six episodes by Qwen3.6-27B (local) in the `codex` harness, three on DEV006 and three on DEV102. Each episode submitted an answer, so there are six answer snapshots. Every snapshot has chemistry evidence. No episode was excluded. The export walked 25 episodes. 19 of them are outside the public scope. They are neither exported nor listed as exclusions.

There are no expert labels, model-judge labels, peak calls, sign-offs or step feedback records. Those files are present and empty, so the format is complete.

## Files

Each format has a JSON Schema in `../schemas/` and one example in `../schemas/examples/`.

| File | One line per | Schema |
|---|---|---|
| `episodes.jsonl` | Episode: one root-to-end path of the attempt tree, with every attempt, the state after each step, run receipts, status, provenance, human contributions, eligibility (all unreviewed) and per-artifact rights | `workbench-episode.v1.json` |
| `excluded.jsonl` | Episode left out, with typed reasons | `excluded-episode.v1.json` |
| `snapshots.jsonl` | Frozen analysis snapshot: the evidence a blind rater sees for one answer (pattern and fit-curve hashes, Rwp, GOF, phase list with amounts, residual peaks, checks). Its digest is the sha256 of the line, and every label keys on it | `snapshot.v1.json` |
| `labels.expert.jsonl` | Rubric label by a human rater (`expert` or `owner`) | `label.v1.json` |
| `labels.model_judge.jsonl` | Rubric label by a model judge. Empty in this release | `label.v1.json` |
| `peak_calls.jsonl` | A rater's call on one unexplained residual peak | `peak-call.v1.json` |
| `signoffs.jsonl` | A rater's closing sign-off on a snapshot | `signoff.v1.json` |
| `step_feedback.jsonl` | Feedback on one intermediate step of a run, separate from the end-state rubric labels | `step-feedback.v1.json` |
| `judge_inputs.jsonl` | Exactly what a model judge reads for one snapshot, which is also the expert's blind view of it | `judge-input.v1.json` |
| `manifest.json` | Counts with denominators, versions, the snapshot index, exclusions, and each file's line count and sha256 | `dataset-manifest.v1.json` |

## How labels are made

- Rubric `agentxrd-rubric-v1`. Per reported phase: "Is this phase supported by the pattern?" and "Is this phase chemically plausible for this sample?". Per analysis: "Are the reported amounts defensible on their basis?" and "Does the fit support the conclusion?". Answers are `yes`, `no` or `cannot_assess`. A `cannot_assess` answer carries its reason and is excluded from agreement.
- The server stamps the rater kind (`expert`, `owner` or `model_judge`) from the signed-in principal. Agreement is computed within one kind only. A model-judge label is never an expert label, and the two kinds live in separate files.
- A rater sees no reference answer, no run identity and no other rater's label until the server holds the required sign-offs. A label made after the reveal is stored with exposure `revealed`.
- Rater ids are pseudonyms (`r-` plus 12 hex characters) in every file.

## What is incomplete

- No imported answer records a verdict, because the agent's response format has no verdict field. Every submitted run carries `NOT_RECORDED`.
- The fit curve that a judge and an expert read is reduced to at most 1,200 points at 4 significant digits. The fit states its native point count, stride, precision and curve sha256. Native resolution, with background and per-phase contributions, is kept only within 0.5 degrees of each listed residual peak.
- The episode roots keep their review-only fields (case source, split and ingest file names).
- Per-artifact rights are unknown except for the scans, and every eligibility decision is unreviewed.
- The `counts` entry `snapshots_judged` counts snapshots selected for judging. It does not count completed judgments. This release carries no judgment.

## Sources and license

The episodes, snapshots and manifest are released under the Apache License 2.0 with this repository. The scans inside the records are CC BY 4.0: DEV006 from the DARA benchmark (Zenodo 10.5281/zenodo.17410051) and DEV102 from the Precursor Genome release (Zenodo 10.5281/zenodo.21285546). Hull distances in the chemistry evidence derive from Materials Project data (CC BY 4.0). `../NOTICE` gives the full attribution.
