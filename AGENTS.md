# AGENTS.md

PhaseEval is where a diffraction expert checks the phase identifications an AI agent makes from powder XRD patterns, and where those checks become records that later models are graded against. `README.md` covers what it does, how to install and run it, and the claims it does not make. This file covers what an agent needs to change the code safely or to drive PhaseEval through its API.

## Map

- `src/phaseeval/workbench/`: the product. `snapshot.py` freezes an answer into the blind view a rater sees; `triage.py` runs the checks and routes answers; `policy.py` derives provenance, eligibility and rights; `labels.py` and `label_routes.py` hold rubric labels, sign-off and the reveal; `review.py` and `executors.py` run the actions, including BGMN refits; `scorecard.py` scores every agent run step by step; `qualification.py` grades models against expert calls; `exports.py` writes the dataset.
- `src/phaseeval/api/`: the FastAPI app, bearer authentication (`auth.py`), the attempt-tree routes (`routes/attempts.py`), and the pages in `assets/`.
- `src/phaseeval/refinement/`: the BGMN refinement that refits use.
- `launcher/`: the double-click launcher's settings (`phaseeval.toml`) and demo store. The locked environment is `.venv` at the repository root.
- `schemas/`: the export formats. `data/README.md` describes the bundled dataset.

## Test

From the repository root:

```sh
uv run --locked --extra test python -m pytest -q -p no:cacheprovider
```

`tests/workbench/test_label_blindness.py` guards blindness with property-based tests. Keep it green, and add a case to it for every route or field a rater can read.

## Rules that keep the records honest

1. **Blindness.** A rater's view carries no reference answer, no run identity and no other rater's label until the server has stored that rater's sign-off. Anything a rater can read goes through the rater projection (`snapshot.blind_payload`, `labels.ExposureLedger`).
2. **Label kinds stay apart.** A model judge writes only `model_judge` labels, through `labels.judge_label`. Present them as model output, and grade judges only against expert labels.
3. **No reference answers or expert labels in the repository.** The dataset ships empty label files so its format is complete.
4. **Claims follow the evidence.** README's "What is measured and what is not" is the boundary for every page, export and document: no accuracy, recall, judge-expert agreement or learning gain that this repository does not measure.
5. **BGMN is third-party.** Refits find or download it at run time (README, "BGMN"). It is never committed.
6. **Exports are versioned.** A change to an exported record changes its type and its schema in `schemas/` together; `tests/workbench/test_export.py` checks that they match, and `scripts/workbench/validate_eval_dataset.py` checks a dataset against them.

## Driving PhaseEval as an agent

Every API route takes `Authorization: Bearer <token>`. The launcher generates an owner token and an owner-rater token on each start. To give an agent its own credential, set `tokens` in `launcher/phaseeval.toml` to a directory holding `principals.json` and the token files, with a principal of type `agent` (or `model_judge` for a judge). The roles on a principal decide what it may read; label-mode roles see only the blind routes.

- `GET /api/attempts/whoami`: check which principal a token is.
- `GET /api/attempts/verbs`: the JSON Schema of the actions.
- `POST /api/attempts/{parent}/actions`: add or remove a phase and refit. This is the same action path the expert's page uses; the refit appears as a child attempt beside its parent.
- `GET /api/attempts/scorecard` (owner): pass, fail or unknown for each step of each run, with the source that decided it.
- `GET /api/attempts/scorecard/pairs.jsonl` (owner): the review-eligible before-and-after pairs. `demo-pairs.jsonl` holds revealed demonstration pairs, which are not training data.
