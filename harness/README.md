# AgentXRD harness

AgentXRD is the agent whose runs PhaseEval reviews. This folder holds the four physics tools it calls and `agent/run_agent.py`, which runs the tools with any OpenAI-compatible model. It does not hold the Harbor setup or the task builder used for the bundled runs. See "Not in this release" below.

## The loop

The six bundled runs were made with harness v2. Each run is one Harbor 0.22.0 trial. The agent loop is the native Codex CLI inside the trial's container. The model is Qwen3.6-27B, served through an OpenAI-compatible local endpoint. Each case has a frozen task that names the case ID and the answer format. The agent reads the task, calls the tools below from its shell, and writes its answer to `/workspace/output/analysis.json`. The episode ends when the agent stops or the time limit is reached.

## Case folder

`casedata.py` reads a case from `INPUT_DIR` (`/workspace/input`) and the tools write to `OUTPUT_DIR` (`/workspace/output`). Every tool also takes `--input-dir`, and `refine.py` and `screen.py` take `--output-dir`. The input folder looks like this.

```
input/
  manifest.json                      {"cases": [ one entry per case ]}
  <pattern_file>                     the observed pattern
  candidates/<CASE_ID>/manifest.json {"case_id": ..., "candidates": [ one row per CIF ]}
  candidates/<CASE_ID>/<file>.cif
```

A case entry carries `case_id`, `pattern_file` and `pattern_sha256`. It may also carry `measurement` (wavelengths), `chemistry_elements`, `instrument_profile` and `material_context` (precursors with weighed masses, processing steps, declared target). A candidate row carries `candidate_id`, `file`, `sha256` and `formula_sum`. Space group and cell are read from the CIF itself.

## The four tools

- `case_brief.py --case <ID>` prints the chemistry, measurement, specimen, precursors with their molar ratios, processing and declared target.
- `candidates.py --case <ID>` prints one line per candidate CIF with formula, space group, crystal system and cell.
- `screen.py --case <ID> --no-search` ranks candidates by how well their simulated lines land on observed peaks and lists strong peaks left unexplained. `screen.py --case <ID> --residual <run-label>` lists the largest residual peaks of a saved fit and the candidates with lines near each one. The tree phase search that `screen.py` runs without `--no-search` is not in this release, so the tool stops and asks for `--no-search`.
- `refine.py --case <ID> --candidates <ID> ... --run-label <label>` runs staged BGMN refinement (Pawley, then Rietveld) with `phaseeval.refinement`. It checks the pattern and CIF hashes first. It uses the AERIS instrument profile and Cu radiation.

`physics/README.md` is the text the agent saw. It gives the working order and the refinement notes.

## Running the tools

Install PhaseEval first (see the top-level README), so that `launcher/.venv` exists and BGMN can be found. Then run the tools from `harness/physics` with the repository's `src` on the path.

```sh
cd harness/physics
PYTHONPATH=../../src ../../launcher/.venv/bin/python case_brief.py --input-dir <input> --case <ID>
```

## What a run writes

Each `refine.py` call writes one folder named by its run label under the output folder. The folder holds `request.json` (the case, candidates and hashes), `result.json` (stages, Rwp, Rexp, GoF, weight fractions, cell shifts, the degradation flag), one `<stage>.csv` curve per finished stage, and the BGMN work files in `fit/`. A label cannot be reused. The agent's answer is `analysis.json` in the output folder. It names `case_id`, `phases` (each with `candidate_id`), `weight_fractions` from 0 to 1, and `fit_artifacts`, the paths of the fit files it rests on.

## Bringing a run into PhaseEval

`src/phaseeval/workbench/importer.py` reads two things. `read_case_package` reads a case folder laid out as above, plus the case's row in a `case_labels.json` and a `custody.json` split-check record. The package must hold exactly the files its manifests name. `read_harbor_run` reads one Harbor trial folder (`jobs/<job>/<trial>/`). That folder needs `config.json`, `result.json`, `agent/trajectory.json`, and the container output under `artifacts/workspace/output/` with `analysis.json` and the run-label folders. Every `request.json` must name the case's pattern and CIF hashes, or the run is refused. Each answered phase must be in the case's candidate library. `WorkbenchStore.import_run` then stores the result. The `phaseeval.workbench.cli import` command wraps these functions. "Run your own agent" in the top-level README gives the commands.

## Not in this release

- The task builder that makes the frozen per-case task and case folder.
- The candidate CIF library. Some of its files are ICSD-derived and cannot be shared. A lab supplies its own CIFs, for example from COD.
- The tree phase search behind `screen.py`.
- The reinforcement learning wrapper.
- An arena for comparing agents.
- The GSAS-II `.gpx` importer.
- The BGMN/Profex importer.
