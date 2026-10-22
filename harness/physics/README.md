# Physics tools

Four tools run inside the diagnostic container. Each takes the case ID your task names and prints a short text summary. Run them with `python3 /workspace/physics/<tool> --case <CASE_ID> ...`.

- `case_brief.py`: chemistry, measurement, specimen, precursors with weighed masses and the molar ratios they imply, processing steps with the furnace trace summarized, and the declared target when one is recorded. The full record stays in `/workspace/input/manifest.json`.
- `candidates.py`: one line per candidate structure with formula, space group, crystal system and cell, read from its CIF and grouped by formula. Structures with the same formula and different space groups are different phases (polymorphs), and they give different patterns.
- `screen.py`: ranks every candidate by how well its simulated Cu K-alpha1 lines land on observed peaks, lists strong observed peaks that a covering set of the best structures does not explain, then runs a tree phase search with quick BGMN fits over all candidates and prints its best hypotheses as candidate sets. Declared precursors are pinned in the search. The search takes about 1 to 3 minutes; `--no-search` skips it. `--residual <run-label>` lists the largest residual peaks of a saved refinement and the candidates with lines near each one.
- `refine.py --candidates <ID> [<ID> ...] --run-label <fresh-label>`: staged BGMN refinement, Pawley then Rietveld stages. It prints each stage's Rwp, Rexp, GoF and convergence; for each phase at the final stage, the weight percent with its esd and the refined cell with its percent shift from the CIF cell; and the `rietveld_degraded_after_good_pawley` flag. The same numbers are in `<run-label>/result.json`. Curves are in `<run-label>/<stage>.csv` (two_theta, observed, calculated, difference).

Set the Bash tool's `timeout` argument to 3600 seconds for `refine.py` and 900 seconds for `screen.py`.

## Working order

1. Read the brief and the candidate table.
2. Screen before refining. Start from the search hypotheses and the ranking, not from formula names.
3. When several structures share a formula, refine each one in turn with the rest of the phase set unchanged, and keep the one the data support: lower Rwp, a flatter residual, and a small cell shift.
4. When the processing history is short or at low temperature, expect unreacted precursors and intermediate phases beside any declared target, and include them in your hypotheses.
5. After each fit, run `screen.py --residual <run-label>` and try candidates for the strongest unexplained residual peaks.
6. When weighed masses are given, convert the refined weight fractions to element molar ratios and check them against the ratios the brief prints. Report any disagreement.
7. Write a provisional `/workspace/output/analysis.json` as soon as one fit is defensible, then improve it as the analysis proceeds.

## Refinement notes

`refine.py` calls the existing staged BGMN backend. It verifies pattern and candidate hashes before execution and saves the numerical work directory, stage results and curves. It does not assign a scientific ACCEPT verdict. Use a fresh run label for each hypothesis. Up to five distinct candidates may be refined together. `--pawley-only` stops after the profile fit; Pawley intensity estimates are not Rietveld weight fractions.

The default command runs Pawley followed by staged Rietveld when the backend's existing quality checks permit it. A poor Pawley fit can prevent Rietveld. Inspect the returned stages rather than assuming all refinements ran. A result with no Rietveld stage does not supply a structure-based quantitative result. When `rietveld_degraded_after_good_pawley` is true, the peak positions fit but the structure model does not: suspect a wrong phase, polymorph or structure model.

The episode ends at its overall time limit. Each numerical stage has its own limit computed from the pattern's point count and the number of phases, so a 300-second shell limit can cut off a multi-stage fit even with two phases. A numerical timeout is an execution limit, not evidence that a phase is absent.

The AERIS profile matches the instrument family recorded in the raw measurement. Its scan-specific calibration remains unverified. The Cu spectrum includes K-alpha 1 and K-alpha 2. Fractions describe modeled crystalline phases, not absolute amorphous content. Candidate provenance is not independently adjudicated. Report limits that affect the conclusion.

Each `refine.py` process uses one BGMN thread. The phase search in `screen.py` runs its quick fits in parallel inside the same container. No live host API is needed. The container receives a clean export of the recorded Git source, public actor inputs and task-owned output storage.
