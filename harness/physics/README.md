# Physics tools

The runner exposes the function tools listed in the task. The physics functions print short summaries; `read_evidence` supplies bounded pages of registered native evidence. There is no shell action.

- `case_brief.py`: chemistry, measurement, specimen, precursors with weighed masses and the molar ratios they imply, processing steps with the furnace trace summarized, and the declared target when one is recorded. Read the full permitted arrays, units and field sources using `read_evidence(handle="context")`. Unknown extensions and custody-sourced fields are excluded before summaries are calculated. Bare legacy fields identify the bundled manifest as their source; richer provenance and absent preparation or calibration facts remain unknown.
- `candidates.py`: one line per candidate structure with formula, space group, crystal system and cell, read from its CIF and grouped by formula. Structures with the same formula and different space groups are different phases (polymorphs), and they give different patterns.
- `screen.py`: ranks every candidate by how well its simulated Cu K-alpha1 lines land on observed peaks, lists strong observed peaks that a covering set of the best structures does not explain, without a tree phase search. The runner always selects `--no-search`. `--residual <run-label>` lists the largest residual peaks of a saved refinement and the candidates with lines near each one.
- `refine.py --candidates <ID> [<ID> ...] --run-label <fresh-label>`: staged BGMN refinement, Pawley then Rietveld stages. It prints each stage's Rwp, Rexp, GoF and convergence; for each phase at the final stage, the weight percent with its esd and the refined cell with its percent shift from the CIF cell; and the `rietveld_degraded_after_good_pawley` flag. The same numbers are in `<run-label>/result.json`. Curves are in `<run-label>/<stage>.csv` (two_theta, observed, calculated, difference).

## Working order

1. Read the brief and the candidate table.
2. Screen before refining. Use the observed peaks and candidate ranking to choose hypotheses.
3. When several structures share a formula, refine each one in turn with the rest of the phase set unchanged, and keep the one the data support: lower Rwp, a flatter residual, and a small cell shift.
4. When the processing history is short or at low temperature, expect unreacted precursors and intermediate phases beside any declared target, and include them in your hypotheses.
5. After each fit, run `screen.py --residual <run-label>` and try candidates for the strongest unexplained residual peaks.
6. When weighed masses are given, convert the refined weight fractions to element molar ratios and check them against the ratios the brief prints. Report any disagreement.
7. Write a provisional `/workspace/output/analysis.json` as soon as one fit is defensible, then improve it as the analysis proceeds.

## Refinement notes

`refine.py` calls the existing staged BGMN backend. It verifies pattern and candidate hashes before execution and saves the numerical work directory, stage results and curves. It does not assign a scientific ACCEPT verdict. Use a fresh run label for each hypothesis. Up to five distinct candidates may be refined together. `--pawley-only` stops after the profile fit; Pawley intensity estimates are not Rietveld weight fractions.

The default command runs Pawley followed by staged Rietveld when the backend's existing quality checks permit it. A poor Pawley fit can prevent Rietveld. Inspect the returned stages rather than assuming all refinements ran. A result with no Rietveld stage does not supply a structure-based quantitative result. When `rietveld_degraded_after_good_pawley` is true, the peak positions fit but the structure model does not: suspect a wrong phase, polymorph or structure model.

The episode ends at its overall time limit. Each numerical stage has its own limit computed from the pattern's point count and the number of phases, so numerical work can stop before all stages complete. A numerical timeout is an execution limit, not evidence that a phase is absent.

The refinement uses the bundled AERIS profile as a model assumption. Whether that model matches the recorded instrument must be checked against the available context; scan-specific calibration remains unverified. The Cu spectrum includes K-alpha 1 and K-alpha 2. Fractions describe modeled crystalline phases, not absolute amorphous content. Candidate provenance is not independently adjudicated. Report limits that affect the conclusion.

Each `refine.py` process uses one BGMN thread. The runner uses the supplied case inputs and its own episode output storage. Evidence reads accept registered logical handles only.
