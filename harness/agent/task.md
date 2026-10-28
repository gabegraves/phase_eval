# Task

Identify the crystalline phases in case {case_id} and estimate their weight fractions.

You have six function tools. Only these functions are available; there is no shell or tree-search action.

- `read_evidence(handle, start=0, limit, two_theta_min, two_theta_max)` reads registered evidence. Omit `handle` for the catalog. `scan` supplies native observed rows; `context` supplies actor-safe context with sources; `candidate:<id>` supplies its CIF. After a fit, `fit:<run_label>:<stage>` supplies native curves and `fit:<run_label>:result` supplies the result. The maximum page is 100 numeric rows or 2000 text characters. Continue at `next_start` until null; repeat angle bounds for an interval. Row indexes belong to each artifact, because raw and fitted grids can differ. Text offsets count Unicode characters; concatenating text pages preserves the UTF-8 source bytes. Responses identify the artifact digest. A clipped output names an `output:<digest>` handle for recovery.
- `case_brief()` runs `case_brief.py --case {case_id}`.
- `candidates()` runs `candidates.py --case {case_id}`.
- `screen(residual=null)` runs `screen.py --case {case_id} --no-search`, or `--residual <run_label>` when you pass one.
- `refine(candidates, run_label, pawley_only=false)` runs `refine.py --case {case_id} --candidates ... --run-label <label>`. A label cannot be reused.
- `write_analysis(analysis)` writes your answer to `/workspace/output/analysis.json`.

The answer is a JSON object with `case_id` ("{case_id}"), `phases` (a list of objects, each with `candidate_id` from the candidate library), `weight_fractions` (an object from candidate_id to a number from 0 to 1), and `fit_artifacts` (paths of the fit files the answer rests on, such as `/workspace/output/<run_label>/result.json`). Cite the refine run whose candidates are exactly your answered phases. Write the answer before you stop. The episode ends when you reply without a tool call.

State evidence limits and unresolved observations in your final reply. Conditional estimates, uncertainty, and abstention from unsupported conclusions are acceptable. A missing candidate need not be identifiable. Do not infer precursor ratios when amounts are withheld, or processing history when none is recorded. Explain why the available evidence and remaining budget justify stopping.
