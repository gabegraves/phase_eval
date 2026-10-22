# Task

Identify the crystalline phases in case {case_id} and estimate their weight fractions.

You have five function tools. They replace the shell commands named above.

- `case_brief()` runs `case_brief.py --case {case_id}`.
- `candidates()` runs `candidates.py --case {case_id}`.
- `screen(residual=null)` runs `screen.py --case {case_id} --no-search`, or `--residual <run_label>` when you pass one.
- `refine(candidates, run_label, pawley_only=false)` runs `refine.py --case {case_id} --candidates ... --run-label <label>`. A label cannot be reused.
- `write_analysis(analysis)` writes your answer to `/workspace/output/analysis.json`.

The answer is a JSON object with `case_id` ("{case_id}"), `phases` (a list of objects, each with `candidate_id` from the candidate library), `weight_fractions` (an object from candidate_id to a number from 0 to 1), and `fit_artifacts` (paths of the fit files the answer rests on, such as `/workspace/output/<run_label>/result.json`). Cite the refine run whose candidates are exactly your answered phases. Write the answer before you stop. The episode ends when you reply without a tool call.
