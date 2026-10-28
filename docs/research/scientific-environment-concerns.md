# Scientific environment: accepted concerns

Preserved from the October 28 working record. These are accepted questions and criteria, not claims of completion.


| ID | Conclusion | Next evidence or action |
|---|---|---|
| C01 | A functioning analysis prototype and trajectory viewer do not yet establish a scientifically valid RL loop. | Demonstrate a qualified task, credible reward, actual optimizer update, and independent evaluation. |
| C02 | Atom Scale's architecture supports capturing experimental context, but does not prove complete experimental history or an implemented agent-learning loop. | Ask for one failed run traced from information available before each action through evaluation and the resulting system change. |
| C03 | Preserve experimental history and agent history, joined at the decision boundary. | Link specimen/preparation/measurement records to exact observations, tool arguments, results, revisions, stopping, model version, and evaluator version. |
| C04 | Unknown preparation or instrument facts must remain unknown. | Expose recorded context with provenance and qualify the claims affected by missing data. Do not invent handling, atmosphere, or per-scan calibration. |
| C05 | Existing resources contain more context than the current DEV006 pilot exposes. | Join selected cases to the source ledger without exposing reference outcomes. The audited rig inventory contains 720 eligible Precursor Genome scans from 643 samples. The original 1,035-experiment ledger has nonempty thermal-profile fields for 981 and actual precursor masses for 969; these source-wide counts are not counts of admitted training cases. |
| C06 | The agent must be able to inspect the evidence used to judge it. | Expose raw patterns and observed/calculated/difference curves, including targeted regions. Saved curve files alone do not prove agent access. |
| C07 | Three supplied candidates support a bounded comparison task, not a broad phase-discovery claim. | Use a frozen, redistributable candidate library; include plausible alternatives and incomplete-library cases. |
| C08 | A first environment can interpret recorded powder-XRD experiments under a limited analysis budget. | Let actions reveal evidence and change saved explanatory models. Physical acquisition and a full experiment simulator are not prerequisites for this scope. |
| C09 | Good outcomes include justified partial identification and unresolved alternatives. | Judge unsupported certainty, ignored residuals, revisions, repaired errors, and stopping against the evidence and actions actually available. |
| C10 | Method assessment does not imply a reward after every action. | Begin with outcome assessment and retained process diagnostics. Add episode-level method preferences or sparse process rewards only when they provide reliable additional signal. |
| C11 | Rewarding a checklist can teach reward exploitation. | Test redundant calls, verbosity, unsupported reference guesses, excessive abstention, alternative valid workflows, and repaired failures. |
| C12 | Expert judgment and reward validation remain necessary. | Collect Gabe's actual judgments, measure within-reviewer consistency, test ordering of contrasting attempts, and qualify any automated judge. Source dataset refinement assessments are not labels of new agent decisions. |
| C13 | Tool execution and token replay have been demonstrated; scientific learning has not. | Latest validation report records zero scientific optimizer updates and no calibrated method reward. Complete the actual trainer path and monitoring before claiming learning. |
| C14 | Compare learned behavior against meaningful simple baselines. | For DEV006, compare all seven nonempty candidate subsets and a fixed procedure under the same configuration and costs. More candidate fits or lower residuals alone are insufficient. |
| C15 | Test whether method supervision adds value rather than presuming it does. | Compare unchanged policy, outcome-based RL, and outcome-plus-method RL with matched data, budgets, and independent scientific outcome assessment. |
| C16 | Keep related specimens, scans, refinement variants, and exposed development cases out of held-out evaluation. | Freeze group-aware splits and record prior exposure; use held-out chemical systems for claims about that kind of generalization. |
| C17 | A public environment must be reproducible from public components. | Package task construction, allowed structures, tool dependencies, reset/step/stop behavior, evaluator, and trainer adapter; identify private dependencies explicitly. |
| C18 | Borrow Periodic's integration of context, tools, and calibrated scientific evaluation. | Do not treat its publications as evidence that every intermediate step needs reward or that our small public benchmark reproduces its private task distribution. |
