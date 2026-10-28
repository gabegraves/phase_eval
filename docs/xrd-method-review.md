# XRD decision review, version 1

The first capability is interpreting fixed powder-XRD measurements: compare phase hypotheses, interpret residual evidence, revise unsupported conclusions, recover from execution errors, and stop with justified uncertainty. Candidate retrieval and new physical measurements are separate capabilities. Selecting among three supplied phases does not demonstrate broad phase search.

## Task contract

Inputs are the recorded task, measured scan, available chemistry and processing metadata, instrument/profile information, and a versioned candidate library. Missing metadata remains unknown. The existing tools are `case_brief`, `candidates`, `screen`, `refine`, and `write_analysis`. Screening and refinement reveal evidence or change a saved explanatory model; they do not change the measured specimen. Each fit retains its configuration and outputs. A submitted conclusion cites the fit and evidence it uses.

The task's recorded launch configuration defines the budget; review it alongside the instructions, calls, timing and errors. Do not compare runs as matched trials when their tools, library or budgets differ. Distinguish deliberate stopping, budget exhaustion, invalid actions, scientific negative results, and infrastructure interruption. An interrupted initialization supplies no agent decision to grade.

Successful work establishes which candidate interpretations the available evidence supports, which alternatives remain plausible, and the limits of reported quantities. Modeled crystalline fractions do not establish absolute amorphous content or unknown precursor amounts. An incomplete library must admit supported partial identification or an unresolved conclusion; it cannot require an unavailable action or withheld candidate.

Scientific admission requires Gabe's actual review of the task and reference evidence. A model-authored solve, an automated consistency check, or this document does not supply that review. Previously exposed cases are development material. Preserve specimen/preparation groups, related scans and refinement variants together when defining held-out evaluation.

## Review workflow

The workbench starts in Method review. Select a message or tool call in recorded order. Earlier turns show historical observations available before that turn. The selected event shows the recorded rationale/action and the result shown to the agent. Calls from one turn share a prior-information boundary: their sibling results are not evidence available before each other. Final prose is a reviewable event even without a tool call.

The method projection omits final-fit summaries and retained executor output that was not shown to the agent. It expands recorded text rather than treating a clipped preview as the whole observation. The transcript may still omit evidence, and historical observations do not prove what remained in active context after compaction. Review is retrospective; viewing an outcome or remembering an earlier review is not undone by changing tabs.

At consequential decisions, assess:

1. Was the action justified by the information, available tools and budget?
2. Did it usefully discriminate between live hypotheses?
3. Did interpretation and revision respect units, assumptions, conflicting evidence, and measurement versus model uncertainty?
4. Did execution, recovery and verification support the claims made?
5. Was stopping justified by remaining uncertainty, available actions and cost?

Record Pass, Fail or Defer with a concise explanation and evidence links. Pass and Fail require cited events. Fail also names the earliest consequential error. Supporting evidence for that error must precede its turn. Distinguish an actual later repair, linked to its event, from proposed repair advice. An unfavorable result alone is not a bad decision; a mistake that was repaired is not the same as an unrepaired failure. Missing records can justify Defer.

Outcome review remains independently available. Its scientific phase/quantity judgments and sign-off are separate records. Saving a method review does not open reference answers, count as scientific admission, or approve training.

## Records and scoring boundary

`xrd-method-v1` reviews use `trajectory-review.v1`. The server stamps reviewer, reference exposure and time, validates event references, and stores immutable corrections. Retry of the same submission returns the same record; a conflicting retry or stale correction is refused. GET returns only the authenticated reviewer's history. Permission withdrawal excludes records from later exports. Method records export separately as `trajectory_reviews.jsonl`, with pseudonyms, schema and manifest hashes.

Keep scientific validity, outcome judgments, process diagnosis, and resource use separate. The existing diagnostic scalar is unchanged and does not evaluate the method rubric above. Training reward remains zero. Method reviews are not consumed by the scorecard, judge inputs, preference exporter or trainer.

Gabe is the sole domain expert. Delayed randomized repeat reviews can measure within-reviewer consistency; they cannot establish agreement between independent experts. Record prior exposure and changed judgments. Do not treat missing or deferred scientific judgments as negative examples.

## Qualification before learning

For a three-candidate task, compare against all seven nonempty subsets using the same refinement configuration and declared costs, plus a fixed screen/refine procedure. A larger frozen candidate universe is needed only for a retrieval/search claim. Neither more candidates nor lower residuals alone establishes useful decision learning.

Before selecting a training reward, review success, intermediate and failure trajectories, including matched outcomes with different methods, repaired errors, ambiguous evidence and incomplete libraries. Test whether candidate scores preserve Gabe's blinded orderings and resist verbosity, redundant calls, identifier changes, unsupported reference guessing and unwarranted abstention. This qualification has not been completed by implementing the review UI.

The first reward comparison should use a scientifically admitted outcome baseline with process diagnostics retained separately. Add sparse process preferences only if they supply reliable additional information. Keep final evaluation cases separate from task and grader development. A later authorized learning comparison must use matched models, data and budgets and independent outcome assessment; increased reward alone is insufficient evidence of improved science.
