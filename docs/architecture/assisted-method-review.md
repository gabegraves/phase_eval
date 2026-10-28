# Agent preparation, bounded resolution, expert exceptions

## Problem

Expert time should resolve consequential scientific uncertainty and qualify the environment. It should not be spent locating tool output, checking JSON, finding broken artifact links, or copying model observations into a form. PhaseEval already separates human judgments, model outcome labels, deterministic checks and training reward. The missing capability is a method-review assistant that prepares an evidence-linked assessment, seeks stronger help when needed, and records what remains unresolved.

## Usage and authority

The owner prepares a frozen packet from an imported agent-answer snapshot. A first-pass judge reads that packet and returns a structured assessment. If the deterministic policy requests help, a separately frozen teacher packet includes the original evidence and first proposal. A teacher, such as Astra, gets one resolution pass. The report groups remaining scientific questions and lists audit invitations. Every original case remains individually linked; answering a grouped question never copies a label onto other cases.

The command-line interface is `scripts/workbench/assist_review.py`: `prepare`, `submit`, `status`, and `report`. Imported model outputs are operator-attested. Recording a model name is not cryptographic proof of which provider ran it. Use a fresh packet-only invocation when possible, and record inherited context or uncertainty honestly. The implementation does not launch a provider or perform proposed numerical experiments.

Run from the repository with its Python environment. Keep the output directory on the owner host, outside the actor's writable workspace:

```sh
PYTHONPATH=src python scripts/workbench/assist_review.py prepare --store "$STORE" --snapshot "$SNAPSHOT" --out "$OUT"
# Give the returned packet's prompt.txt, input.json and response-schema.json to the reviewer.
PYTHONPATH=src python scripts/workbench/assist_review.py submit --out "$OUT" --packet "$PACKET" --response "$RESPONSE" --model "$MODEL" --run-id "$RUN_ID" --context fresh_packet
# If requested, review and submit the returned teacher packet in the same way.
PYTHONPATH=src python scripts/workbench/assist_review.py status --out "$OUT" --packet "$PACKET"
PYTHONPATH=src python scripts/workbench/assist_review.py report --out "$OUT"
```

Use the original first-pass packet for `status`. Preserve rejected responses and any repair feedback alongside execution records; an accepted-proposal record alone does not capture failed model attempts. A repair using feedback has inherited context. The combined owner report contains model suggestions: give a blind auditor only the original packet identified by an audit invitation, not the combined report.

An agent-resolved status is a provisional operational disposition. It does not establish a correct phase identification, an expert label, scientific admission, calibrated reward, or learning gain. Existing human judgments and scientific gates retain their authority.

## Shape

`assisted_review.py` owns immutable packets and model proposals. The exact prompt, projected observations, response schema and teacher-parent proposal are bound by hashes. External responses have a closed schema and validated event references. A correction or different model invocation cannot silently replace an existing accepted result. The CLI adapts files; it does not implement scientific routing.

`snapshot.method_trace` remains the source of scrubbed recorded observations. It supplies no reference answer, human correction, or retained executor output the actor never saw. `triage.py` owns a separate versioned assisted disposition policy and deterministic seeded audit selection. Existing outcome `RouteDecision` records retain their original numerical-check semantics; the assisted workflow does not invent check failures to encode scientific questions.

Each output directory belongs to a frozen policy version. The importer rejects older policies rather than reinterpreting them under new routing rules. Retain their original report and code revision for replay; prepare a separate output directory for a revised policy.

V1 reviews the complete historical trace. Prior-event citations prevent impossible evidence links but do not make a whole-trace review prefix-blind. A future decision-time reviewer must receive a physically truncated packet in a fresh invocation, containing only prior observations and the proposed action. A separate retrospective pass can then examine the result, updates, repairs and stopping. Private model reasoning is neither requested nor required.

The observable review contract is: judgment, concise explanation, supporting event references, earliest consequential error if present, observed repair if present, proposed action, and unresolved question. Suggestions are not recorded as executed repairs. Trace text is untrusted evidence; it cannot authorize tools, change policy, select a route, or write an expert record.

## Routing and expert time

| Situation | Next owner |
| --- | --- |
| Invalid response, broken references, corrupt or missing record | Agent/operator repairs the evidence or execution path |
| Routine supported assessment | Provisional model resolution, subject to audit |
| First-pass failure, deferral, novelty or scientific question | One stronger-teacher pass |
| Necessary observation unavailable | Explicit unresolved result; no invented measurement |
| Remaining scientific question, novelty or material disagreement | Expert receives a focused question with cited evidence |
| Seeded audit selection | Expert reviews original evidence independently of proposals |

The sampled population and pending cases remain visible. A small audit rate is a provisional workload choice, not a demonstrated safety or accuracy bound. Audits must include apparently good runs; only inspecting disagreements cannot detect shared blind spots. Track missed consequential failures, unsupported flags, unresolved cases, teacher escalation and expert minutes per useful correction. Report uncertainty and selection bias before interpreting those metrics.

Initial task-family admission, unresolved physics assumptions and permission to use a reward still require the designated domain expert. Those decisions should be prepared once with reusable evidence and a clear scope. They are not a request for the expert to annotate every routine trajectory. With one expert, blind repeats measure within-reviewer consistency; they do not establish independent expert agreement.

Expert corrections may become cited guidance in a new prompt or a proposed policy revision, within their permitted use. Preserve their scope and exposure. Compare the revision on separately held-out judgments. A future escalation rule can catch the same failure family; it must not extrapolate one expert approval to unseen cases or rewrite historical records.

## Model choice and teaching

Astra can act as reviewer and teacher using this observable contract without disclosing private reasoning. Its recommendations need evidence and, when they propose a different experiment or tool action, execution-based verification before becoming successful training demonstrations. Open-weight models offer reproducible deployments, inspectable output traces and a different model family; those are useful comparison axes, not guarantees of scientific correctness.

Qwen's official [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) supplies open weights and describes thinking and non-thinking modes. Kimi's [official model documentation](https://www.kimi.com/code/docs/en/) lists K3.1 Pro and its required thinking mode. This design does not assume K3.1 weights or a training license have been verified. Model availability and deployment identifiers must be pinned when a benchmark is actually run.

Visible reasoning is an optional diagnostic, not ground truth about internal causation. Research has demonstrated that models can omit influences on their answers from their visible reasoning ([Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410)). Evaluate teachers on supported conclusions, valid actions, recovery, uncertainty and measured expert agreement. Do not reward long explanations or select a teacher merely because it exposes more text.

Keep student policy learning separate from judge calibration. The intended progression is a baseline, a small verified teacher-demonstration SFT comparison, then RL only after task admission and reward qualification. Train observable tool decisions and supported concise rationales. Do not treat a retrospective critique as a successful rollout, or recycle development cases as held-out evaluation.

## Synthesis decision

Four independent candidates favored frozen packets over immediate background-service integration. A separate live-queue candidate established the alternative shape: authenticated runner jobs, atomic stage transitions, an exposure ledger and an assisted inbox. The cross-judge scored A and B 9/10 and the live alternative 6/10. The chosen implementation uses B's immutable custody boundary and A's complete bounded workflow, operator-attested provenance and explicit hindsight-only ceiling. Parent selection includes routing now because prepare/import alone would not address expert workload.

Rejected additions include a provider SDK, generic model registry, job leases, consensus scores, an unmeasured robustness probe and a learned grader. Existing outcome JudgeIdentity and RouteDecision are not repurposed: their contracts are different. Grouped question presentation retains every case and confers no new authority. No calibration or scientific accuracy is inferred from schema validity.

## Implementation boundary and next step

The first implementation is an owner-host packet/import/report workflow. It does not automatically invoke the local model or Astra, alter the browser's human-review fields, hide cases from its queue, execute repairs, export training data or enable reward. This ceiling avoids misrepresenting an incomplete automation path as a calibrated autonomous reviewer.

The next live adapter should use these same records, one integrated queue and an explicit assistance-exposure receipt. Assisted exceptions can show model proposals immediately with their provenance; blind audit tasks must withhold them until the expert's initial judgment is stored. Both should provide a pattern view, the relevant chronology segment and a direct route to the complete trace. That is where the existing human form becomes a focused adjudication surface rather than the default preparation task.
