# XRD decision evaluation

This context describes scientific decisions made while interpreting and refining powder XRD measurements, and the evidence used to review and evaluate those decisions.

## Language

**Capability**: A scientific decision-making ability an evaluation is intended to distinguish, such as interpreting residual evidence, revising a phase hypothesis or recognizing an unsupported conclusion.
_Avoid_: Tool usage as a synonym for scientific capability.

**Task contract**: The scientific objective, permitted observations and actions, resource budget, required evidence and success conditions. The same specimen can support different task contracts.

**Trial**: One scheduled execution of a task, including setup, agent interaction and verification. It can fail before the agent acts.

**Agent episode**: The bounded interaction history produced by an agent working on a task. An interrupted episode remains evidence of observed behavior, without necessarily establishing a scientific outcome.

**Observable trajectory**: The ordered instructions, observations, actions, returned results, stated rationales and artifacts available for reconstructing an agent episode. It does not require access to private chain-of-thought.

**Consequential decision**: A choice that changes the scientific interpretation, the next evidence gathered, the resources spent or the decision to stop. It is judged against information available at that time.

**Scientific outcome**: The supported conclusion, uncertainty or justified unresolved result established by an agent episode. It is distinct from successful software execution and from agreement with a published phase list.

**Process diagnostic**: An evidence-backed assessment of a decision or workflow, retained for review and analysis. Its existence does not authorize its use as a training reward.

**Training reward**: The validated signal permitted to influence policy optimization. A diagnostic score is not a training reward merely because it is numerical.

**Scientific admission**: An expert decision that a specified task and its reference evidence support the intended scientific evaluation. It does not itself establish an episode's correctness or authorize a reward design.

**Review protocol**: The order, information exposure and questions used to collect a reviewer’s judgments. Scientific labels collected under different protocols are not automatically interchangeable.

**Attempt group**: A set of trials declared before execution under the same task and policy conditions. Failed and missing trials remain members; multiple fits within an episode are not separate trials.

**Reward profile**: The observed availability and variation of a specified reward across an attempt group, together with the missingness and exclusion reasons. It describes the signal available for learning without establishing scientific validity.
