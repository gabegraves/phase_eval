# GIWAXS development verification

29 October 2026. The real synchrotron bundle runs through PhaseEval's existing runner, with inspectable reduction choices and an owner method packet. **This proves engineering execution and evidence integrity, not scientific reward validity or learning.**

## Reproduction evidence

Source: [Wu, Kraus and colleagues, Electron Microscopy and GIWAXS Raw Data and Code](https://zenodo.org/records/18484796), DOI `10.5281/zenodo.18484796`, CC BY 4.0. The archive SHA256 is `fcc15f7f3a7d2b2824435975dbcbdabc1f16a29ff1755264857c32a9df42079f`; its 11,312,981 bytes contain the raw detector image, supplied geometry and mask used here.

The comparator is the earlier local execution of the unchanged deposited notebook's scientific cells. Its NPZ SHA256, recorded before this implementation, is `0df7397a3f1b7ced9b533cd98c8dc40387ee207fa9656a27ddf6a228b3d41103`. Original notebook SHA256: `4f6bb73a60969221ecc9e2c93183b2cb485ee745fb79ee6f6393948b53d1994c`. The only added notebook cell exported arrays and checked them. The authors deposited plots, not independent numerical reference arrays; this comparison is not validation against withheld physical truth.

The final recorded invocation finished at `2026-10-29T14:09:14.628467+00:00`. It used the command in the [developer instructions](../../harness/giwaxs/README.md), with the original archive, prior baseline, recorded baseline digest and isolated interpreter supplied explicitly. Its external output directory is named `giwaxs-environment-proof-2026-10-29-reviewed`. Exact absolute arguments are retained in `proof.json`.

| Check | Measured result |
| --- | --- |
| Real execution | Three new reciprocal transforms and native detector decoding through `Runner`; only provider replies scripted. |
| Map reproduction | 2048×2048; qᵣ/qz axes exactly equal to the prior notebook; maximum absolute intensity difference **0.0**. |
| Notebook profile | 2,048 points; axis exactly equal; maximum absolute intensity difference **0.0**. Engineering tolerances were rtol 1e−6, atol 1e−5. |
| Repeat | New worker invocation produced exactly equal map, coverage, axes and profile arrays. |
| Native inspection | Raw, mask, reciprocal-map and profile pages each delivered 12 native rows with hashes and continuation. This is access-path proof, not a full visual analysis. |
| Method chronology | 24 main-episode requests: 20 successful and 4 expected rejections; later valid actions and a cited partial answer preserved. |
| Rejections | Incompatible q/s comparison, collapsed nearest-bin window, unregistered handle, and a separate tampered-input setup attempt refused. |
| Stopping | Successful submission prevented an additional transform proposed in the same tool batch; that request remains recorded as rejected. |
| Owner review | Main, no-reduction partial and setup-failed packets exported; the readable main summary was inspected. One synthetic acceptance control passed and three invalid annotation requests were rejected for their intended reasons; no scientific annotations collected. |
| Accounting | 4 scheduled, 3 attempted, 2 comparable, 1 invalid, 1 missing; no unscheduled attempts in this group. |
| Scientific signal | **0 known scientific rewards, 4 missing, null mean and spread.** |

Worker versions observed: Python 3.10.15, pyFAI 2023.3.0, pygix 2022.11.14, NumPy 1.26.4, Fabio 2025.10.0, SciPy 1.11.4 and silx 1.1.2. PhaseEval's extraction/comparison runtime used NumPy 2.5.4. These versions are part of the comparison identity; installed-package byte identity and full original-environment equivalence remain unknown.

## What the method comparison shows

The notebook's 2D transform omits the mask. A second transform applies the supplied mask; neither choice is automatically scored as scientifically correct. Over qᵣ = 2π × [0.3, 3] nm⁻¹, **456 common fully covered bins** were compared. The exact notebook-mean profiles differ by **0.0 relative L2 on that subset**. The comparator excludes bins outside the window or without full coverage in either profile, including coverage lost through masking. This does not measure the mask's total effect or establish its correctness. Narrowing qz from [0.1, 0.5) to [0.2, 0.4) produced **0.07918989094332912 relative L2 change (about 7.92%)** on common support. The comparison records 1,592 excluded left bins from window/support filters and 1,729 usable masked bins across the full profile. These are descriptive comparisons, not uncertainty estimates or rewards.

The earlier standalone diagnostic reported a tiny mask difference of about 4.52e−8. Its helper used `np.nanmean(values.copy())`, while the notebook and this environment use the ordinary mean on the original strip layout. A follow-up under pinned NumPy 1.26.4 confirmed that the masked maps are exactly identical to that earlier diagnostic; the old helper reproduces its old profile exactly, including the 0.0009765625 maximum rounding difference. The current 0.0 result is recorded rather than forcing agreement with a different reduction helper. Both results demonstrate why a mask-call checklist is insufficient scientific supervision.

The requested notebook window selects 13 rows, with actual sample centers from 0.09704606164039897 to 0.4804449536922703 nm⁻¹ and excluded upper coordinate 0.51239486136326 nm⁻¹. Across the full masked profile, 242 bins have no coverage and 77 have partial coverage. The packet records these facts and distinguishes them from measured zero/negative intensity. It also records the narrower window's six realized rows and the separate coverage-aware extraction.

## Integrity and method assessment

The owner packet binds inputs, artifact lineage, requested/executed geometry, requested/realized ROI, coordinate convention, runtime, receipt, chronology and submission. It keeps shown observations separate from full executor output and preserves exact source copies for later review. Raw and trial files remain external hash-bound references; retaining the whole proof directory is required.

Existing PhaseEval judgment validators check supporting evidence, earliest error and a later repair. Packet event references include runner UUIDs and recorded message IDs. Accepted annotation structure still carries unqualified reviewer provenance, null scientific reward and false scientific qualification. This does not establish expert agreement, evaluator accuracy or whether a scientific judgment is right. No rater-facing route or blindness claim was added.

The proof script is bound as the declared scripted policy revision. This four-slot group is **plumbing/accounting coverage with contrasting programmed branches**, not repeated stochastic policy samples. The profiler's `independent_episodes=2` describes distinct executed episodes, not two independent specimens or a policy-performance estimate. Earlier development directories remain separate: the first exposed a script label-reuse mistake and failed; the next passed before script-version binding was added. Neither was erased or silently substituted into this final group.

## Verification receipts

- Proof script SHA256: `0efa51a435b3b514b2fcdf312494322e128954c37b01d53605c3f92b7829718a`.
- Code and proof script were committed at execution: `df75035c45893f6feb7bca9f031c43eafaf4e979`. The immediate follow-up to that scripted proof changed only this verification document; later live-run fixes are recorded below.
- Comparison fingerprint: `e85b8d2232e74fcc12e4dff1b5593ed5b00cde836194960c1990764002fe3f34`.
- `proof.json` SHA256: `4529cec09456768aedd80433ab314753dd9e5d75c01efb7c583020b68553d5a1`.
- Main packet SHA256: `3d9520c7dd9fd26d9bf3c3db10230e1112698e99caa47e6d5af2f3b48d9de9db`.
- Partial packet SHA256: `775ea225ebaf800abdcb1f8d265989be299107ef0a9ea5ca60ed678615a06870`.
- Failed packet SHA256: `9d773e28c3a96b53841c182971b7273da186d394c881c443c1f647b1d859e33e`.

The final real-data proof passed after the review fixes. Its accepted annotation control is an in-memory synthetic fixture, not a collected human judgment. Each rejected control checks its intended error, so an unrelated packet failure cannot satisfy the proof.

The full repository command, `uv run --project launcher --with pytest --with hypothesis --with jsonschema python -m pytest`, passed **635 tests**, with **46 skips** and **11 pre-existing warnings**, in 62.15 seconds. Skips cover unavailable private smoke/custody data and platform/solver integrations, including Linux bubblewrap and BGMN slice requirements. Warnings concern existing FastAPI/Starlette deprecations and unregistered integration/slow marks. No separate lint/type-check command is configured; `git diff --check` passed.

Code review completed as run `20261029-093224-db3c894c`: nine local lenses plus an independence-verified Claude peer. All four actionable findings were fixed and verified: native-read recovery and label isolation; qz/s numerical coverage; unbound calibration/EDF dependency rejection; and preservation of post-submit requests. Three findings received a separate validator pass; the native-read issue had cross-model corroboration. No actionable finding remains. The final readable packet was inspected after regeneration.

Geometry support is deliberately limited to the deposited self-contained PONI v2 generic Detector form. External spline/detector files and EDF binary payloads are refused. Runtime verification failures fail closed; they do not establish that input bytes actually changed.

## What remains before scientific RL

This slice enables real reduction, observable method choices, reproducible comparisons, structural review validation and complete attempted-group accounting now. It does not supply scan-linked calibration uncertainty, full correction/preparation histories, scientifically qualified interpretation references, expert method/outcome judgments, scorer calibration, independent cases, representative repeated live policy sampling or an optimizer update. Laboratory reconstruction and the paper's final fitted structural quantities remain outside scope. All crops, perturbations and related measurements are one exposed development family.

The next scientific step is to collect qualified judgments on contrasting attempts and test whether an evaluator distinguishes them for defensible reasons, including valid alternative methods and justified partial answers. Reward design and RL experiments follow that qualification; successful tool execution and valid JSON do not replace it.

## Live-run environment corrections and reruns

Later on 29 October, live attempts exposed interface and replay limitations. The following development groups are separate from the scripted proof above and from the original 27B/Astra comparison. All originals and failed attempts were retained.

- `95456aa72c212c9c0f862d370399c2ab0faf004c`: citable context, explicit geometry units/orientation and orthogonal strip semantics, numeric region selection, compact summaries and clearer handle errors. The first 27B rerun produced 21 successful and 4 rejected requests but overflowed its context before submission: the serving tokenizer measured 71,041 tokens against 65,536. Astra submitted partial after 25 successful requests.
- `cf8892516ee5407df1eae52e95333419e5bcce44`: explicit omission of older numeric rows from replay, preserving every latest-batch page and the four most recent pages overall; exact request snapshots; request/response-count validation. Both subsequent attempts submitted partial: 27B had 43 successful/10 rejected requests, Astra 19/1. Those traces revealed a parser incompatibility: the installed `qwen3_xml` parser inferred string/null from unresolved selection references. Some affected calls also had independently invalid limits or fields.
- `905459403711e0dc2f9e5b374836b44e873b739e`: explicit object/null schema types for selection and numeric/text limit descriptions. Runtime validation remains strict. Exact installed parser helpers reproduce the original defect and correctly process all three selection kinds plus null after the fix; three malformed cases remain rejected.

The final, separately declared pair used the standard local Qwen3.6 27B FP8 service and requested GPT-6 Astra/high through the rig's Codex CLI. Controllers and diffraction tools ran on the rig; Astra inference was hosted. Different adapters, system instructions and decoding, plus concurrent workloads, prevent a pure model-only or speed comparison. No immutable served model revision is attested for Astra. Each arm had one attempt, a 30-decision ceiling and 900 seconds.

| Final attempt | Outcome | Model decisions | Successful / rejected tools | Wall time |
| --- | --- | ---: | ---: | ---: |
| 27B | Decision budget exhausted; no submission | 30 | 53 / 3 | 509.4 s |
| Astra | Cited partial submission | 6 | 16 / 0 | 139.2 s |

27B's largest request was 54,330 tokens plus 2,048 reserved output tokens, below 65,536. It did not overflow context. Its final selection is a real object, correctly rejected because detector-kind coordinates were requested on a map. The other rejected reads use a literal `None` handle and an oversized numeric limit. Twenty-two 27B requests omit older numeric rows explicitly; original observations remain saved. Astra's six requests require no such omissions.

All six final profiles independently recompute exactly on the rig. All three maps match corresponding prior transforms, both frozen packets rebuild, and bound files verify in local mirrors. Astra's reported map/coverage counts, sampled local maxima and comparison statistics match saved arrays. Its near-axis qz profile has 78 usable bins, all partially covered by 1–11 of 26 selected strip positions. These checks establish arithmetic and execution, not physical validity.

The earlier context-corrected 27B submission remains a contrasting review case: it reports a cited qz profile's maximum near 1.4 nm⁻¹, while its saved maximum is near 0.289 nm⁻¹, and makes structural assignments while leaving indexing unresolved. The numerical discrepancy is directly checkable; physical assignments require expert adjudication. Neither this model-assisted observation nor Astra's cautious wording constitutes an expert label.

The actor prompt does not explicitly expose the decision ceiling or remaining-turn count. Settle that budget contract before treating a cutoff as a negative stopping-method label. The fixed recent-page replay policy also needs to be considered when reviewing what evidence was available at a particular decision. It is not a total-token guarantee.

Final tests: **684 passed, 46 skipped, 11 existing warnings**, 64.90 seconds; **70 focused tests** passed after the final limit-description clarification. Diff checks passed. Interface review `20261029-161742-b87fc324` completed without actionable findings. Context review `20261029-164333-8a977f68` found two issues (premature newest-page omission and incomplete request lists); both were fixed with red/green regressions. A separate read-only targeted review of the parser correction found no actionable issue; it is not represented as part of the earlier full review.

Final packet hashes: 27B `d1b6545f814440fcb54e7a06059138064c314ee938fecb419d7a85be454cb012`; Astra `16cd1bcb310b1208cb1654537eaf2ff7bed8a2ddadbc2bc336a768165259ca05`. External owner artifacts are in `giwaxs-parser-rerun-2026-10-29`: `REPORT.md`, `TRAJECTORIES.md`, `EXPERT-REVIEW.md`, request snapshots, frozen packets and verification receipts. The custody chain verifies unchanged earlier records: 191 context-run, 153 interface-run and 189 original rig files; local checks also preserve the original comparison's 198 files and the earlier 7B run's 46 files. These counts refer to separate directories, not deduplicated cases.

Scientific reward remains null and qualification false throughout. The next gates are expert adjudication, a settled task/budget contract, blinded evaluator qualification and independent held-out specimen/study families. No optimizer update or RL improvement was demonstrated.
