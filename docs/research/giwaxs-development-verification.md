# GIWAXS development verification

29 October 2026. The real synchrotron bundle runs through PhaseEval's existing runner, with inspectable reduction choices and an owner method packet. **This proves engineering execution and evidence integrity, not scientific reward validity or learning.**

## Reproduction evidence

Source: [Wu, Kraus and colleagues, Electron Microscopy and GIWAXS Raw Data and Code](https://zenodo.org/records/18484796), DOI `10.5281/zenodo.18484796`, CC BY 4.0. The archive SHA256 is `fcc15f7f3a7d2b2824435975dbcbdabc1f16a29ff1755264857c32a9df42079f`; its 11,312,981 bytes contain the raw detector image, supplied geometry and mask used here.

The comparator is the earlier local execution of the unchanged deposited notebook's scientific cells. Its NPZ SHA256, recorded before this implementation, is `0df7397a3f1b7ced9b533cd98c8dc40387ee207fa9656a27ddf6a228b3d41103`. Original notebook SHA256: `4f6bb73a60969221ecc9e2c93183b2cb485ee745fb79ee6f6393948b53d1994c`. The only added notebook cell exported arrays and checked them. The authors deposited plots, not independent numerical reference arrays; this comparison is not validation against withheld physical truth.

The final recorded invocation finished at `2026-10-29T13:18:05.193000+00:00`. It used the command in the [developer instructions](../../harness/giwaxs/README.md), with the original archive, prior baseline, recorded baseline digest and isolated interpreter supplied explicitly. Its external output directory is named `giwaxs-environment-proof-2026-10-29-final`. Exact absolute arguments are retained in `proof.json`.

| Check | Measured result |
| --- | --- |
| Real execution | Three new reciprocal transforms and native detector decoding through `Runner`; only provider replies scripted. |
| Map reproduction | 2048×2048; qᵣ/qz axes exactly equal to the prior notebook; maximum absolute intensity difference **0.0**. |
| Notebook profile | 2,048 points; axis exactly equal; maximum absolute intensity difference **0.0**. Engineering tolerances were rtol 1e−6, atol 1e−5. |
| Repeat | New worker invocation produced exactly equal map, coverage, axes and profile arrays. |
| Native inspection | Raw, mask, reciprocal-map and profile pages each delivered 12 native rows with hashes and continuation. This is access-path proof, not a full visual analysis. |
| Method chronology | 23 main-episode actions: 20 successful and 3 expected rejections; later valid actions and a cited partial answer preserved. |
| Rejections | Incompatible q/s comparison, collapsed nearest-bin window, unregistered handle, and a separate tampered-input setup attempt refused. |
| Stopping | Successful submission prevented an additional transform proposed in the same tool batch. |
| Owner review | Main, no-reduction partial and setup-failed packets exported; the readable main summary was inspected. Three invalid annotation requests rejected; no scientific annotations collected. |
| Accounting | 4 scheduled, 3 attempted, 2 comparable, 1 invalid, 1 missing; no unscheduled attempts in this group. |
| Scientific signal | **0 known scientific rewards, 4 missing, null mean and spread.** |

Worker versions observed: Python 3.10.15, pyFAI 2023.3.0, pygix 2022.11.14, NumPy 1.26.4, Fabio 2025.10.0, SciPy 1.11.4 and silx 1.1.2. PhaseEval's extraction/comparison runtime used NumPy 2.5.4. These versions are part of the comparison identity; installed-package byte identity and full original-environment equivalence remain unknown.

## What the method comparison shows

The notebook's 2D transform omits the mask. A second transform applies the supplied mask; neither choice is automatically scored as scientifically correct. Over qᵣ = 2π × [0.3, 3] nm⁻¹, **456 common fully covered bins** were compared. The exact notebook-mean path produced **0.0 relative L2 difference** for mask application. Narrowing qz from [0.1, 0.5) to [0.2, 0.4) produced **0.07918989094332912 relative L2 change (about 7.92%)**. These are descriptive sensitivities, not uncertainty estimates or rewards.

The earlier standalone diagnostic reported a tiny mask difference of about 4.52e−8. Its helper used `np.nanmean(values.copy())`, while the notebook and this environment use the ordinary mean on the original strip layout. A follow-up under pinned NumPy 1.26.4 confirmed that the masked maps are exactly identical to that earlier diagnostic; the old helper reproduces its old profile exactly, including the 0.0009765625 maximum rounding difference. The current 0.0 result is recorded rather than forcing agreement with a different reduction helper. Both results demonstrate why a mask-call checklist is insufficient scientific supervision.

The requested notebook window selects 13 rows, with actual sample centers from 0.09704606164039897 to 0.4804449536922703 nm⁻¹ and excluded upper coordinate 0.51239486136326 nm⁻¹. Across the full masked profile, 242 bins have no coverage and 77 have partial coverage. The packet records these facts and distinguishes them from measured zero/negative intensity. It also records the narrower window's six realized rows and the separate coverage-aware extraction.

## Integrity and method assessment

The owner packet binds inputs, artifact lineage, requested/executed geometry, requested/realized ROI, coordinate convention, runtime, receipt, chronology and submission. It keeps shown observations separate from full executor output and preserves exact source copies for later review. Raw and trial files remain external hash-bound references; retaining the whole proof directory is required.

Existing PhaseEval judgment validators check supporting evidence, earliest error and a later repair. Packet event references include runner UUIDs and recorded message IDs. Accepted annotation structure still carries unqualified reviewer provenance, null scientific reward and false scientific qualification. This does not establish expert agreement, evaluator accuracy or whether a scientific judgment is right. No rater-facing route or blindness claim was added.

The proof script is bound as the declared scripted policy revision. This four-slot group is **plumbing/accounting coverage with contrasting programmed branches**, not repeated stochastic policy samples. The profiler's `independent_episodes=2` describes distinct executed episodes, not two independent specimens or a policy-performance estimate. Earlier development directories remain separate: the first exposed a script label-reuse mistake and failed; the next passed before script-version binding was added. Neither was erased or silently substituted into this final group.

## Verification receipts

- Proof script SHA256: `fe1ec27955952e653637e719f2eec7f893b3a34a431bd27acec8b72408239dd3`.
- Runner implementation Git HEAD at execution: `8b739b95376b84a53ff0d76d4dd647b74dc866e3`; the then-uncommitted proof script is separately hash-bound above.
- Comparison fingerprint: `9baf711d379c1c941d9e36096c5da5be6598be9fe6809f87c7048472fb17d805`.
- `proof.json` SHA256: `3713dad2008cd2fd36c37a4008600c9f9ffd973a5df6aa89cfa22a0b6d2c8c70`.
- Main packet SHA256: `2ac3a98cc95c7ec433d3ee351ad0139d77ce4b71588c6637086e594d9bd06d60`.
- Partial packet SHA256: `550c9b87b1c446b9b148999bf9ec0885f652c2643c5c6538fe3ed69dae25b384`.
- Failed packet SHA256: `de1b33bae69bc8b65d47004039a3ff7c720521aae86fda054b514f50a2b50343`.

The real-data proof was the U4 integration check, without a duplicate mock physics layer. The focused GIWAXS/runner/review/reward-profile command passed **100 tests**. Full-repository verification and code-review evidence belong to the final delivery record; this unit does not claim those have already completed.

## What remains before scientific RL

This slice enables real reduction, observable method choices, reproducible comparisons, structural review validation and complete attempted-group accounting now. It does not supply scan-linked calibration uncertainty, full correction/preparation histories, scientifically qualified interpretation references, expert method/outcome judgments, scorer calibration, independent cases, live policy sampling or an optimizer update. Laboratory reconstruction and the paper's final fitted structural quantities remain outside scope. All crops, perturbations and related measurements are one exposed development family.

The next scientific step is to collect qualified judgments on contrasting attempts and test whether an evaluator distinguishes them for defensible reasons, including valid alternative methods and justified partial answers. Reward design and RL experiments follow that qualification; successful tool execution and valid JSON do not replace it.
