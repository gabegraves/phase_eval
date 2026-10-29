# GIWAXS development environment

This task runs detector-image reduction through the existing PhaseEval agent runner. It preserves geometry and mask choices, extraction windows, coordinate definitions, coverage, observations, failures and cited conclusions. An owner can inspect the resulting method packet and supply an explicitly unqualified annotation. Scientific reward is always unavailable.

The initial case is the synchrotron branch of [Wu, Kraus and colleagues' deposited data](https://zenodo.org/records/18484796), DOI `10.5281/zenodo.18484796`, CC BY 4.0. Keep attribution with the case. The laboratory branch and final paper fitting results are outside this implementation. All related data are one exposed development family, not independent train/test cases.

## Isolated physics runtime

Run PhaseEval with its existing Python 3.12 launcher. Create a separate Python 3.10.15 environment for the archived scientific implementation:

```sh
uv venv --python 3.10.15 /absolute/path/giwaxs-runtime
uv pip install --python /absolute/path/giwaxs-runtime/bin/python -r harness/giwaxs/requirements.txt
```

The runner probes actual versions before provider calls and rechecks them before use. The pins reproduce the known notebook implementation; they are not a claim of complete original-environment or cross-platform equivalence. pyFAI may build from source. Do not replace the launcher's dependencies with this environment.

Geometry is limited to the deposited self-contained PONI v2 generic `Detector` form with inline numeric pixel/shape configuration. External spline files, detector files and EDF binary payloads are rejected; supporting those requires explicit manifest bindings for every dependency.

## Prepare the independent notebook baseline

Download `GIWAXS.zip` from the linked record. Its SHA256 must be `fcc15f7f3a7d2b2824435975dbcbdabc1f16a29ff1755264857c32a9df42079f` (11,312,981 bytes). Inspect the deposited notebook before executing it. For notebook use, add `matplotlib==3.8.4` and `ipykernel` to the isolated environment and select that interpreter as the notebook kernel.

Run the unchanged scientific cells in `GIWAXS/synchrotron/Synchrotron-GIWAXS.ipynb`, with its raw TIFF, PONI and EDF files in the working directory. Append this export cell after execution:

```python
np.savez_compressed("baseline-arrays.npz", intensity=ii_2d, qr=qr, qz=qz, ip=ip)
```

Record the NPZ's SHA256 immediately, along with the original notebook hash, interpreter/package versions and executed notebook. Pass that previously recorded digest to the proof. The baseline is a local notebook re-execution, not numerical ground truth deposited by the authors. Never generate it from the new environment's outputs.

## Run the real-data proof

From the repository root, supply absolute paths and a new output directory outside the repository:

```sh
uv run --project launcher python scripts/prove_giwaxs_environment.py \
  --archive /absolute/path/GIWAXS.zip \
  --giwaxs-python /absolute/path/giwaxs-runtime/bin/python \
  --baseline /absolute/path/baseline-arrays.npz \
  --baseline-sha256 PREVIOUSLY_RECORDED_SHA256 \
  --out /absolute/path/new-giwaxs-proof
```

The script replaces only provider replies. It executes real decoding and three real reciprocal transforms through `Runner`, checks the prior map/profile baseline and a deterministic repeat, records mask/window sensitivities, and exercises rejected coordinates/windows/handles. Four scheduled slots retain a full reduction, a no-reduction partial answer, a tampered-input setup failure and one missing attempt. Inspect:

- `proof.json`: exact invocation, versions, hashes, numerical differences, expected rejections and limitations.
- `profile.json`: complete attempt accounting and zero known scientific rewards, with null mean/spread.
- `reviews/main/METHOD.md` and `packet.json`: ordered method, actual requested/executed parameters, realized windows, shown/full observations and evidence links.
- `reviews/partial/` and `reviews/failed/`: unresolved and failed cases remain inspectable.

Keep the entire proof directory. Packets preserve the bound source code, but their raw/trial files remain hash-bound at the recorded paths. The scripted proof demonstrates engineering behavior. Separate live-model development attempts and their limitations are recorded in [the verification record](../../docs/research/giwaxs-development-verification.md); they do not establish general model performance or scientific reward validity. See the [measured verification record](../../docs/research/giwaxs-development-verification.md).

## Agent tools and interpretation boundaries

| Tool | Behavior |
| --- | --- |
| `read_evidence` | Citable, hash-bound context; exact text pages or native detector/map/profile rows with compact summaries and optional numeric selection. |
| `transform` | Explicit geometry and `supplied`/`omit` mask policy, pinned worker, immutable map and receipt. Alternative geometry remains an assumption. |
| `extract` | Nearest-bin, upper-exclusive strip; requested and realized bounds; `notebook` or `coverage_aware` mean. |
| `compare` | Compatible profiles over common covered support; interpolation never bridges internal uncovered gaps. |
| `submit` | Cited claims or partial answer, limitations and evidence needed; successful submission ends execution. Further requests in the same batch remain recorded as rejected. |

Use a fresh label for every transform, extraction and comparison, including across different tools. An allocated label remains consumed after failure.

Map coordinates use **q = 2π/d** in nm⁻¹. Extracted profiles may explicitly use **s = 1/d**, also in nm⁻¹; unit equality does not make the conventions interchangeable. The extraction `axis` is the output axis: `qr` averages a `qz` strip and `qz` averages a `qr` strip. Strip bounds always describe q, regardless of output convention. Receipts name the ROI axis separately from output extent and usable support; a support bounding interval may contain gaps. Notebook aggregation preserves fill contributions for reproduction and flags incomplete coverage. Coverage-aware aggregation excludes uncovered cells; measured zero and negative intensities remain measurements. A mask call or narrower window does not earn scientific credit automatically.

Acquisition and transform incidence/tilt angles are **degrees**. PONI rotations are **radians** and lengths are **meters**. The tool schema describes pygix's four sample orientations in detector image coordinates.

Numeric `read_evidence` calls can add `selection`: profiles use `{"kind":"profile","lower":0,"upper":1}` in their declared output convention; maps use `{"kind":"map","qr_min":0,"qr_max":1,"qz_min":0,"qz_max":1}` in q nm⁻¹; detector images/masks use `{"kind":"detector","row_min":0,"row_max":9,"column_min":0,"column_max":9}` in zero-based pixels. These illustrative bounds are inclusive and must lie within the actual artifact extent. Selection changes neither the data nor coverage. Repeat the same selection with `next_start`; offsets count selected rows and `row_indexes` retain native flat indexes. Omitting selection preserves native paging. Summaries describe the full artifact, including comparison metrics without the potentially long index list; full data and receipts remain available.

Before each GIWAXS chat call, replay keeps every numeric page from the latest assistant/tool batch and the four most recent numeric pages overall in full, replacing other older `rows` and `row_indexes` with explicit reread notices. All messages, tool call IDs, metadata, hashes, summaries, coordinates, selection and paging handles remain; text and errors remain exact. The actor can reread older pages and retain concise findings in visible assistant messages. This fixed policy reduces repeated numeric data but does not guarantee a model's token limit. Original recorded tool observations remain unchanged in the chronology. Powder replay is unchanged.

Each GIWAXS call saves its exact messages and tool schema in a unique `agent/provider-request-NNNN.json` before invoking chat, including custom runner chat methods and failed calls. The receipt binds each file's hash and identifies omitted message indexes. Provider credentials are not included. Owner review verifies and links these files in `METHOD.md`, distinguishing recorded tool observations from inputs supplied to chat in a particular request. A snapshot does not prove the provider processed the request. When request logs are present, their count must match recorded responses, with at most one additional pending or failed call. Historical trials without request logs remain readable; review does not create missing logs.

The proof creates a versioned `inputs/manifest.json` that can serve as a development case for a real provider:

```sh
uv run --project launcher python -m harness.agent.run_agent \
  --task-kind giwaxs --input-dir /absolute/path/proof/inputs --case synchrotron_s16 \
  --giwaxs-python /absolute/path/giwaxs-runtime/bin/python \
  --base-url https://YOUR_PROVIDER/v1 --model YOUR_MODEL \
  --out /absolute/path/new-trial --max-turns 30 --time-limit 900
```

Use the runner's existing API-key option and predeclared-group options for actual collections. Ordinary powder-XRD calls remain the default task kind. A GIWAXS reduction cannot receive a powder scorecard.

## Owner review

```sh
uv run --project launcher python -m harness.giwaxs.review \
  --trial /absolute/path/trial --out /absolute/path/new-owner-packet
uv run --project launcher python -m harness.giwaxs.review \
  --packet /absolute/path/new-owner-packet/packet.json --annotation /absolute/path/annotation.json
```

For a valid run with no registry, also pass `--input-dir`. The [annotation schema](../../schemas/giwaxs-method-annotation.v1.json) requires the packet digest, unqualified reviewer provenance, verdict and explanation. Evidence references use recorded event UUIDs or displayed `message:<turn_index>` IDs. Supporting evidence must precede the error's decision; a repair must be a later agent turn. Validation rejects invented events, future evidence, stale files and forged reward/qualification fields. It does **not** establish that an accepted judgment is scientifically correct. The CLI validates an already saved annotation; it creates no expert label or training reward.

Before scientific RL, obtain scan-linked calibration/correction history, representative independent cases, qualified outcome and method judgments, disagreement handling, and a validated scorer. Then measure reward availability and variation on real policy rollouts before any optimizer experiment. This slice adds no blind-rater route, scientific reward calibration, held-out benchmark or training loop.
