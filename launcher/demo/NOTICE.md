# PhaseEval demo data

The launcher unpacks `phaseeval-demo.tar.xz` into its state folder on first start.
Its SHA-256 is `f791c07e48b2d9d3fa24581d21d8268c30aff8ea978a22b75708d79bf9c0814f`.

- `store/` is a review store rebuilt from the six Qwen3.6-27B AgentXRD runs in the public export (three on DEV006, three on DEV102). It holds their attempts, answer snapshots and routing decisions. It holds no label, sign-off, judge manifest or reference answer.
- `dataset/` is the public-scope export of that store, which the scorecard reads. It holds the same six runs as `data/dataset-v0-public/`, but its snapshot digests differ because the store was rebuilt. It holds no expert label and no model-judge label.

The DEV006 scan comes from the DARA benchmark and the DEV102 scan and synthesis record from the Precursor Genome release, both CC BY 4.0. The candidate crystal structures come from the Crystallography Open Database (CC0). The repository's `NOTICE` gives the full attribution.

BGMN is not included. See "BGMN" in the repository's `README.md`.

The content-addressed rebuild recipe is committed in `tools/demo/rewrite_store.py`.
From the repository root, with a copy of the original archive, run:

```sh
PYTHONPATH=src uv run --locked python tools/demo/rewrite_store.py original.tar.xz rebuilt.tar.xz '<old-base-url>' 'http://redacted.invalid/v1' hash-map.json
PYTHONPATH=src uv run --locked python tools/demo/compare_exports.py original.tar.xz rebuilt.tar.xz hash-map.json '<old-base-url>' 'http://redacted.invalid/v1'
```

The comparison uses the root locked environment. The recipe remaps blob and
record hashes, re-derives episode IDs with the exporter, and checks their references and
sizes. The comparison validates episode IDs and stored blob sizes in both bundled datasets
and fresh CLI exports before comparing their records after remapping.
Reapplying the replacement reaches a fixed point. Archive members are sorted with zero mtimes.
Both tools treat episode artifact rights, judge manifest entries, attempt input hashes,
imported receipt references, manifest root and leaf lists, and assignment snapshot lists as
unordered collections. The comparison also treats JSONL records as an unordered collection.
It preserves order in other arrays, including tool calls and coordinates.

Two provenance placeholders are accepted by the leak scanner: `/home/coder/svn-repositories/`
is the upstream COD CIF provenance prefix; `/private-runs/` is the original export's
redaction placeholder for host paths. It is not a macOS `/private/` system path.
Demo job identities are intentional fixture data. Artifact tests omit the scanner's
`demo_run` rule; that rule rejects these identities when enabled.
