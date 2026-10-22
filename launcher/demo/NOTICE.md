# PhaseEval demo data

The launcher unpacks `phaseeval-demo.tar.xz` into its state folder on first start.

- `store/` is a review store rebuilt from the six Qwen3.6-27B AgentXRD runs in the public export (three on DEV006, three on DEV102). It holds their attempts, answer snapshots and routing decisions. It holds no label, sign-off, judge manifest or reference answer.
- `dataset/` is the public-scope export of that store, which the scorecard reads. It holds the same six runs as `data/dataset-v0-public/`, but its snapshot digests differ because the store was rebuilt. It holds no expert label and no model-judge label.

The DEV006 scan comes from the DARA benchmark and the DEV102 scan and synthesis record from the Precursor Genome release, both CC BY 4.0. The candidate crystal structures come from the Crystallography Open Database (CC0). The repository's `NOTICE` gives the full attribution.

BGMN is not included. See "BGMN" in the repository's `README.md`.
