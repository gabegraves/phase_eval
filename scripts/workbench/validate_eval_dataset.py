"""Validate an eval dataset directory (W6 export; plan U6, N4).

    PYTHONPATH=src .venv/bin/python scripts/workbench/validate_eval_dataset.py <dataset dir> \\
        [--custody <worktree>/data/custody/workbench]

Checks that every line validates against its schema under schemas/, no custody field appears
anywhere, no run-identity key (reference_match, job, model, harness, series, episode_id) appears
in snapshots or judge inputs, label files never mix rater kinds, every label, peak call and
sign-off names a snapshot in snapshots.jsonl, rater ids are pseudonyms, the manifest's counts,
hashes and judged flags match the files, and (with --custody) no custody canary appears.
"""

import argparse
import json
import sys
from pathlib import Path

from build_eval_dataset import canaries

from phaseeval.workbench.exports import validate_dataset

WORKTREE = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--custody", type=Path, default=None)
    args = parser.parse_args()
    schemas = {p.name: json.loads(p.read_text()) for p in sorted((WORKTREE / "schemas").glob("*.json"))}
    problems = validate_dataset(args.dataset.expanduser(), schemas, canaries(args.custody))
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problems in {args.dataset}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
