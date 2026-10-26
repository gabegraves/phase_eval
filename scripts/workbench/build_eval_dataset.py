"""Build the eval dataset over the overnight store (W6 export; plan U6, N4; origin R11, R18, R19, R33).

    PYTHONPATH=src .venv/bin/python scripts/workbench/build_eval_dataset.py \\
        [--store ~/.local/state/axa-workbench/overnight] [--out ~/.local/state/axa-workbench/dataset-v0] \\
        [--scope private|public] [--custody <worktree>/data/custody/workbench]

The store and its manifest come from scripts/workbench/load_store.py. The output directory sits
outside the repo by default. It holds episodes.jsonl, excluded.jsonl, snapshots.jsonl,
labels.expert.jsonl, labels.model_judge.jsonl, peak_calls.jsonl, signoffs.jsonl,
step_feedback.jsonl, judge_inputs.jsonl, manifest.json and, in the private scope only,
judge_manifest.json, expectations.jsonl (Qualifier.expectations of every frozen qualification set,
with the custody reference reader on this host when --custody is given) and reports.jsonl.
The rater identity map and the judge manifest's frozen copy stay under the store root. The build
validates its own output and exits non-zero on any problem.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from qualify import CustodyReferences

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench import exports, policy, snapshot
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.labels import ExposureLedger, LabelStore
from phaseeval.workbench.model import AttemptDigest, RootAttempt
from phaseeval.workbench.qualification import Qualifier
from phaseeval.workbench.seams import NoReference, public_seams
from phaseeval.workbench.store import WorkbenchStore
from phaseeval.kernel.ledger import KeyShape, Mode

WORKTREE = Path(__file__).resolve().parents[2]
STATE = Path.home() / ".local/state/axa-workbench"
DATASET_VERSION = "agentxrd-workbench-eval-v0"


def canaries(custody: Path | None) -> list[str]:
    """Every string stored under a "canary" key in each case's custody record."""
    if custody is None:
        return []
    found: list[str] = []

    def walk(value: object, under_canary: bool) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, under_canary or key == "canary")
        elif isinstance(value, list):
            for item in value:
                walk(item, under_canary)
        elif under_canary and isinstance(value, str) and value:
            found.append(value)

    for record in sorted(custody.glob("*/custody.json")):
        walk(json.loads(record.read_text()), False)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=STATE / "overnight")
    parser.add_argument("--manifest", type=Path, default=None, help="default <store>.manifest.json")
    parser.add_argument("--out", type=Path, default=None, help="default ~/.local/state/axa-workbench/dataset-v0[-public]")
    parser.add_argument("--scope", choices=("private", "public"), default="private")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--custody", type=Path, default=None, help="scan the output for each case's custody canaries")
    parser.add_argument("--now", type=datetime.fromisoformat, default=None)
    parser.add_argument(
        "--freeze-judge-manifest",
        action="store_true",
        help="freeze the judge manifest in the store now (at judge launch); once frozen it is always enforced",
    )
    args = parser.parse_args()

    store_root = args.store.expanduser().resolve()
    manifest_path = args.manifest or store_root.with_name(store_root.name + ".manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    load_manifest = json.loads(manifest_bytes)
    out = (args.out or STATE / ("dataset-v0" if args.scope == "private" else "dataset-v0-public")).expanduser()
    if WORKTREE in out.resolve().parents:
        sys.exit("the dataset directory must sit outside the repo")
    now = args.now or datetime.now(UTC)

    store = WorkbenchStore(
        store_root,
        default_registry().run,
        public_seams(),
        lambda: now,
        import_root=Path(load_manifest["runs"]),
        exposure=ExposureLedger(store_root),
    )
    owner = Principal(load_manifest["owner"], PrincipalType.USER, frozenset({"owner"}))
    roots = tuple(AttemptDigest(d) for d in sorted(load_manifest["roots"].values()))
    cases = []
    for root in roots:
        attempt = store.walk(root)[0][1]
        assert isinstance(attempt, RootAttempt)
        cases.append(attempt.case)
    inputs = exports.ExportInputs(
        dataset_version=args.dataset_version,
        eligibility=policy.ELIGIBILITY_LEDGER_V0,
        rights=policy.rights_registry_v0(tuple(cases)),
        now=now,
    )
    manifests = store.ledger.collection('judge-manifests', exports.JudgeManifest, Mode.ONCE, KeyShape(('safe_id',)), owner=__name__)
    judge_manifest = manifests.get((args.dataset_version,))
    if args.freeze_judge_manifest and (judge_manifest is not None or args.scope != "private"):
        sys.exit("the judge manifest freezes once, from a private build")
    pseudonyms = exports.RaterPseudonyms(store_root / "private", ledger=store.ledger)
    labels = LabelStore(store)
    reference = CustodyReferences(args.custody) if args.custody else NoReference()
    dataset = exports.build_dataset(
        store,
        labels,
        owner,
        roots,
        inputs,
        "internal_evaluation" if args.scope == "private" else "public_release",
        pseudonyms,
        build_snapshot=snapshot.build_snapshot,
        review_payload=exports.blind_view,
        scope=args.scope,
        judge_manifest=judge_manifest,
        freeze_judge_manifest=args.freeze_judge_manifest,
        store_manifest=load_manifest,
        store_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        qualifier=Qualifier(store, labels, reference),
    )
    if args.freeze_judge_manifest:
        manifests.put((args.dataset_version,), dataset.judge_manifest)
    written = exports.write_dataset(dataset, out, canaries=canaries(args.custody))
    pseudonyms.write_map()
    os.chmod(out, 0o700)

    print(f"dataset {args.dataset_version} ({args.scope}) -> {out}")
    for count in written.counts:
        print(f"  {count.name}: {count.count} of {count.denominator} {count.denominator_name}")
    print(f"  exclusions: {len(written.exclusions)}")
    state = "frozen" if dataset.judge_manifest.frozen else "provisional (freeze at judge launch)"
    print(f"  judge manifest: {state}, {sum(e.judged for e in dataset.judge_manifest.entries)} judged entries")
    sys.exit(0)


if __name__ == "__main__":
    main()
