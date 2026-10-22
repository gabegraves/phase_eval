"""Bring a lab's own run into its local PhaseEval store, and export the store as a dataset.

    python -m phaseeval.workbench.cli import --case-dir D --run-dir R [--case ID] [--store S]
    python -m phaseeval.workbench.cli export --out DIR [--store S]

With no --store, both commands use the store the launcher serves: launcher/phaseeval.toml (or
$PHASEEVAL_CONFIG), and the demo store under the state dir when the config says "demo".
Neither command does any network I/O.

A lab's own case has no labels and no reference answer. When the case dir has no
case_labels.json or custody.json beside it, import writes the minimal records that mark it as
a user case (split unassigned, no reference, no labels) and says so. It never invents an answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phaseeval.api.auth import Principal, PrincipalType

from . import exports, importer, policy, snapshot
from .executors import default_registry
from .labels import ExposureLedger, LabelStore
from .model import AttemptDigest, RootAttempt
from .qualification import Qualifier
from .seams import NoReference, public_seams
from .store import ActionRefused, WorkbenchStore

OWNER = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
IMPORTER = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
DATASET_VERSION = "phaseeval-local-v0"
USER_EXPOSURE = "unexposed"
USER_TASK_KIND = "identification"
# What an export must never carry: local paths, hostnames or IPv4 addresses.
LEAKS = re.compile(rb"/Users/|/home/|/private/|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


class CliError(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"error: {message}")


def launcher_store() -> Path:
    from phaseeval import launcher

    config_path = Path(os.environ.get("PHASEEVAL_CONFIG", launcher.REPO / "launcher" / "phaseeval.toml"))
    config = launcher.LaunchConfig.load(config_path)
    if config.store == "demo":
        config.state_dir.mkdir(parents=True, exist_ok=True)
        return launcher.demo_paths(config.state_dir)[0]
    return config.store


def open_store(root: Path, import_root: Path) -> WorkbenchStore:
    root = root.expanduser().resolve()
    return WorkbenchStore(
        root,
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=import_root,
        exposure=ExposureLedger(root),
    )


def user_case_records(case_dir: Path, case_id: str, out_dir: Path) -> tuple[Path, Path]:
    """case_labels.json and custody.json for a lab's own case: unassigned split, no reference
    answer, no labels. Only the fields the importer reads are written."""
    labels = out_dir / "case_labels.json"
    custody = out_dir / "custody.json"
    labels.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": case_id,
                        "source": {"kind": importer.USER_CASE_COLLECTION, "citation": f"User case {case_id}."},
                        "exposure": USER_EXPOSURE,
                        "split": "unassigned",
                        "chemical_system": None,
                        "confirmation_tag": USER_TASK_KIND,
                    }
                ]
            },
            indent=2,
        )
        + "\n"
    )
    custody.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "user_case": True,
                "reference_answer": None,
                "split_check": {
                    "assigned_split": "unassigned",
                    "verdict": "unassigned",
                    "chemical_system": None,
                    "split_manifest_content_sha256": None,
                },
                "labels": {"confirmation_tag": USER_TASK_KIND, "exposure": USER_EXPOSURE},
            },
            indent=2,
        )
        + "\n"
    )
    return labels, custody


def _case_id(case_dir: Path, wanted: str | None) -> str:
    try:
        (entry,) = json.loads((case_dir / "manifest.json").read_bytes())["cases"]
    except (OSError, ValueError, KeyError) as error:
        raise CliError(f"{case_dir} has no readable manifest.json with one case: {error}") from error
    if wanted is not None and entry["case_id"] != wanted:
        raise CliError(f"{case_dir} holds case {entry['case_id']}, not {wanted}")
    return entry["case_id"]


def cmd_import(args: argparse.Namespace) -> int:
    case_dir, run_dir = args.case_dir.resolve(), args.run_dir.resolve()
    case_id = _case_id(case_dir, args.case)
    store_root = args.store or launcher_store()
    labels, custody = case_dir / "case_labels.json", case_dir / "custody.json"
    if not labels.is_file() or not custody.is_file():
        records = store_root.expanduser().resolve().parent / "user-cases" / case_id
        records.mkdir(parents=True, exist_ok=True)
        labels, custody = user_case_records(case_dir, case_id, records)
        print(f"{case_id}: no case_labels.json or custody.json, so it is imported as a user case.")
        print("It has no reference answer and no labels. Nothing is scored against a reference.")
    store = open_store(store_root, run_dir.parent)
    try:
        case, blobs = importer.read_case_package(case_dir, labels, custody=custody)
        root = store.load_case(OWNER, case, blobs)
        action, result, run_blobs = importer.read_harbor_run(run_dir, case)
        child = store.import_run(IMPORTER, root, action, result, run_blobs)
    except FileNotFoundError as error:
        raise CliError(f"a file the import needs is missing: {error.filename}") from error
    except (importer.PackageError, importer.BindingError, ActionRefused, ValueError) as error:
        raise CliError(f"refused: {error}") from error
    print(f"imported run {action.harbor.trial_id} for case {case_id}")
    print(f"root {root}")
    print(f"attempt {child}")
    print(f"store {store.root}")
    return 0


def _leaks(out: Path) -> list[str]:
    return [
        f"{path.name}: {match.group().decode()}"
        for path in sorted(out.rglob("*"))
        if path.is_file()
        for match in [LEAKS.search(path.read_bytes())]
        if match
    ]


def cmd_export(args: argparse.Namespace) -> int:
    store_root = (args.store or launcher_store()).expanduser().resolve()
    store = open_store(store_root, store_root)
    roots = store.roots(OWNER)
    cases = tuple(a.case for r in roots for _, a in store.walk(r)[:1] if isinstance(a, RootAttempt))
    now = datetime.now(UTC)
    inputs = exports.ExportInputs(
        dataset_version=DATASET_VERSION,
        eligibility=policy.ELIGIBILITY_LEDGER_V0,
        rights=policy.rights_registry_v0(cases),
        now=now,
    )
    labels = LabelStore(store)
    dataset = exports.build_dataset(
        store,
        labels,
        OWNER,
        roots,
        inputs,
        "internal_evaluation",
        exports.RaterPseudonyms(store_root / "private"),
        build_snapshot=snapshot.build_snapshot,
        review_payload=exports.blind_view,
        qualifier=Qualifier(store, labels, NoReference()),
    )
    written = exports.write_dataset(dataset, args.out)
    if leaks := _leaks(args.out):
        raise CliError(f"the export at {args.out} carries local paths or addresses: {leaks}")
    print(f"exported {len(roots)} cases to {args.out}")
    for count in written.counts:
        print(f"  {count.name}: {count.count} of {count.denominator} {count.denominator_name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import", help="import one run folder under its case")
    imp.add_argument("--case-dir", type=Path, required=True)
    imp.add_argument("--run-dir", type=Path, required=True)
    imp.add_argument("--case", default=None)
    imp.add_argument("--store", type=Path, default=None)
    exp = commands.add_parser("export", help="export the store as a dataset directory")
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--store", type=Path, default=None)
    args = parser.parse_args(argv)
    return cmd_import(args) if args.command == "import" else cmd_export(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
