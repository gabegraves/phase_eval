"""Prepare and import owner-side assisted method reviews without invoking a model provider."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench.assisted_review import AssistedReviews
from phaseeval.workbench.labels import ExposureLedger
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import WorkbenchStore


def _nothing(*_: object) -> None:
    raise RuntimeError("assisted review never executes workbench actions")


def _reviews(store_root: Path, out: Path, owner_id: str | None = None) -> AssistedReviews:
    store_root = store_root.expanduser().resolve()
    manifest_path = store_root.with_name(store_root.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else {}
    owner = Principal(owner_id or manifest.get("owner", "owner"), PrincipalType.USER, frozenset({"owner"}))
    store = WorkbenchStore(store_root, _nothing, public_seams(), lambda: datetime.now(UTC), import_root=Path(manifest.get("runs", store_root)), exposure=ExposureLedger(store_root))
    return AssistedReviews(store, out, owner)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--store", type=Path, required=True)
    prepare.add_argument("--snapshot", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--out", type=Path, required=True)
    submit.add_argument("--packet", required=True)
    submit.add_argument("--response", type=Path, required=True)
    submit.add_argument("--model", required=True)
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--context", choices=("fresh_packet", "inherited_context", "unknown"), default="unknown")
    status = sub.add_parser("status")
    status.add_argument("--out", type=Path, required=True)
    status.add_argument("--packet", required=True)
    report = sub.add_parser("report")
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = _reviews(args.store, args.out, args.owner).prepare(args.snapshot)
    else:
        helper = AssistedReviews(None, args.out)
        if args.command == "submit":
            result = helper.submit(args.packet, args.response.read_bytes(), model=args.model, run_id=args.run_id, context=args.context)
        else:
            result = helper.status(args.packet) if args.command == "status" else helper.report()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
