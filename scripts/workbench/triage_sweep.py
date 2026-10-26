"""Route every answered agent answer in a workbench store under the provisional triage policy.

R2c (AXA-43zwk.17.3). Runs triage.Triage(store, labels, PROVISIONAL_2026_10_21).sweep as the
store's admin owner over every root that owner may export. Each answered imported trial's frozen
agent_answer snapshot (frozen here only when none exists, KTD10) is routed and its decision
published once under routing/<snapshot>/<policy_version>.json. A rerun is a no-op: it publishes
nothing and prints the same counts. The decisions are operator data: the dashboard shows them to
the admin owner, and a rater sees one only in the revealed payload after signing that snapshot off.

    PYTHONPATH=src .venv/bin/python scripts/workbench/triage_sweep.py \\
        [--store ~/.local/state/axa-workbench/overnight] [--owner <the manifest's owner>] [--case DEV006 ...]

The owner defaults to the one in <store>.manifest.json (load_store.py), else "owner". --case limits
the sweep to those cases' roots in the manifest. This is the one sweep entry point:
.claude/skills/verify-axa/bin/triage-sweep.py calls `sweep` here.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench.labels import StageProjection, LabelStore
from phaseeval.workbench.seams import public_seams
from phaseeval.workbench.store import WorkbenchStore
from phaseeval.workbench.triage import PROVISIONAL_2026_10_21 as POLICY
from phaseeval.workbench.triage import ROUTES, Triage, TriageQueue

DEFAULT_STORE = Path.home() / ".local/state/axa-workbench/overnight"


def executes_nothing(*_: object) -> None:
    raise RuntimeError("the triage sweep runs no executor")


def recorded(store_root: Path) -> set[Path]:
    """The decision files under this policy version, to tell new decisions from recorded ones."""
    return set(store_root.glob(f"routing/*/{POLICY.version}.json"))


def sweep(
    store_root: Path, owner_id: str | None = None, cases: list[str] | None = None
) -> tuple[TriageQueue, int, str]:
    """Run the sweep: the expert queue, how many decisions this run published, and the owner."""
    store_root = store_root.expanduser().resolve()
    if not (store_root / "roots").is_dir():
        raise SystemExit(f"no workbench store at {store_root}")
    manifest_path = store_root.with_name(store_root.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else {}
    owner_id = owner_id or manifest.get("owner", "owner")
    owner = Principal(owner_id, PrincipalType.USER, frozenset({"owner"}))
    roots = None if cases is None else tuple(manifest["roots"][case] for case in cases)
    store = WorkbenchStore(
        store_root,
        executes_nothing,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=Path(manifest.get("runs", store_root)),
        exposure=StageProjection(store_root),
    )
    before = recorded(store_root)
    queue = Triage(store, LabelStore(store), POLICY).sweep(owner, roots)
    return queue, len(recorded(store_root) - before), owner_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument(
        "--owner", default=None, help="admin owner principal id (default: the manifest's)"
    )
    parser.add_argument("--case", action="append", default=None, help="sweep only this case's root")
    args = parser.parse_args(argv)
    store_root = args.store.expanduser().resolve()
    queue, new, owner_id = sweep(store_root, args.owner, args.case)
    n = sum(queue.counts.values())
    status = "provisional" if POLICY.provisional else "final"
    print(f"store {store_root}, owner {owner_id}, policy {POLICY.version} ({status})")
    print(f"routed {n} answered agent answers: {new} new decisions, {n - new} already recorded")
    for name in ROUTES:
        print(f"  {name:<15} {queue.counts[name]} / {n}")
    print(f"  a routing check not computable {queue.not_computable} / {n}")
    print(f"  flagged only by unmeasured checks {queue.unmeasured_flags} / {n}")


if __name__ == "__main__":
    main()
