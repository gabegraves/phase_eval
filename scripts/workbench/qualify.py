"""Qualify producers and model judges against a lab's own judgments, on the owner's host.

U16b (AXA-43zwk.26). One CLI over qualification.Qualifier for every subject:

    PYTHONPATH=src .venv/bin/python scripts/workbench/qualify.py \\
        [--store ~/.local/state/axa-workbench/overnight] [--owner owner] [--custody DIR] COMMAND

    freeze --lab lab-a --name demo-2026-10-22 [--case DEV006 --case DEV102] --rater-kind owner \\
        --policy-version V --justified-by TEXT [--min-raters 1] [--identity candidate_id] \\
        [--require-blind] [--amount-tolerance WT_PCT] [--split-seed SEED]
    expectations --set DIGEST
    run --set DIGEST --producer MODEL:HARNESS[:VERSION]
    run --set DIGEST --judge MODEL:PROMPT_SHA256 [--tuned-on SET_DIGEST:tune|held_out ...]
    plan --set DIGEST --producer MODEL:HARNESS[:VERSION]

Prints JSON. Every number is agreement with this lab's judgments, never accuracy.

CustodyReferences is the only reference path (seams.ReferenceReader). It reads
<custody>/<case>/custody.json reference_phase_coverage, the layout snapshot._reference reads,
and lives in this owner-host script so src/axa never imports it (design L9 moves it into the
private seams build). Without --custody or $AXA_WORKBENCH_CUSTODY every stance reads "absent".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from phaseeval.api.auth import Principal, PrincipalType
from phaseeval.workbench.executors import default_registry
from phaseeval.workbench.labels import ExposureLedger, LabelStore
from phaseeval.workbench.model import ChildAttempt, Completed, ImportedRunResult
from phaseeval.workbench.qualification import (
    JudgeVersion,
    NoFrozenSplit,
    Producer,
    QualificationPolicy,
    Qualifier,
    TunedOn,
    TunedOnHeldOut,
)
from phaseeval.workbench.seams import SealedReference, SealedReferencePhase, public_seams
from phaseeval.workbench.store import WorkbenchStore

DEFAULT_STORE = Path.home() / ".local/state/axa-workbench/overnight"


class CustodyReferences:
    """seams.ReferenceReader over custody.json reference_phase_coverage."""

    def __init__(self, custody: Path | None) -> None:
        self._custody = custody

    def reference(self, case_id: str) -> SealedReference | None:
        path = self._custody / case_id / "custody.json" if self._custody else None
        if path is None or not path.is_file():
            return None
        record = json.loads(path.read_bytes())
        return SealedReference(
            case_id=case_id,
            source="custody reference_phase_coverage",
            phases=tuple(
                SealedReferencePhase(
                    formula=p["formula"],
                    space_group=p.get("space_group"),
                    weight_percent=p.get("weight_percent"),
                    candidates=tuple(p.get("shipped_candidates", [])),
                )
                for p in record.get("reference_phase_coverage", [])
            ),
        )


def recorded_agents(store: WorkbenchStore, owner: Principal) -> set[tuple[str, str, str]]:
    """(model_id, harness, harness_version) of every imported run under the owner's roots."""
    found = set()
    for root in store.roots(owner):
        for _, attempt in store.walk(root):
            if isinstance(attempt, ChildAttempt) and isinstance(attempt.outcome, Completed):
                result = attempt.outcome.result
                if isinstance(result, ImportedRunResult):
                    found.add((result.agent.model_id, result.agent.harness, result.agent.harness_version))
    return found


def producer(spec: str, agents: set[tuple[str, str, str]]) -> Producer:
    """MODEL:HARNESS[:VERSION]. Without a version, the one version the store holds for that
    model and harness; several or none is an error naming what exists."""
    model_id, _, rest = spec.partition(":")
    harness, _, version = rest.partition(":")
    if not version:
        versions = sorted(v for m, h, v in agents if (m, h) == (model_id, harness))
        if len(versions) != 1:
            known = sorted(f"{m}:{h}:{v}" for m, h, v in agents)
            raise SystemExit(f"--producer {spec}: versions {versions}; recorded agents: {known}")
        version = versions[0]
    return Producer(model_id=model_id, harness=harness, harness_version=version)


def judge(spec: str, tuned: list[str]) -> JudgeVersion:
    """MODEL:PROMPT_SHA256 plus every SET_DIGEST:tune|held_out the prompt was revised against."""
    model, _, prompt = spec.rpartition(":")
    try:
        tuned_on = tuple(
            TunedOn(set_digest=digest, partition=partition)
            for digest, _, partition in (t.partition(":") for t in tuned)
        )
        return JudgeVersion(model=model, prompt_sha256=prompt, tuned_on=tuned_on)
    except ValidationError as error:
        raise SystemExit(f"--judge MODEL:PROMPT_SHA256 [--tuned-on SET_DIGEST:tune|held_out]: {error}") from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--owner", default="owner", help="the lab owner principal id")
    parser.add_argument("--custody", type=Path, default=os.environ.get("AXA_WORKBENCH_CUSTODY"))
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--lab", required=True)
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--case", action="append", help="case id; every owned root when omitted")
    freeze.add_argument("--rater-kind", action="append", required=True, choices=("expert", "owner"))
    freeze.add_argument("--policy-version", required=True)
    freeze.add_argument("--justified-by", required=True)
    freeze.add_argument("--min-raters", type=int, default=1)
    freeze.add_argument("--identity", default="formula_and_space_group", choices=("candidate_id", "formula_and_space_group"))
    freeze.add_argument("--require-blind", action="store_true")
    freeze.add_argument("--amount-tolerance", type=float, default=None, help="wt%% points; omitted: amounts never graded")
    freeze.add_argument("--split-seed", default=None)
    for name in ("expectations", "run", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--set", required=True, dest="set_digest")
        if name != "expectations":
            command.add_argument("--producer")
        if name == "run":
            command.add_argument("--judge", help="MODEL:PROMPT_SHA256")
            command.add_argument("--tuned-on", action="append", default=[], help="SET_DIGEST:tune|held_out")
    args = parser.parse_args(argv)

    owner = Principal(args.owner, PrincipalType.USER, frozenset({"owner"}))
    root = args.store.resolve()
    # Qualification never imports a run, so import paths resolve nowhere outside the store.
    store = WorkbenchStore(
        root,
        default_registry().run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=root,
        exposure=ExposureLedger(root),
    )
    qualifier = Qualifier(store, LabelStore(store), CustodyReferences(args.custody))

    if args.command == "freeze":
        roots = tuple(
            root
            for root in store.roots(owner)
            if args.case is None or store.walk(root)[0][1].case.case_id in args.case
        )
        policy = QualificationPolicy(
            version=args.policy_version,
            justified_by=args.justified_by,
            rater_kinds=tuple(args.rater_kind),
            require_blind=args.require_blind,
            min_raters=args.min_raters,
            phase_identity=args.identity,
            amount_abs_wt_pct=args.amount_tolerance,
            expectation_rule="lab-expectation-v2",
        )
        digest = qualifier.freeze(owner, lab=args.lab, name=args.name, roots=roots, policy=policy, split_seed=args.split_seed)
        out: object = {"set": digest, "roots": len(roots)}
    elif args.command == "expectations":
        out = {root: e.model_dump(mode="json") for root, e in qualifier.expectations(owner, args.set_digest).items()}
    elif args.command == "plan":
        out = {"roots": qualifier.plan(owner, args.set_digest, producer(args.producer, recorded_agents(store, owner)))}
    else:
        if (args.producer is None) == (args.judge is None):
            raise SystemExit("run takes exactly one of --producer and --judge")
        subject = producer(args.producer, recorded_agents(store, owner)) if args.producer else judge(args.judge, args.tuned_on)
        try:
            out = qualifier.run(owner, args.set_digest, subject).model_dump(mode="json")
        except (NoFrozenSplit, TunedOnHeldOut) as refused:
            raise SystemExit(f"{type(refused).__name__}: {refused}") from None
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
