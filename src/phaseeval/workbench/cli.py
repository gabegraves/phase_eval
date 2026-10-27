"""Bring a lab's own run into its local PhaseEval store, and export the store as a dataset.

    python -m phaseeval.workbench.cli import --case-dir D --run-dir R [--case ID] [--store S]
    python -m phaseeval.workbench.cli import-gpx --case-dir D --project P.gpx [--case ID] [--store S]
    python -m phaseeval.workbench.cli import-bgmn --case-dir D --run-dir R [--control F.sav] [--case ID] [--store S]
    python -m phaseeval.workbench.cli export --out DIR [--store S]

import-gpx and import-bgmn bring a refinement finished in GSAS-II, BGMN or Profex into the same
review queue as an agent run (importers.external). A .gpx is a Python pickle: it is read only
inside a bubblewrap sandbox, so import-gpx works on Linux with bwrap installed on a merged-/usr
layout (/bin and /lib as links into /usr: Debian, Ubuntu, Fedora, Arch) and refuses elsewhere.
Set PHASEEVAL_GSASII_SOURCE to the GSAS-II source directory the sandbox may read. What a hostile
project can cost is bounded as importers.gsasii documents, with one residual: memory held in
shared memory (memfd, System V) and the number of processes inside are not limited before the
read's deadline, so such a project can take the host's free memory and PIDs for up to 300 s,
after which the sandbox is killed whole. This is a local, single-user command.

With no --store, both commands use the store the launcher serves: launcher/phaseeval.toml (or
$PHASEEVAL_CONFIG), and the demo store under the state dir when the config says "demo".
Neither command does any network I/O.

A lab's own case has no labels and no reference answer. When the case dir has no
case_labels.json or custody.json beside it, import marks it as a user case
(split unassigned, no reference, no labels) and says so. It never invents an answer.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from phaseeval.api.auth import CaseOperation, Principal, PrincipalType

from . import exports, importer, policy
from .executors import RefitAdmission, default_registry
from .labels import StageProjection, LabelStore
from .model import AttemptDigest, Record, RootAttempt
from .qualification import Qualifier
from .seams import NoReference, public_seams
from .store import ActionRefused, WorkbenchStore

OWNER = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
IMPORTER = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
# The launcher's label-mode principal (launcher.RATER), the one /workbench signs in as.
REVIEWERS = ("owner-rater",)
DATASET_VERSION = "phaseeval-local-v0"


class Settings(Record):
    store: Path | None = None
    principals: Path | None = None
    dataset: Path | None = Path.home() / ".local/state/axa-workbench/dataset-v0"
    import_root: Path | None = None
    bgmn: Path | None = Path.home() / ".axa" / "bgmn"
    custody: Path | None = None
    refit_slots: int = Field(default=2, ge=1, le=16)
    refits_per_principal: int = Field(default=1, ge=1)
    refit_max_seconds: int = Field(default=1800, ge=60)
    max_body_bytes: int = 1024 * 1024
    bind_host: Literal["127.0.0.1", "::1"] = "127.0.0.1"

    @field_validator("store", "principals", "dataset", "import_root", "bgmn", "custody")
    @classmethod
    def absolute_paths(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("settings paths must be absolute")
        return value

    @model_validator(mode="after")
    def paired_store(self):
        if (self.store is None) != (self.principals is None):
            raise ValueError("store and principals must be configured together")
        return self

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        names = {
            "store": ("STORE", "WORKBENCH_ROOT"),
            "principals": ("PRINCIPALS", "API_PRINCIPALS_FILE"),
            "dataset": ("DATASET", "WORKBENCH_DATASET"),
            "import_root": ("IMPORT_ROOT", "WORKBENCH_IMPORT_ROOT"),
            "bgmn": ("BGMN_DIR", "BGMN_DIR"),
            "custody": ("CUSTODY", "WORKBENCH_CUSTODY"),
            "bind_host": ("BIND_HOST", "BIND_HOST"),
        }
        values = {}
        for field, (current, old) in names.items():
            current, old = f"PHASEEVAL_{current}", f"AXA_{old}"
            if current in env:
                value = env[current]
            elif old in env:
                value = env[old]
                warnings.warn(f"{old} is deprecated; use {current}", FutureWarning, stacklevel=2)
            else:
                continue
            values[field] = value or None
        return cls(**values)


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


def open_store(root: Path, import_root: Path, *, settings: Settings | None = None) -> WorkbenchStore:
    from phaseeval.refinement.bgmn.installer import BGMNInstaller

    settings = settings or Settings.from_env(os.environ)
    installer = BGMNInstaller(install_dir=settings.bgmn or BGMNInstaller.INSTALL_DIR)
    root = root.expanduser().resolve()
    return WorkbenchStore(
        root,
        default_registry(installer=installer, enabled=settings.bgmn is not None).run,
        public_seams(),
        lambda: datetime.now(UTC),
        import_root=import_root,
        exposure=StageProjection(root),
        admission=RefitAdmission(settings.refit_slots, settings.refits_per_principal),
    )


def _queued(store: WorkbenchStore, root: AttemptDigest, child: AttemptDigest, what: str, case_id: str) -> int:
    snap = queue_for_review(store, root, child)
    print(f"imported {what} for case {case_id}")
    print(f"root {root}")
    print(f"attempt {child}")
    print(f"snapshot {snap} queued for {', '.join(REVIEWERS)}")
    print(f"store {store.root}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    try:
        inputs = importer.read_case(args.case_dir, case_id=args.case)
        case, blobs = importer.case_manifest(inputs)
        action, result, run_blobs = importer.read_harbor_run(run_dir, case)
        store = open_store(args.store or launcher_store(), run_dir.parent)
        root = store.load_case(OWNER, case, blobs)
        child = store.import_run(IMPORTER, root, action, result, run_blobs)
    except FileNotFoundError as error:
        raise CliError(f"a file the import needs is missing: {error.filename}") from error
    except (importer.PackageError, importer.BindingError, ActionRefused, ValueError) as error:
        raise CliError(f"refused: {error}") from error
    if inputs.labels is None:
        print(f"{case.case_id}: no case_labels.json or custody.json, so it is imported as a user case.")
        print("It has no reference answer and no labels. Nothing is scored against a reference.")
    return _queued(store, root, child, f"run {action.harbor.trial_id}", case.case_id)


def _phase_map(pairs: list[str] | None) -> dict[str, str]:
    """--phase NAME=CANDIDATE_ID, repeatable: a refined phase bound to a case candidate by hand."""
    mapping: dict[str, str] = {}
    for pair in pairs or ():
        name, sep, candidate = pair.partition("=")
        if not sep or not name or not candidate:
            raise CliError(f"--phase takes NAME=CANDIDATE_ID, not {pair!r}")
        mapping[name] = candidate
    return mapping


def _import_external(args: argparse.Namespace, import_root: Path, read: Any) -> int:
    """Load the case as `cmd_import` does, then `read(case, pattern_bytes)` -> (action, result,
    blobs) from importers.external, written and queued the same way."""
    from .importers import bgmn, gsasii

    try:
        inputs = importer.read_case(args.case_dir, case_id=args.case)
        case, blobs = importer.case_manifest(inputs)
        action, result, run_blobs = read(case, blobs[case.pattern.sha256])
        store = open_store(args.store or launcher_store(), import_root)
        root = store.load_case(OWNER, case, blobs)
        child = store.import_run(IMPORTER, root, action, result, run_blobs)
    except FileNotFoundError as error:
        raise CliError(f"a file the import needs is missing: {error.filename}") from error
    except gsasii.SandboxUnavailable as error:
        raise CliError(f"refused: a .gpx is opened only inside bubblewrap: {error}") from error
    except (
        importer.PackageError,
        importer.BindingError,
        bgmn.BundleError,
        gsasii.GpxImportError,
        ActionRefused,
        ValueError,
    ) as error:
        raise CliError(f"refused: {error}") from error
    from .importers import external

    described = external.describe(result)
    print(f"source {described['source']} ({described['software']}); stage {described['stage']}; Rwp {described['rwp_percent']}")
    for candidate, weight in described["phases"].items():
        print(f"  {candidate}: {'no weight fraction' if weight is None else f'{weight:.2f} wt%'}")
    for gap in described["missing"]:
        print(f"  not recorded: {gap}")
    if inputs.labels is None:
        print(f"{case.case_id}: no case_labels.json or custody.json, so it is imported as a user case.")
        print("It has no reference answer and no labels. Nothing is scored against a reference.")
    return _queued(store, root, child, f"{described['source']} {action.harbor.trial_id}", case.case_id)


def cmd_import_gpx(args: argparse.Namespace) -> int:
    from .importers import external, gsasii

    project = args.project.resolve()
    phase_map = _phase_map(args.phase)

    def read(case: Any, pattern: bytes) -> external.Imported:
        analysis = gsasii.read_gpx(project)  # inside bubblewrap; never unpickled here
        return external.from_gpx(analysis, project.read_bytes(), case, pattern, phase_map)

    return _import_external(args, project.parent, read)


def cmd_import_bgmn(args: argparse.Namespace) -> int:
    from .importers import bgmn, external

    run_dir = args.run_dir.resolve()
    phase_map = _phase_map(args.phase)

    def read(case: Any, pattern: bytes) -> external.Imported:
        analysis, members = bgmn.read_bundle(run_dir, control=args.control)
        return external.from_bgmn(analysis, members, case, pattern, phase_map)

    return _import_external(args, run_dir.parent, read)


def queue_for_review(store: WorkbenchStore, root: AttemptDigest, child: AttemptDigest) -> str:
    """Grant each reviewer the root as the demo store does, freeze the run's agent answer and
    index it in each reviewer's items. Idempotent: one snapshot, listed once."""
    for reviewer in REVIEWERS:
        store.grant(OWNER, root, reviewer, (CaseOperation.INSPECT, CaseOperation.ACTION))
    labels = LabelStore(store)
    digest = store.snapshots.freeze(OWNER, child, "agent_answer")
    with store.ledger.transaction():
        if not any(digest in c.snapshots for c in labels._projection._facts_locked(root).cohorts if c.root == root):
            latest = labels.latest_assignment(root)
            raters = tuple(dict.fromkeys((*(latest.raters if latest else ()), *REVIEWERS)))
            labels._assign_locked(OWNER, root, raters, (digest,))
    return digest


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
        exports.RaterPseudonyms(store_root / "private", ledger=store.ledger),
        review_payload=exports.blind_view,
        qualifier=Qualifier(store, labels, NoReference()),
    )
    try:
        written = exports.write_dataset(dataset, args.out)
    except (ValueError, FileExistsError) as exc:
        raise CliError(str(exc)) from exc
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
    gpx = commands.add_parser("import-gpx", help="import a finished GSAS-II project (.gpx) under its case")
    gpx.add_argument("--case-dir", type=Path, required=True)
    gpx.add_argument("--project", type=Path, required=True, help="the .gpx; read only inside bubblewrap")
    bgm = commands.add_parser("import-bgmn", help="import saved BGMN or Profex outputs (a directory or a Profex ZIP) under its case")
    bgm.add_argument("--case-dir", type=Path, required=True)
    bgm.add_argument("--run-dir", type=Path, required=True, help="the directory or Profex project ZIP holding the .sav and its outputs")
    bgm.add_argument("--control", default=None, help="the .sav to read when the bundle holds several")
    for sub in (gpx, bgm):
        sub.add_argument("--case", default=None)
        sub.add_argument("--store", type=Path, default=None)
        sub.add_argument("--phase", action="append", default=None, metavar="NAME=CANDIDATE_ID",
                         help="bind a refined phase to a case candidate; repeatable")
    exp = commands.add_parser("export", help="export the store as a dataset directory")
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--store", type=Path, default=None)
    args = parser.parse_args(argv)
    commands_by_name = {
        "import": cmd_import,
        "import-gpx": cmd_import_gpx,
        "import-bgmn": cmd_import_bgmn,
        "export": cmd_export,
    }
    return commands_by_name[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
