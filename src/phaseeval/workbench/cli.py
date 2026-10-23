"""Bring a lab's own run into its local PhaseEval store, and export the store as a dataset.

    python -m phaseeval.workbench.cli import --case-dir D --run-dir R [--case ID] [--store S]
    python -m phaseeval.workbench.cli import-gpx --case-dir D --project P.gpx [--case ID] [--store S]
    python -m phaseeval.workbench.cli import-bgmn --case-dir D --run-dir R [--control F.sav] [--case ID] [--store S]
    python -m phaseeval.workbench.cli export --out DIR [--store S]

import-gpx and import-bgmn bring a refinement finished in GSAS-II, BGMN or Profex into the same
review queue as an agent run (importers.external). A .gpx is a Python pickle: it is read only
inside a bubblewrap sandbox, so import-gpx works on Linux with bwrap installed and refuses
elsewhere. Set PHASEEVAL_GSASII_SOURCE to the GSAS-II source directory the sandbox may read.

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
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phaseeval.api.auth import CaseOperation, Principal, PrincipalType

from . import exports, importer, policy, snapshot
from .executors import default_registry
from .labels import ExposureLedger, LabelStore
from .model import AttemptDigest, RootAttempt
from .qualification import Qualifier
from .seams import NoReference, public_seams
from .store import ActionRefused, WorkbenchStore, _atomic_write_bytes

OWNER = Principal("owner", PrincipalType.USER, frozenset({"owner"}))
IMPORTER = Principal("importer", PrincipalType.SERVICE, frozenset({"importer"}))
# The launcher's label-mode principal (launcher.RATER), the one /workbench signs in as.
REVIEWERS = ("owner-rater",)
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


# The harness tools' defaults when a harness case omits them (harness/physics/screen.py and
# refine.py): a Cu K-alpha doublet and the default BGMN profile, which restricts quantification.
HARNESS_KALPHA = (1.540598, 1.544426)
HARNESS_DEFAULT_PROFILE = "Aeris-fds-Pixcel1d-Medipix3"
HARNESS_TASK = Path(__file__).resolve().parents[3] / "harness" / "agent" / "task.md"
# A user's candidate is never adjudicated: not reviewed, not eligible for anything.
USER_ELIGIBILITY = "not_adjudicated"
USER_DECLARED = "packaging"


def _elements(formula_sum: str) -> list[str]:
    from pymatgen.core import Composition

    return sorted(str(element) for element in Composition(formula_sum).elements)


def _source_paths(value: Any, path: str) -> list[str]:
    """Every field path read_case_package asks a field_sources row for."""
    if isinstance(value, dict):
        return [path] + [p for k, v in value.items() for p in _source_paths(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [path] + [p for i, v in enumerate(value) for p in _source_paths(v, f"{path}.{i}")]
    return [path]


def harness_case_package(case_dir: Path, case_id: str, out_dir: Path) -> Path:
    """A harness-layout case folder (harness/README.md) as the package read_case_package reads,
    written under `out_dir`. The case folder is not changed. Pattern and CIF bytes are copied
    unchanged, so the importer's hash checks still run against what the agent read. Candidates
    are user-supplied and not adjudicated. Every field is user-declared, and a field the case
    omits takes the harness tools' default, which the profile marks as a default fallback."""
    manifest = json.loads((case_dir / "manifest.json").read_bytes())
    (entry,) = manifest["cases"]
    if "field_sources" in entry:
        return case_dir
    candidates_name = f"candidates/{case_id}/manifest.json"
    library = json.loads((case_dir / candidates_name).read_bytes())
    if "chemistry_elements" not in entry:
        raise CliError(f"{case_id} declares no chemistry_elements")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "candidates" / case_id).mkdir(parents=True)
    shutil.copyfile(case_dir / entry["pattern_file"], out_dir / entry["pattern_file"])
    rows = []
    for record in library["candidates"]:
        name = f"candidates/{case_id}/{record['file']}"
        shutil.copyfile(case_dir / name, out_dir / name)
        rows.append(
            {
                **record,
                "elements": record.get("elements") or _elements(record["formula_sum"]),
                "source_library_id": record.get("source_library_id", f"user {record['candidate_id']}"),
                "eligibility_status": USER_ELIGIBILITY,
            }
        )
    (out_dir / candidates_name).write_text(json.dumps({**library, "candidates": rows}, indent=2) + "\n")

    measurement = {"instrument": None, "detector": None, **entry.get("measurement", {})}
    measurement.setdefault("wavelength_angstrom", HARNESS_KALPHA[0])
    measurement.setdefault("kalpha2_wavelength_angstrom", HARNESS_KALPHA[1])
    profile = entry.get("instrument_profile") or {
        "profile": HARNESS_DEFAULT_PROFILE,
        "resolver_confidence": "fallback",
        "warnings": ["instrument_profile_default_fallback"],
    }
    material = {"precursors": [], "processing": [], **entry.get("material_context", {})}
    normalized = {
        **entry,
        "measurement": measurement,
        "instrument_profile": profile,
        "material_context": material,
    }
    paths = [
        "chemistry_elements",
        "measurement.instrument",
        "measurement.wavelength_angstrom",
        "instrument_profile.profile",
        *_source_paths(material, "material_context"),
    ]
    normalized["field_sources"] = {
        path: {"class": USER_DECLARED, "locator": f"user-declared manifest.json {path}"}
        for path in paths
    }
    task = manifest.get("task") or HARNESS_TASK.read_text().replace("{case_id}", case_id)
    out = {**manifest, "task": task, "cases": [normalized]}
    (out_dir / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    return out_dir


def _case_id(case_dir: Path, wanted: str | None) -> str:
    try:
        (entry,) = json.loads((case_dir / "manifest.json").read_bytes())["cases"]
    except (OSError, ValueError, KeyError) as error:
        raise CliError(f"{case_dir} has no readable manifest.json with one case: {error}") from error
    if wanted is not None and entry["case_id"] != wanted:
        raise CliError(f"{case_dir} holds case {entry['case_id']}, not {wanted}")
    return entry["case_id"]


def _case_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, str]:
    """The case package to load and its label and custody records, writing the user-case records
    when the case dir has none: (store root, package dir, labels, custody, case id)."""
    case_dir = args.case_dir.resolve()
    case_id = _case_id(case_dir, args.case)
    store_root = args.store or launcher_store()
    labels, custody = case_dir / "case_labels.json", case_dir / "custody.json"
    if not labels.is_file() or not custody.is_file():
        records = store_root.expanduser().resolve().parent / "user-cases" / case_id
        records.mkdir(parents=True, exist_ok=True)
        labels, custody = user_case_records(case_dir, case_id, records)
        print(f"{case_id}: no case_labels.json or custody.json, so it is imported as a user case.")
        print("It has no reference answer and no labels. Nothing is scored against a reference.")
        case_dir = harness_case_package(case_dir, case_id, records / "package")
    return store_root, case_dir, labels, custody, case_id


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
    store_root, case_dir, labels, custody, case_id = _case_inputs(args)
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
    return _queued(store, root, child, f"run {action.harbor.trial_id}", case_id)


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

    store_root, case_dir, labels, custody, case_id = _case_inputs(args)
    store = open_store(store_root, import_root)
    try:
        case, blobs = importer.read_case_package(case_dir, labels, custody=custody)
        root = store.load_case(OWNER, case, blobs)
        action, result, run_blobs = read(case, blobs[case.pattern.sha256])
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
    return _queued(store, root, child, f"{described['source']} {action.harbor.trial_id}", case_id)


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
    append it to each reviewer's review-queue.v1 file. Idempotent: one snapshot, listed once."""
    from .page_routes import ReviewQueue, queue_path

    for reviewer in REVIEWERS:
        store.grant(OWNER, root, reviewer, (CaseOperation.INSPECT, CaseOperation.ACTION))
    labels = LabelStore(store)
    existing = [d for d, s in labels._snapshots() if s.origin == child and s.kind == "agent_answer"]
    digest = existing[0] if existing else labels.put_snapshot(
        snapshot.build_snapshot(store, OWNER, child, "agent_answer")
    )
    # The rater projection admits an imported answer only as a snapshot of the rater's cohort,
    # so the owner assigns it, keeping the root's latest assignment and adding to it.
    latest = labels.latest_assignment(root)
    raters = tuple(dict.fromkeys((*(latest.raters if latest else ()), *REVIEWERS)))
    cohort = tuple(dict.fromkeys((*(latest.snapshots if latest else ()), digest)))
    if latest is None or (latest.raters, latest.snapshots) != (raters, cohort):
        labels.assign(OWNER, root, raters, cohort)
    for reviewer in REVIEWERS:
        path = queue_path(store, reviewer)
        queue = (
            ReviewQueue.model_validate(json.loads(path.read_bytes()))
            if path.is_file()
            else ReviewQueue(principal_id=reviewer)
        )
        if digest not in queue.snapshots:
            queue = queue.model_copy(update={"snapshots": (*queue.snapshots, digest)})
            _atomic_write_bytes(path, (queue.model_dump_json(indent=2) + "\n").encode())
    return digest


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
