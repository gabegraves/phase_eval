"""Import pre-generated Harbor trials as imported answers with a display trajectory (R28).

Target path: src/phaseeval/workbench/importer.py (public). Entry point:

    python -m phaseeval.workbench.importer --root <digest> --run <dir relative to the import root>

The CLI is a thin client: it posts the relative run path to POST /api/attempts/{root}/imports
as the importer service principal, and the server resolves it inside the import root with
store.resolve_inbox_path, parses it here, and calls WorkbenchStore.import_run.

This module is the parse boundary for the frozen U2 case packages (data/workbench/cases/) and
for Harbor trial directories (the U6 layout under data/custody/workbench/runs/). Inside the tree
a case or an imported run is typed data; nothing downstream reads a package or Harbor file again.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import JsonValue, TypeAdapter

from . import executors
from .model import (
    AdapterIdentity,
    BlobDigest,
    CaseManifest,
    Complete,
    DocumentChannel,
    Digest,
    FitSummary,
    ImportedAnswer,
    ImportedInvalid,
    ImportedRunEnd,
    ImportedRunResult,
    ImportedSubmitted,
    ImportedUnsuccessful,
    ImportRun,
    Incomplete,
    InfraCause,
    RecordedAgent,
    Record,
    RecordedToolCall,
    RelativeName,
    SafeID,
)

PACKET_ADAPTER = AdapterIdentity(adapter_id="harbor_packet", version="1")

# The U2 packager's actor-safe field source classes (package_cases.ACTOR_SAFE_SOURCES). A
# custody class has no row, so a package that carries one is refused.
SOURCE_CHANNELS: Mapping[str, DocumentChannel] = {
    "packaging": "packet",
    "pattern_data": "packet",
    "candidate_library": "packet",
    "instrument_header": "instrument_export",
    "instrument_resolver": "instrument_registry",
    "declared_recipe": "synthesis_recipe",
    "synthesis_record": "dataset_metadata",
    "derived_from_synthesis_record": "dataset_metadata",
}
EXPOSURES = {"development-exposed": "development_exposed", "unexposed": "unexposed"}
COLLECTIONS = {
    "10.5281/zenodo.17410051": "dara_zenodo_17410051",
    "10.5281/zenodo.21285546": "precursor_genome_zenodo_21285546",
}
# K-alpha1 in angstrom -> tube anode. The package records the wavelength the instrument used;
# BGMN takes the anode symbol (LAMBDA=CU), never the float (executors.bgmn_wavelength).
TUBE_K_ALPHA1 = {1.540598: "Cu"}

# Precursor Genome workflow actions and step fields -> ProcessingStep. An unlisted action or
# key refuses the package rather than dropping evidence the agents saw.
STEP_ACTIONS = {
    "PowderDosing": "dose",
    "Heating": "heat",
    "RecoverPowder": "recover",
    "Diffraction": "diffract",
}
STEP_FIELDS = {
    "equipment": "equipment",
    "furnace": "furnace",
    "temperature_c": "temperature_c",
    "duration_min": "duration_min",
    "atmosphere": "atmosphere",
    "ramp_rate_c_per_min": "heating_rate_c_per_min",
    "met_target_mass": "met_target_mass",
    "mass_recovered_mg": "mass_mg",
    "mass_for_xrd_mg": "mass_mg",
}
PRECURSOR_FIELDS = {
    "formula": "formula",
    "name": "name",
    "cas_number": "cas_number",
    "purity": "purity",
}
PRECURSOR_MASSES_G = {"actual_mass_g": "amount", "target_mass_g": "target_amount"}

# The U6 response contract (instruction.md, tests/test.sh) reports 0..1 weight fractions.
ANSWER_WEIGHT_UNIT = "fraction"
# Codex's context-window refusal. Harbor files it as NonZeroAgentExitCodeError or, when its
# regex hits "separate limit" in the README, ApiRateLimitError (u6-agent-runs.md).
CONTEXT_OVERFLOW = re.compile(rb"prompt contains at least \d+ input tokens")
# Harbor exception types that are external failures. Anything unlisted is refused, not guessed.
INFRA_CAUSES: Mapping[str, InfraCause] = {"ApiRateLimitError": "provider_error"}
OUTPUT = PurePosixPath("artifacts/workspace/output")
CONTAINER_OUTPUT = PurePosixPath("/workspace/output")


class PackageError(ValueError):
    """The case package, its labels and its custody record disagree."""


class BindingError(ValueError):
    """The trial did not read this case's files, so it cannot sit under this root."""


class CaseFile(Record):
    name: RelativeName
    sha256: Digest
    data: bytes


class CaseLabels(Record):
    exposure: Literal["development_exposed", "unexposed"]
    split: str
    task_kind: str
    source: Mapping[str, JsonValue]
    chemical_system: str | None
    manifest_sha256: Digest | None


class CaseInput(Record):
    case_id: SafeID
    layout: Literal["package", "harness"]
    manifest: Mapping[str, JsonValue]
    candidates_manifest: Mapping[str, JsonValue]
    files: tuple[CaseFile, ...]
    labels: CaseLabels | None


def _case_file(directory: Path, name: str) -> CaseFile:
    name = TypeAdapter(RelativeName).validate_python(name)
    path = directory.absolute() / name
    try:
        with ExitStack() as stack:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            descriptor = os.open(path.anchor, flags)
            stack.callback(os.close, descriptor)
            for part in path.parts[1:-1]:
                descriptor = os.open(part, flags, dir_fd=descriptor)
                stack.callback(os.close, descriptor)
            stream = stack.enter_context(os.fdopen(os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
            ), "rb"))
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise PackageError("a case file must be a regular file inside its folder")
            data = stream.read(64 * 1024 * 1024 + 1)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise PackageError("case paths must not contain symlinks and must name regular files") from error
        raise
    if len(data) > 64 * 1024 * 1024:
        raise PackageError("a case file exceeds 64 MiB")
    return CaseFile(name=name, sha256=BlobDigest(hashlib.sha256(data).hexdigest()), data=data)


def _case_labels(case_id: str, labels: Path, custody: Path) -> CaseLabels:
    if labels.is_symlink() or custody.is_symlink():
        raise PackageError("case labels and custody must not be symlinks")
    label = _label_row(labels, case_id)
    check = _split_check(custody, case_id, label)
    return CaseLabels(
        exposure=_lookup(EXPOSURES, label["exposure"], "exposure"),
        split=label["split"],
        task_kind=label["confirmation_tag"],
        source=label["source"],
        chemical_system=label["chemical_system"],
        manifest_sha256=check["split_manifest_content_sha256"],
    )


def read_case(directory: Path, *, case_id: SafeID | None = None) -> CaseInput:
    """Read either case layout without creating a package or changing the store."""
    manifest_file = _case_file(directory, "manifest.json")
    try:
        manifest = json.loads(manifest_file.data)
        (entry,) = manifest["cases"]
        cid = TypeAdapter(SafeID).validate_python(entry["case_id"])
        pattern_name = TypeAdapter(RelativeName).validate_python(entry["pattern_file"])
    except (KeyError, TypeError, ValueError) as error:
        raise PackageError(f"{directory} has no readable manifest.json with one case: {error}") from error
    if case_id is not None:
        TypeAdapter(SafeID).validate_python(case_id)
        if cid != case_id:
            raise PackageError(f"{directory} holds case {cid}, not {case_id}")
    candidates_name = f"candidates/{cid}/manifest.json"
    candidates_file = _case_file(directory, candidates_name)
    library = json.loads(candidates_file.data)
    names = [
        f"candidates/{cid}/{TypeAdapter(RelativeName).validate_python(c['file'])}"
        for c in library["candidates"]
    ]
    if library["case_id"] != cid:
        raise PackageError("the candidate library names another case")
    files = (
        manifest_file, candidates_file, _case_file(directory, pattern_name),
        *(_case_file(directory, name) for name in names),
    )
    layout = "package" if "field_sources" in entry else "harness"
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise PackageError("case files must not be symlinks")
    if layout == "package":
        actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
        if actual - {"case_labels.json", "custody.json"} != {f.name for f in files}:
            raise PackageError("package files differ from its manifests")
    labels, custody = directory / "case_labels.json", directory / "custody.json"
    return CaseInput(
        case_id=cid, layout=layout, manifest=manifest,
        candidates_manifest=library, files=files,
        labels=_case_labels(cid, labels, custody) if labels.is_file() and custody.is_file() else None,
    )


def case_manifest(case: CaseInput) -> tuple[CaseManifest, Mapping[str, bytes]]:
    """Build root evidence from the bytes already read, preserving existing root digests."""
    files = {f.name: f.data for f in case.files}
    case_id = case.case_id
    candidates_name = f"candidates/{case_id}/manifest.json"
    manifest_bytes, candidates_bytes = (
        _harness_documents(case) if case.layout == "harness"
        else (files["manifest.json"], files[candidates_name])
    )
    manifest, library = json.loads(manifest_bytes), json.loads(candidates_bytes)
    (entry,) = manifest["cases"]
    pattern = files[entry["pattern_file"]]
    blobs: dict[BlobDigest, bytes] = {}
    pattern_ref = _artifact(pattern, "application/vnd.axa.pxrd.xy", "observed_pattern", blobs)
    if pattern_ref["sha256"] != entry["pattern_sha256"]:
        raise PackageError(
            f"{entry['pattern_file']} does not hash to its manifest's pattern_sha256"
        )
    document = _artifact(manifest_bytes, "application/json", "context_document", blobs)
    candidates_document = _artifact(candidates_bytes, "application/json", "context_document", blobs)

    candidates = []
    for record in library["candidates"]:
        cif = _artifact(
            files[f"candidates/{case_id}/{record['file']}"],
            "chemical/x-cif",
            "candidate_cif",
            blobs,
        )
        if cif["sha256"] != record["sha256"]:
            raise PackageError(f"{record['file']} does not hash to its candidate record")
        candidates.append(
            {
                "candidate_id": record["candidate_id"],
                "formula_sum": record["formula_sum"],
                "elements": record["elements"],
                "cif": cif,
                "source_library_id": record["source_library_id"],
                "eligibility_status": record["eligibility_status"],
            }
        )

    label = case.labels
    source = _case_source(label.source) if label else {
        "kind": "experimental", "collection": USER_CASE_COLLECTION, "citation": f"User case {case_id}."
    }
    sourced = _Sourcer(entry["field_sources"], document["sha256"])
    manifest_record = {
        "case_id": case_id,
        "source": source,
        "exposure": label.exposure if label else "unexposed",
        "split": {
            "split": label.split if label else "unassigned",
            "manifest_sha256": label.manifest_sha256 if label else None,
            "chemical_system": label.chemical_system if label else None,
        },
        "task_kind": label.task_kind if label else "identification",
        "task": manifest["task"],
        "pattern": pattern_ref,
        "measurement": _measurement(entry, pattern, sourced),
        "material": _material(entry, sourced),
        "candidates": candidates,
        "ingest": {
            "instrument_adapter": PACKET_ADAPTER.model_dump(),
            "context_adapters": [PACKET_ADAPTER.model_dump()],
            "candidate_selection": PACKET_ADAPTER.model_dump(),
            "sources": [
                {"blob": document, "original_name": "manifest.json"},
                {"blob": candidates_document, "original_name": candidates_name},
            ],
        },
    }
    return CaseManifest.model_validate(manifest_record), blobs


def read_case_package(
    case_dir: Path, case_labels: Path, *, custody: Path | None = None
) -> tuple[CaseManifest, Mapping[BlobDigest, bytes]]:
    case = read_case(case_dir)
    labels = _case_labels(case.case_id, case_labels, custody or _custody_record(case_labels, case.case_id))
    return case_manifest(case.model_copy(update={"labels": labels}))


# The harness tools' defaults when a harness case omits them (harness/physics/screen.py and
# refine.py): a Cu K-alpha doublet and the default BGMN profile, which restricts quantification.
HARNESS_KALPHA = (1.540598, 1.544426)
HARNESS_DEFAULT_PROFILE = "Aeris-fds-Pixcel1d-Medipix3"
HARNESS_TASK = Path(__file__).resolve().parents[3] / "harness" / "agent" / "task.md"
USER_ELIGIBILITY = "not_adjudicated"
USER_DECLARED = "packaging"


def _elements(formula_sum: str) -> list[str]:
    from pymatgen.core import Composition

    return sorted(str(element) for element in Composition(formula_sum).elements)


def _source_paths(value: Any, path: str) -> list[str]:
    """Provenance paths for user-declared material fields."""
    if isinstance(value, dict):
        return [path] + [p for k, v in value.items() for p in _source_paths(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [path] + [p for i, v in enumerate(value) for p in _source_paths(v, f"{path}.{i}")]
    return [path]


def _harness_documents(case: CaseInput) -> tuple[bytes, bytes]:
    manifest = case.manifest
    (entry,) = manifest["cases"]
    case_id = case.case_id
    library = case.candidates_manifest
    if "chemistry_elements" not in entry:
        raise PackageError(f"{case_id} declares no chemistry_elements")
    rows = []
    for record in library["candidates"]:
        rows.append(
            {
                **record,
                "elements": record.get("elements") or _elements(record["formula_sum"]),
                "source_library_id": record.get("source_library_id", f"user {record['candidate_id']}"),
                "eligibility_status": USER_ELIGIBILITY,
            }
        )
    candidates_bytes = (json.dumps({**library, "candidates": rows}, indent=2) + "\n").encode()

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
    return (json.dumps(out, indent=2) + "\n").encode(), candidates_bytes


def read_harbor_run(
    run_dir: Path, case: CaseManifest
) -> tuple[ImportRun, ImportedRunResult, Mapping[BlobDigest, bytes]]:
    """One Harbor trial directory (jobs/<job>/<trial>/) as an imported run under `case`.

    The trial binds to the case by content: every refine request it recorded must name this
    case's pattern and CIF sha256s, which refine.py checked when it ran. The answer is what
    analysis.json reported, with no verdict invented. A cited file that is absent is named in
    Incomplete and nothing stands in for it (R11)."""
    config = _json(run_dir / "config.json")
    result = _json(run_dir / "result.json")
    output = run_dir / OUTPUT
    requests = {
        path.relative_to(run_dir).as_posix(): _json(path)
        for path in sorted(output.rglob("request.json"))
        if _regular(path)
    }
    action = ImportRun.model_validate(
        {
            "harbor": {
                "job_name": Path(config["trials_dir"]).name,
                "trial_id": result["trial_name"],
                "task_sha256": result["task_checksum"],
                "pattern_sha256": case.pattern.sha256,
                "cif_sha256": _bind(requests, case),
            }
        }
    )
    end = _end(run_dir, result, case)
    missing: list[str] = []
    trajectory = _trajectory(run_dir, missing)
    basis_fit = None
    if isinstance(end, ImportedSubmitted):
        analysis = _json(output / "analysis.json")
        basis_fit = _basis_fit(run_dir, analysis, end.answer, requests, case, missing)
    timeout = config["agent"]["override_timeout_sec"]
    execution = result["agent_execution"]
    imported = ImportedRunResult(
        agent=RecordedAgent(
            model_id=result["agent_info"]["model_info"]["name"],
            model_version=None,
            harness=result["agent_info"]["name"],
            harness_version=result["agent_info"]["version"],
        ),
        budget_wall_seconds=None if timeout is None else int(timeout),
        wall_seconds=(
            _time(execution["finished_at"]) - _time(execution["started_at"])
        ).total_seconds(),
        cost_usd=result["agent_result"]["cost_usd"],
        end=end,
        basis_fit=basis_fit,
        trajectory=trajectory,
        completeness=Incomplete(missing=tuple(missing)) if missing else Complete(),
    )
    blobs = {
        BlobDigest(hashlib.sha256(data).hexdigest()): data
        for data in (p.read_bytes() for p in sorted(run_dir.rglob("*")) if _regular(p))
    }
    if isinstance(end, ImportedSubmitted) and end.answer_sha256 not in blobs:
        raise BindingError("the answer file the import names is not one of its receipts")
    return action, imported, blobs


# --- case package ------------------------------------------------------------------------------


class _Sourcer:
    """Wraps a package value with the provenance its field_sources row names."""

    def __init__(self, field_sources: Mapping[str, Mapping[str, str]], document: str) -> None:
        self._rows = field_sources
        self._document = document

    def __call__(self, value: object, path: str) -> dict[str, Any]:
        row = self._rows.get(path)
        if row is None:
            raise PackageError(f"{path} has no field_sources row")
        return {
            "value": value,
            "source": {
                "kind": "document",
                "channel": _lookup(SOURCE_CHANNELS, row["class"], f"{path} source class"),
                "adapter": PACKET_ADAPTER.model_dump(),
                "document": self._document,
                "locator": row["locator"],
            },
        }


def _measurement(entry: Mapping[str, Any], pattern: bytes, sourced: _Sourcer) -> dict[str, Any]:
    m, profile = entry["measurement"], entry["instrument_profile"]
    if profile.get("declared_by_source", profile["profile"]) != profile["profile"]:
        raise PackageError("the source declares a profile other than the resolved one")
    if "kalpha2_wavelength_angstrom" not in m:
        raise PackageError("a single-line wavelength has no tube anode mapping here")
    two_theta = [float(line.split()[0]) for line in pattern.decode().splitlines() if line.strip()]
    return {
        "instrument": sourced(
            {
                "vendor": None,
                "model": m["instrument"],
                "geometry": None,
                "detector": m["detector"],
                "source_format": "harbor_packet",
            },
            "measurement.instrument",
        ),
        "radiation": sourced(
            {"kind": "tube", "anode": _lookup(TUBE_K_ALPHA1, m["wavelength_angstrom"], "K-alpha1")},
            "measurement.wavelength_angstrom",
        ),
        "wavelength_provenance": None,
        "profile": sourced(
            {
                "instrument_profile": profile["profile"],
                "resolver_confidence": profile["resolver_confidence"],
                "reason_codes": profile["warnings"],
            },
            "instrument_profile.profile",
        ),
        "two_theta_deg": (min(two_theta), max(two_theta)),
    }


def _material(entry: Mapping[str, Any], sourced: _Sourcer) -> dict[str, Any]:
    context = dict(entry["material_context"])
    amounts = context.pop("precursor_amounts", None)
    if amounts is not None and not amounts.startswith("withheld"):
        raise PackageError(f"precursor_amounts is not a withheld notice: {amounts!r}")
    if context.pop("element_evidence", entry["chemistry_elements"]) != entry["chemistry_elements"]:
        raise PackageError("element_evidence differs from chemistry_elements")
    material = {
        "elements": sourced(entry["chemistry_elements"], "chemistry_elements"),
        "elements_basis": "declared",
        "precursors": [
            _precursor(record, f"material_context.precursors.{i}", sourced)
            for i, record in enumerate(context.pop("precursors"))
        ],
        "processing": [
            _step(record, i, sourced) for i, record in enumerate(context.pop("processing", ()))
        ],
    }
    if "specimen" in context:
        material["specimen"] = sourced(context.pop("specimen"), "material_context.specimen")
    target = dict(context.pop("declared_target", None) or {})
    if target:
        path = "material_context.declared_target"
        material["declared_target"] = sourced(target.pop("formula"), f"{path}.formula")
        material["declared_target_stoichiometry"] = sourced(
            target.pop("stoichiometry"), f"{path}.stoichiometry"
        )
    if context or target:
        raise PackageError(f"material context fields without a slot: {sorted({*context, *target})}")
    return material


def _precursor(record: Mapping[str, Any], path: str, sourced: _Sourcer) -> dict[str, Any]:
    precursor = {}
    for key, value in record.items():
        if key in PRECURSOR_FIELDS:
            precursor[PRECURSOR_FIELDS[key]] = sourced(value, f"{path}.{key}")
        elif key in PRECURSOR_MASSES_G:
            amount = {"value": value, "unit": "g"}
            precursor[PRECURSOR_MASSES_G[key]] = sourced(amount, f"{path}.{key}")
        else:
            raise PackageError(f"{path}.{key} has no Precursor slot")
    return precursor


def _step(record: Mapping[str, Any], index: int, sourced: _Sourcer) -> dict[str, Any]:
    path = f"material_context.processing.{index}"
    step: dict[str, Any] = {"order": index + 1}
    for key, value in record.items():
        if key == "action":
            operation = _lookup(STEP_ACTIONS, value, f"{path}.action")
            step["operation"] = sourced(operation, f"{path}.action")
        elif key == "temperature_profile":
            trace = {
                "time_min": value["time_minutes"],
                "temperature_c": value["temperature_celsius"],
            }
            step["temperature_profile"] = sourced(trace, f"{path}.temperature_profile.time_minutes")
        elif key in STEP_FIELDS:
            step[STEP_FIELDS[key]] = sourced(value, f"{path}.{key}")
        else:
            raise PackageError(f"{path}.{key} has no ProcessingStep slot")
    return step


def _artifact(
    data: bytes, media_type: str, role: str, blobs: dict[BlobDigest, bytes]
) -> dict[str, Any]:
    digest = BlobDigest(hashlib.sha256(data).hexdigest())
    blobs[digest] = data
    return {"sha256": digest, "size_bytes": len(data), "media_type": media_type, "role": role}


def _label_row(case_labels: Path, case_id: str) -> Mapping[str, Any]:
    rows = [row for row in json.loads(_case_file(case_labels.parent, case_labels.name).data)["cases"] if row["case_id"] == case_id]
    if len(rows) != 1:
        raise PackageError(f"{case_labels.name} has {len(rows)} rows for {case_id}")
    return rows[0]


USER_CASE_COLLECTION = "user_case"


def _case_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if source.get("kind") == USER_CASE_COLLECTION:
        return {"kind": "experimental", "collection": USER_CASE_COLLECTION, "citation": source["citation"]}
    return {
        "kind": "experimental",
        "collection": _lookup(COLLECTIONS, source["doi"], "dataset doi"),
        "citation": f"{source['creators']}. {source['dataset']}. doi:{source['doi']}. {source['license']}.",
    }


def _custody_record(case_labels: Path, case_id: str) -> Path:
    return case_labels.parent.parent / "custody/workbench" / case_id / "custody.json"


def _split_check(custody: Path, case_id: str, label: Mapping[str, Any]) -> Mapping[str, Any]:
    record = json.loads(_case_file(custody.parent, custody.name).data)
    check = record["split_check"]
    agrees = (
        record["case_id"] == case_id
        and check["assigned_split"] == check["verdict"] == label["split"]
        and check["chemical_system"] == label["chemical_system"]
        and record["labels"]["confirmation_tag"] == label["confirmation_tag"]
        and record["labels"]["exposure"] == label["exposure"]
    )
    if not agrees:
        raise PackageError(f"{custody} disagrees with the {case_id} label row")
    return check


# --- Harbor trial ------------------------------------------------------------------------------


def _bind(requests: Mapping[str, Mapping[str, Any]], case: CaseManifest) -> dict[str, str]:
    """The CIFs the trial's refines read, after checking every recorded hash against the case."""
    if not requests:
        raise BindingError(
            "the trial recorded no refine request, so nothing binds it to a case by hash"
        )
    library = {c.candidate_id: c.cif.sha256 for c in case.candidates}
    read: dict[str, str] = {}
    for name, request in requests.items():
        if request["pattern_sha256"] != case.pattern.sha256:
            raise BindingError(
                f"{name} read pattern {request['pattern_sha256']}, not {case.case_id}'s {case.pattern.sha256}"
            )
        for candidate in request["candidates"]:
            if library.get(candidate["candidate_id"]) != candidate["sha256"]:
                raise BindingError(
                    f"{name} read {candidate['candidate_id']} {candidate['sha256']}, "
                    f"which is not in {case.case_id}'s candidate library"
                )
            read[candidate["candidate_id"]] = candidate["sha256"]
    return read


def _end(run_dir: Path, result: Mapping[str, Any], case: CaseManifest) -> ImportedRunEnd:
    exception = result["exception_info"]
    answer = run_dir / OUTPUT / "analysis.json"
    if exception is None:
        if not answer.is_file():
            return ImportedUnsuccessful(reason="no_submission")
        data = answer.read_bytes()
        return ImportedSubmitted(
            answer=_answer(json.loads(data), case),
            answer_sha256=BlobDigest(hashlib.sha256(data).hexdigest()),
        )
    kind = exception["exception_type"]
    codex = run_dir / "agent/codex.txt"
    # The context window is part of the run's budget, so an overflow is the model's own limit.
    if kind == "AgentTimeoutError" or (
        codex.is_file() and CONTEXT_OVERFLOW.search(codex.read_bytes())
    ):
        return ImportedUnsuccessful(reason="budget_exhausted")
    if kind in INFRA_CAUSES:
        return ImportedInvalid(
            cause=INFRA_CAUSES[kind], detail=f"{kind}: {exception['exception_message'][:500]}"
        )
    raise ValueError(f"unclassified Harbor ending {kind} in {result['trial_name']}")


def _answer(analysis: Mapping[str, Any], case: CaseManifest) -> ImportedAnswer:
    if analysis["case_id"] != case.case_id:
        raise BindingError(f"the answer is for {analysis['case_id']}, not {case.case_id}")
    phases = tuple(p["candidate_id"] for p in analysis["phases"])
    fractions: Mapping[str, float | None] = analysis.get("weight_fractions") or {}
    library = {c.candidate_id for c in case.candidates}
    if outside := sorted({*phases, *fractions} - library):
        raise BindingError(
            f"the answer names {outside}, which are not in {case.case_id}'s candidate library"
        )
    if out_of_range := {k: v for k, v in fractions.items() if v is not None and not 0 <= v <= 1}:
        raise ValueError(f"weight fractions outside 0..1: {out_of_range}")
    return ImportedAnswer(
        phases=phases,
        reported_weight_percent={
            k: None if v is None else round(v * 100, 10) for k, v in fractions.items()
        },
        reported_weight_unit=ANSWER_WEIGHT_UNIT if fractions else None,
    )


def _basis_fit(
    run_dir: Path,
    analysis: Mapping[str, Any],
    answer: ImportedAnswer,
    requests: Mapping[str, Mapping[str, Any]],
    case: CaseManifest,
    missing: list[str],
) -> FitSummary | None:
    """The fit the answer rests on: the one cited refine run of exactly the answered phases.
    analysis.json does not name it. Every cited file that is absent is named in `missing`."""
    cited = [
        PurePosixPath(path).relative_to(CONTAINER_OUTPUT)
        for path in analysis.get("fit_artifacts") or ()
        if PurePosixPath(path).is_relative_to(CONTAINER_OUTPUT)
    ]
    missing.extend(
        (OUTPUT / rel).as_posix() for rel in cited if not _regular(run_dir / OUTPUT / rel)
    )
    if all(v is None for v in answer.reported_weight_percent.values()):
        return None
    runs = sorted({OUTPUT / rel.parts[0] for rel in cited if len(rel.parts) > 1})
    basis = [
        run
        for run in runs
        if _phase_set(requests.get((run / "request.json").as_posix())) == set(answer.phases)
    ]
    if len(basis) != 1:
        missing.append(f"basis fit: {len(basis)} cited refine runs fit exactly the answered phases")
        return None
    (run,) = basis
    request = requests[(run / "request.json").as_posix()]
    stage = "pawley" if request["config"]["pawley_only"] else "rietveld"
    required = [
        run / "result.json",
        run / f"fit/{case.case_id}_{stage}.lst",
        run / f"fit/{case.case_id}_{stage}.dia",
    ]
    absent = [path.as_posix() for path in required if not _regular(run_dir / path)]
    if absent:
        missing.extend(path for path in absent if path not in missing)
        return None
    record = executors.read_recorded_execution(run_dir / run, binary_sha256=None)
    # The recorded result carries no final summary: its quantification depends on the case's
    # profile, which the executor applies the same way to a live refine.
    restricted = case.measurement.profile.value.restricts_quantification
    return executors._final(record.result, restricts_quantification=restricted)


def _phase_set(request: Mapping[str, Any] | None) -> set[str]:
    return {c["candidate_id"] for c in request["candidates"]} if request else set()


def _trajectory(run_dir: Path, missing: list[str]) -> tuple[RecordedToolCall, ...]:
    """Recorded tool calls, in order, for display. They are never actions in the tree (R28)."""
    path = run_dir / "agent/trajectory.json"
    if not path.is_file():
        missing.append("agent/trajectory.json")
        return ()
    calls = [call for step in _json(path)["steps"] for call in step.get("tool_calls") or ()]
    return tuple(
        RecordedToolCall(
            index=index,
            tool=call["function_name"],
            arguments_sha256=hashlib.sha256(_canonical(call["arguments"])).hexdigest(),
            summary=_summary(call["arguments"]),
        )
        for index, call in enumerate(calls)
    )


def _summary(arguments: object) -> str:
    if isinstance(arguments, dict) and isinstance(arguments.get("cmd"), str):
        return arguments["cmd"][:500]
    return _canonical(arguments).decode()[:500]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _regular(path: Path) -> bool:
    """A regular file, not followed through a symlink: the fit/bgmnwin/ links point into the
    container's /root and fail to stat on the host."""
    return not path.is_symlink() and path.is_file()


def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _lookup[K, V](table: Mapping[K, V], key: K, what: str) -> V:
    if key not in table:
        raise PackageError(f"no mapping for {what} {key!r}")
    return table[key]


def main() -> None:  # pragma: no cover
    from .cli import main as cli_main

    cli_main()
