"""Import saved BGMN or Profex outputs as a reviewable analysis.

A bundle is a directory, or a Profex project archive (ZIP), holding one BGMN control file
(`.sav`) and the files it names: the
instrument profile (VERZERR), the scan (VAL[n]), the phases (STRUC[n]) and the outputs LIST
(`.lst`), DIAGRAMM (`.dia`) and OUTPUT (`.par`). Profex projects are BGMN projects, so one reader
serves both.

The import claims no more than COPY. Every number is read where the source report printed it and
nothing is refit. BGMN writes refined values back into its .sav, so restarting a saved project
continues from them: a new run, not this one again. A value the .lst does not print stays None
with the reason. BGMN records no convergence flag and nothing about whether its instrument profile
is the measuring instrument's, so those stay "unknown" and are never read off Rpb, GEWICHT or a
missing value.

The stage is read from the saved run itself. BGMN computes each phase's intensities from its
STRUC file's structure unless that file sets LeBail=1, which turns the fit into a Le Bail
extraction; that line is how the harness writes its Pawley stage (refinement.bgmn.control_file),
so a bundle whose .str files are all present names its own stage. When a .str is missing, the
control file's harness stage suffix (_pawley, _rietveld_lattice, _rietveld_profile, _rietveld)
is used; with neither, the stage is not knowable and the import refuses rather than call the fit
Rietveld.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
import zipfile
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from phaseeval.refinement.bgmn import result_parser

from ..executors import _finite, read_lst_facts
from ..model import ArtifactRef, ArtifactRole, BlobDigest, Cell, MediaType, Record, StageName

COPY = (
    "Imported saved BGMN outputs; Rwp copied from the source report. "
    "Restarting the saved project is a new continuation."
)
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_ASSIGNMENT = re.compile(
    r"^\s*(?P<key>STRUC\[\d+\]|VAL\[\d+\]|LIST|OUTPUT|DIAGRAMM|VERZERR)\s*=\s*(?P<value>\S+)", re.MULTILINE
)
# A .sav key that names a member decides its role; STRUC, VERZERR and unnamed files are raw_export.
_KEY_ROLES: Mapping[str, tuple[MediaType, ArtifactRole]] = {
    "VAL": ("application/vnd.axa.pxrd.xy", "observed_pattern"),
    "LIST": ("text/plain", "native_lst"),
    "DIAGRAMM": ("text/plain", "native_dia"),
    "OUTPUT": ("text/plain", "native_par"),
}
_STATISTICS = ("Rwp", "Rexp", "Rp", "Rpb")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_TEXTURE_GEWICHT = re.compile(rf"MeanValue\(GEWICHT\)\s*=\s*(?P<value>{_NUMBER})(?![\w.])")
_PREFIXED_GEWICHT = re.compile(rf"(?P<value>{_NUMBER})\+-\S*")
_PHASE = re.compile(r"^\s*PHASE\s*=\s*(?P<name>[^\s/]+)", re.MULTILINE)
_DEFINITION = re.compile(r"^\s*(?P<name>\w+)\s*=\s*(?P<expression>[^=%\n]+?)\s*$", re.MULTILINE)
_GOAL = re.compile(r"^\s*GOAL\[\d+\]\s*=\s*(?P<name>\w+)\s*(?:=[^\n]*)?$", re.MULTILINE)
_PRINTED_GOAL = re.compile(r"^(?P<name>\w+)=(?P<center>[^+\s]+)\+-(?P<error>\S+)\s*$", re.MULTILINE)
_ITERATIONS = re.compile(r"^(\d+) iteration steps?\s*$", re.MULTILINE)
_GRAIN_DIAMETER = re.compile(r"^[^%/\n]*?(?<![\w.])d\s*=+\s*(?P<value>[^\s/]*)", re.MULTILINE)
# BGMN's LeBail switch in a .str; the harness writes LeBail=1 into every .str of its Pawley stage.
_LEBAIL = re.compile(r"^\s*LeBail\s*=\s*(\S+)", re.MULTILINE)
_HARNESS_STAGE = re.compile(r"_(pawley|rietveld_lattice|rietveld_profile|rietveld)\.sav$", re.IGNORECASE)

WeightBasis = Literal["recorded_goal", "derived_gewicht", "unavailable"]


class BundleError(ValueError):
    """A bundle that does not hold one BGMN control file with the LIST and DIAGRAMM it names."""


class Statistic(Record):
    """A global R value in percent as the source .lst printed it. Without one, `value` is None
    and `absent` says why: nothing is defaulted."""

    value: float | None
    absent: str | None = None

    @model_validator(mode="after")
    def _value_or_reason(self) -> Self:
        if (self.value is None) == (self.absent is None):
            raise ValueError("a statistic holds its value or the reason it is absent")
        return self


class ImportedPhase(Record):
    name: str = Field(min_length=1, max_length=500)  # as the .lst names it (BGMN PHASE=)
    structure: str | None  # the STRUC member whose PHASE= is this name; None when not in the bundle
    weight_percent: float | None  # on RecordedFit.weight_basis
    weight_percent_error: float | None  # the recorded GOAL error; never set on a derived weight
    cell: Cell
    spacegroup_no: int | None


class RecordedFit(Record):
    """The source .lst and .dia as printed."""

    rwp: Statistic
    rexp: Statistic
    rp: Statistic
    rpb: Statistic
    gof: float | None  # Rwp/Rexp when both are printed
    convergence: Literal["unknown"] = "unknown"
    stage: StageName  # from the STRUC files' LeBail lines, else the control file's harness suffix
    stage_basis: str = Field(max_length=2000)  # how the stage was decided
    profile: Literal["unknown"] = "unknown"  # whether VERZERR describes the measuring instrument
    iterations: int | None
    measured_points: int | None
    peaks: int | None
    parameters: int | None
    # recorded_goal: the .lst's Q GOAL center and error, where the .sav defines each Q as one
    # phase over the sum of every phase. derived_gewicht: normalized GEWICHT, times exp(my*d*3/4)
    # when the .sav sets d, an estimate with no error. weight_detail says which, or why there is
    # no weight.
    weight_basis: WeightBasis
    weight_detail: str = Field(max_length=2000)
    phases: tuple[ImportedPhase, ...]
    curve: ArtifactRef  # the .dia as an evidence.Curve CSV, background kept (K2)


class BundleMember(Record):
    ref: ArtifactRef  # the member's original bytes, hashed unchanged
    named_by: tuple[str, ...]  # the .sav keys naming it, e.g. ("STRUC[1]",); () if none does


class ImportedBgmnAnalysis(Record):
    """Saved BGMN outputs and every file they came from. Shown for review and never replayed as
    an action."""

    software: Literal["bgmn"] = "bgmn"
    description: str = COPY
    control: str  # the .sav member name
    bgmn_version: str | None  # from the .lst header
    fit: RecordedFit
    members: dict[str, BundleMember]  # member name -> digest and role, for every file in the bundle
    # The .sav references with no member, as written (e.g. "VAL[1]=scan.xy"); empty when the
    # bundle holds every file the .sav names.
    missing_members: tuple[str, ...]


def read_bundle(path: Path, control: str | None = None) -> tuple[ImportedBgmnAnalysis, dict[BlobDigest, bytes]]:
    """A directory or a Profex project archive -> the analysis and the bytes of every file.
    `control` names the .sav to read when the bundle holds several, e.g. one per stage."""
    members = _zip_members(path) if zipfile.is_zipfile(path) else _directory_members(path)
    controls = sorted(name for name in members if name.lower().endswith(".sav"))
    if control is not None:
        if control not in controls:
            raise BundleError(f"the bundle holds no control file {control!r}; its .sav files are {controls or 'none'}")
        controls = [control]
    if len(controls) != 1:
        raise BundleError(f"a bundle holds one BGMN control file (.sav), found {controls or 'none'}; name one with --control")
    control = controls[0]
    sav = _text(members[control])
    references = [
        (match["key"], match["value"], PureWindowsPath(match["value"]).name) for match in _ASSIGNMENT.finditer(sav)
    ]
    named = {key: name for key, _, name in references}
    lst_name, dia_name = named.get("LIST"), named.get("DIAGRAMM")
    if lst_name not in members or dia_name not in members:
        raise BundleError(f"{control} names LIST={lst_name} and DIAGRAMM={dia_name}; both must be in the bundle")
    lst_text = _text(members[lst_name])
    with tempfile.TemporaryDirectory() as scratch:
        lst_path, dia_path = Path(scratch) / "fit.lst", Path(scratch) / "fit.dia"
        lst_path.write_text(lst_text)
        dia_path.write_text(_text(members[dia_name]))
        lst, dia = result_parser.parse_lst(lst_path), result_parser.parse_dia(dia_path)
    if dia is None or not lst["phases"]:
        raise BundleError(f"{lst_name} or {dia_name} holds no finished fit")

    curve = _curve_csv(dia)
    blobs = {BlobDigest(hashlib.sha256(curve).hexdigest()): curve}
    structures: dict[str, str] = {}
    for key, _, name in references:
        if key.startswith("STRUC[") and name in members and (phase := _PHASE.search(_text(members[name]))):
            structures.setdefault(phase["name"], name)
    weights, errors, basis, detail = _weights(sav, lst_text, lst["phases"])
    stage, stage_basis = _stage(control, references, members)
    statistics = {name: _statistic(lst_text, name) for name in _STATISTICS}
    rwp, rexp = statistics["Rwp"].value, statistics["Rexp"].value
    iterations = _ITERATIONS.search(lst_text)
    facts = read_lst_facts(lst_text)
    fit = RecordedFit(
        rwp=statistics["Rwp"],
        rexp=statistics["Rexp"],
        rp=statistics["Rp"],
        rpb=statistics["Rpb"],
        gof=rwp / rexp if rwp is not None and rexp else None,
        stage=stage,
        stage_basis=stage_basis,
        iterations=int(iterations[1]) if iterations else None,
        measured_points=facts.measured_points,
        peaks=facts.peaks,
        parameters=facts.parameters,
        weight_basis=basis,
        weight_detail=detail,
        phases=tuple(
            ImportedPhase(
                name=name,
                structure=structures.get(name),
                weight_percent=weights.get(name),
                weight_percent_error=errors.get(name),
                cell=_cell(data),
                spacegroup_no=data.get("spacegroup_no"),
            )
            for name, data in lst["phases"].items()
        ),
        curve=_ref(curve, "text/csv", "fit_curve"),
    )

    files: dict[str, BundleMember] = {}
    for name, data in sorted(members.items()):
        keys = tuple(key for key, _, referenced in references if referenced == name)
        kinds = [_KEY_ROLES[key.split("[")[0]] for key in keys if key.split("[")[0] in _KEY_ROLES]
        media_type, role = kinds[0] if kinds else ("application/octet-stream", "raw_export")
        files[name] = BundleMember(ref=_ref(data, media_type, role), named_by=keys)
        blobs[BlobDigest(hashlib.sha256(data).hexdigest())] = data
    analysis = ImportedBgmnAnalysis(
        control=control,
        bgmn_version=facts.bgmn_version,
        fit=fit,
        members=files,
        missing_members=tuple(f"{key}={value}" for key, value, name in references if name not in members),
    )
    return analysis, blobs


def _stage(control: str, references: list[tuple[str, str, str]], members: Mapping[str, bytes]) -> tuple[StageName, str]:
    """The fit's stage as the saved run records it: a Le Bail extraction (LeBail=1 in any STRUC
    file) is the Pawley stage; structures without it are Rietveld, narrowed to the harness's
    rietveld_lattice or rietveld_profile when the control file is named for one. Without every
    STRUC file the control file's harness suffix decides; without that too, BundleError."""
    strucs = [name for key, _, name in references if key.startswith("STRUC[")]
    missing = [name for name in strucs if name not in members]
    suffix = _HARNESS_STAGE.search(control)
    named: StageName | None = suffix[1].lower() if suffix else None  # type: ignore[assignment]
    if strucs and not missing:
        le_bail = [
            name
            for name in strucs
            if any(m[1].split("/")[0] not in ("", "0") for m in _LEBAIL.finditer(_text(members[name])))
        ]
        if le_bail:
            return "pawley", f"LeBail=1 in {', '.join(le_bail)}: intensities extracted, not computed from the structures"
        stage: StageName = named if named in ("rietveld_lattice", "rietveld_profile") else "rietveld"
        basis = "no STRUC file sets LeBail: intensities computed from the structures"
        return stage, basis + (f"; the control file is named for the {stage} stage" if stage != "rietveld" else "")
    if named is not None:
        return named, f"STRUC file(s) {missing or 'none named'} not in the bundle; the control file {control} is named for the {named} stage"
    raise BundleError(
        f"the stage of {control} cannot be determined: its STRUC file(s) {missing} are not in the bundle, so "
        "a Le Bail (LeBail=1) fit cannot be told from a Rietveld one, and its name carries no harness stage. "
        "Include the .str files the control file names."
    )


def _statistic(lst_text: str, name: str) -> Statistic:
    match = re.search(rf"(?<!\w){name}=(\S*)", lst_text)
    if match is None:
        return Statistic(value=None, absent=f"the .lst prints no {name}")
    percent = re.fullmatch(rf"({_NUMBER})%", match[1])
    value = float(percent[1]) if percent else math.nan
    if not math.isfinite(value):
        return Statistic(value=None, absent=f"the .lst prints {name}={match[1]}, not a finite percent")
    return Statistic(value=value)


def _weights(
    sav: str, lst_text: str, phases: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, float], dict[str, float], WeightBasis, str]:
    """Recorded Q GOAL center/error pairs when the .sav defines each Q as one phase over the sum
    of every phase and the .lst prints all of them with usable centers; else normalized GEWICHT,
    grain-corrected when the .sav sets d, labelled derived, with no error; else none, with the
    reason."""
    recorded = _recorded_goals(sav, lst_text, list(phases))
    if isinstance(recorded, dict):
        weights = {name: center for name, (center, _) in recorded.items()}
        errors = {name: error for name, (_, error) in recorded.items()}
        detail = "Q GOAL centers and errors as the .lst printed them; the .sav defines each Q as the phase over the sum of every phase"
        return weights, errors, "recorded_goal", detail
    gewicht = {name: _gewicht(name, data.get("_raw_section", {})) for name, data in phases.items()}
    unread = [name for name, value in gewicht.items() if value is None]
    if unread:
        return {}, {}, "unavailable", f"{recorded}; the .lst prints no readable GEWICHT for {', '.join(unread)}"
    settings = {match["value"] for match in _GRAIN_DIAMETER.finditer(sav)}
    if settings:
        return _grain_corrected_weights(recorded, settings, {name: {**data, "gewicht": gewicht[name]} for name, data in phases.items()})
    total = sum(value for value in gewicht.values() if value is not None)
    if not total > 0:
        return {}, {}, "unavailable", f"{recorded}; the printed GEWICHT values sum to {total}"
    weights = {name: value / total * 100 for name, value in gewicht.items() if value is not None}
    return weights, {}, "derived_gewicht", f"{recorded}; normalized GEWICHT is a derived estimate with no recorded error"


def _grain_corrected_weights(
    recorded: str, settings: set[str], phases: dict[str, dict[str, Any]]
) -> tuple[dict[str, float], dict[str, float], WeightBasis, str]:
    """Normalized GEWICHT x exp(my*d*3/4). GEWICHT omits that factor, so without it the amounts
    are unknown rather than uncorrected."""
    value = next(iter(settings))
    diameter = float(value) if len(settings) == 1 and re.fullmatch(_NUMBER, value) else 0.0
    if not 0 < diameter < math.inf:
        return {}, {}, "unavailable", f"{recorded}; the .sav sets d to {sorted(settings)}, not one positive number"
    weights, reason = result_parser._grain_corrected_percent(phases, diameter)
    if weights is None:
        return {}, {}, "unavailable", f"{recorded}; {reason}"
    detail = f"{recorded}; normalized GEWICHT x exp(my*d*3/4) at d={diameter} is a derived estimate with no recorded error"
    return weights, {}, "derived_gewicht", detail


def _recorded_goals(sav: str, lst_text: str, phases: list[str]) -> dict[str, tuple[float, float]] | str:
    """Each phase's (center, error) in percent, or why the recorded GOALs are not used."""
    definitions = {match["name"]: re.sub(r"\s+", "", match["expression"]) for match in _DEFINITION.finditer(sav)}
    goals = {match["name"] for match in _GOAL.finditer(sav)}
    global_part = lst_text.split("Local parameters and GOALs for phase", 1)[0]
    printed = {match["name"]: (match["center"], match["error"]) for match in _PRINTED_GOAL.finditer(global_part)}
    recorded: dict[str, tuple[float, float]] = {}
    for phase in phases:
        targets = [goal for goal in sorted(goals) if _is_share(definitions.get(goal, ""), phase, definitions, phases)]
        if len(targets) != 1:
            return f"the .sav defines no single GOAL that is {phase} over the sum of every phase"
        center_text, error_text = printed.get(targets[0], ("", ""))
        center, error = _percent(center_text), _percent(error_text)
        if center is None or error is None:
            return f"the .lst prints no {targets[0]}=center+-error"
        if error >= abs(center):
            return (
                f"the .lst prints {targets[0]}={center_text}+-{error_text}: BGMN rounds a GOAL to its esd, so an esd "
                "at least as large as the center leaves no usable center and no usable formal error"
            )
        recorded[phase] = (center, error)
    return recorded


def _is_share(expression: str, phase: str, definitions: Mapping[str, str], phases: list[str]) -> bool:
    numerator, _, total = expression.partition("/")
    return numerator == phase and sorted(definitions.get(total, "").split("+")) == sorted(phases)


def _percent(text: str) -> float | None:
    """A printed fraction as percent, exactly as printed (0.410 -> 41.0)."""
    try:
        fraction = Decimal(text)
    except InvalidOperation:
        return None
    return float(fraction * 100) if fraction.is_finite() else None


def _gewicht(phase: str, section: Mapping[str, Any]) -> float | None:
    """The printed GEWICHT as a finite number, else None: absent, ERROR, UNDEF or unreadable is
    never zero. Besides `0.0153+-0.0012`, a texture fit prints `SPHAR2, MeanValue(GEWICHT)=1.465`
    and some fits print the phase name first, `NiO0.3827+-0.0049`; the name is stripped whole,
    since a name can end in digits."""
    raw = section.get("GEWICHT")
    value: float | None = None
    if isinstance(raw, tuple):
        value = float(raw[0])
    elif isinstance(raw, int | float) and not isinstance(raw, bool):
        value = float(raw)
    elif isinstance(raw, str):
        match = _TEXTURE_GEWICHT.search(raw)
        if match is None and raw.startswith(phase):
            match = _PREFIXED_GEWICHT.fullmatch(raw.removeprefix(phase))
        value = float(match["value"]) if match else None
    return value if value is not None and math.isfinite(value) and value >= 0 else None


def _cell(data: Mapping[str, Any]) -> Cell:
    """The printed cell, with the parameters BGMN omits filled from its lattice setting."""
    keys = ("a", "b", "c", "alpha", "beta", "gamma")
    lattice = result_parser._complete_lattice_params(
        {key: data[key] for key in keys if key in data},
        lattice_system=data.get("lattice_system"),
        spacegroup_no=data.get("spacegroup_no"),
        unique_axis=data.get("unique_axis"),
    )
    return Cell(**{key: _finite(lattice.get(key)) for key in keys})


def _ref(data: bytes, media_type: MediaType, role: ArtifactRole) -> ArtifactRef:
    return ArtifactRef(
        sha256=BlobDigest(hashlib.sha256(data).hexdigest()),
        size_bytes=len(data),
        media_type=media_type,
        role=role,
    )


def _text(data: bytes) -> str:
    """BGMN reads only its platform's line endings, so a Windows Profex project arrives as CRLF."""
    return data.decode("latin-1").replace("\r\n", "\n")


def _curve_csv(dia: Mapping[str, object]) -> bytes:
    """The evidence.Curve CSV, background kept: the residual-peak check needs it (K2)."""
    columns = ("two_theta", "observed", "calculated", "difference", "background")
    rows = zip(*(dia[column] for column in columns), strict=True)
    lines = [",".join(columns), *(",".join(repr(float(value)) for value in row) for row in rows)]
    return ("\n".join(lines) + "\n").encode()


def _directory_members(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir():
        raise BundleError(f"{directory} is neither a directory nor a ZIP archive")
    members: dict[str, bytes] = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        members[path.name] = path.read_bytes()
    _bounded(members)
    return members


def _zip_members(archive: Path) -> dict[str, bytes]:
    """Members are read in memory, never extracted. One top folder is allowed; a member path that
    is absolute, climbs out, nests deeper or repeats another member's name is refused."""
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(archive) as bundle:
        infos = [info for info in bundle.infolist() if not info.is_dir()]
        if sum(info.file_size for info in infos) > MAX_BUNDLE_BYTES:
            raise BundleError("the archive expands past the bundle size limit")
        for info in infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or len(name.parts) > 2:
                raise BundleError(f"unsafe or nested archive member {info.filename!r}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise BundleError(f"archive member {info.filename!r} is too large")
            if name.name in members:
                raise BundleError(f"two archive members are named {name.name!r}")
            members[name.name] = bundle.read(info)
    _bounded(members)
    return members


def _bounded(members: Mapping[str, bytes]) -> None:
    if any(len(data) > MAX_MEMBER_BYTES for data in members.values()):
        raise BundleError("a bundle member is past the size limit")
    if sum(len(data) for data in members.values()) > MAX_BUNDLE_BYTES:
        raise BundleError("the bundle is past the size limit")
