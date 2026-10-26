"""Workbench records: one immutable attempt per action, linked by parent digest.

Target path: src/phaseeval/workbench/model.py (public). Replaces src/phaseeval/api/attempt_models.py:
AttemptBundle, CaptureAction, CaptureCompleteness and AttemptSource are deleted; CaptureArtifact
becomes ArtifactRef; ReviewerNote becomes review.Note; SafeID and Digest move here.

Base: arena candidate 2 (attempt tree). Grafts from candidates 1 and 3 and the revision-3
additions are tagged where they land: [C1], [C3], [rev3].

Invariants this module owns:
- An attempt's id is sha256(canonical_bytes(attempt)); a child names its parent's digest (R3).
- A root is a pure function of case evidence: no actor, no time (R1).
- Truth is unrepresentable: CaseManifest has no truth field.
- Analysis state is never stored; it is a fold of `transition` over the path.
- The actor is stamped from the authenticated principal by `actor_for`, never read from a
  request (R1, KTD12).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, NewType, Self, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, model_serializer, model_validator

from phaseeval.api.auth import AuthorizationError, Principal, PrincipalType
from phaseeval.orchestrator.governance import validate_requested_data_type
from phaseeval.refinement.instrument_resolver import ProfileConfidence

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

HEX64 = r"^[0-9a-f]{64}$"
SafeID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Digest = Annotated[str, Field(pattern=HEX64)]

AttemptDigest = NewType("AttemptDigest", str)
BlobDigest = NewType("BlobDigest", str)
AttemptRef = Annotated[AttemptDigest, Field(pattern=HEX64)]
BlobRef = Annotated[BlobDigest, Field(pattern=HEX64)]

# Opaque id inside one root's candidate library (e.g. CIF006427), as build_packet.py issues them.
CandidateId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
Element = Annotated[str, Field(pattern=r"^[A-Z][a-z]?$")]
# The closed measurement vocabulary stays in governance.py.
RequestedDataType = Annotated[str, AfterValidator(validate_requested_data_type)]


class _FreeTextMarker:
    pass


FREE_TEXT = _FreeTextMarker()
FreeText = Annotated[str, FREE_TEXT]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def canonical_bytes(record: BaseModel) -> bytes:
    """The one serialization every digest is taken over (same form as AttemptStore today)."""
    return (
        json.dumps(
            record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def digest_of(record: BaseModel) -> AttemptDigest:
    return AttemptDigest(hashlib.sha256(canonical_bytes(record)).hexdigest())


# ---------------------------------------------------------------------------
# Retained artifacts. Who may see a role is decided once, in review.ARTIFACT_AUDIENCE [C3];
# the artifact carries no visibility flag of its own.
# ---------------------------------------------------------------------------

MediaType = Literal[
    "application/json",
    "application/x-ndjson",
    "text/plain",
    "text/csv",
    "chemical/x-cif",
    "application/vnd.axa.pxrd.xy",
    "application/octet-stream",
]
ArtifactRole = Literal[
    "observed_pattern",
    "candidate_cif",
    "fit_curve",  # csv: two_theta,observed,calculated,difference (+background when retained)
    "native_lst",
    "native_dia",
    "native_par",
    "raw_export",  # the instrument's original file; review-only (ids and names can carry answers)
    "context_document",  # a context file as received; review-only
    "harbor_receipt",  # an imported trial's result, request, trajectory or log file [C1]
    "custody_withheld",  # bytes that tripped the canary scan; never actor-visible
]
StageName = Literal["pawley", "rietveld_lattice", "rietveld_profile", "rietveld"]


class ArtifactRef(Record):
    sha256: BlobRef
    size_bytes: int = Field(ge=0, le=64 * 1024 * 1024)
    media_type: MediaType
    role: ArtifactRole
    stage: StageName | None = None


# ---------------------------------------------------------------------------
# Case evidence. Every context field is Sourced: it says where it came from (R8).
# ---------------------------------------------------------------------------


class AdapterIdentity(Record):
    adapter_id: SafeID  # e.g. xrdml, xy, context_json, harbor_packet
    version: str


DocumentChannel = Literal[
    "instrument_export",
    "instrument_registry",
    "xrf",
    "eds",
    "eln",
    "lims",
    "synthesis_recipe",
    "dataset_metadata",
    "packet",
    "operator_note",  # free text: may carry answers, so never actor-safe (review.ACTOR_SAFE_CHANNELS)
    "custody_record",  # e.g. weighed amounts on an unheated mixture (R8, KD11): expert-only
]


class DocumentSource(Record):
    """A field read by an adapter from a source document or record."""

    kind: Literal["document"] = "document"
    channel: DocumentChannel
    adapter: AdapterIdentity
    document: BlobRef | None  # the retained raw_export / context_document blob
    locator: str | None = None  # record id or JSON pointer inside that document


class CorrectionSource(Record):
    """A field set by a human correction attempt. Stamped by `transition`, never sent by a
    client: the correcting attempt's digest exists only after it is written. This is the
    "expert as a named source" of R8."""

    kind: Literal["correction"] = "correction"
    attempt: AttemptRef
    principal_id: SafeID


ContextSource = Annotated[DocumentSource | CorrectionSource, Field(discriminator="kind")]


class Sourced[T](Record):
    value: T
    source: ContextSource


class TubeRadiation(Record):
    """Laboratory tube: BGMN LAMBDA=<anode>, K-alpha1/K-alpha2 doublet included."""

    kind: Literal["tube"] = "tube"
    anode: Literal["Cu", "Co", "Mo", "Fe", "Cr", "Ag"]


class MonochromaticRadiation(Record):
    """Single line: BGMN SYNCHROTRON=<nm>. Choosing this for a Cu tube scan drops K-alpha2."""

    kind: Literal["monochromatic"] = "monochromatic"
    wavelength_angstrom: float = Field(gt=0.1, lt=3.0)


Radiation = Annotated[TubeRadiation | MonochromaticRadiation, Field(discriminator="kind")]

ProfileReasonCode = Literal[
    "instrument_profile_proxy", "instrument_profile_default_fallback", "instrument_profile_missing"
]


class ProfileAssignment(Record):
    """Copied from phaseeval.refinement.instrument_resolver.resolve_pattern_instrument_policy and the
    metrology guard's reason codes (metrology_guard.py:283-294); no new proxy mapping (KTD6).
    A stand-in restricts quantitative claims and never blocks identification (R7, AE6)."""

    instrument_profile: SafeID | None  # BGMN .geq name; None only when the resolver found none
    resolver_confidence: ProfileConfidence
    reason_codes: tuple[ProfileReasonCode, ...] = ()

    @model_validator(mode="after")
    def stand_ins_are_caveated(self) -> Self:
        codes = set(self.reason_codes)
        if self.resolver_confidence == "fallback" and not codes & {
            "instrument_profile_proxy",
            "instrument_profile_default_fallback",
        }:
            raise ValueError("a fallback profile names whether it is a proxy or the default")
        # infer_profile_from_metadata picks a representative family from manufacturer, model or
        # detector tokens: a stand-in, so it carries the proxy caveat (AE6).
        if self.resolver_confidence in ("inferred", "screened") and "instrument_profile_proxy" not in codes:
            raise ValueError("an inferred profile is a stand-in and carries instrument_profile_proxy")
        if (self.resolver_confidence == "missing") != (self.instrument_profile is None):
            raise ValueError("a profile is absent exactly when the resolver reports it missing")
        if self.resolver_confidence == "matched" and codes:
            raise ValueError("a matched profile carries no stand-in reason code")
        return self

    @property
    def restricts_quantification(self) -> bool:
        return self.resolver_confidence != "matched"

    @property
    def blocks_accept(self) -> bool:
        """Only no profile at all forces UNKNOWN (owner 2026-09-03); a proxy caveats."""
        return self.resolver_confidence == "missing" or (
            "instrument_profile_default_fallback" in self.reason_codes
        )


class InstrumentIdentity(Record):
    """Structured identity only. Free-text header fields (sample id, name, comment, preparer)
    are custody and never become fields here (R9)."""

    vendor: str | None
    model: str | None
    geometry: str | None
    detector: str | None
    source_format: Literal["xrdml", "xy", "xye", "harbor_packet"]


class MeasurementContext(Record):
    """Acquisition context the tools consume. Corrected with CorrectMeasurement."""

    instrument: Sourced[InstrumentIdentity]
    radiation: Sourced[Radiation]
    wavelength_provenance: str | None  # ResolvedInstrumentPolicy.wavelength_provenance, verbatim
    profile: Sourced[ProfileAssignment]
    two_theta_deg: tuple[float, float]  # measured from the pattern itself

    def sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


class Amount(Record):
    value: float = Field(gt=0)
    unit: Literal["g", "mg", "mmol", "mol_ratio", "wt_fraction"]


class Precursor(Record):
    formula: Sourced[str]
    amount: Sourced[Amount] | None = None  # the amount actually used
    target_amount: Sourced[Amount] | None = None  # the amount the recipe asked for
    name: Sourced[str] | None = None
    cas_number: Sourced[str] | None = None
    purity: Sourced[float] | None = None  # mass fraction, 0..1


# dose, heat, recover and diffract are the Precursor Genome workflow steps, kept as recorded:
# a ledger "Heating" says nothing of whether it was a calcine, anneal or reaction.
ProcessingOperation = Literal[
    "mix",
    "mill",
    "press",
    "dry",
    "calcine",
    "anneal",
    "sinter",
    "react",
    "quench",
    "dose",
    "heat",
    "recover",
    "diffract",
    "other",
]
CoolingMode = Literal["furnace_cool", "controlled", "air_quench", "water_quench", "unknown"]


class TemperatureProfile(Record):
    """A logged furnace trace: one temperature per time point."""

    time_min: tuple[float, ...]
    temperature_c: tuple[float, ...]

    @model_validator(mode="after")
    def paired(self) -> Self:
        if len(self.time_min) != len(self.temperature_c):
            raise ValueError("a temperature profile pairs every time with a temperature")
        return self


class ProcessingStep(Record):
    """One ordered step. Processing history decides which phases can form, so each quantity
    is typed and sourced rather than a free-text note (KD6)."""

    order: int = Field(ge=1)
    operation: Sourced[ProcessingOperation]
    equipment: Sourced[str] | None = None
    furnace: Sourced[str] | None = None
    temperature_c: Sourced[float] | None = None
    duration_min: Sourced[float] | None = None
    # A sourced None is a field the record lacks, e.g. "the ledger has no atmosphere field".
    # It is what an expert correction fills in (DEV102).
    atmosphere: Sourced[str | None] | None = None
    heating_rate_c_per_min: Sourced[float] | None = None
    temperature_profile: Sourced[TemperatureProfile] | None = None
    cooling: Sourced[CoolingMode] | None = None
    cooling_rate_c_per_min: Sourced[float] | None = None
    met_target_mass: Sourced[bool] | None = None  # dose: every precursor met its target mass
    mass_mg: Sourced[float] | None = None  # recover: mass recovered; diffract: mass measured


class MaterialContext(Record):
    """What went into the sample and what was done to it. Evidence the agent may see, never
    truth. A declared target is what the chemist intended, not what formed (R8)."""

    elements: Sourced[tuple[Element, ...]]
    elements_basis: Literal["measured", "declared"]
    precursors: tuple[Precursor, ...] = ()
    processing: tuple[ProcessingStep, ...] = ()
    declared_target: Sourced[str] | None = None
    declared_target_stoichiometry: Sourced[str] | None = None  # e.g. "Bi2O3-2GeO2"
    # What the specimen is, as declared, e.g. "Unheated physical mixture of the declared
    # precursors." The U6 agents read it in every package.
    specimen: Sourced[str] | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if [step.order for step in self.processing] != list(range(1, len(self.processing) + 1)):
            raise ValueError("processing steps are numbered 1..n in order")
        return self

    def sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self)).hexdigest()


class MeasurementPatch(Record):
    """Plain values from the corrector. `transition` wraps each changed field in
    Sourced(value, CorrectionSource(this attempt)); there is no source field to send [C3]."""

    radiation: Radiation | None = None
    instrument_profile: SafeID | None = None

    @model_validator(mode="after")
    def changes_something(self) -> Self:
        if self.radiation is None and self.instrument_profile is None:
            raise ValueError("a measurement correction changes at least one field")
        return self


class ProcessingStepInput(Record):
    operation: ProcessingOperation
    temperature_c: float | None = None
    duration_min: float | None = None
    atmosphere: str | None = None
    heating_rate_c_per_min: float | None = None
    cooling: CoolingMode | None = None
    cooling_rate_c_per_min: float | None = None


class PrecursorInput(Record):
    formula: str
    amount: Amount | None = None


class MaterialPatch(Record):
    """Section replacement with plain values. `transition` diffs field by field, so only the
    changed fields get a CorrectionSource and the parent keeps the original (AE5)."""

    elements: tuple[Element, ...] | None = None
    precursors: tuple[PrecursorInput, ...] | None = None
    processing: tuple[ProcessingStepInput, ...] | None = None
    declared_target: str | None = None

    @model_validator(mode="after")
    def changes_something(self) -> Self:
        if all(getattr(self, name) is None for name in type(self).model_fields):
            raise ValueError("a material correction changes at least one section")
        return self


def _corrected(old: dict | None, new: JsonValue, source: dict) -> dict | None:
    """One Sourced field after a correction, in JSON form. Repeating the old value keeps the old
    source; a new value is sourced to the corrector; None removes the value."""
    if new is None:
        return None
    if old is not None and old["value"] == new:
        return old
    return {"value": new, "source": source}


def apply_measurement_patch(
    context: MeasurementContext, patch: MeasurementPatch, source: CorrectionSource
) -> MeasurementContext:
    """A corrected profile is re-resolved, so its confidence is the resolver's. A profile the
    resolver does not know raises ValueError: the store refuses that correction."""
    from types import SimpleNamespace

    from phaseeval.refinement.instrument_resolver import resolve_pattern_instrument_policy

    data = context.model_dump(mode="json")
    src = source.model_dump(mode="json")
    if patch.radiation is not None:
        data["radiation"] = _corrected(data["radiation"], patch.radiation.model_dump(mode="json"), src)
    if patch.instrument_profile is not None:
        pattern = SimpleNamespace(metadata={"instrument_profile": patch.instrument_profile})
        if resolve_pattern_instrument_policy(pattern).profile_confidence != "matched":
            raise ValueError(f"{patch.instrument_profile} is not a known instrument profile")
        assignment = ProfileAssignment(
            instrument_profile=patch.instrument_profile, resolver_confidence="matched"
        )
        data["profile"] = _corrected(data["profile"], assignment.model_dump(mode="json"), src)
    return MeasurementContext.model_validate(data)


def apply_material_patch(
    context: MaterialContext, patch: MaterialPatch, source: CorrectionSource
) -> MaterialContext:
    """Per field: an unchanged value keeps its source. Precursors and steps align by position;
    added ones are all-new, removed ones drop."""
    data = context.model_dump(mode="json")
    new = patch.model_dump(mode="json")
    src = source.model_dump(mode="json")
    for name in ("elements", "declared_target"):
        if new[name] is not None:
            data[name] = _corrected(data[name], new[name], src)
    if new["precursors"] is not None:
        old = data["precursors"]
        data["precursors"] = [
            {
                field: _corrected(old[i][field] if i < len(old) else None, value, src)
                for field, value in precursor.items()
            }
            for i, precursor in enumerate(new["precursors"])
        ]
    if new["processing"] is not None:
        old = data["processing"]
        data["processing"] = [
            {"order": i + 1}
            | {
                field: _corrected(old[i][field] if i < len(old) else None, value, src)
                for field, value in step.items()
            }
            for i, step in enumerate(new["processing"])
        ]
    return MaterialContext.model_validate(data)


class CaseSource(Record):
    """Where the case came from. Rights are not here: they are versioned policy inputs
    (policy.RightsRegistry), because a license check can change while the root cannot."""

    kind: Literal["experimental", "synthetic"]
    collection: SafeID
    citation: str


# Benchmark collections the plan admits only from dev chemical systems (R10, KD9).
BENCHMARK_COLLECTIONS: frozenset[str] = frozenset(
    {"dara_zenodo_17410051", "precursor_genome_zenodo_21285546"}
)
# Never loaded, never published (R10; measurement plan R23).
PROTECTED_COLLECTIONS: frozenset[str] = frozenset({"terminal_bench_science", "xrdbench"})


class SplitAssignment(Record):
    """Copied from the holdout split manifest; the workbench never chooses a split. A lab
    case outside every benchmark is `unassigned`. No test or held-out member exists (R10)."""

    split: Literal["dev", "unassigned"]
    manifest_sha256: Digest | None
    chemical_system: str | None  # e.g. "Bi-C-Li-Ni-O", as the split manifest keys it

    @model_validator(mode="after")
    def dev_cites_the_manifest(self) -> Self:
        assigned = self.manifest_sha256 is not None and self.chemical_system is not None
        if (self.split == "dev") != assigned:
            raise ValueError("a dev assignment cites the split manifest and chemical system")
        return self


class CandidateEntry(Record):
    candidate_id: CandidateId
    formula_sum: str
    elements: tuple[Element, ...]
    cif: ArtifactRef
    source_library_id: str  # e.g. "COD 1530018"; review-only
    eligibility_status: Literal["not_adjudicated", "support_only", "quarantined"]


class SourceFile(Record):
    blob: ArtifactRef  # role raw_export or context_document
    original_name: str  # review-only: lab and dataset filenames often name the phases


class IngestProvenance(Record):
    """Review-only and deterministic (no principal, no clock). Who loaded the root and when is
    a review.LoadEvent annotation."""

    instrument_adapter: AdapterIdentity
    context_adapters: tuple[AdapterIdentity, ...]
    candidate_selection: AdapterIdentity
    sources: tuple[SourceFile, ...]


# Measurement plan R12: a case whose visible context names the phases to find is a
# confirmation task, scored and reported apart from identification (R30).
TaskKind = Literal["identification", "confirmation"]


class CaseManifest(Record):
    """The evidence for one case. There is no field to hold truth."""

    schema_version: Literal["axa.workbench.case.v1"] = "axa.workbench.case.v1"
    case_id: SafeID  # opaque; never a filename, because DARA names encode precursors and amounts
    source: CaseSource
    exposure: Literal["development_exposed", "unexposed"]
    split: SplitAssignment
    task_kind: TaskKind  # the R30 confirmation tag, decided by the TaskKindRule seam
    task: str = Field(min_length=1, max_length=8000)
    pattern: ArtifactRef
    measurement: MeasurementContext
    material: MaterialContext
    candidates: tuple[CandidateEntry, ...] = Field(min_length=1, max_length=2000)
    ingest: IngestProvenance
    # A new physical measurement is new evidence: a new root citing the request it answers.
    follows_request: AttemptRef | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        ids = [c.candidate_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids are unique")
        if self.pattern.role != "observed_pattern":
            raise ValueError("the pattern artifact has role observed_pattern")
        if any(c.cif.role != "candidate_cif" for c in self.candidates):
            raise ValueError("every candidate CIF has role candidate_cif")
        if any(f.blob.role not in ("raw_export", "context_document") for f in self.ingest.sources):
            raise ValueError("source files are raw exports or context documents")
        if self.source.collection in PROTECTED_COLLECTIONS:
            raise ValueError("protected evaluation collections never enter the workbench")
        if self.source.collection in BENCHMARK_COLLECTIONS and (
            self.split.split != "dev" or self.exposure != "development_exposed"
        ):
            raise ValueError("benchmark cases come from dev systems and are development-exposed")
        return self


class RootAttempt(Record):
    schema_version: Literal["axa.attempt.v3"] = "axa.attempt.v3"
    kind: Literal["root"] = "root"
    case: CaseManifest


# ---------------------------------------------------------------------------
# Actors, stamped from the authenticated principal (R1, KTD12)
# ---------------------------------------------------------------------------


class CompletionReceipt(Record):
    """Per-completion model identity (measurement plan R19). Recorded even on mismatch."""

    model_requested: str
    model_observed: str
    response_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None  # None means unknown, never zero (R20)

    @property
    def identity_verified(self) -> bool:
        return self.model_observed == self.model_requested


# The Harbor launcher mints agent tokens with this role only for the task container (R31).
HARBOR_ROLE = "harbor_episode"
Environment = Literal["harbor_container", "host_development"]
ReferenceExposure = Literal["exposed", "unexposed"]
ServiceRole = Literal["importer", "harness", "budget_guard", "sealed_evaluator"]


class AgentActor(Record):
    kind: Literal["agent"] = "agent"
    principal_id: SafeID
    environment: Environment  # host_development episodes are never exported (R31)
    completion: CompletionReceipt | None  # None on lifecycle verbs, or until a RunClose attests it


class HumanActor(Record):
    kind: Literal["human"] = "human"
    principal_id: SafeID
    mode: Literal["intervene"] = "intervene"  # observe and label are deferred; they never act
    revealed_review: Literal[True] = True  # the expert saw the agent's output (R16)
    reference_exposure: ReferenceExposure  # whether the expert had seen the reference (R16)


class ServiceActor(Record):
    kind: Literal["service"] = "service"
    principal_id: SafeID
    role: ServiceRole


Actor = Annotated[AgentActor | HumanActor | ServiceActor, Field(discriminator="kind")]
ActorKind = Literal["agent", "human", "service"]


def environment_for(principal: Principal) -> Environment:
    return "harbor_container" if HARBOR_ROLE in principal.roles else "host_development"


def actor_for(
    principal: Principal,
    *,
    completion: CompletionReceipt | None,
    reference_exposure: ReferenceExposure,
) -> Actor:
    """The only constructor of an attempt's actor. `reference_exposure` comes from the root's
    exposure ledger in the store, defaulting to exposed; the request supplies nothing."""
    match principal.principal_type:
        case PrincipalType.AGENT:
            return AgentActor(
                principal_id=principal.principal_id,
                environment=environment_for(principal),
                completion=completion,
            )
        case PrincipalType.USER:
            return HumanActor(
                principal_id=principal.principal_id, reference_exposure=reference_exposure
            )
        case PrincipalType.SERVICE:
            roles = [role for role in principal.roles if role in get_args(ServiceRole)]
            if len(roles) != 1:
                raise AuthorizationError("a service principal holds exactly one service role")
            return ServiceActor(principal_id=principal.principal_id, role=roles[0])
        case PrincipalType.MODEL_JUDGE:
            raise AuthorizationError("a model judge only labels snapshots; it never acts")
    raise AssertionError(principal.principal_type)


# ---------------------------------------------------------------------------
# Actions. One union for agent and human (R2). Each class declares its own properties, so the
# executor registry, the page's action bar and the MCP tool list derive from the classes.
# ---------------------------------------------------------------------------


class _Action(Record):
    executes: ClassVar[bool]  # runs an executor from executors.default_registry
    terminal: ClassVar[bool] = False  # a leaf: nothing may follow it
    lifecycle: ClassVar[bool] = False  # harness bookkeeping, not a model decision
    authors: ClassVar[frozenset[ActorKind]]


def _distinct(candidates: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates are distinct")
    return candidates


Candidates = Annotated[
    tuple[CandidateId, ...], Field(min_length=1, max_length=5), AfterValidator(_distinct)
]


class AddPhases(_Action):
    """Phase edits are their own action types and execute nothing (R13). The page posts an
    edit and its refine as one gesture that records both attempts."""

    executes: ClassVar[bool] = False
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent", "human"})
    verb: Literal["add_phases"] = "add_phases"
    candidates: Candidates


class RemovePhases(_Action):
    executes: ClassVar[bool] = False
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent", "human"})
    verb: Literal["remove_phases"] = "remove_phases"
    candidates: Candidates


class ReplacePhase(_Action):
    executes: ClassVar[bool] = False
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent", "human"})
    verb: Literal["replace_phase"] = "replace_phase"
    old: CandidateId
    new: CandidateId

    @model_validator(mode="after")
    def changes(self) -> Self:
        if self.old == self.new:
            raise ValueError("a replacement names two different candidates")
        return self


class Refine(_Action):
    """BGMN on the current hypothesis's phase set, synchronously, in an isolated directory
    (KTD3). There is no phase argument: the phase set comes only from phase edits."""

    executes: ClassVar[bool] = True
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent", "human"})
    verb: Literal["refine"] = "refine"
    recipe: Literal["staged_rietveld", "pawley_only"] = "staged_rietveld"


class CorrectMeasurement(_Action):
    """Correct acquisition context (R13). Like a phase edit it records and executes nothing;
    the page posts it and a refine as one gesture, so `refine` is the only executing verb.
    Human only: an agent is not a source of measurement facts."""

    executes: ClassVar[bool] = False
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"human"})
    verb: Literal["correct_measurement"] = "correct_measurement"
    patch: MeasurementPatch


class CorrectMaterial(_Action):
    """Correct material context (R13). BGMN reads none of it, so nothing executes;
    re-analysis is a continuation run started from this attempt (AE5, KD12). Human only:
    processing history comes from a recording source or the expert (R8)."""

    executes: ClassVar[bool] = False
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"human"})
    verb: Literal["correct_material"] = "correct_material"
    patch: MaterialPatch


# --- answers ---------------------------------------------------------------------------------


class NextAction(StrEnum):
    """Typed follow-ups for UNKNOWN, from AGENTS.md (Gates Are Safety Invariants) [C3]."""

    EXPAND_CANDIDATES = "EXPAND_CANDIDATES"
    RUN_POLYMORPH_SEARCH = "RUN_POLYMORPH_SEARCH"
    RUN_PROFILE_MICROSTRUCTURE_PROBE = "RUN_PROFILE_MICROSTRUCTURE_PROBE"
    REQUEST_XRF_OR_EDS = "REQUEST_XRF_OR_EDS"
    REQUEST_HIGH_ANGLE_SCAN = "REQUEST_HIGH_ANGLE_SCAN"
    REQUEST_PRIMARY_TRUTH_SOURCE = "REQUEST_PRIMARY_TRUTH_SOURCE"
    QUARANTINE_CANDIDATE = "QUARANTINE_CANDIDATE"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class Uncertainty(Record):
    identification: str = Field(min_length=1, max_length=4000)
    quantification: str = Field(min_length=1, max_length=4000)


class MeasurementRequest(Record):
    requested_data_type: RequestedDataType
    reason: str = Field(min_length=1)
    expected_discriminator: str = Field(min_length=1)


class Accept(Record):
    """Numbers are not typed in. `basis` is an ancestor attempt whose fit reached a Rietveld
    stage; phases and weight fractions are read from it."""

    verdict: Literal["ACCEPT"] = "ACCEPT"
    basis: AttemptRef
    uncertainty: Uncertainty
    limitations: tuple[str, ...] = ()


class SetOfHypotheses(Record):
    verdict: Literal["SET"] = "SET"
    ranked: tuple[AttemptRef, ...] = Field(min_length=2)
    discriminating_measurement: MeasurementRequest
    uncertainty: Uncertainty
    limitations: tuple[str, ...] = ()


class Unknown(Record):
    verdict: Literal["UNKNOWN"] = "UNKNOWN"
    reason: str = Field(min_length=1)
    next_action: NextAction  # required: UNKNOWN without a typed next action is refused (R15)
    best_partial: AttemptRef | None = None
    uncertainty: Uncertainty
    limitations: tuple[str, ...] = ()


class RequestMoreData(Record):
    verdict: Literal["REQUEST_MORE_DATA"] = "REQUEST_MORE_DATA"
    measurement: MeasurementRequest  # the typed next action (R15)
    ranked: tuple[AttemptRef, ...] = ()
    uncertainty: Uncertainty
    limitations: tuple[str, ...] = ()


Answer = Annotated[
    Accept | SetOfHypotheses | Unknown | RequestMoreData, Field(discriminator="verdict")
]
Verdict = Literal["ACCEPT", "SET", "UNKNOWN", "REQUEST_MORE_DATA"]


class Submit(_Action):
    """An agent's final answer. The expert's conclusion is a labels.py rubric label and sign-off,
    so there is one
    way for each actor to conclude."""

    executes: ClassVar[bool] = False
    terminal: ClassVar[bool] = True
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent"})
    verb: Literal["submit"] = "submit"
    answer: Answer


# --- run lifecycle ----------------------------------------------------------------------------


class Budget(Record):
    max_model_actions: int = Field(ge=1)
    max_wall_seconds: int = Field(ge=1)  # the local model gets at least 1,720 s (KTD5)
    max_executor_seconds: int = Field(ge=1)
    max_cost_usd: float | None = None
    max_consecutive_rejections: int = Field(default=3, ge=1)


class ServiceCheck(Record):
    name: SafeID  # e.g. bgmn_binary, attempt_api, model_endpoint
    ok: bool
    detail: str


class Preflight(Record):
    """Measurement plan R19 checks, measured by the harness before the first model call."""

    context_window_tokens: int
    p95_episode_tokens: int
    output_reserve_tokens: int
    numeric_context_lossless: bool
    runtime_cap_seconds: int
    measured_p95_runtime_seconds: int
    services: tuple[ServiceCheck, ...]

    def failures(self) -> tuple[str, ...]:
        raise NotImplementedError  # TODO: name each failed check; empty means the run may start


class HarnessIdentity(Record):
    name: SafeID  # e.g. harbor-local-mcp, harbor-codex-cli
    version: str
    git_sha: str
    instructions_sha256: Digest


class HarborTaskRef(Record):
    """The Harbor trial a scored episode ran in (R31). Trials mount Harbor task copies of a
    case, never data/workbench/cases/ itself, so a trial binds to its root by content: the
    pattern and every CIF its refines read, by sha256, each equal to the root's (R31)."""

    job_name: SafeID
    trial_id: SafeID
    task_sha256: Digest  # Harbor's task_checksum
    pattern_sha256: Digest
    cif_sha256: dict[CandidateId, Digest] = Field(min_length=1)


class RunStart(_Action):
    """Starts one live agent run. A start whose parent is the root is independent; any other
    start is a continuation (assisted)."""

    executes: ClassVar[bool] = False
    lifecycle: ClassVar[bool] = True
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent"})
    verb: Literal["run_start"] = "run_start"
    harness: HarnessIdentity
    model_requested: str
    budget: Budget
    preflight: Preflight | None  # required inside Harbor; see ChildAttempt.coherent
    harbor: HarborTaskRef | None


InfraCause = Literal[
    "provider_error",
    "host_crash",
    "service_down",
    "model_identity_mismatch",
    "executor_unavailable",
    "executor_hang",
]


class BudgetExhausted(Record):
    kind: Literal["budget_exhausted"] = "budget_exhausted"
    limit: Literal["model_actions", "wall_seconds", "executor_seconds", "cost_usd"]


class MalformedToolUse(Record):
    kind: Literal["malformed_tool_use"] = "malformed_tool_use"
    consecutive_rejections: int


class InfraFailure(Record):
    kind: Literal["infra_failure"] = "infra_failure"
    cause: InfraCause
    detail: str


class Cancelled(Record):
    kind: Literal["cancelled"] = "cancelled"
    by: SafeID


HaltReason = Annotated[
    BudgetExhausted | MalformedToolUse | InfraFailure | Cancelled, Field(discriminator="kind")
]


def halt_validity(reason: HaltReason) -> Literal["invalid", "unsuccessful"]:
    """External failures are invalid, the actor's own are unsuccessful (R11, AE2)."""
    match reason:
        case BudgetExhausted() | MalformedToolUse():
            return "unsuccessful"
        case InfraFailure() | Cancelled():
            return "invalid"
    raise AssertionError(reason)


class Halt(_Action):
    executes: ClassVar[bool] = False
    terminal: ClassVar[bool] = True
    lifecycle: ClassVar[bool] = True
    authors: ClassVar[frozenset[ActorKind]] = frozenset({"agent", "service"})
    verb: Literal["halt"] = "halt"
    reason: HaltReason


Action = Annotated[
    AddPhases
    | RemovePhases
    | ReplacePhase
    | Refine
    | CorrectMeasurement
    | CorrectMaterial
    | Submit
    | RunStart
    | Halt,
    Field(discriminator="verb"),
]
ACTION_CLASSES: tuple[type[_Action], ...] = (
    AddPhases,
    RemovePhases,
    ReplacePhase,
    Refine,
    CorrectMeasurement,
    CorrectMaterial,
    Submit,
    RunStart,
    Halt,
)
# The union and the tuple name the same classes; import fails when one gains a member alone.
if set(get_args(get_args(Action)[0])) != set(ACTION_CLASSES):
    raise RuntimeError("Action and ACTION_CLASSES disagree")


class ImportRun(Record):
    """A pre-generated Harbor trial entering the tree as one imported answer (R28) [C1]. Not a
    member of `Action`: `store.act` cannot parse it, so an import is never presented as an
    action through the shared path. Only `store.import_run` writes it."""

    verb: Literal["import_run"] = "import_run"
    harbor: HarborTaskRef
    # A refinement finished in other software (importers.external) names the software here; a
    # Harbor trial's import (importer.read_harbor_run) never sets it. The review keys its
    # external-import handling on this field, not on a name the run supplies.
    external_source: Literal["gsasii_project", "bgmn_saved_output"] | None = None

    @model_serializer(mode="wrap")
    def _without_an_unset_source(self, handler: Any) -> dict[str, Any]:
        """A Harbor trial's import serializes as it did before this field existed: the key is
        written only when it is set. Every digest (the attempt's, the idempotency claim's, an
        export step's) is taken over these bytes, so records of agent runs made before and after
        the field are the same bytes, and the field is read back as None when absent."""
        data = handler(self)
        if self.external_source is None:
            data.pop("external_source", None)
        return data


class Unparsed(Record):
    """An agent tool call that failed to parse. Written so the agent sees its error and
    malformed-use counts derive from the tree."""

    verb: Literal["unparsed"] = "unparsed"
    raw: JsonValue


AttemptAction = Annotated[Action | ImportRun | Unparsed, Field(discriminator="verb")]


# ---------------------------------------------------------------------------
# Results, replay manifests and outcomes (parsed from executor wire output at the executor)
# ---------------------------------------------------------------------------


class Cell(Record):
    """Lengths in angstrom, angles in degrees. None is a value the parser could not read."""

    a: float | None
    b: float | None
    c: float | None
    alpha: float | None
    beta: float | None
    gamma: float | None


class PhaseFit(Record):
    candidate_id: CandidateId
    weight_percent: float | None
    # BGMN prints the esd on the final .lst `Q<phase>=v+-e` line; the axa parser drops it.
    weight_percent_error: float | None
    cell: Cell
    spacegroup_no: int | None = None


class StageSummary(Record):
    stage: StageName
    rwp_percent: float | None
    gof: float | None
    rexp_percent: float | None
    converged: bool | None
    iterations: int  # recorded, never compared on replay
    # From the .lst header line; parse_lst drops it. Replay compares these exactly (U4).
    measured_points: int | None
    peaks: int | None
    parameters: int | None
    phases: tuple[PhaseFit, ...]
    curve: ArtifactRef | None


class FitSummary(Record):
    stage: StageName
    # Pawley-only fits carry no weight fraction; a stand-in profile restricts the claim (R7).
    quantification: Literal["quantitative", "restricted_profile", "not_quantitative"]
    rwp_percent: float
    gof: float | None
    phases: tuple[PhaseFit, ...]
    curve: ArtifactRef | None


class BackendIdentity(Record):
    name: Literal["bgmn"] = "bgmn"
    version: str
    binary_sha256: Digest | None


class RefineResult(Record):
    kind: Literal["refine"] = "refine"
    stages: tuple[StageSummary, ...]
    final: FitSummary | None
    refinement_failed: bool
    failure_reason: str | None


class RecordedAgent(Record):
    model_id: str
    model_version: str | None
    harness: SafeID
    harness_version: str


class RecordedToolCall(Record):
    """One row of an imported trajectory: shown in the workbench, never replayed as an
    action (R28)."""

    index: int = Field(ge=0)
    tool: str
    arguments_sha256: Digest
    summary: str = Field(max_length=500)  # e.g. "refine CIF006427,CIF007676,CIF016107"
    files: tuple[BlobRef, ...] = ()


class ImportedAnswer(Record):
    """What a trial's answer file reported. The U6 response contract has no verdict field, so
    none is invented (R11): NOT_RECORDED. Reported weights are normalized to percent; the
    unit the trial wrote them in is kept, and a reported null stays None."""

    verdict: Verdict | Literal["NOT_RECORDED"] = "NOT_RECORDED"
    phases: tuple[CandidateId, ...]
    reported_weight_percent: dict[CandidateId, float | None] = Field(default_factory=dict)
    reported_weight_unit: Literal["fraction", "percent"] | None = None
    next_action: NextAction | None = None
    measurement: MeasurementRequest | None = None

    @model_validator(mode="after")
    def typed_next_action(self) -> Self:
        if (self.reported_weight_unit is None) != (not self.reported_weight_percent):
            raise ValueError("reported weights name the unit they were reported in, and only they do")
        if self.verdict == "UNKNOWN" and self.next_action is None:
            raise ValueError("an imported UNKNOWN carries a typed next action")
        if self.verdict == "REQUEST_MORE_DATA" and self.measurement is None:
            raise ValueError("an imported REQUEST_MORE_DATA names the measurement")
        if self.verdict in ("ACCEPT", "SET") and not self.phases:
            raise ValueError("an imported ACCEPT or SET lists phases")
        return self


class ImportedSubmitted(Record):
    end: Literal["submitted"] = "submitted"
    answer: ImportedAnswer
    # The sha256 of the answer file the importer parsed, one of the import's receipts. A trial's
    # drafts can report the same phases and amounts, so the answer's own words are read from
    # these bytes only. None on an import written before this was recorded: its words are
    # unavailable, and no other receipt stands in for them.
    answer_sha256: BlobRef | None = None


class ImportedUnsuccessful(Record):
    end: Literal["unsuccessful"] = "unsuccessful"
    reason: Literal["budget_exhausted", "malformed_tool_use", "no_submission"]


class ImportedInvalid(Record):
    """An infrastructure failure: there is no answer to read as UNKNOWN [C3]."""

    end: Literal["invalid"] = "invalid"
    cause: InfraCause
    detail: str


ImportedRunEnd = Annotated[
    ImportedSubmitted | ImportedUnsuccessful | ImportedInvalid, Field(discriminator="end")
]


class Complete(Record):
    kind: Literal["complete"] = "complete"


class Incomplete(Record):
    """Missing evidence is named, never synthesized (R11)."""

    kind: Literal["incomplete"] = "incomplete"
    missing: tuple[str, ...] = Field(min_length=1)


class ImportedRunResult(Record):
    kind: Literal["imported_run"] = "imported_run"
    agent: RecordedAgent
    budget_wall_seconds: int | None
    wall_seconds: float | None
    cost_usd: float | None  # None means unknown (R20)
    end: ImportedRunEnd
    basis_fit: FitSummary | None  # the refinement the answer rests on, parsed from its files
    trajectory: tuple[RecordedToolCall, ...]
    completeness: Annotated[Complete | Incomplete, Field(discriminator="kind")]

    @model_validator(mode="after")
    def evidence_or_named_gap(self) -> Self:
        answered = isinstance(self.end, ImportedSubmitted) and self.end.answer.verdict == "ACCEPT"
        if answered and self.basis_fit is None and isinstance(self.completeness, Complete):
            raise ValueError("an ACCEPT without its basis fit is marked incomplete")
        return self


ToolResult = Annotated[RefineResult | ImportedRunResult, Field(discriminator="kind")]


class PackageVersion(Record):
    name: str
    version: str


class ThreadSettings(Record):
    omp_num_threads: str | None
    openblas_num_threads: str | None
    mkl_num_threads: str | None


class ReplayEnvironment(Record):
    """The named allowlist of non-secret environment fields a replay manifest freezes (R4).
    extra="forbid": a new field is a reviewed edit here, so no secret can ride along."""

    python_version: str
    os_release: str
    machine: str
    axa_git_sha: str
    axa_source_dirty: bool
    bgmn: BackendIdentity
    packages: tuple[PackageVersion, ...]
    threads: ThreadSettings


RecipeDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{16}$")]


class ResolvedRecipe(Record):
    recipe: Literal["staged_rietveld", "pawley_only"]
    recipe_digest: RecipeDigest  # executors.recipe_digest; includes the thread count
    bgmn_threads: int = Field(ge=1)
    instrument_profile: SafeID
    geq_sha256: Digest
    ger_sha256: Digest
    bgmn_wavelength: str | float  # LAMBDA symbol or SYNCHROTRON value actually written


class PhaseInput(Record):
    candidate_id: CandidateId
    cif_sha256: Digest


class ReplayManifest(Record):
    """Freezes what a re-execution needs (R4). Every field but `environment` must match
    exactly for a re-execution to count as a replay (U4)."""

    pattern_sha256: Digest
    # BGMN STRUC order; reordering one DEV006 set moved Li2CO3 by 0.51 points (KTD3).
    phases: tuple[PhaseInput, ...] = Field(min_length=1)
    recipe: ResolvedRecipe
    # Git blob sha of the parser that wrote the compared numbers. None: a recorded run that
    # does not name it (R11).
    parser_revision: Annotated[str, Field(min_length=1)] | None
    # Version from the run's own .lst header. binary_sha256 None: a recorded run that did not
    # capture its binary (R11).
    backend: BackendIdentity
    environment: ReplayEnvironment | None  # None: a recorded run that did not capture it (R11)


class ExecutionRecord(Record):
    """One BGMN refine as a replay compares it: a live outcome's manifest and result, or a
    recorded run read back from its files."""

    manifest: ReplayManifest
    result: RefineResult


class LiveExecution(Record):
    origin: Literal["live"] = "live"
    manifest: ReplayManifest

    @model_validator(mode="after")
    def environment_captured(self) -> Self:
        """Unknown is allowed only on recorded and imported records."""
        manifest = self.manifest
        if manifest.environment is None:
            raise ValueError("a live execution captures its environment")
        if manifest.parser_revision is None:
            raise ValueError("a live execution names its parser revision")
        if manifest.backend.binary_sha256 is None:
            raise ValueError("a live execution hashes its BGMN binary")
        if manifest.environment.bgmn != manifest.backend:
            raise ValueError("a live execution's backend is the one its environment captured")
        return self


class ImportedRecord(Record):
    """Captured elsewhere and admitted as history [C1]."""

    origin: Literal["imported"] = "imported"
    receipts: tuple[ArtifactRef, ...] = Field(min_length=1)


class RecordedPlayback(Record):
    """The demo fallback: a recorded execution shown in place of a live one, labeled as
    playback on the page and in the export (AE8)."""

    origin: Literal["playback"] = "playback"
    recording_sha256: Digest
    recorded_manifest: ReplayManifest


ExecutionOrigin = Annotated[
    LiveExecution | ImportedRecord | RecordedPlayback, Field(discriminator="origin")
]


class Contamination(Record):
    """A canary from custody appeared in bytes bound for, or produced by, the actor (R9)."""

    canary_sha256: Digest
    channel: Literal["tool_output", "model_output", "transcript"]


class ToolIdentity(Record):
    """What handled the attempt. Present on every child, including store-only verbs (R1)."""

    name: SafeID  # e.g. bgmn_refine, workbench_store, harbor_importer
    version: str
    axa_git_sha: str
    backend: BackendIdentity | None = None


CustodyScan = Literal["clean", "match", "not_configured"]


class Completed(Record):
    status: Literal["completed"] = "completed"
    result: ToolResult
    parent_state_sha256: Digest  # commitment to the derived state it ran from; checked on replay
    artifacts: tuple[ArtifactRef, ...] = ()
    elapsed_seconds: float = Field(ge=0)
    # not_configured: no scanner was injected, so R9 cannot be shown and every training use is
    # unreviewable downstream. Never silently "clean".
    custody_scan: CustodyScan
    contamination: Contamination | None = None
    origin: ExecutionOrigin


class Recorded(Record):
    """A non-executing action (phase edit, material correction, submit, lifecycle): written,
    nothing ran."""

    status: Literal["recorded"] = "recorded"


class Refused(Record):
    """A valid action the store or executor would not run. Codes name observations."""

    status: Literal["refused"] = "refused"
    reason_code: SafeID  # e.g. candidate_not_in_case_library, basis_not_ancestor, no_phases
    detail: str


class InfraFailed(Record):
    status: Literal["infra_failed"] = "infra_failed"
    cause: InfraCause
    detail: str


class Rejected(Record):
    status: Literal["rejected"] = "rejected"
    errors: tuple[str, ...] = Field(min_length=1)


ToolOutcome = Annotated[
    Completed | Recorded | Refused | InfraFailed | Rejected, Field(discriminator="status")
]


# ---------------------------------------------------------------------------
# The attempt
# ---------------------------------------------------------------------------


class ChildAttempt(Record):
    """R1: actor, typed action, inputs by digest, tool and version, parent, time."""

    schema_version: Literal["axa.attempt.v3"] = "axa.attempt.v3"
    kind: Literal["child"] = "child"
    parent: AttemptRef
    root: AttemptRef  # determined by the parent; stored for O(1) ACL checks
    depth: int = Field(ge=1)
    actor: Actor
    action: AttemptAction
    tool: ToolIdentity
    inputs: tuple[BlobRef, ...]  # every blob the handler read (pattern, CIFs, curves, receipts)
    rationale: FreeText = Field(default="", max_length=20_000)  # never part of replay
    outcome: ToolOutcome
    idempotency_key: SafeID  # claimed under store.IdempotencyScope
    created_at_utc: datetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        action, actor, outcome = self.action, self.actor, self.outcome
        if isinstance(action, Unparsed):
            if not (isinstance(outcome, Rejected) and isinstance(actor, AgentActor)):
                raise ValueError("only an agent's unparsed call exists, and it is rejected")
            return self
        if isinstance(outcome, Rejected):
            raise ValueError("a parsed action is never rejected")
        if isinstance(action, ImportRun):
            if not (isinstance(actor, ServiceActor) and actor.role == "importer"):
                raise ValueError("only the importer service writes an imported run")
            if not (
                isinstance(outcome, Completed)
                and isinstance(outcome.origin, ImportedRecord)
                and isinstance(outcome.result, ImportedRunResult)
            ):
                raise ValueError("an imported run completes with an imported origin and result")
            return self
        cls = type(action)
        if actor.kind not in cls.authors:
            raise ValueError(f"{actor.kind} may not author {action.verb}")
        if isinstance(actor, ServiceActor) and not (
            isinstance(action, Halt) and actor.role in ("budget_guard", "harness")
        ):
            raise ValueError("services author only halts, as budget guard or harness")
        if isinstance(outcome, Completed) and not cls.executes:
            raise ValueError(f"{action.verb} executes nothing, so it is recorded, not completed")
        if isinstance(outcome, Recorded) and cls.executes:
            raise ValueError(f"{action.verb} executes, so it cannot be merely recorded")
        if isinstance(outcome, Completed) and not isinstance(outcome.result, RefineResult):
            raise ValueError("a refine completes with a refine result")
        if isinstance(outcome, Completed) and isinstance(outcome.origin, ImportedRecord):
            raise ValueError("an imported origin belongs only to an imported run")
        if isinstance(action, RunStart) and isinstance(actor, AgentActor):
            harbor = actor.environment == "harbor_container"
            if harbor and (action.harbor is None or action.preflight is None):
                raise ValueError("a Harbor run names its task and passes preflight")
            if not harbor and action.harbor is not None:
                raise ValueError("a host session cannot claim a Harbor task")
        return self


Attempt = Annotated[RootAttempt | ChildAttempt, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Derived analysis state (never stored)
# ---------------------------------------------------------------------------


class Hypothesis(Record):
    phases: tuple[CandidateId, ...]
    # The attempt whose fit is current. None after a phase edit, until the next refine.
    fit_basis: AttemptRef | None
    fit: FitSummary | None
    fit_origin: Literal["live", "imported", "playback"] | None
    measurement_sha256: Digest | None  # measurement context the fit ran under
    material_sha256: Digest  # material context in force when the phase set was chosen


class AnalysisState(Record):
    root: AttemptRef
    measurement: MeasurementContext
    material: MaterialContext
    hypothesis: Hypothesis | None

    @property
    def fit_stale(self) -> bool:
        """The fit ran under a measurement context that has since been corrected."""
        h = self.hypothesis
        return h is not None and h.fit is not None and h.measurement_sha256 != self.measurement.sha256()

    @property
    def analysis_stale(self) -> bool:
        """Material context changed after this phase set was chosen: re-analysis is due."""
        return self.hypothesis is not None and self.hypothesis.material_sha256 != self.material.sha256()


Step = tuple[AttemptDigest, ChildAttempt]


def initial_state(root_digest: AttemptDigest, root: RootAttempt) -> AnalysisState:
    return AnalysisState(
        root=root_digest, measurement=root.case.measurement, material=root.case.material, hypothesis=None
    )


def edited_phases(
    phases: tuple[CandidateId, ...], edit: AddPhases | RemovePhases | ReplacePhase
) -> tuple[CandidateId, ...]:
    """BGMN STRUC order is part of the execution (KTD3): an add appends in the order given, a
    remove keeps the rest in order, a replacement substitutes in place."""
    match edit:
        case AddPhases(candidates=added):
            return phases + tuple(c for c in added if c not in phases)
        case RemovePhases(candidates=removed):
            return tuple(c for c in phases if c not in removed)
        case ReplacePhase(old=old, new=new):
            return tuple(new if c == old else c for c in phases)
    raise AssertionError(edit)


def transition(state: AnalysisState, step: Step) -> AnalysisState:
    """Pure reducer. Only phase edits, completed refines, imports and corrections change state."""
    digest, attempt = step
    hypothesis = state.hypothesis
    correction = CorrectionSource(attempt=digest, principal_id=attempt.actor.principal_id)
    match attempt.action, attempt.outcome:
        case (AddPhases() | RemovePhases() | ReplacePhase()) as edit, Recorded():
            hypothesis = Hypothesis(
                phases=edited_phases(hypothesis.phases if hypothesis else (), edit),
                fit_basis=None,
                fit=None,
                fit_origin=None,
                measurement_sha256=None,
                material_sha256=state.material.sha256(),
            )
        case Refine(), Completed(result=RefineResult() as result, origin=origin) if hypothesis:
            hypothesis = hypothesis.model_copy(
                update={
                    "fit_basis": digest,
                    "fit": result.final,
                    "fit_origin": origin.origin,
                    "measurement_sha256": state.measurement.sha256(),
                }
            )
        case CorrectMeasurement(patch=patch), Recorded():
            measurement = apply_measurement_patch(state.measurement, patch, correction)
            return state.model_copy(update={"measurement": measurement})
        case CorrectMaterial(patch=patch), Recorded():
            material = apply_material_patch(state.material, patch, correction)
            return state.model_copy(update={"material": material})
        case ImportRun(), Completed(
            result=ImportedRunResult(end=ImportedSubmitted(answer=answer), basis_fit=fit)
        ):
            hypothesis = Hypothesis(
                phases=answer.phases,
                fit_basis=digest if fit else None,
                fit=fit,
                fit_origin="imported" if fit else None,
                measurement_sha256=state.measurement.sha256() if fit else None,
                material_sha256=state.material.sha256(),
            )
        case _:
            return state
    return state.model_copy(update={"hypothesis": hypothesis})


def state_at(root: tuple[AttemptDigest, RootAttempt], steps: tuple[Step, ...]) -> AnalysisState:
    state = initial_state(*root)
    for step in steps:
        state = transition(state, step)
    return state


def state_sha256(state: AnalysisState) -> Digest:
    """What Completed.parent_state_sha256 commits to: the derived state an execution ran from."""
    return hashlib.sha256(canonical_bytes(state)).hexdigest()
