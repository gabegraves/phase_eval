"""Versioned GIWAXS development evidence; scientific qualification stays unavailable."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

SafeId = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')]
Sha256 = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Text = Annotated[str, Field(min_length=1, max_length=4000)]


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    @model_validator(mode='before')
    @classmethod
    def exact_literals(cls, value):
        if isinstance(value, dict):
            if 'schema_version' in value and type(value['schema_version']) is not int:
                raise ValueError('schema_version must be an integer')
            for key, expected in (('scientific_qualified', False), ('development_only', True)):
                if key in value and value[key] is not expected:
                    raise ValueError(f'{key} must be literal {expected}')
        return value


class FileRef(Contract):
    file: Text
    sha256: Sha256


class Source(Contract):
    doi: Text
    license: Text
    attribution: Text
    archive_sha256: Sha256


ORIENTATION = ('pygix sample orientation: 1=horizontal plane, +qz bottom-to-top; '
               '2=vertical plane, +qz left-to-right; 3=horizontal plane, +qz top-to-bottom; '
               '4=vertical plane, +qz right-to-left in detector image coordinates.')


class Acquisition(Contract):
    incident_angle: float = Field(ge=0, le=90, description='Incidence angle in degrees, not radians.')
    sample_orientation: int = Field(ge=1, le=4, description=ORIENTATION)
    tilt_angle: float = Field(ge=-180, le=180, description='Sample tilt angle in degrees, not radians.')


class GiwaxsCase(Contract):
    case_id: SafeId
    raw: FileRef
    geometry: FileRef = Field(description='PONI geometry: rotations rot1/rot2/rot3 in radians; distance, poni1/poni2, pixels and wavelength in meters.')
    mask: FileRef
    source: Source
    context: dict[str, Text]
    unknowns: list[Text] = Field(min_length=1)
    acquisition: Acquisition
    development_only: Literal[True] = True


class GiwaxsManifest(Contract):
    schema_version: Literal[1]
    task_kind: Literal['giwaxs']
    cases: list[GiwaxsCase] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_cases(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError('case IDs must be distinct')
        return self


class Coordinate(Contract):
    definition: Literal['q=2pi/d', 's=1/d']
    unit: Literal['nm^-1'] = 'nm^-1'
    axis: Literal['qr', 'qz']


class Runtime(Contract):
    python: Text
    pyFAI: Text
    pygix: Text
    numpy: Text
    fabio: Text
    scipy: Text
    silx: Text


class ArtifactReceipt(Contract):
    schema_version: Literal[1] = 1
    task_kind: Literal['giwaxs'] = 'giwaxs'
    case_id: SafeId
    kind: Literal['map', 'profile', 'comparison']
    handle: Text
    parents: dict[str, Sha256]
    inputs: dict[str, Sha256]
    outputs: dict[str, Sha256]
    parameters: dict
    coordinate: Coordinate
    runtime: Runtime
    warnings: list[Text]
    scientific_reward: None = None
    scientific_qualified: Literal[False] = False


class ProfileSelection(Contract):
    kind: Literal['profile']
    lower: float = Field(description='Inclusive minimum in the profile output coordinate convention and nm^-1.')
    upper: float = Field(description='Inclusive maximum in the profile output coordinate convention and nm^-1.')


class MapSelection(Contract):
    kind: Literal['map']
    qr_min: float = Field(description='Inclusive minimum qr, q=2pi/d in nm^-1.')
    qr_max: float = Field(description='Inclusive maximum qr, q=2pi/d in nm^-1.')
    qz_min: float = Field(description='Inclusive minimum qz, q=2pi/d in nm^-1.')
    qz_max: float = Field(description='Inclusive maximum qz, q=2pi/d in nm^-1.')


class DetectorSelection(Contract):
    kind: Literal['detector']
    row_min: int = Field(ge=0, description='Inclusive zero-based detector row.')
    row_max: int = Field(ge=0, description='Inclusive zero-based detector row.')
    column_min: int = Field(ge=0, description='Inclusive zero-based detector column.')
    column_max: int = Field(ge=0, description='Inclusive zero-based detector column.')


class ReadEvidence(Contract):
    handle: Text | None = None
    start: int = Field(default=0, ge=0, description='Page offset within the selected rows, or native rows/text when selection is omitted.')
    limit: int | None = Field(default=None, ge=1, le=2000, description='At most 100 rows for numeric evidence or 2000 characters for text. Omit to use the appropriate default.')
    selection: Annotated[ProfileSelection | MapSelection | DetectorSelection, Field(discriminator='kind')] | None = Field(
        default=None, json_schema_extra={'type': ['object', 'null']}, description='Optional inclusive numeric range/region, within native bounds. Repeat the identical selection with next_start; row_indexes remain native flat indexes. Text cannot be selected.')


class Transform(Contract):
    label: SafeId
    mask_policy: Literal['supplied', 'omit']
    incident_angle: float = Field(ge=0, le=90, description='Incidence angle in degrees, not radians.')
    sample_orientation: int = Field(ge=1, le=4, description=ORIENTATION)
    tilt_angle: float = Field(ge=-180, le=180, description='Sample tilt angle in degrees, not radians.')
    distance_scale: float = Field(default=1.0, gt=0, le=10)


class Extract(Contract):
    label: SafeId
    map_handle: Text
    axis: Literal['qr', 'qz'] = Field(default='qr', description='OUTPUT axis. qr averages a qz strip; qz averages a qr strip.')
    lower: float = Field(description='Lower bound of the ORTHOGONAL averaging strip in q=2pi/d, nm^-1, even when output is s=1/d.')
    upper: float = Field(description='Upper bound of the ORTHOGONAL averaging strip in q=2pi/d, nm^-1; nearest-bin upper-exclusive.')
    mode: Literal['notebook', 'coverage_aware']
    coordinate_definition: Literal['q=2pi/d', 's=1/d'] = 'q=2pi/d'

    @model_validator(mode='after')
    def interval(self):
        if self.lower >= self.upper:
            raise ValueError('ROI must have increasing bounds')
        return self


class Compare(Contract):
    label: SafeId
    left: Text
    right: Text
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode='after')
    def interval(self):
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError('comparison interval must increase')
        return self


class Claim(Contract):
    text: Text
    evidence: list[Text] = Field(min_length=1)


class Submission(Contract):
    schema_version: Literal[1] = 1
    status: Literal['supported', 'partial']
    claims: list[Claim]
    limitations: list[Text] = Field(min_length=1)
    evidence_needed: list[Text] = Field(min_length=1)
    scientific_reward: None = None
    scientific_qualified: Literal[False] = False

    @model_validator(mode='after')
    def supported_claims(self):
        if self.status == 'supported' and not self.claims:
            raise ValueError('supported submission requires cited claims')
        return self


ACTION_TYPES = {'read_evidence': ReadEvidence, 'transform': Transform, 'extract': Extract,
                'compare': Compare, 'submit': Submission}
TOOLS = [{'type': 'function', 'function': {'name': name,
          'description': description, 'parameters': ACTION_TYPES[name].model_json_schema()}}
         for name, description in {
             'read_evidence': 'Read exact text or numeric pages with compact summaries; omit handle for catalog. Optional selection targets an inclusive profile range or 2D map/detector region; paginate with identical selection and next_start.',
             'transform': 'Transform raw data using supplied PONI (rotations radians, lengths meters); incident/tilt angles are degrees and sample_orientation follows pygix detector conventions. Alternatives are exploratory assumptions.',
             'extract': 'Extract a profile along the OUTPUT axis by averaging an ORTHOGONAL nearest-bin, upper-exclusive strip. Bounds are q=2pi/d in nm^-1 even for s=1/d output. Inspect separate output extent and usable support.',
             'compare': 'Compare profiles over common covered support, without bridging coverage gaps.',
             'submit': 'Submit cited claims or a partial result; scientific validity remains unassessed.',
         }.items()]


class SubmissionReceipt(Submission):
    case_id: SafeId
    citations: dict[str, Sha256]


class EvidenceFile(Contract):
    root: Text
    file: Text
    kind: Literal['detector', 'text', 'map', 'profile']
    sha256: Sha256


class EvidenceRegistry(Contract):
    schema_version: Literal[1] = 1
    task_kind: Literal['giwaxs'] = 'giwaxs'
    case_id: SafeId
    manifest_sha256: Sha256
    inputs_root: Text
    inputs: dict[str, Sha256]
    files: dict[str, EvidenceFile]
    artifacts: dict[str, ArtifactReceipt]
    scientific_reward: None = None
    scientific_qualified: Literal[False] = False


class MethodEvent(Contract):
    event_id: Text
    turn_index: int = Field(ge=0)
    call_index: int = Field(ge=0)
    function_name: Text
    arguments: dict | list | str | int | float | bool | None
    transport_status: Literal['started', 'completed', 'failed']
    outcome: Literal['success', 'rejected', 'failure'] | None
    shown_observation: str | None
    executor_observation: str | None
    executor_sha256: Sha256 | None
    clipped: bool


class MethodPacket(Contract):
    schema_version: Literal[1] = 1
    task_kind: Literal['giwaxs'] = 'giwaxs'
    audience: Literal['owner_development'] = 'owner_development'
    case_id: str  # Failed setup may have an invalid attempted identifier; it is display data only.
    case: GiwaxsCase | None
    receipt: dict
    trajectory: dict
    submission: SubmissionReceipt | None
    events: list[MethodEvent]
    message_events: dict[str, int]
    artifacts: dict[str, ArtifactReceipt]
    bound_files: dict[str, EvidenceFile]
    integrity_checks: list[Text]
    unassessed_questions: list[Text]
    scientific_reward: None = None
    scientific_qualified: Literal[False] = False


class MethodAnnotation(Contract):
    schema_version: Literal[1] = 1
    protocol_version: Literal['giwaxs-development-v1'] = 'giwaxs-development-v1'
    packet_sha256: Sha256
    reviewer_source: Literal['human_unqualified', 'model_judge']
    reviewer_identity: Text
    reviewer_model: Text | None = None
    verdict: Literal['pass', 'fail', 'defer']
    note: Text
    evidence: list[Text] = Field(default_factory=list, max_length=100)
    earliest_error: Text | None = None
    observed_repair: Text | None = None
    repair_status: Literal['not_applicable', 'repaired', 'unrepaired', 'unknown'] = 'not_applicable'
    proposed_repair: str = Field(default='', max_length=4000)
    scientific_reward: None = None
    scientific_qualified: Literal[False] = False

    @model_validator(mode='after')
    def unqualified_provenance(self):
        if (self.reviewer_source == 'model_judge') != (self.reviewer_model is not None):
            raise ValueError('only a model judge must name its model')
        if any(not value.strip() for value in (self.note, self.reviewer_identity, self.reviewer_model or 'human')):
            raise ValueError('review explanation and identity must not be blank')
        return self
