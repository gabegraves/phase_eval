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


class Acquisition(Contract):
    incident_angle: float = Field(ge=0, le=90)
    sample_orientation: int = Field(ge=1, le=4)
    tilt_angle: float = Field(ge=-180, le=180)


class GiwaxsCase(Contract):
    case_id: SafeId
    raw: FileRef
    geometry: FileRef
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


class ReadEvidence(Contract):
    handle: Text | None = None
    start: int = Field(default=0, ge=0)
    limit: int | None = Field(default=None, ge=1, le=2000)


class Transform(Contract):
    label: SafeId
    mask_policy: Literal['supplied', 'omit']
    incident_angle: float = Field(ge=0, le=90)
    sample_orientation: int = Field(ge=1, le=4)
    tilt_angle: float = Field(ge=-180, le=180)
    distance_scale: float = Field(default=1.0, gt=0, le=10)


class Extract(Contract):
    label: SafeId
    map_handle: Text
    axis: Literal['qr', 'qz'] = 'qr'
    lower: float
    upper: float
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
             'read_evidence': 'Read exact registered text or native numeric pages; omit handle for catalog.',
             'transform': 'Transform raw detector data using supplied PONI and explicit mask/geometry choices.',
             'extract': 'Extract a nearest-bin, upper-exclusive mean strip; bounds are q in nm^-1.',
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
