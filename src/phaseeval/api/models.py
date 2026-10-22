"""Pydantic models for FastAPI endpoints."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pymatgen.core.periodic_table import Element


class UploadResponse(BaseModel):
    """Response from file upload."""

    file_id: str
    filename: str
    num_points: int
    two_theta_range: tuple[float, float]
    metadata: dict[str, Any]


class AnalysisConfig(BaseModel):
    """Configuration for analysis pipeline."""

    backend: str = Field(default="bgmn", description="Refinement backend (bgmn/gsas)")
    max_phases: int = Field(default=3, ge=1, le=5, description="Maximum phases to identify")
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    top_k: int = Field(default=5, description="Number of top hypotheses to return")
    enable_calibration: bool = Field(default=True, description="Enable temperature scaling")
    enable_quantification: bool = Field(default=True, description="Enable phase quantification")
    known_elements: list[str] | None = None
    instrument_profile: str | None = None

    @field_validator("known_elements")
    @classmethod
    def validate_known_elements(cls, known_elements: list[str] | None) -> list[str] | None:
        """Reject element symbols that cannot define a chemistry prior."""
        if known_elements is None:
            return None
        for symbol in known_elements:
            try:
                Element(symbol)
            except ValueError as exc:
                raise ValueError(f"Unknown or invalid element symbol: {symbol}") from exc
        return known_elements


class AnalysisRequest(BaseModel):
    """Request for pattern analysis."""

    file_ids: list[str]
    config: AnalysisConfig | None = None


class PhaseResult(BaseModel):
    """Single phase result."""

    phase_id: str
    composition: str
    weight_percent: float | None = None
    confidence: float
    rwp: float | None = None


class SetPhaseResult(BaseModel):
    """Reference to a phase within a SET payload."""

    phase_id: str
    composition: str | None = None


class HypothesisPhaseResult(BaseModel):
    """Per-phase details within a structured SET hypothesis."""

    phase_id: str
    composition: str | None = None
    conditional_weight_fraction: float | None = None


class HypothesisResult(BaseModel):
    """Multi-hypothesis result for a pattern."""

    rank: int
    name: str | None = None
    phase_ids: list[str] = Field(default_factory=list)
    phases: list[HypothesisPhaseResult] = Field(default_factory=list)
    conditional_weight_fractions: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    posterior_probability: float | None = None
    fom: float | None = None
    rwp: float | None = None
    bayes_factor_vs_best: float | None = None
    is_distinguishable: bool | None = None
    pawley_fallback: bool | None = None


class SetPayloadResult(BaseModel):
    """Structured SET payload surfaced through the API."""

    common_core_phases: list[SetPhaseResult] = Field(default_factory=list)
    differentiator_phases: list[SetPhaseResult] = Field(default_factory=list)
    hypotheses: list[HypothesisResult] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Analysis result for a single pattern."""

    sample_id: str
    file_id: str | None = None
    phases: list[PhaseResult]
    num_phases: int
    top_phase: str | None = None
    needs_review: bool
    review_reason: str | None = None
    confidence: float | None = None
    rwp: float | None = None
    outcome: str = "ok"
    reason_codes: list[str] = Field(default_factory=list)
    set_payload: SetPayloadResult | None = None
    terminal_route: dict[str, Any] | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class PipelineStateEnum(str, Enum):  # noqa: UP042
    """Pipeline state enum matching orchestrator states."""

    INIT = "INIT"
    LOADING = "LOADING"
    PREPROCESSING = "PREPROCESSING"
    RETRIEVAL = "RETRIEVAL"
    SCORING = "SCORING"
    REFINEMENT = "REFINEMENT"
    QUANTIFICATION = "QUANTIFICATION"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class BatchStatus(BaseModel):
    """Status of a batch analysis job."""

    job_id: str
    total: int
    completed: int
    failed: int
    current_file: str | None = None
    state: PipelineStateEnum
    progress: float = Field(ge=0.0, le=1.0)
    error_message: str | None = None


class PatternData(BaseModel):
    """Uploaded pattern data."""

    two_theta: list[float]
    intensities: list[float]
    sample_id: str
    metadata: dict[str, Any]


class ExportFormat(str, Enum):  # noqa: UP042
    """Export format options."""

    JSON = "json"
    CSV = "csv"
    ZIP = "zip"


class ExportRequest(BaseModel):
    """Request for exporting results."""

    job_id: str
    format: ExportFormat = ExportFormat.JSON
    include_provenance: bool = True


class ExportResponse(BaseModel):
    """Response from export."""

    download_url: str
    filename: str
    size_bytes: int


class PhaseSearchQuery(BaseModel):
    """Query for phase database search."""

    query: str
    database: str = Field(default="cod", description="Database to search (cod/mp)")
    limit: int = Field(default=50, ge=1, le=500)


class PhaseInfo(BaseModel):
    """Phase information from database."""

    phase_id: str
    composition: str
    space_group: str | None = None
    crystal_system: str | None = None
    database: str
    cif_available: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    pipeline_ready: bool
    num_uploaded_patterns: int
    num_active_jobs: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    status_code: int
