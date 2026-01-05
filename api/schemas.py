"""
Pydantic schemas for API request/response validation.

Provides comprehensive validation for all X-Ray API endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class RunStatus(str, Enum):
    """Status of a run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Status of a step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# Nested Schemas
# ============================================================================

class RunMetadata(BaseModel):
    """Metadata about a run."""
    version: Optional[str] = None
    environment: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom: Dict[str, Any] = Field(default_factory=dict)


class RunContext(BaseModel):
    """Contextual information about a run."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class FinalOutput(BaseModel):
    """Final output of a run."""
    success: bool = True
    result_ids: List[str] = Field(default_factory=list)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    error: Optional[str] = None


class RunSummary(BaseModel):
    """Summary statistics for a run."""
    total_candidates_considered: int = 0
    total_candidates_filtered: int = 0
    final_output_count: int = 0
    total_filters_applied: int = 0
    rejection_rate: float = 0.0
    avg_step_duration_ms: float = 0.0


class Candidate(BaseModel):
    """A candidate item considered during a step."""
    id: str
    score: Optional[float] = None
    source: Optional[str] = None
    rank: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class FilterMetrics(BaseModel):
    """Metrics for a filter application."""
    input_count: int = 0
    output_count: int = 0
    removed_count: Optional[int] = None
    execution_time_ms: Optional[int] = None


class StepFilter(BaseModel):
    """A filter applied during a step."""
    type: str
    name: Optional[str] = None
    id: Optional[str] = None
    order: Optional[int] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[FilterMetrics] = None
    removed_sample: List[str] = Field(default_factory=list)
    removed_reasons: Dict[str, str] = Field(default_factory=dict)


class StepInputs(BaseModel):
    """Inputs to a step."""
    data: Dict[str, Any] = Field(default_factory=dict)
    schema_version: Optional[str] = None
    source_step_id: Optional[str] = None


class StepOutputs(BaseModel):
    """Outputs from a step."""
    count: int = 0
    result_ids: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    schema_version: Optional[str] = None
    storage_ref: Optional[str] = None


class StepCandidates(BaseModel):
    """Candidates considered in a step."""
    total_count: int = 0
    items: List[Candidate] = Field(default_factory=list)
    schema_version: Optional[str] = None
    storage_ref: Optional[str] = None


class ReasoningFactor(BaseModel):
    """A factor contributing to a decision."""
    name: str
    value: float
    weight: Optional[float] = None
    contribution: Optional[float] = None
    description: Optional[str] = None


class StepReasoning(BaseModel):
    """Reasoning and decision explanation for a step."""
    algorithm: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    factors: List[ReasoningFactor] = Field(default_factory=list)
    decision_path: List[Dict[str, Any]] = Field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    model_info: Optional[Dict[str, Any]] = None


class StepDebug(BaseModel):
    """Debug information for a step."""
    logs: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    trace: Dict[str, Any] = Field(default_factory=dict)


class StepTiming(BaseModel):
    """Timing information for a step."""
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


# ============================================================================
# Run Schemas
# ============================================================================

class RunCreate(BaseModel):
    """Schema for creating a new run."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the run")
    pipeline: Optional[str] = Field(None, max_length=100, description="Pipeline type")
    metadata: Optional[RunMetadata] = None
    context: Optional[RunContext] = None
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()


class RunUpdate(BaseModel):
    """Schema for updating a run (e.g., completing or failing)."""
    status: Optional[RunStatus] = None
    final_output: Optional[FinalOutput] = None
    summary: Optional[RunSummary] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


class RunResponse(BaseModel):
    """Schema for run response."""
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    
    run_id: str
    name: str
    pipeline: Optional[str] = None
    status: RunStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="run_metadata")
    context: Optional[Dict[str, Any]] = None
    final_output: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    step_count: int = 0


class RunWithStepsResponse(RunResponse):
    """Schema for run response including all steps."""
    steps: List["StepResponse"] = Field(default_factory=list)


# ============================================================================
# Step Schemas
# ============================================================================

class StepCreate(BaseModel):
    """Schema for creating a new step."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the step")
    step_type: Optional[str] = Field(None, max_length=100)
    status: StepStatus = StepStatus.SUCCESS
    parent_step_id: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Complex data
    inputs: Optional[StepInputs] = None
    candidates: Optional[StepCandidates] = None
    filters: List[StepFilter] = Field(default_factory=list)
    outputs: Optional[StepOutputs] = None
    reasoning: Optional[StepReasoning] = None
    debug: Optional[StepDebug] = None
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()


class StepResponse(BaseModel):
    """Schema for step response."""
    model_config = ConfigDict(from_attributes=True)
    
    step_id: str
    run_id: str
    step_index: int
    name: str
    step_type: Optional[str] = None
    status: StepStatus
    parent_step_id: Optional[str] = None
    child_step_ids: Optional[List[str]] = None
    
    # Timing
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Complex data (stored as JSON)
    inputs: Optional[Dict[str, Any]] = None
    candidates: Optional[Dict[str, Any]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    outputs: Optional[Dict[str, Any]] = None
    reasoning: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None


# ============================================================================
# Query Schemas
# ============================================================================

class RunQuery(BaseModel):
    """Schema for querying runs."""
    status: Optional[RunStatus] = None
    pipeline: Optional[str] = None
    name_contains: Optional[str] = Field(None, max_length=255)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_duration_ms: Optional[int] = Field(None, ge=0)
    max_duration_ms: Optional[int] = Field(None, ge=0)
    
    # Step characteristics (from summary)
    rejection_rate_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    rejection_rate_max: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_candidates: Optional[int] = Field(None, ge=0)
    
    # Pagination
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    
    # Sorting
    order_by: str = Field(default="created_at")
    order_desc: bool = True
    
    # Context filters
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Metadata filters
    environment: Optional[str] = None
    tags: Optional[List[str]] = None


class RunListResponse(BaseModel):
    """Schema for paginated run list response."""
    runs: List[RunResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None


# Forward reference resolution
RunWithStepsResponse.model_rebuild()
