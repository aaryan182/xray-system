"""
Data models for the X-Ray SDK.

These models represent the core entities in the X-Ray debugging system:
runs, steps, candidates, filters, and reasoning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


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


@dataclass
class RunMetadata:
    """Metadata about a run."""
    version: Optional[str] = None
    environment: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.version: result["version"] = self.version
        if self.environment: result["environment"] = self.environment
        if self.tags: result["tags"] = self.tags
        if self.custom: result["custom"] = self.custom
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["RunMetadata"]:
        if not data: return None
        return cls(version=data.get("version"), environment=data.get("environment"),
                   tags=data.get("tags", []), custom=data.get("custom", {}))


@dataclass
class RunContext:
    """Contextual information about a run."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.user_id: result["user_id"] = self.user_id
        if self.session_id: result["session_id"] = self.session_id
        if self.trace_id: result["trace_id"] = self.trace_id
        if self.request_id: result["request_id"] = self.request_id
        if self.custom: result["custom"] = self.custom
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["RunContext"]:
        if not data: return None
        return cls(user_id=data.get("user_id"), session_id=data.get("session_id"),
                   trace_id=data.get("trace_id"), request_id=data.get("request_id"),
                   custom=data.get("custom", {}))


@dataclass
class FinalOutput:
    """Final output of a run."""
    success: bool = True
    result_ids: List[str] = field(default_factory=list)
    result_data: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"success": self.success}
        if self.result_ids: result["result_ids"] = self.result_ids
        if self.result_data: result["result_data"] = self.result_data
        if self.summary: result["summary"] = self.summary
        if self.error: result["error"] = self.error
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["FinalOutput"]:
        if not data: return None
        return cls(success=data.get("success", True), result_ids=data.get("result_ids", []),
                   result_data=data.get("result_data", {}), summary=data.get("summary"),
                   error=data.get("error"))


@dataclass
class Candidate:
    """A candidate item considered during a step."""
    id: str
    score: Optional[float] = None
    source: Optional[str] = None
    rank: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    rejection_reason: Optional[str] = None
    rejected_by_filter: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id}
        if self.score is not None: result["score"] = self.score
        if self.source: result["source"] = self.source
        if self.rank is not None: result["rank"] = self.rank
        if self.data: result["data"] = self.data
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Candidate":
        return cls(id=data["id"], score=data.get("score"), source=data.get("source"),
                   rank=data.get("rank"), data=data.get("data", {}))


@dataclass
class FilterMetrics:
    """Metrics for a filter application."""
    input_count: int = 0
    output_count: int = 0
    removed_count: Optional[int] = None
    execution_time_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"input_count": self.input_count, "output_count": self.output_count}
        if self.removed_count is not None: result["removed_count"] = self.removed_count
        if self.execution_time_ms is not None: result["execution_time_ms"] = self.execution_time_ms
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["FilterMetrics"]:
        if not data: return None
        return cls(input_count=data.get("input_count", 0), output_count=data.get("output_count", 0),
                   removed_count=data.get("removed_count"), execution_time_ms=data.get("execution_time_ms"))


@dataclass
class StepFilter:
    """A filter applied during a step."""
    type: str
    name: Optional[str] = None
    id: Optional[str] = None
    order: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[FilterMetrics] = None
    removed_sample: List[str] = field(default_factory=list)
    removed_reasons: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.name: result["name"] = self.name
        if self.id: result["id"] = self.id
        if self.order is not None: result["order"] = self.order
        if self.config: result["config"] = self.config
        if self.metrics: result["metrics"] = self.metrics.to_dict()
        if self.removed_sample: result["removed_sample"] = self.removed_sample
        if self.removed_reasons: result["removed_reasons"] = self.removed_reasons
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StepFilter":
        return cls(type=data["type"], name=data.get("name"), id=data.get("id"),
                   order=data.get("order"), config=data.get("config", {}),
                   metrics=FilterMetrics.from_dict(data.get("metrics")),
                   removed_sample=data.get("removed_sample", []),
                   removed_reasons=data.get("removed_reasons", {}))


@dataclass
class StepInputs:
    """Inputs to a step."""
    data: Dict[str, Any] = field(default_factory=dict)
    schema_version: Optional[str] = None
    source_step_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.data: result["data"] = self.data
        if self.schema_version: result["schema_version"] = self.schema_version
        if self.source_step_id: result["source_step_id"] = self.source_step_id
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["StepInputs"]:
        if not data: return None
        return cls(data=data.get("data", {}), schema_version=data.get("schema_version"),
                   source_step_id=data.get("source_step_id"))


@dataclass
class StepOutputs:
    """Outputs from a step."""
    count: int = 0
    result_ids: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    schema_version: Optional[str] = None
    storage_ref: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"count": self.count}
        if self.result_ids: result["result_ids"] = self.result_ids
        if self.data: result["data"] = self.data
        if self.schema_version: result["schema_version"] = self.schema_version
        if self.storage_ref: result["storage_ref"] = self.storage_ref
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["StepOutputs"]:
        if not data: return None
        return cls(count=data.get("count", 0), result_ids=data.get("result_ids", []),
                   data=data.get("data", {}), schema_version=data.get("schema_version"),
                   storage_ref=data.get("storage_ref"))


@dataclass
class ReasoningFactor:
    """A factor contributing to a decision."""
    name: str
    value: float
    weight: Optional[float] = None
    contribution: Optional[float] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "value": self.value}
        if self.weight is not None: result["weight"] = self.weight
        if self.contribution is not None: result["contribution"] = self.contribution
        if self.description: result["description"] = self.description
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningFactor":
        return cls(name=data["name"], value=data["value"], weight=data.get("weight"),
                   contribution=data.get("contribution"), description=data.get("description"))


@dataclass
class StepReasoning:
    """Reasoning and decision explanation for a step."""
    algorithm: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    factors: List[ReasoningFactor] = field(default_factory=list)
    decision_path: List[Dict[str, Any]] = field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    model_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.algorithm: result["algorithm"] = self.algorithm
        if self.explanation: result["explanation"] = self.explanation
        if self.confidence is not None: result["confidence"] = self.confidence
        if self.factors: result["factors"] = [f.to_dict() for f in self.factors]
        if self.decision_path: result["decision_path"] = self.decision_path
        if self.alternatives_considered: result["alternatives_considered"] = self.alternatives_considered
        if self.model_info: result["model_info"] = self.model_info
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["StepReasoning"]:
        if not data: return None
        factors = [ReasoningFactor.from_dict(f) for f in data.get("factors", [])]
        return cls(algorithm=data.get("algorithm"), explanation=data.get("explanation"),
                   confidence=data.get("confidence"), factors=factors,
                   decision_path=data.get("decision_path", []),
                   alternatives_considered=data.get("alternatives_considered", []),
                   model_info=data.get("model_info"))


@dataclass
class StepTiming:
    """Timing information for a step."""
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["StepTiming"]:
        if not data: return None
        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ended_at = data.get("ended_at")
        if isinstance(ended_at, str):
            ended_at = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        return cls(started_at=started_at, ended_at=ended_at, duration_ms=data.get("duration_ms"))


@dataclass
class StepDebug:
    """Debug information for a step."""
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.logs: result["logs"] = self.logs
        if self.warnings: result["warnings"] = self.warnings
        if self.errors: result["errors"] = self.errors
        if self.metrics: result["metrics"] = self.metrics
        if self.trace: result["trace"] = self.trace
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["StepDebug"]:
        if not data: return None
        return cls(logs=data.get("logs", []), warnings=data.get("warnings", []),
                   errors=data.get("errors", []), metrics=data.get("metrics", {}),
                   trace=data.get("trace", {}))


@dataclass 
class StepCandidates:
    """Candidates considered in a step."""
    total_count: int = 0
    items: List[Candidate] = field(default_factory=list)
    schema_version: Optional[str] = None
    storage_ref: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"total_count": self.total_count}
        if self.items: result["items"] = [c.to_dict() for c in self.items]
        if self.schema_version: result["schema_version"] = self.schema_version
        if self.storage_ref: result["storage_ref"] = self.storage_ref
        return result
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["StepCandidates"]:
        if not data: return None
        items = [Candidate.from_dict(c) for c in data.get("items", [])]
        return cls(total_count=data.get("total_count", 0), items=items,
                   schema_version=data.get("schema_version"), storage_ref=data.get("storage_ref"))


@dataclass
class RunSummary:
    """Summary statistics for a run."""
    total_candidates_considered: int = 0
    total_candidates_filtered: int = 0
    final_output_count: int = 0
    total_filters_applied: int = 0
    avg_step_duration_ms: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> Optional["RunSummary"]:
        if not data: return None
        return cls(total_candidates_considered=data.get("total_candidates_considered", 0),
                   total_candidates_filtered=data.get("total_candidates_filtered", 0),
                   final_output_count=data.get("final_output_count", 0),
                   total_filters_applied=data.get("total_filters_applied", 0),
                   avg_step_duration_ms=data.get("avg_step_duration_ms", 0.0))


@dataclass
class Step:
    """A single step in a run."""
    step_id: str
    run_id: str
    step_index: int
    name: str
    status: StepStatus
    step_type: Optional[str] = None
    timing: Optional[StepTiming] = None
    parent_step_id: Optional[str] = None
    child_step_ids: List[str] = field(default_factory=list)
    inputs: Optional[StepInputs] = None
    candidates: Optional[StepCandidates] = None
    filters: List[StepFilter] = field(default_factory=list)
    outputs: Optional[StepOutputs] = None
    reasoning: Optional[StepReasoning] = None
    debug: Optional[StepDebug] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Step":
        filters = [StepFilter.from_dict(f) for f in data.get("filters", [])]
        return cls(
            step_id=data["step_id"], run_id=data["run_id"], step_index=data["step_index"],
            name=data["name"], status=StepStatus(data["status"]), step_type=data.get("step_type"),
            timing=StepTiming.from_dict(data.get("timing")), parent_step_id=data.get("parent_step_id"),
            child_step_ids=data.get("child_step_ids", []), inputs=StepInputs.from_dict(data.get("inputs")),
            candidates=StepCandidates.from_dict(data.get("candidates")), filters=filters,
            outputs=StepOutputs.from_dict(data.get("outputs")),
            reasoning=StepReasoning.from_dict(data.get("reasoning")),
            debug=StepDebug.from_dict(data.get("debug")))


@dataclass
class Run:
    """A single execution run."""
    run_id: str
    name: str
    status: RunStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: Optional[RunMetadata] = None
    context: Optional[RunContext] = None
    step_refs: List[str] = field(default_factory=list)
    step_count: int = 0
    summary: Optional[RunSummary] = None
    final_output: Optional[FinalOutput] = None
    steps: List[Step] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Run":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        return cls(
            run_id=data["run_id"], name=data["name"], status=RunStatus(data["status"]),
            created_at=created_at, completed_at=completed_at, duration_ms=data.get("duration_ms"),
            metadata=RunMetadata.from_dict(data.get("metadata")),
            context=RunContext.from_dict(data.get("context")),
            step_refs=data.get("step_refs", []), step_count=data.get("step_count", 0),
            summary=RunSummary.from_dict(data.get("summary")),
            final_output=FinalOutput.from_dict(data.get("final_output")), steps=steps)
