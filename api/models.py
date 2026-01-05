"""
SQLAlchemy database models for the X-Ray debugging system.

These models represent the normalized storage schema for runs and steps.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, JSON, 
    Enum as SQLEnum, ForeignKey, Index
)
from sqlalchemy.orm import relationship
import enum

from .database import Base


class RunStatusEnum(str, enum.Enum):
    """Status of a run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatusEnum(str, enum.Enum):
    """Status of a step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class Run(Base):
    """
    A run represents a single execution of a decision pipeline.
    Contains metadata, context, and references to its steps.
    """
    __tablename__ = "runs"

    run_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    pipeline = Column(String(100), nullable=True, index=True)
    status = Column(SQLEnum(RunStatusEnum), nullable=False, default=RunStatusEnum.RUNNING)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    
    # JSON fields for flexible nested data
    run_metadata = Column(JSON, nullable=True)  # version, environment, tags, custom
    context = Column(JSON, nullable=True)   # user_id, session_id, trace_id, request_id, custom
    final_output = Column(JSON, nullable=True)  # success, result_ids, result_data, summary, error
    summary = Column(JSON, nullable=True)  # Computed statistics
    
    step_count = Column(Integer, default=0)
    
    # Relationship to steps
    steps = relationship("Step", back_populates="run", cascade="all, delete-orphan")

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_runs_status", "status"),
        Index("ix_runs_created_at", "created_at"),
        Index("ix_runs_pipeline_created", "pipeline", "created_at"),
        Index("ix_runs_status_created", "status", "created_at"),
    )


class Step(Base):
    """
    A step represents a single stage in a decision pipeline.
    Contains inputs, candidates, filters, outputs, and reasoning.
    """
    __tablename__ = "steps"

    step_id = Column(String(64), primary_key=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    step_index = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    step_type = Column(String(100), nullable=True, index=True)
    status = Column(SQLEnum(StepStatusEnum), nullable=False, default=StepStatusEnum.PENDING)
    
    parent_step_id = Column(String(64), nullable=True)
    
    # Timing information
    started_at = Column(DateTime, nullable=True, index=True)
    ended_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    
    # JSON fields for complex nested data
    inputs = Column(JSON, nullable=True)     # data, schema_version, source_step_id
    candidates = Column(JSON, nullable=True)  # total_count, items, storage_ref
    filters = Column(JSON, nullable=True)     # Array of filter objects
    outputs = Column(JSON, nullable=True)     # count, result_ids, data, storage_ref
    reasoning = Column(JSON, nullable=True)   # algorithm, explanation, confidence, factors
    debug = Column(JSON, nullable=True)       # logs, warnings, errors, metrics, trace
    
    # Child step references (stored as JSON array of step_ids)
    child_step_ids = Column(JSON, nullable=True)
    
    # Relationship back to run
    run = relationship("Run", back_populates="steps")

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_steps_run_id", "run_id"),
        Index("ix_steps_run_index", "run_id", "step_index"),
        Index("ix_steps_type_status", "step_type", "status"),
        Index("ix_steps_status", "status"),
    )
