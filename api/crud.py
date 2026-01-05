"""
CRUD operations for the X-Ray debugging system.

Provides database operations for runs and steps.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, func, Float, Integer

from . import models, schemas


# ============================================================================
# Helper Functions
# ============================================================================

def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid.uuid4().hex[:16]}"


def generate_step_id() -> str:
    """Generate a unique step ID."""
    return f"step_{uuid.uuid4().hex[:16]}"


# ============================================================================
# Run CRUD Operations
# ============================================================================

def create_run(db: Session, run_data: schemas.RunCreate) -> models.Run:
    """
    Create a new run in the database.
    
    Args:
        db: Database session
        run_data: Run creation data
        
    Returns:
        Created Run model instance
    """
    run_id = generate_run_id()
    
    db_run = models.Run(
        run_id=run_id,
        name=run_data.name,
        pipeline=run_data.pipeline,
        status=models.RunStatusEnum.RUNNING,
        created_at=datetime.utcnow(),
        run_metadata=run_data.metadata.model_dump() if run_data.metadata else None,
        context=run_data.context.model_dump() if run_data.context else None,
        step_count=0,
    )
    
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    
    return db_run


def get_run(db: Session, run_id: str) -> Optional[models.Run]:
    """
    Get a run by ID.
    
    Args:
        db: Database session
        run_id: Run ID to retrieve
        
    Returns:
        Run model instance or None if not found
    """
    return db.query(models.Run).filter(models.Run.run_id == run_id).first()


def get_run_with_steps(db: Session, run_id: str) -> Optional[models.Run]:
    """
    Get a run by ID with all its steps loaded.
    
    Args:
        db: Database session
        run_id: Run ID to retrieve
        
    Returns:
        Run model instance with steps or None if not found
    """
    return (
        db.query(models.Run)
        .filter(models.Run.run_id == run_id)
        .first()
    )


def update_run(db: Session, run_id: str, run_update: schemas.RunUpdate) -> Optional[models.Run]:
    """
    Update a run's status and other fields.
    
    Args:
        db: Database session
        run_id: Run ID to update
        run_update: Update data
        
    Returns:
        Updated Run model instance or None if not found
    """
    db_run = get_run(db, run_id)
    if not db_run:
        return None
    
    update_data = run_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if field == "status" and value:
            db_run.status = models.RunStatusEnum(value)
        elif field == "final_output" and value:
            db_run.final_output = value
        elif field == "summary" and value:
            db_run.summary = value
        elif field == "completed_at" and value:
            db_run.completed_at = value
        elif field == "duration_ms" and value is not None:
            db_run.duration_ms = value
    
    # Auto-set completed_at if status is terminal
    if run_update.status and run_update.status != schemas.RunStatus.RUNNING:
        if not db_run.completed_at:
            db_run.completed_at = datetime.utcnow()
        if not db_run.duration_ms and db_run.created_at:
            delta = db_run.completed_at - db_run.created_at
            db_run.duration_ms = int(delta.total_seconds() * 1000)
    
    db.commit()
    db.refresh(db_run)
    
    return db_run


def query_runs(
    db: Session,
    query: schemas.RunQuery
) -> Tuple[List[models.Run], int]:
    """
    Query runs with filters and pagination.
    
    Args:
        db: Database session
        query: Query parameters
        
    Returns:
        Tuple of (list of runs, total count)
    """
    q = db.query(models.Run)
    
    # Apply filters
    filters = []
    
    if query.status:
        filters.append(models.Run.status == models.RunStatusEnum(query.status))
    
    if query.pipeline:
        filters.append(models.Run.pipeline == query.pipeline)
    
    if query.name_contains:
        filters.append(models.Run.name.ilike(f"%{query.name_contains}%"))
    
    if query.created_after:
        filters.append(models.Run.created_at >= query.created_after)
    
    if query.created_before:
        filters.append(models.Run.created_at <= query.created_before)
    
    if query.min_duration_ms is not None:
        filters.append(models.Run.duration_ms >= query.min_duration_ms)
    
    if query.max_duration_ms is not None:
        filters.append(models.Run.duration_ms <= query.max_duration_ms)

    # Step characteristics from summary
    if query.rejection_rate_min is not None:
        filters.append(
            func.cast(func.json_extract(models.Run.summary, '$.rejection_rate'), Float) >= query.rejection_rate_min
        )
    
    if query.rejection_rate_max is not None:
        filters.append(
            func.cast(func.json_extract(models.Run.summary, '$.rejection_rate'), Float) <= query.rejection_rate_max
        )
        
    if query.min_candidates is not None:
        filters.append(
            func.cast(func.json_extract(models.Run.summary, '$.total_candidates_considered'), Integer) >= query.min_candidates
        )
    
    # Context filters (using JSON path queries - SQLite compatible)
    if query.user_id:
        filters.append(
            func.json_extract(models.Run.context, '$.user_id') == query.user_id
        )
    
    if query.session_id:
        filters.append(
            func.json_extract(models.Run.context, '$.session_id') == query.session_id
        )
    
    if query.trace_id:
        filters.append(
            func.json_extract(models.Run.context, '$.trace_id') == query.trace_id
        )
    
    # Metadata filters
    if query.environment:
        filters.append(
            func.json_extract(models.Run.run_metadata, '$.environment') == query.environment
        )
    
    if filters:
        q = q.filter(and_(*filters))
    
    # Get total count before pagination
    total = q.count()
    
    # Apply sorting
    order_column = getattr(models.Run, query.order_by, models.Run.created_at)
    if query.order_desc:
        q = q.order_by(desc(order_column))
    else:
        q = q.order_by(asc(order_column))
    
    # Apply pagination
    runs = q.offset(query.offset).limit(query.limit).all()
    
    return runs, total


def delete_run(db: Session, run_id: str) -> bool:
    """
    Delete a run and all its steps.
    
    Args:
        db: Database session
        run_id: Run ID to delete
        
    Returns:
        True if deleted, False if not found
    """
    db_run = get_run(db, run_id)
    if not db_run:
        return False
    
    db.delete(db_run)
    db.commit()
    return True


# ============================================================================
# Step CRUD Operations
# ============================================================================

def create_step(db: Session, run_id: str, step_data: schemas.StepCreate) -> Optional[models.Step]:
    """
    Create a new step for a run.
    
    Args:
        db: Database session
        run_id: Run ID to add step to
        step_data: Step creation data
        
    Returns:
        Created Step model instance or None if run not found
    """
    # Verify run exists
    db_run = get_run(db, run_id)
    if not db_run:
        return None
    
    # Get next step index
    step_index = db_run.step_count
    step_id = generate_step_id()
    
    # Convert Pydantic models to dicts for JSON storage
    db_step = models.Step(
        step_id=step_id,
        run_id=run_id,
        step_index=step_index,
        name=step_data.name,
        step_type=step_data.step_type,
        status=models.StepStatusEnum(step_data.status),
        parent_step_id=step_data.parent_step_id,
        started_at=step_data.started_at,
        ended_at=step_data.ended_at,
        duration_ms=step_data.duration_ms,
        inputs=step_data.inputs.model_dump() if step_data.inputs else None,
        candidates=step_data.candidates.model_dump() if step_data.candidates else None,
        filters=[f.model_dump() for f in step_data.filters] if step_data.filters else None,
        outputs=step_data.outputs.model_dump() if step_data.outputs else None,
        reasoning=step_data.reasoning.model_dump() if step_data.reasoning else None,
        debug=step_data.debug.model_dump() if step_data.debug else None,
    )
    
    # Update run step count
    db_run.step_count = step_index + 1
    
    db.add(db_step)
    db.commit()
    db.refresh(db_step)
    
    return db_step


def get_step(db: Session, step_id: str) -> Optional[models.Step]:
    """
    Get a step by ID.
    
    Args:
        db: Database session
        step_id: Step ID to retrieve
        
    Returns:
        Step model instance or None if not found
    """
    return db.query(models.Step).filter(models.Step.step_id == step_id).first()


def get_steps_for_run(db: Session, run_id: str) -> List[models.Step]:
    """
    Get all steps for a run, ordered by step_index.
    
    Args:
        db: Database session
        run_id: Run ID to get steps for
        
    Returns:
        List of Step model instances
    """
    return (
        db.query(models.Step)
        .filter(models.Step.run_id == run_id)
        .order_by(asc(models.Step.step_index))
        .all()
    )


def update_run_summary(db: Session, run_id: str) -> Optional[models.Run]:
    """
    Compute and update the summary statistics for a run.
    
    Args:
        db: Database session
        run_id: Run ID to update
        
    Returns:
        Updated Run model instance or None if not found
    """
    db_run = get_run(db, run_id)
    if not db_run:
        return None
    
    steps = get_steps_for_run(db, run_id)
    
    total_candidates = 0
    total_filtered = 0
    total_filters = 0
    total_duration = 0
    duration_count = 0
    
    for step in steps:
        if step.candidates:
            total_candidates += step.candidates.get("total_count", 0)
        if step.outputs:
            output_count = step.outputs.get("count", 0)
            if step.candidates:
                candidate_count = step.candidates.get("total_count", 0)
                total_filtered += max(0, candidate_count - output_count)
        if step.filters:
            total_filters += len(step.filters)
        if step.duration_ms:
            total_duration += step.duration_ms
            duration_count += 1
    
    summary = {
        "total_candidates_considered": total_candidates,
        "total_candidates_filtered": total_filtered,
        "final_output_count": steps[-1].outputs.get("count", 0) if steps and steps[-1].outputs else 0,
        "total_filters_applied": total_filters,
        "rejection_rate": total_filtered / total_candidates if total_candidates > 0 else 0,
        "avg_step_duration_ms": total_duration / duration_count if duration_count > 0 else 0,
    }
    
    db_run.summary = summary
    db.commit()
    db.refresh(db_run)
    
    return db_run
