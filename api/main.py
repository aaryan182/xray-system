"""
X-Ray Debugging System API

FastAPI backend for the X-Ray debugging system that captures and analyzes
multi-step decision processes.
"""

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import crud, schemas
from .database import get_db, init_db

# Initialize FastAPI app
app = FastAPI(
    title="X-Ray Debugging API",
    description="API for capturing and analyzing multi-step decision processes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Startup Events
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# Run Endpoints
# ============================================================================

@app.post(
    "/api/v1/runs",
    response_model=schemas.RunResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Runs"],
    summary="Create a new run",
    description="Creates a new run to track a decision pipeline execution."
)
def create_run(
    run_data: schemas.RunCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new run.
    
    - **name**: Name of the run (required)
    - **metadata**: Optional metadata (version, environment, tags, custom)
    - **context**: Optional context (user_id, session_id, trace_id, request_id, custom)
    """
    db_run = crud.create_run(db, run_data)
    return db_run


@app.get(
    "/api/v1/runs/query",
    response_model=schemas.RunListResponse,
    tags=["Runs"],
    summary="Query runs",
    description="Search and filter runs with pagination."
)
def query_runs(
    status: Optional[schemas.RunStatus] = Query(None, description="Filter by status"),
    pipeline: Optional[str] = Query(None, description="Filter by pipeline type"),
    name_contains: Optional[str] = Query(None, max_length=255, description="Filter by name substring"),
    created_after: Optional[datetime] = Query(None, description="Filter by creation time"),
    created_before: Optional[datetime] = Query(None, description="Filter by creation time"),
    min_duration_ms: Optional[int] = Query(None, ge=0, description="Minimum duration in milliseconds"),
    max_duration_ms: Optional[int] = Query(None, ge=0, description="Maximum duration in milliseconds"),
    rejection_rate_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum rejection rate"),
    rejection_rate_max: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum rejection rate"),
    min_candidates: Optional[int] = Query(None, ge=0, description="Minimum total candidates considered"),
    user_id: Optional[str] = Query(None, description="Filter by user ID in context"),
    session_id: Optional[str] = Query(None, description="Filter by session ID in context"),
    trace_id: Optional[str] = Query(None, description="Filter by trace ID in context"),
    environment: Optional[str] = Query(None, description="Filter by environment in metadata"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    order_by: str = Query("created_at", description="Field to order by"),
    order_desc: bool = Query(True, description="Order descending"),
    db: Session = Depends(get_db)
):
    """
    Query runs with filters.
    
    Supports filtering by:
    - Status
    - Pipeline type
    - Name (substring match)
    - Creation time range
    - Duration range
    - Rejection rate range (computed from summary)
    - Minimum candidates considered
    - Context fields (user_id, session_id, trace_id)
    - Metadata fields (environment)
    
    Results are paginated and sortable.
    """
    query = schemas.RunQuery(
        status=status,
        pipeline=pipeline,
        name_contains=name_contains,
        created_after=created_after,
        created_before=created_before,
        min_duration_ms=min_duration_ms,
        max_duration_ms=max_duration_ms,
        rejection_rate_min=rejection_rate_min,
        rejection_rate_max=rejection_rate_max,
        min_candidates=min_candidates,
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        environment=environment,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_desc=order_desc,
    )
    
    runs, total = crud.query_runs(db, query)
    
    return schemas.RunListResponse(
        runs=[schemas.RunResponse.model_validate(r) for r in runs],
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(runs) < total,
    )


@app.get(
    "/api/v1/runs/{run_id}",
    response_model=schemas.RunWithStepsResponse,
    tags=["Runs"],
    summary="Get a run by ID",
    description="Retrieves a run and all its steps."
)
def get_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a run by ID with all its steps.
    
    - **run_id**: The unique run identifier
    """
    db_run = crud.get_run_with_steps(db, run_id)
    if not db_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    # Get steps ordered by index
    steps = crud.get_steps_for_run(db, run_id)
    
    # Convert to response format
    response = schemas.RunWithStepsResponse.model_validate(db_run)
    response.steps = [schemas.StepResponse.model_validate(s) for s in steps]
    
    return response


@app.patch(
    "/api/v1/runs/{run_id}",
    response_model=schemas.RunResponse,
    tags=["Runs"],
    summary="Update a run",
    description="Updates a run's status, final output, or other fields."
)
def update_run(
    run_id: str,
    run_update: schemas.RunUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a run (e.g., mark as completed or failed).
    
    - **run_id**: The unique run identifier
    - **status**: New status (completed, failed, cancelled)
    - **final_output**: Final output data
    - **summary**: Summary statistics
    """
    db_run = crud.update_run(db, run_id, run_update)
    if not db_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    return db_run


@app.delete(
    "/api/v1/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Runs"],
    summary="Delete a run",
    description="Deletes a run and all its steps."
)
def delete_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a run and all its steps.
    
    - **run_id**: The unique run identifier
    """
    deleted = crud.delete_run(db, run_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    return None


# ============================================================================
# Step Endpoints
# ============================================================================

@app.post(
    "/api/v1/runs/{run_id}/steps",
    response_model=schemas.StepResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Steps"],
    summary="Add a step to a run",
    description="Creates a new step for an existing run."
)
def create_step(
    run_id: str,
    step_data: schemas.StepCreate,
    db: Session = Depends(get_db)
):
    """
    Add a step to an existing run.
    
    - **run_id**: The run to add the step to
    - **name**: Name of the step (required)
    - **step_type**: Optional type classification
    - **status**: Step status (default: success)
    - **inputs**: Input data
    - **candidates**: Candidates considered
    - **filters**: Filters applied
    - **outputs**: Output data
    - **reasoning**: Decision reasoning
    - **debug**: Debug information
    """
    db_step = crud.create_step(db, run_id, step_data)
    if not db_step:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    return db_step


@app.get(
    "/api/v1/runs/{run_id}/steps/{step_id}",
    response_model=schemas.StepResponse,
    tags=["Steps"],
    summary="Get a step by ID",
    description="Retrieves a specific step from a run."
)
def get_step(
    run_id: str,
    step_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific step.
    
    - **run_id**: The run ID
    - **step_id**: The step ID
    """
    db_step = crud.get_step(db, step_id)
    if not db_step or db_step.run_id != run_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Step {step_id} not found in run {run_id}"
        )
    return db_step


@app.get(
    "/api/v1/runs/{run_id}/steps",
    response_model=list[schemas.StepResponse],
    tags=["Steps"],
    summary="Get all steps for a run",
    description="Retrieves all steps for a run, ordered by step index."
)
def get_steps(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all steps for a run.
    
    - **run_id**: The run ID
    """
    # Verify run exists
    db_run = crud.get_run(db, run_id)
    if not db_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    
    steps = crud.get_steps_for_run(db, run_id)
    return [schemas.StepResponse.model_validate(s) for s in steps]


# ============================================================================
# Stats Endpoint
# ============================================================================

@app.post(
    "/api/v1/runs/{run_id}/compute-summary",
    response_model=schemas.RunResponse,
    tags=["Runs"],
    summary="Compute run summary",
    description="Computes and updates summary statistics for a run."
)
def compute_run_summary(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Compute summary statistics for a run.
    
    - **run_id**: The run ID
    """
    db_run = crud.update_run_summary(db, run_id)
    if not db_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )
    return db_run


# ============================================================================
# Root
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """API root endpoint."""
    return {
        "api": "X-Ray Debugging API",
        "version": "1.0.0",
        "docs": "/docs",
    }
