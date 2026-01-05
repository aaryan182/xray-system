"""
X-Ray SDK - Python SDK for the X-Ray Debugging System API.

This SDK provides a convenient interface for tracking multi-step decision
processes, including runs, steps, candidates, filters, and reasoning.

Example Usage:
    from xray_sdk import XRayClient
    
    client = XRayClient(base_url="http://localhost:8000/api/v1")
    
    # Start a run
    run = client.start_run(
        name="Product Recommendation",
        context={"user_id": "user_123", "session_id": "sess_456"}
    )
    
    # Add a step
    step = client.add_step(
        name="Candidate Generation",
        step_type="candidate_generation",
        inputs={"query": "wireless headphones"},
        candidates=[{"id": "prod_001", "score": 0.95}]
    )
    
    # Complete the run
    client.complete_run(success=True, result_ids=["prod_001"])
"""

from xray_sdk.client import XRayClient
from xray_sdk.models import (
    Run,
    RunStatus,
    Step,
    StepStatus,
    Candidate,
    StepFilter,
    FilterMetrics,
    StepInputs,
    StepOutputs,
    StepReasoning,
    ReasoningFactor,
    RunContext,
    RunMetadata,
    FinalOutput,
)
from xray_sdk.decorators import (
    trace_step,
    trace_run,
    trace_step_method,
    set_global_client,
    get_global_client,
)
from xray_sdk.config import XRayConfig

__version__ = "1.0.0"
__all__ = [
    # Main client
    "XRayClient",
    # Configuration
    "XRayConfig",
    # Models
    "Run",
    "RunStatus",
    "Step",
    "StepStatus",
    "Candidate",
    "StepFilter",
    "FilterMetrics",
    "StepInputs",
    "StepOutputs",
    "StepReasoning",
    "ReasoningFactor",
    "RunContext",
    "RunMetadata",
    "FinalOutput",
    # Decorators
    "trace_step",
    "trace_run",
    "trace_step_method",
    "set_global_client",
    "get_global_client",
]
