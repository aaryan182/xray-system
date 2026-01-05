"""
Decorators for easy X-Ray tracing integration.

Provides decorators for automatically tracing function executions
as steps within an X-Ray run.
"""

import functools
import inspect
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from xray_sdk.models import StepStatus

F = TypeVar("F", bound=Callable[..., Any])

# Global client reference for simple decorator usage
_global_client = None


def set_global_client(client):
    """
    Set a global XRayClient for use with decorators.
    
    This allows decorators to work without explicitly passing the client.
    
    Example:
        from xray_sdk import XRayClient
        from xray_sdk.decorators import set_global_client, trace_step
        
        client = XRayClient()
        set_global_client(client)
        
        @trace_step(run_id="run_abc123")
        def my_function(x, y):
            return x + y
    """
    global _global_client
    _global_client = client


def get_global_client():
    """Get the global XRayClient instance."""
    return _global_client


def trace_step(
    run_id: Optional[str] = None,
    step_name: Optional[str] = None,
    step_type: Optional[str] = None,
    client: Optional[Any] = None,
    capture_args: bool = True,
    capture_result: bool = True,
    capture_exception: bool = True,
    exclude_args: Optional[List[str]] = None,
) -> Callable[[F], F]:
    """
    Decorator to automatically trace a function as a step in an X-Ray run.
    
    Automatically captures:
    - Function name as step name (or custom step_name)
    - Function arguments as step inputs
    - Return value as step output
    - Execution time in milliseconds
    - Any exceptions that occur
    
    Usage:
        @trace_step(run_id='abc123', step_name='filter_products')
        def filter_by_price(products, min_price, max_price):
            return [p for p in products if min_price <= p['price'] <= max_price]
        
        # Or with automatic function name:
        @trace_step(run_id='abc123')
        def generate_candidates(query, limit=100):
            return search(query, limit)
    
    Args:
        run_id: The run ID to add this step to. Required unless using global client
                with active run context.
        step_name: Custom step name. Defaults to function name.
        step_type: Type/category of step (e.g., 'filtering', 'ranking').
        client: XRayClient instance. Uses global client if not provided.
        capture_args: Whether to capture function arguments as inputs.
        capture_result: Whether to capture return value as output.
        capture_exception: Whether to capture exception details on failure.
        exclude_args: List of argument names to exclude from capture.
    
    Returns:
        Decorated function that automatically traces execution.
    
    Example with all features:
        from xray_sdk import XRayClient
        from xray_sdk.decorators import trace_step, set_global_client
        
        client = XRayClient(base_url="http://localhost:8000/api/v1")
        set_global_client(client)
        
        @trace_step(run_id='run_abc123', step_name='filter_products', step_type='filtering')
        def filter_by_price(products, min_price, max_price):
            filtered = [p for p in products if min_price <= p['price'] <= max_price]
            return filtered
        
        # This will automatically:
        # 1. Create a step named 'filter_products' in run 'run_abc123'
        # 2. Record inputs: {"products": [...], "min_price": 10, "max_price": 100}
        # 3. Track execution time
        # 4. Record output: {"result": [...], "count": 5}
        # 5. Mark step as success or failure
    """
    def decorator(func: F) -> F:
        # Determine step name from function if not provided
        actual_step_name = step_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the client to use
            xray_client = client or _global_client
            
            if xray_client is None:
                # No client available, just run the function normally
                return func(*args, **kwargs)
            
            # Determine the run ID
            actual_run_id = run_id
            
            # If no run_id provided, try to get from current run context
            if actual_run_id is None:
                current_run = xray_client.get_current_run()
                if current_run:
                    actual_run_id = current_run.run_id
            
            if actual_run_id is None:
                # No run ID available, just run the function normally
                return func(*args, **kwargs)
            
            # Capture function inputs
            inputs_data = {}
            if capture_args:
                inputs_data = _capture_function_inputs(
                    func, args, kwargs, exclude_args or []
                )
            
            # Record start time
            start_time = time.time()
            started_at = datetime.utcnow().isoformat() + "Z"
            
            # Prepare step data
            step_data = {
                "name": actual_step_name,
                "status": StepStatus.RUNNING.value,
            }
            
            if step_type:
                step_data["step_type"] = step_type
            
            if inputs_data:
                step_data["inputs"] = {"data": inputs_data}
            
            # Create the step via API
            step_id = None
            try:
                endpoint = f"/runs/{actual_run_id}/steps"
                response = xray_client._make_request_sync("POST", endpoint, step_data)
                if response:
                    step_id = response.get("step_id")
            except Exception as e:
                # Log but don't fail the function if step creation fails
                import logging
                logging.getLogger("xray_sdk").warning(f"Failed to create step: {e}")
            
            # Execute the function
            exception_info = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                exception_info = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc() if capture_exception else None,
                }
                raise
                
            finally:
                # Calculate execution time
                end_time = time.time()
                execution_time_ms = int((end_time - start_time) * 1000)
                ended_at = datetime.utcnow().isoformat() + "Z"
                
                # Prepare update payload
                update_data = {}
                
                if exception_info:
                    # Step failed
                    update_data["status"] = StepStatus.FAILED.value
                    update_data["debug"] = {
                        "errors": [f"{exception_info['type']}: {exception_info['message']}"],
                        "metrics": {"execution_time_ms": execution_time_ms},
                    }
                    if exception_info.get("traceback"):
                        update_data["debug"]["trace"] = {
                            "exception_traceback": exception_info["traceback"]
                        }
                else:
                    # Step succeeded
                    update_data["status"] = StepStatus.SUCCESS.value
                    
                    # Capture output
                    if capture_result and result is not None:
                        output_data = _capture_output(result)
                        output_data["execution_time_ms"] = execution_time_ms
                        update_data["outputs"] = output_data
                    else:
                        update_data["outputs"] = {
                            "count": 0,
                            "data": {"execution_time_ms": execution_time_ms}
                        }
                
                # Update the step
                if step_id and xray_client:
                    try:
                        endpoint = f"/runs/{actual_run_id}/steps/{step_id}"
                        xray_client._make_request_async("PATCH", endpoint, update_data)
                    except Exception:
                        pass  # Don't fail if update fails
        
        return wrapper  # type: ignore
    
    return decorator


def _capture_function_inputs(
    func: Callable,
    args: tuple,
    kwargs: dict,
    exclude: List[str]
) -> Dict[str, Any]:
    """Capture function arguments as a dictionary."""
    inputs = {}
    
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, value in bound_args.arguments.items():
            # Skip excluded parameters
            if param_name in exclude:
                continue
            
            # Skip 'self' and 'cls'
            if param_name in ("self", "cls"):
                continue
            
            # Serialize the value
            inputs[param_name] = _safe_serialize(value)
    
    except Exception:
        # If we can't get the signature, capture what we can
        for i, arg in enumerate(args):
            inputs[f"arg_{i}"] = _safe_serialize(arg)
        for key, value in kwargs.items():
            if key not in exclude:
                inputs[key] = _safe_serialize(value)
    
    return inputs


def _capture_output(result: Any) -> Dict[str, Any]:
    """Capture function return value as output data."""
    output = {"count": 0, "data": {}}
    
    try:
        # Handle list of items with IDs
        if isinstance(result, list):
            output["count"] = len(result)
            
            # Check if items have 'id' field
            if result and isinstance(result[0], dict) and "id" in result[0]:
                output["result_ids"] = [item["id"] for item in result if "id" in item]
            
            output["data"]["items"] = _safe_serialize(result)
        
        # Handle single dict with ID
        elif isinstance(result, dict):
            output["count"] = 1
            if "id" in result:
                output["result_ids"] = [result["id"]]
            output["data"]["item"] = _safe_serialize(result)
        
        # Handle primitive types
        elif isinstance(result, (str, int, float, bool)):
            output["count"] = 1
            output["data"]["value"] = result
        
        # Handle other types
        else:
            output["count"] = 1
            output["data"]["result"] = _safe_serialize(result)
    
    except Exception:
        output["data"]["result"] = str(result)
    
    return output


def _safe_serialize(value: Any, max_depth: int = 3, max_items: int = 100) -> Any:
    """
    Safely serialize a value for JSON, with depth and size limits.
    
    Args:
        value: Value to serialize.
        max_depth: Maximum nesting depth.
        max_items: Maximum items in lists/dicts.
    
    Returns:
        JSON-serializable value.
    """
    if max_depth <= 0:
        return str(value)
    
    if value is None:
        return None
    
    if isinstance(value, (str, int, float, bool)):
        return value
    
    if isinstance(value, (list, tuple)):
        items = list(value)[:max_items]
        return [_safe_serialize(item, max_depth - 1, max_items) for item in items]
    
    if isinstance(value, dict):
        items = list(value.items())[:max_items]
        return {
            str(k): _safe_serialize(v, max_depth - 1, max_items)
            for k, v in items
        }
    
    # Handle datetime
    if isinstance(value, datetime):
        return value.isoformat()
    
    # Try to_dict method
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    
    # Try __dict__
    if hasattr(value, "__dict__"):
        return _safe_serialize(value.__dict__, max_depth - 1, max_items)
    
    # Fallback to string representation
    return str(value)


# =============================================================================
# Additional Decorators
# =============================================================================

def trace_run(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    client: Optional[Any] = None,
) -> Callable[[F], F]:
    """
    Decorator to wrap a function execution in an X-Ray run.
    
    This decorator will:
    1. Start a new run before function execution
    2. Complete the run after successful execution
    3. Fail the run if an exception occurs
    
    Usage:
        from xray_sdk.decorators import trace_run, set_global_client
        
        set_global_client(XRayClient())
        
        @trace_run(name="My Pipeline", tags=["production"])
        def execute_pipeline(query: str):
            result = step_one(query)
            return step_two(result)
    
    Args:
        name: Run name. Defaults to function name.
        user_id: User ID for the run context.
        session_id: Session ID for the run context.
        tags: Tags for the run.
        client: XRayClient instance. Uses global client if not provided.
    
    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        run_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            xray_client = client or _global_client
            
            if xray_client is None:
                # No client found, just execute normally
                return func(*args, **kwargs)
            
            with xray_client.run(
                name=run_name,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
            ) as run:
                return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def trace_step_method(
    step_name: Optional[str] = None,
    step_type: Optional[str] = None,
    client_attr: str = "xray_client",
    run_attr: str = "xray_run",
    capture_args: bool = True,
    capture_result: bool = True,
    exclude_args: Optional[List[str]] = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing methods on classes that have xray_client and xray_run attributes.
    
    This is useful for class-based pipelines where the client and run are stored
    as instance attributes.
    
    Usage:
        class RecommendationPipeline:
            def __init__(self):
                self.xray_client = XRayClient()
                self.xray_run = None
            
            @trace_step_method(step_type="candidate_generation")
            def generate_candidates(self, query: str, limit: int = 100):
                return self.search(query, limit)
            
            def execute(self, query: str):
                with self.xray_client.run("Recommendations") as run:
                    self.xray_run = run
                    return self.generate_candidates(query)
    
    Args:
        step_name: Custom step name. Defaults to method name.
        step_type: Type/category of step.
        client_attr: Attribute name for XRayClient on self.
        run_attr: Attribute name for current run on self.
        capture_args: Whether to capture method arguments.
        capture_result: Whether to capture return value.
        exclude_args: Arguments to exclude from capture.
    
    Returns:
        Decorated method.
    """
    def decorator(func: F) -> F:
        actual_step_name = step_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get client and run from self
            xray_client = getattr(self, client_attr, None)
            run = getattr(self, run_attr, None)
            
            if run is None and xray_client is not None:
                run = xray_client.get_current_run()
            
            if run is None:
                # No run context, execute normally
                return func(self, *args, **kwargs)
            
            # Capture inputs (exclude 'self')
            inputs_data = {}
            if capture_args:
                inputs_data = _capture_function_inputs(
                    func, (self,) + args, kwargs, (exclude_args or []) + ["self"]
                )
            
            # Create step
            step = run.add_step(
                name=actual_step_name,
                step_type=step_type,
                inputs=inputs_data if inputs_data else None,
                status=StepStatus.RUNNING,
            )
            
            start_time = time.time()
            
            try:
                result = func(self, *args, **kwargs)
                
                execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Capture output
                if capture_result and result is not None:
                    output = _capture_output(result)
                    output["data"]["execution_time_ms"] = execution_time_ms
                    step.complete(
                        result_ids=output.get("result_ids"),
                        output_data=output.get("data"),
                        count=output.get("count"),
                    )
                else:
                    step.complete(output_data={"execution_time_ms": execution_time_ms})
                
                return result
                
            except Exception as e:
                step.fail(error=str(e))
                raise
        
        return wrapper  # type: ignore
    
    return decorator
