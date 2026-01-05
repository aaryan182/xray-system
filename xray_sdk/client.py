"""
Main XRayClient class for the X-Ray SDK.

Provides a high-level interface for tracking runs, steps, and candidates
with asynchronous, non-blocking API calls and graceful failure handling.
"""

import json
import logging
import os
import queue
import random
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

try:
    import requests
except ImportError:
    requests = None

from xray_sdk.config import XRayConfig, RetryConfig
from xray_sdk.models import (
    Run, RunStatus, RunMetadata, RunContext, FinalOutput,
    Step, StepStatus, StepInputs, StepOutputs, StepCandidates,
    Candidate, StepFilter, FilterMetrics, StepReasoning, ReasoningFactor,
)

logger = logging.getLogger("xray_sdk")


class XRayError(Exception):
    """Base exception for X-Ray SDK errors."""
    pass


class APIError(XRayError):
    """Exception for API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_body: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class QueueFullError(XRayError):
    """Exception when the local queue is full."""
    pass


class _RequestQueue:
    """Thread-safe queue for storing failed requests."""
    
    def __init__(self, config: XRayConfig):
        self.config = config
        self._queue: queue.Queue = queue.Queue(maxsize=config.queue_config.max_queue_size)
        self._persistence_path = self._get_persistence_path()
        self._lock = threading.Lock()
        self._load_persisted()
    
    def _get_persistence_path(self) -> Optional[Path]:
        if not self.config.queue_config.persist_to_disk:
            return None
        path = self.config.queue_config.persistence_path
        if path:
            return Path(path)
        return Path.home() / ".xray_sdk" / "queue"
    
    def _load_persisted(self):
        """Load any persisted queue items from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        try:
            queue_file = self._persistence_path / "queue.json"
            if queue_file.exists():
                with open(queue_file, "r") as f:
                    items = json.load(f)
                for item in items:
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        break
                queue_file.unlink()
                logger.info(f"Loaded {len(items)} queued items from disk")
        except Exception as e:
            logger.warning(f"Failed to load persisted queue: {e}")
    
    def enqueue(self, method: str, endpoint: str, payload: Dict) -> bool:
        """Add a request to the queue. Returns True if successful."""
        item = {
            "method": method,
            "endpoint": endpoint,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
            "retries": 0,
        }
        try:
            self._queue.put_nowait(item)
            self._persist()
            return True
        except queue.Full:
            logger.warning("Request queue is full, dropping request")
            return False
    
    def dequeue_batch(self, size: int) -> List[Dict]:
        """Get a batch of items from the queue."""
        items = []
        for _ in range(size):
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items
    
    def requeue(self, items: List[Dict]):
        """Put items back in the queue (e.g., after failed flush)."""
        for item in items:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                break
        self._persist()
    
    def _persist(self):
        """Persist queue to disk for crash recovery."""
        if not self._persistence_path:
            return
        try:
            self._persistence_path.mkdir(parents=True, exist_ok=True)
            items = []
            temp_queue = queue.Queue()
            while True:
                try:
                    item = self._queue.get_nowait()
                    items.append(item)
                    temp_queue.put_nowait(item)
                except queue.Empty:
                    break
            while True:
                try:
                    self._queue.put_nowait(temp_queue.get_nowait())
                except queue.Empty:
                    break
            queue_file = self._persistence_path / "queue.json"
            with open(queue_file, "w") as f:
                json.dump(items, f)
        except Exception as e:
            logger.warning(f"Failed to persist queue: {e}")
    
    @property
    def size(self) -> int:
        return self._queue.qsize()


class XRayClient:
    """
    Main client for the X-Ray Debugging System API.
    
    Provides methods to track execution runs, steps, candidates, and reasoning
    with automatic async handling and graceful failure recovery.
    
    Example:
        client = XRayClient(base_url="http://localhost:8000/api/v1")
        
        with client.run("Product Recommendations", user_id="user_123") as run:
            step = run.add_step(
                name="Candidate Generation",
                step_type="candidate_generation",
                inputs={"query": "headphones"}
            )
            step.add_candidates([
                {"id": "prod_1", "score": 0.95},
                {"id": "prod_2", "score": 0.89}
            ])
            step.complete(result_ids=["prod_1", "prod_2"])
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[XRayConfig] = None,
        **kwargs
    ):
        """
        Initialize the XRayClient.
        
        Args:
            base_url: Base URL for the API (overrides config).
            api_key: API key for authentication (overrides config).
            config: Full configuration object. If not provided, uses defaults
                    or environment variables.
            **kwargs: Additional config overrides.
        """
        if requests is None:
            raise ImportError("requests library is required. Install with: pip install requests")
        
        if config is None:
            config = XRayConfig.from_env()
        
        if base_url:
            config = config.with_overrides(base_url=base_url)
        if api_key:
            config = config.with_overrides(api_key=api_key)
        if kwargs:
            config = config.with_overrides(**kwargs)
        
        self.config = config
        self._session = requests.Session()
        self._setup_session()
        
        # Request queue for handling failures
        self._queue = _RequestQueue(config)
        
        # Thread pool for async requests
        self._executor: Optional[ThreadPoolExecutor] = None
        if config.async_config.enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=config.async_config.max_workers,
                thread_name_prefix="xray_"
            )
        
        # Background flush thread
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_flush = threading.Event()
        self._start_flush_thread()
        
        # Current run context (for decorator support)
        self._current_run: Optional["RunBuilder"] = None
        self._run_lock = threading.Lock()
        
        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
    
    def _setup_session(self):
        """Configure the requests session."""
        if self.config.api_key:
            self._session.headers[self.config.api_key_header] = self.config.api_key
        self._session.headers["Content-Type"] = "application/json"
        self._session.headers["Accept"] = "application/json"
        self._session.headers["User-Agent"] = "xray-sdk-python/1.0.0"
    
    def _start_flush_thread(self):
        """Start the background thread for flushing queued requests."""
        if self._queue.size == 0 and not self.config.queue_config.persist_to_disk:
            return
        
        def _flush_loop():
            while not self._stop_flush.is_set():
                self._flush_queue()
                self._stop_flush.wait(self.config.queue_config.flush_interval_seconds)
        
        self._flush_thread = threading.Thread(target=_flush_loop, daemon=True, name="xray_flush")
        self._flush_thread.start()
    
    def _flush_queue(self):
        """Attempt to flush queued requests to the API."""
        if self._queue.size == 0:
            return
        
        batch_size = self.config.queue_config.batch_size
        items = self._queue.dequeue_batch(batch_size)
        
        failed_items = []
        for item in items:
            try:
                self._make_request_sync(item["method"], item["endpoint"], item["payload"])
                logger.debug(f"Successfully flushed queued request to {item['endpoint']}")
            except Exception as e:
                item["retries"] = item.get("retries", 0) + 1
                if item["retries"] < self.config.retry_config.max_retries:
                    failed_items.append(item)
                else:
                    logger.error(f"Dropping request after max retries: {e}")
        
        if failed_items:
            self._queue.requeue(failed_items)
    
    def _make_request_sync(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict] = None,
        retry: bool = True
    ) -> Optional[Dict]:
        """Make a synchronous HTTP request with retry logic."""
        # Dry-run mode: log and return mock response
        if self.config.dry_run:
            return self._make_dry_run_response(method, endpoint, payload)
        
        url = f"{self.config.base_url}{endpoint}"
        retry_config = self.config.retry_config
        
        last_error = None
        for attempt in range(retry_config.max_retries + 1 if retry else 1):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=self.config.async_config.request_timeout_seconds,
                )
                
                if response.status_code >= 400:
                    # Check if this is a retryable status code
                    if response.status_code in retry_config.retry_on_status_codes and retry:
                        if attempt < retry_config.max_retries:
                            delay = retry_config.initial_delay_seconds * (2 ** attempt)
                            delay = min(delay, retry_config.max_delay_seconds)
                            logger.debug(f"Got {response.status_code}, retrying in {delay}s")
                            time.sleep(delay)
                            continue  # Retry the request
                        # Max retries exhausted
                        raise APIError(
                            f"Request failed with status {response.status_code} after {attempt + 1} attempts",
                            status_code=response.status_code
                        )
                    
                    # Non-retryable error
                    try:
                        body = response.json()
                    except:
                        body = {"message": response.text}
                    raise APIError(
                        f"API error: {body.get('message', 'Unknown error')}",
                        status_code=response.status_code,
                        response_body=body
                    )
                
                if response.status_code == 204:
                    return None
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retry_config.max_retries and retry:
                    delay = retry_config.initial_delay_seconds * (2 ** attempt)
                    delay = min(delay, retry_config.max_delay_seconds)
                    logger.debug(f"Request failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                    continue
                raise APIError(f"Request failed: {e}")
        
        raise last_error or APIError("Request failed after all retries")
    
    def _make_request_async(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict] = None,
        callback: Optional[Callable[[Optional[Dict], Optional[Exception]], None]] = None,
    ):
        """Make an asynchronous HTTP request (non-blocking)."""
        if not self._executor:
            # Fallback to sync if async is disabled
            try:
                result = self._make_request_sync(method, endpoint, payload)
                if callback:
                    callback(result, None)
            except Exception as e:
                self._handle_request_failure(method, endpoint, payload)
                if callback:
                    callback(None, e)
            return
        
        def _do_request():
            try:
                result = self._make_request_sync(method, endpoint, payload)
                if callback:
                    callback(result, None)
            except Exception as e:
                logger.debug(f"Async request failed, queueing: {e}")
                self._handle_request_failure(method, endpoint, payload)
                if callback:
                    callback(None, e)
        
        self._executor.submit(_do_request)
    
    def _handle_request_failure(self, method: str, endpoint: str, payload: Optional[Dict]):
        """Handle a failed request by queueing it for retry."""
        if payload is None:
            return  # Don't queue GET requests
        if not self._queue.enqueue(method, endpoint, payload):
            logger.error("Failed to queue request and queue is full")
    
    def _make_dry_run_response(
        self, 
        method: str, 
        endpoint: str, 
        payload: Optional[Dict]
    ) -> Dict:
        """
        Generate a mock response for dry-run mode.
        
        Logs the operation and returns realistic mock data with generated IDs.
        Useful for development and testing without a running API server.
        """
        from datetime import datetime
        
        logger.info(f"[DRY-RUN] {method} {endpoint}")
        if payload and self.config.debug:
            logger.debug(f"[DRY-RUN] Payload: {json.dumps(payload, default=str)[:500]}...")
        
        # Generate response based on endpoint pattern
        if "/runs" in endpoint and "/steps" in endpoint:
            # Step creation
            return {
                "step_id": f"step_{uuid.uuid4().hex[:12]}",
                "run_id": endpoint.split("/runs/")[1].split("/")[0] if "/runs/" in endpoint else "unknown",
                "step_index": 0,
                "name": payload.get("name", "unnamed") if payload else "unnamed",
                "status": "running",
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        elif "/runs" in endpoint and method == "POST":
            # Run creation
            return {
                "run_id": f"run_{uuid.uuid4().hex[:12]}",
                "name": payload.get("name", "unnamed") if payload else "unnamed",
                "status": "running",
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        elif "/runs" in endpoint and method == "PATCH":
            # Run update
            run_id = endpoint.split("/runs/")[1].split("/")[0] if "/runs/" in endpoint else "unknown"
            return {
                "run_id": run_id,
                "status": payload.get("status", "completed") if payload else "completed",
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
        else:
            # Generic response
            return {"success": True, "dry_run": True}
    
    # =========================================================================
    # Public API Methods
    # =========================================================================
    
    def start_run(
        self,
        name: str,
        metadata: Optional[Union[Dict, RunMetadata]] = None,
        context: Optional[Union[Dict, RunContext]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        environment: Optional[str] = None,
        blocking: bool = False,
    ) -> "RunBuilder":
        """
        Start a new execution run.
        
        Args:
            name: Human-readable name for the run.
            metadata: Run metadata (dict or RunMetadata object).
            context: Run context (dict or RunContext object).
            user_id: User ID (shorthand for context.user_id).
            session_id: Session ID (shorthand for context.session_id).
            trace_id: Trace ID (shorthand for context.trace_id).
            tags: Tags for categorization.
            environment: Environment name.
            blocking: If True, wait for API response before returning.
            
        Returns:
            RunBuilder for adding steps and completing the run.
        """
        # Build metadata
        if isinstance(metadata, dict):
            meta = RunMetadata(**metadata)
        elif metadata:
            meta = metadata
        else:
            meta = RunMetadata()
        
        if tags:
            meta.tags = tags
        if environment:
            meta.environment = environment
        
        # Merge with default metadata
        for key, value in self.config.default_metadata.items():
            if not hasattr(meta, key) or getattr(meta, key) is None:
                if hasattr(meta, key):
                    setattr(meta, key, value)
                else:
                    meta.custom[key] = value
        
        # Build context
        if isinstance(context, dict):
            ctx = RunContext(**context)
        elif context:
            ctx = context
        else:
            ctx = RunContext()
        
        if user_id:
            ctx.user_id = user_id
        if session_id:
            ctx.session_id = session_id
        if trace_id:
            ctx.trace_id = trace_id
        
        payload = {"name": name}
        meta_dict = meta.to_dict()
        if meta_dict:
            payload["metadata"] = meta_dict
        ctx_dict = ctx.to_dict()
        if ctx_dict:
            payload["context"] = ctx_dict
        
        builder = RunBuilder(self, name, payload)
        
        if blocking:
            try:
                response = self._make_request_sync("POST", "/runs", payload)
                builder._run_id = response["run_id"]
                builder._created = True
            except Exception as e:
                logger.error(f"Failed to create run: {e}")
                builder._run_id = f"run_{uuid.uuid4().hex[:12]}"
                builder._pending_create = payload
        else:
            builder._run_id = f"run_{uuid.uuid4().hex[:12]}"
            
            def _on_create(response, error):
                if response:
                    builder._run_id = response["run_id"]
                    builder._created = True
                else:
                    builder._pending_create = payload
            
            self._make_request_async("POST", "/runs", payload, _on_create)
        
        return builder
    
    @contextmanager
    def run(
        self,
        name: str,
        metadata: Optional[Union[Dict, RunMetadata]] = None,
        context: Optional[Union[Dict, RunContext]] = None,
        **kwargs
    ):
        """
        Context manager for a run. Automatically completes or fails the run.
        
        Example:
            with client.run("My Pipeline", user_id="user_123") as run:
                run.add_step("Step 1", inputs={"key": "value"})
                # ... more steps
            # Run is automatically completed when exiting the context
        """
        run_builder = self.start_run(name, metadata=metadata, context=context, **kwargs)
        
        with self._run_lock:
            previous_run = self._current_run
            self._current_run = run_builder
        
        try:
            yield run_builder
            run_builder.complete(success=True)
        except Exception as e:
            run_builder.fail(error=str(e))
            raise
        finally:
            with self._run_lock:
                self._current_run = previous_run
    
    def get_current_run(self) -> Optional["RunBuilder"]:
        """Get the current run (if in a run context)."""
        return self._current_run
    
    def get_run(self, run_id: str, include_steps: bool = True) -> Run:
        """
        Retrieve a run by ID.
        
        Args:
            run_id: The run ID.
            include_steps: Whether to include step details.
            
        Returns:
            Run object with details.
        """
        params = f"?include_steps={str(include_steps).lower()}"
        response = self._make_request_sync("GET", f"/runs/{run_id}{params}")
        return Run.from_dict(response)
    
    def close(self):
        """Close the client and flush any pending requests."""
        self._stop_flush.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2)
        
        # Final flush attempt
        self._flush_queue()
        
        if self._executor:
            self._executor.shutdown(
                wait=True,
                cancel_futures=False
            )
        
        self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RunBuilder:
    """Builder for constructing and managing a run."""
    
    def __init__(self, client: XRayClient, name: str, create_payload: Dict):
        self._client = client
        self._name = name
        self._run_id: Optional[str] = None
        self._created = False
        self._pending_create: Optional[Dict] = None
        self._step_index = 0
        self._steps: List["StepBuilder"] = []
        self._completed = False
    
    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"RunBuilder(name={self._name!r}, run_id={self._run_id!r}, "
            f"steps={len(self._steps)}, completed={self._completed})"
        )
    
    @property
    def run_id(self) -> str:
        return self._run_id or "pending"
    
    def add_step(
        self,
        name: str,
        step_type: Optional[str] = None,
        inputs: Optional[Union[Dict, StepInputs]] = None,
        candidates: Optional[List[Union[Dict, Candidate]]] = None,
        filters: Optional[List[Union[Dict, StepFilter]]] = None,
        outputs: Optional[Union[Dict, StepOutputs]] = None,
        reasoning: Optional[Union[Dict, StepReasoning]] = None,
        status: StepStatus = StepStatus.RUNNING,
        parent_step_id: Optional[str] = None,
        blocking: bool = False,
    ) -> "StepBuilder":
        """
        Add a step to this run.
        
        Args:
            name: Human-readable step name.
            step_type: Type/category of step.
            inputs: Step inputs.
            candidates: Candidates considered.
            filters: Filters applied.
            outputs: Step outputs.
            reasoning: Decision reasoning.
            status: Initial step status.
            parent_step_id: Parent step for nesting.
            blocking: Wait for API response.
            
        Returns:
            StepBuilder for further modifications.
        """
        step_builder = StepBuilder(
            self._client, self, self._step_index, name, step_type,
            inputs, candidates, filters, outputs, reasoning, status, parent_step_id
        )
        self._steps.append(step_builder)
        self._step_index += 1
        
        step_builder._send(blocking=blocking)
        
        return step_builder
    
    def complete(
        self,
        success: bool = True,
        result_ids: Optional[List[str]] = None,
        result_data: Optional[Dict] = None,
        summary: Optional[str] = None,
    ):
        """Complete the run successfully."""
        if self._completed:
            return
        
        final_output = FinalOutput(
            success=success,
            result_ids=result_ids or [],
            result_data=result_data or {},
            summary=summary,
        )
        
        payload = {
            "status": RunStatus.COMPLETED.value,
            "final_output": final_output.to_dict(),
        }
        
        self._client._make_request_async("PATCH", f"/runs/{self.run_id}", payload)
        self._completed = True
    
    def fail(self, error: Optional[str] = None):
        """Mark the run as failed."""
        if self._completed:
            return
        
        payload = {
            "status": RunStatus.FAILED.value,
            "final_output": {"success": False, "error": error},
        }
        
        self._client._make_request_async("PATCH", f"/runs/{self.run_id}", payload)
        self._completed = True


class StepBuilder:
    """Builder for constructing and managing a step with smart candidate tracking."""
    
    # Configuration for smart candidate sampling
    SAMPLE_THRESHOLD = 1000  # Sample when more than this many candidates
    SAMPLE_SIZE = 100  # Number of candidates to sample
    
    def __init__(
        self,
        client: XRayClient,
        run: RunBuilder,
        index: int,
        name: str,
        step_type: Optional[str],
        inputs: Optional[Union[Dict, StepInputs]],
        candidates: Optional[List[Union[Dict, Candidate]]],
        filters: Optional[List[Union[Dict, StepFilter]]],
        outputs: Optional[Union[Dict, StepOutputs]],
        reasoning: Optional[Union[Dict, StepReasoning]],
        status: StepStatus,
        parent_step_id: Optional[str],
    ):
        self._client = client
        self._run = run
        self._index = index
        self._name = name
        self._step_type = step_type
        self._status = status
        self._parent_step_id = parent_step_id
        
        # Process inputs
        if isinstance(inputs, dict):
            self._inputs = StepInputs(data=inputs)
        else:
            self._inputs = inputs
        
        # Candidate tracking with smart sampling support
        self._candidates: List[Candidate] = []
        self._candidates_total_count: int = 0
        self._candidates_mode: Literal["detailed", "summary"] = "detailed"
        self._candidates_sampled: bool = False
        self._candidates_stats: Dict[str, Any] = {}
        
        # Track rejected candidates separately for efficient lookup
        self._rejected_candidates: Dict[str, Dict[str, Any]] = {}
        # Format: {candidate_id: {"reason": str, "filter": Optional[str], "data": Dict}}
        
        if candidates:
            self._add_candidates_internal(candidates)
        
        # Process filters
        self._filters: List[StepFilter] = []
        if filters:
            for f in filters:
                if isinstance(f, dict):
                    self._filters.append(StepFilter(**f))
                else:
                    self._filters.append(f)
        
        # Process outputs
        if isinstance(outputs, dict):
            self._outputs = StepOutputs(**outputs)
        else:
            self._outputs = outputs
        
        # Process reasoning
        if isinstance(reasoning, dict):
            self._reasoning = StepReasoning(**reasoning)
        else:
            self._reasoning = reasoning
        
        self._step_id: Optional[str] = None
        self._created = False
    
    def _add_candidates_internal(self, candidates: List[Union[Dict, Candidate]]):
        """Internal method to add candidates without mode parameter."""
        for c in candidates:
            if isinstance(c, dict):
                self._candidates.append(Candidate(**c))
            else:
                self._candidates.append(c)
        self._candidates_total_count = len(self._candidates)
    
    @property
    def step_id(self) -> str:
        return self._step_id or f"step_{self._index:03d}"
    
    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return (
            f"StepBuilder(name={self._name!r}, index={self._index}, "
            f"status={self._status.value}, candidates={self._candidates_total_count}, "
            f"rejected={len(self._rejected_candidates)})"
        )
    
    def _build_payload(self) -> Dict:
        """Build the API request payload with smart candidate handling."""
        payload = {"name": self._name, "status": self._status.value}
        
        if self._step_type:
            payload["step_type"] = self._step_type
        if self._parent_step_id:
            payload["parent_step_id"] = self._parent_step_id
        if self._inputs:
            payload["inputs"] = self._inputs.to_dict()
        
        # Build candidates payload with smart sampling
        if self._candidates or self._candidates_total_count > 0:
            payload["candidates"] = self._build_candidates_payload()
        
        if self._filters:
            payload["filters"] = [f.to_dict() for f in self._filters]
        if self._outputs:
            payload["outputs"] = self._outputs.to_dict()
        if self._reasoning:
            payload["reasoning"] = self._reasoning.to_dict()
        
        return payload
    
    def _build_candidates_payload(self) -> Dict:
        """Build the candidates section of the payload with stats and sampling."""
        candidates_payload = {
            "total_count": self._candidates_total_count,
        }
        
        # Add items (potentially sampled)
        if self._candidates:
            candidates_payload["items"] = [c.to_dict() for c in self._candidates]
        
        # Add statistics if available
        if self._candidates_stats:
            candidates_payload["statistics"] = self._candidates_stats
        
        # Add sampling info if applicable
        if self._candidates_sampled:
            candidates_payload["sampled"] = True
            candidates_payload["sample_size"] = len(self._candidates)
        
        # Add rejected candidates summary
        if self._rejected_candidates:
            candidates_payload["rejected"] = self._build_rejection_summary()
        
        return candidates_payload
    
    def _build_rejection_summary(self) -> Dict:
        """Build a summary of rejected candidates grouped by reason."""
        rejection_by_reason: Dict[str, Dict[str, Any]] = {}
        
        for candidate_id, info in self._rejected_candidates.items():
            reason = info.get("reason", "unknown")
            if reason not in rejection_by_reason:
                rejection_by_reason[reason] = {"count": 0, "sample_ids": []}
            rejection_by_reason[reason]["count"] += 1
            # Keep up to 10 sample IDs per reason for debugging
            if len(rejection_by_reason[reason]["sample_ids"]) < 10:
                rejection_by_reason[reason]["sample_ids"].append(candidate_id)
        
        return {
            "total_count": len(self._rejected_candidates),
            "by_reason": rejection_by_reason,
        }
    
    def _send(self, blocking: bool = False):
        """Send the step creation request."""
        payload = self._build_payload()
        endpoint = f"/runs/{self._run.run_id}/steps"
        
        if blocking:
            try:
                response = self._client._make_request_sync("POST", endpoint, payload)
                self._step_id = response.get("step_id")
                self._created = True
            except Exception as e:
                logger.error(f"Failed to create step: {e}")
        else:
            def _on_create(response, error):
                if response:
                    self._step_id = response.get("step_id")
                    self._created = True
            
            self._client._make_request_async("POST", endpoint, payload, _on_create)
    
    def add_candidates(
        self,
        candidates: List[Union[Dict, Candidate]],
        source: Optional[str] = None,
        mode: Literal["detailed", "summary"] = "detailed",
        sample_size: Optional[int] = None,
        score_key: str = "score",
    ) -> "StepBuilder":
        """
        Add candidates to this step with smart sampling support.
        
        When there are more than 1000 candidates (configurable via SAMPLE_THRESHOLD),
        the SDK automatically:
        - Samples 100 random candidates (configurable)
        - Computes summary statistics (min/max/avg/median scores, etc.)
        - Stores the total count for accurate reporting
        
        Args:
            candidates: List of candidates (dicts or Candidate objects).
            source: Source identifier for these candidates.
            mode: 'detailed' stores all candidates, 'summary' forces sampling.
            sample_size: Override default sample size (default: 100).
            score_key: Key to use for score statistics (default: 'score').
        
        Returns:
            Self for method chaining.
        
        Example:
            # Detailed mode (default) - stores all if < 1000
            step.add_candidates(products)
            
            # Summary mode - always sample and compute stats
            step.add_candidates(products, mode='summary')
            
            # Auto-sampling happens when > 1000 candidates
            step.add_candidates(large_product_list)  # Automatically sampled
        """
        self._candidates_mode = mode
        actual_sample_size = sample_size or self.SAMPLE_SIZE
        
        # Convert all to Candidate objects first
        all_candidates: List[Candidate] = []
        for c in candidates:
            if isinstance(c, dict):
                c_copy = c.copy()
                if source and "source" not in c_copy:
                    c_copy["source"] = source
                all_candidates.append(Candidate(**c_copy))
            else:
                if source and not c.source:
                    c.source = source
                all_candidates.append(c)
        
        total_count = len(all_candidates)
        self._candidates_total_count = total_count
        
        # Determine if we should sample
        should_sample = (
            mode == "summary" or 
            total_count > self.SAMPLE_THRESHOLD
        )
        
        if should_sample and total_count > actual_sample_size:
            # Compute statistics before sampling
            self._candidates_stats = self._compute_candidate_stats(
                all_candidates, score_key
            )
            
            # Random sampling
            sampled = random.sample(all_candidates, actual_sample_size)
            self._candidates = sampled
            self._candidates_sampled = True
            
            logger.debug(
                f"Sampled {actual_sample_size} candidates from {total_count} total"
            )
        else:
            # Store all candidates
            self._candidates = all_candidates
            self._candidates_sampled = False
            
            # Still compute stats if in summary mode
            if mode == "summary":
                self._candidates_stats = self._compute_candidate_stats(
                    all_candidates, score_key
                )
        
        return self
    
    def _compute_candidate_stats(
        self, 
        candidates: List[Candidate],
        score_key: str = "score"
    ) -> Dict[str, Any]:
        """Compute summary statistics for candidates."""
        stats: Dict[str, Any] = {
            "total_count": len(candidates),
        }
        
        # Extract scores
        scores = []
        for c in candidates:
            score = c.score
            if score is None and c.data and score_key in c.data:
                score = c.data[score_key]
            if score is not None:
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    pass
        
        if scores:
            stats["score_stats"] = {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "count": len(scores),
            }
            if len(scores) >= 2:
                stats["score_stats"]["stdev"] = statistics.stdev(scores)
        
        # Count by source
        source_counts: Dict[str, int] = {}
        for c in candidates:
            src = c.source or "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1
        
        if source_counts:
            stats["by_source"] = source_counts
        
        return stats
    
    def reject_candidate(
        self,
        candidate_id: str,
        reason: str,
        filter_name: Optional[str] = None,
        data: Optional[Dict] = None,
    ) -> "StepBuilder":
        """
        Mark a candidate as rejected with a reason.
        
        Rejection information is tracked separately and included in the
        step's candidates payload with a summary grouped by reason.
        
        Args:
            candidate_id: ID of the candidate to reject.
            reason: Human-readable reason for rejection (e.g., 'price_too_high').
            filter_name: Optional name of the filter that rejected this candidate.
            data: Optional additional data about the rejection.
        
        Returns:
            Self for method chaining.
        
        Example:
            step.reject_candidate('prod_123', reason='price_too_high')
            step.reject_candidate('prod_456', reason='out_of_stock', filter_name='availability')
            step.reject_candidate(
                'prod_789', 
                reason='low_score',
                data={'threshold': 0.5, 'actual_score': 0.3}
            )
        """
        # Store in rejected candidates map
        self._rejected_candidates[candidate_id] = {
            "reason": reason,
            "filter": filter_name,
            "data": data or {},
        }
        
        # Also update the candidate object if it exists in our list
        for c in self._candidates:
            if c.id == candidate_id:
                c.accepted = False
                c.rejection_reason = reason
                c.rejected_by_filter = filter_name
                break
        
        return self
    
    def reject_candidates(
        self,
        candidate_ids: List[str],
        reason: str,
        filter_name: Optional[str] = None,
    ) -> "StepBuilder":
        """
        Reject multiple candidates with the same reason.
        
        Convenience method for bulk rejections (e.g., after a filter).
        
        Args:
            candidate_ids: List of candidate IDs to reject.
            reason: Reason for rejection.
            filter_name: Optional filter name.
        
        Returns:
            Self for method chaining.
        
        Example:
            filtered_out = ['prod_1', 'prod_2', 'prod_3']
            step.reject_candidates(filtered_out, reason='price_out_of_range')
        """
        for cid in candidate_ids:
            self.reject_candidate(cid, reason, filter_name)
        return self
    
    def get_rejection_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all rejected candidates grouped by reason.
        
        Returns:
            Dict with rejection statistics.
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for cid, info in self._rejected_candidates.items():
            reason = info.get("reason", "unknown")
            if reason not in summary:
                summary[reason] = {"count": 0, "sample_ids": [], "filters": set()}
            summary[reason]["count"] += 1
            if len(summary[reason]["sample_ids"]) < 10:
                summary[reason]["sample_ids"].append(cid)
            if info.get("filter"):
                summary[reason]["filters"].add(info["filter"])
        
        # Convert sets to lists for JSON serialization
        for reason in summary:
            summary[reason]["filters"] = list(summary[reason]["filters"])
        
        return {
            "total_rejected": len(self._rejected_candidates),
            "by_reason": summary,
        }
    
    def add_filter(
        self,
        filter_type: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        input_count: int = 0,
        output_count: int = 0,
        removed_ids: Optional[List[str]] = None,
        removed_reasons: Optional[Dict[str, str]] = None,
        execution_time_ms: Optional[int] = None,
    ) -> "StepBuilder":
        """Add a filter to this step."""
        metrics = FilterMetrics(
            input_count=input_count,
            output_count=output_count,
            removed_count=input_count - output_count if input_count > output_count else None,
            execution_time_ms=execution_time_ms,
        )
        
        step_filter = StepFilter(
            type=filter_type,
            name=name or filter_type,
            order=len(self._filters),
            config=config or {},
            metrics=metrics,
            removed_sample=removed_ids[:10] if removed_ids else [],
            removed_reasons=removed_reasons or {},
        )
        self._filters.append(step_filter)
        return self
    
    def set_reasoning(
        self,
        algorithm: Optional[str] = None,
        explanation: Optional[str] = None,
        confidence: Optional[float] = None,
        factors: Optional[List[Union[Dict, ReasoningFactor]]] = None,
    ) -> "StepBuilder":
        """Set reasoning information for this step."""
        factor_list = []
        if factors:
            for f in factors:
                if isinstance(f, dict):
                    factor_list.append(ReasoningFactor(**f))
                else:
                    factor_list.append(f)
        
        self._reasoning = StepReasoning(
            algorithm=algorithm,
            explanation=explanation,
            confidence=confidence,
            factors=factor_list,
        )
        return self
    
    def complete(
        self,
        result_ids: Optional[List[str]] = None,
        output_data: Optional[Dict] = None,
        count: Optional[int] = None,
    ):
        """Complete this step successfully."""
        self._status = StepStatus.SUCCESS
        self._outputs = StepOutputs(
            count=count or len(result_ids or []),
            result_ids=result_ids or [],
            data=output_data or {},
        )
        
        payload = {
            "status": StepStatus.SUCCESS.value,
            "outputs": self._outputs.to_dict(),
        }
        if self._reasoning:
            payload["reasoning"] = self._reasoning.to_dict()
        
        endpoint = f"/runs/{self._run.run_id}/steps/{self.step_id}"
        self._client._make_request_async("PATCH", endpoint, payload)
    
    def fail(self, error: Optional[str] = None):
        """Mark this step as failed."""
        self._status = StepStatus.FAILED
        payload = {"status": StepStatus.FAILED.value}
        
        endpoint = f"/runs/{self._run.run_id}/steps/{self.step_id}"
        self._client._make_request_async("PATCH", endpoint, payload)
