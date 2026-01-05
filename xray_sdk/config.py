"""
Configuration module for X-Ray SDK.

Provides configuration options for the XRayClient including
API endpoints, retry behavior, and async queue settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RetryConfig:
    """Configuration for retry behavior on API failures."""
    
    max_retries: int = 3
    """Maximum number of retry attempts for failed requests."""
    
    initial_delay_seconds: float = 0.5
    """Initial delay between retries (doubles with each retry)."""
    
    max_delay_seconds: float = 30.0
    """Maximum delay between retries."""
    
    retry_on_status_codes: tuple = (429, 500, 502, 503, 504)
    """HTTP status codes that should trigger a retry."""
    
    exponential_backoff: bool = True
    """Whether to use exponential backoff for retries."""


@dataclass
class QueueConfig:
    """Configuration for the local queue used during API failures."""
    
    max_queue_size: int = 1000
    """Maximum number of items to queue locally."""
    
    flush_interval_seconds: float = 5.0
    """How often to attempt flushing the queue."""
    
    persist_to_disk: bool = True
    """Whether to persist queued items to disk for crash recovery."""
    
    persistence_path: Optional[str] = None
    """Path for persisting queue data. Defaults to ~/.xray_sdk/queue."""
    
    batch_size: int = 50
    """Number of items to send in each batch when flushing."""


@dataclass
class AsyncConfig:
    """Configuration for asynchronous request handling."""
    
    enabled: bool = True
    """Whether to send data asynchronously (non-blocking)."""
    
    max_workers: int = 4
    """Maximum number of worker threads for async requests."""
    
    request_timeout_seconds: float = 30.0
    """Timeout for individual API requests."""
    
    shutdown_timeout_seconds: float = 10.0
    """Timeout when shutting down the async executor."""


@dataclass
class XRayConfig:
    """
    Main configuration class for the X-Ray SDK.
    
    Example:
        config = XRayConfig(
            base_url="https://api.xray-system.com/v1",
            api_key="your-api-key",
            async_config=AsyncConfig(enabled=True, max_workers=4)
        )
        client = XRayClient(config=config)
    """
    
    base_url: str = "http://localhost:8000/api/v1"
    """Base URL for the X-Ray API."""
    
    api_key: Optional[str] = None
    """API key for authentication (if required)."""
    
    api_key_header: str = "X-API-Key"
    """Header name for the API key."""
    
    default_timeout_seconds: float = 30.0
    """Default timeout for API requests."""
    
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    """Configuration for retry behavior."""
    
    queue_config: QueueConfig = field(default_factory=QueueConfig)
    """Configuration for local queue during failures."""
    
    async_config: AsyncConfig = field(default_factory=AsyncConfig)
    """Configuration for asynchronous requests."""
    
    default_metadata: Dict[str, Any] = field(default_factory=dict)
    """Default metadata to include in all runs."""
    
    environment: Optional[str] = None
    """Environment name (e.g., production, staging, development)."""
    
    debug: bool = False
    """Enable debug logging."""
    
    dry_run: bool = False
    """
    Enable dry-run mode for testing without API calls.
    
    When enabled:
    - No HTTP requests are made to the API
    - All operations are logged at INFO level
    - Runs and steps get generated IDs
    - Useful for development and testing
    """
    
    @classmethod
    def from_env(cls) -> "XRayConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            XRAY_BASE_URL: Base URL for the API
            XRAY_API_KEY: API key for authentication
            XRAY_ENVIRONMENT: Environment name
            XRAY_DEBUG: Enable debug mode (true/false)
            XRAY_ASYNC_ENABLED: Enable async mode (true/false)
            XRAY_MAX_RETRIES: Maximum retry attempts
            XRAY_QUEUE_SIZE: Maximum queue size
            XRAY_PERSIST_QUEUE: Persist queue to disk (true/false)
        
        Returns:
            XRayConfig instance configured from environment variables.
        """
        def parse_bool(value: Optional[str], default: bool = False) -> bool:
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes")
        
        def parse_int(value: Optional[str], default: int) -> int:
            if value is None:
                return default
            try:
                return int(value)
            except ValueError:
                return default
        
        def parse_float(value: Optional[str], default: float) -> float:
            if value is None:
                return default
            try:
                return float(value)
            except ValueError:
                return default
        
        retry_config = RetryConfig(
            max_retries=parse_int(os.getenv("XRAY_MAX_RETRIES"), 3),
        )
        
        queue_config = QueueConfig(
            max_queue_size=parse_int(os.getenv("XRAY_QUEUE_SIZE"), 1000),
            persist_to_disk=parse_bool(os.getenv("XRAY_PERSIST_QUEUE"), True),
            persistence_path=os.getenv("XRAY_QUEUE_PATH"),
        )
        
        async_config = AsyncConfig(
            enabled=parse_bool(os.getenv("XRAY_ASYNC_ENABLED"), True),
            max_workers=parse_int(os.getenv("XRAY_MAX_WORKERS"), 4),
            request_timeout_seconds=parse_float(
                os.getenv("XRAY_REQUEST_TIMEOUT"), 30.0
            ),
        )
        
        return cls(
            base_url=os.getenv("XRAY_BASE_URL", "http://localhost:8000/api/v1"),
            api_key=os.getenv("XRAY_API_KEY"),
            environment=os.getenv("XRAY_ENVIRONMENT"),
            debug=parse_bool(os.getenv("XRAY_DEBUG"), False),
            dry_run=parse_bool(os.getenv("XRAY_DRY_RUN"), False),
            retry_config=retry_config,
            queue_config=queue_config,
            async_config=async_config,
        )
    
    def with_overrides(self, **kwargs) -> "XRayConfig":
        """
        Create a new config with specified overrides.
        
        Args:
            **kwargs: Configuration fields to override.
            
        Returns:
            New XRayConfig instance with overrides applied.
        """
        current = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_key_header": self.api_key_header,
            "default_timeout_seconds": self.default_timeout_seconds,
            "retry_config": self.retry_config,
            "queue_config": self.queue_config,
            "async_config": self.async_config,
            "default_metadata": self.default_metadata.copy(),
            "environment": self.environment,
            "debug": self.debug,
            "dry_run": self.dry_run,
        }
        current.update(kwargs)
        return XRayConfig(**current)
