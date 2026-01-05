"""
Example usage of the X-Ray SDK.

This file demonstrates the various features of the X-Ray SDK for
tracking multi-step decision processes.
"""

from xray_sdk import (
    XRayClient, XRayConfig,
    Candidate, StepFilter, FilterMetrics,
    ReasoningFactor,
)
from xray_sdk.decorators import trace_step, trace_run


# =============================================================================
# Example 1: Basic Usage with Context Manager
# =============================================================================

def example_basic_usage():
    """Basic example showing run and step creation."""
    
    # Initialize client (uses default localhost URL)
    client = XRayClient(
        base_url="http://localhost:8000/api/v1",
        api_key="your-api-key"  # Optional
    )
    
    # Start a run with context manager (auto-completes)
    with client.run(
        name="Product Recommendation Pipeline",
        user_id="user_123",
        session_id="sess_456",
        tags=["production", "recommendations"]
    ) as run:
        
        # Add a candidate generation step
        step1 = run.add_step(
            name="Vector Search",
            step_type="candidate_generation",
            inputs={"query": "wireless headphones", "limit": 100}
        )
        
        # Add candidates that were found
        step1.add_candidates([
            {"id": "prod_001", "score": 0.95, "data": {"name": "Sony WH-1000XM5"}},
            {"id": "prod_002", "score": 0.89, "data": {"name": "Bose QC45"}},
            {"id": "prod_003", "score": 0.85, "data": {"name": "Apple AirPods Max"}},
        ], source="vector_search")
        
        step1.complete(result_ids=["prod_001", "prod_002", "prod_003"], count=3)
        
        # Add a filtering step
        step2 = run.add_step(
            name="Apply Business Rules",
            step_type="filtering",
            inputs={"candidate_ids": ["prod_001", "prod_002", "prod_003"]}
        )
        
        # Add a filter with metrics
        step2.add_filter(
            filter_type="price_range",
            name="Price Range Filter",
            config={"min_price": 50, "max_price": 500},
            input_count=3,
            output_count=2,
            removed_ids=["prod_003"],
            removed_reasons={"prod_003": "Price exceeds maximum of $500"},
            execution_time_ms=12
        )
        
        # Mark a candidate as rejected with reason
        step2.reject_candidate(
            "prod_003",
            reason="Price exceeds budget",
            filter_name="price_range"
        )
        
        step2.complete(result_ids=["prod_001", "prod_002"], count=2)
        
        # Add a ranking step with reasoning
        step3 = run.add_step(
            name="Final Ranking",
            step_type="ranking",
            inputs={"candidate_ids": ["prod_001", "prod_002"]}
        )
        
        step3.set_reasoning(
            algorithm="hybrid_ranking",
            explanation="Combined semantic similarity with popularity and user preferences",
            confidence=0.92,
            factors=[
                {"name": "semantic_similarity", "value": 0.95, "weight": 0.5},
                {"name": "popularity_score", "value": 0.88, "weight": 0.3},
                {"name": "user_preference_match", "value": 0.75, "weight": 0.2},
            ]
        )
        
        step3.complete(result_ids=["prod_001", "prod_002"])
    
    # Run is automatically completed here
    print("Run completed successfully!")
    
    client.close()


# =============================================================================
# Example 2: Using Decorators
# =============================================================================

class RecommendationPipeline:
    """Example pipeline using decorators for automatic tracing."""
    
    def __init__(self):
        self.xray_client = XRayClient()
        self.xray_run = None  # Will be set by @trace_run
    
    @trace_run(name="Recommendation Pipeline", tags=["ml-pipeline"])
    def execute(self, query: str, user_id: str):
        """Main pipeline execution - automatically wrapped in a run."""
        candidates = self.generate_candidates(query)
        filtered = self.apply_filters(candidates)
        ranked = self.rank_candidates(filtered)
        return ranked
    
    @trace_step(step_type="candidate_generation")
    def generate_candidates(self, query: str):
        """Generate initial candidates - automatically traced as a step."""
        # Simulated search
        return [
            {"id": "prod_001", "score": 0.95, "name": "Product A"},
            {"id": "prod_002", "score": 0.89, "name": "Product B"},
            {"id": "prod_003", "score": 0.82, "name": "Product C"},
        ]
    
    @trace_step(step_type="filtering")
    def apply_filters(self, candidates):
        """Apply business filters - automatically traced as a step."""
        # Simulated filtering
        return [c for c in candidates if c["score"] > 0.85]
    
    @trace_step(step_type="ranking")
    def rank_candidates(self, candidates):
        """Final ranking - automatically traced as a step."""
        return sorted(candidates, key=lambda x: x["score"], reverse=True)


def example_decorators():
    """Example using decorators for automatic tracing."""
    pipeline = RecommendationPipeline()
    result = pipeline.execute("wireless headphones", "user_123")
    print(f"Got {len(result)} recommendations")
    pipeline.xray_client.close()


# =============================================================================
# Example 3: Configuration from Environment
# =============================================================================

def example_env_config():
    """Example using environment-based configuration."""
    import os
    
    # Set environment variables
    os.environ["XRAY_BASE_URL"] = "https://api.xray-system.com/v1"
    os.environ["XRAY_API_KEY"] = "your-api-key"
    os.environ["XRAY_ENVIRONMENT"] = "production"
    os.environ["XRAY_ASYNC_ENABLED"] = "true"
    os.environ["XRAY_MAX_RETRIES"] = "5"
    
    # Configuration is automatically loaded from environment
    config = XRayConfig.from_env()
    
    client = XRayClient(config=config)
    
    with client.run("My Pipeline") as run:
        run.add_step("Step 1", inputs={"key": "value"})
    
    client.close()


# =============================================================================
# Example 4: Handling API Failures Gracefully
# =============================================================================

def example_failure_handling():
    """
    Example showing graceful failure handling.
    
    The SDK automatically:
    - Retries failed requests with exponential backoff
    - Queues requests locally when API is unavailable
    - Persists queue to disk for crash recovery
    - Flushes queue in background when API becomes available
    """
    from xray_sdk.config import RetryConfig, QueueConfig, AsyncConfig
    
    config = XRayConfig(
        base_url="http://localhost:8000/api/v1",
        
        # Configure retry behavior
        retry_config=RetryConfig(
            max_retries=5,
            initial_delay_seconds=0.5,
            max_delay_seconds=30,
            exponential_backoff=True,
        ),
        
        # Configure local queue for failures
        queue_config=QueueConfig(
            max_queue_size=1000,
            flush_interval_seconds=5,
            persist_to_disk=True,
            persistence_path="~/.xray_sdk/queue",
        ),
        
        # Configure async behavior
        async_config=AsyncConfig(
            enabled=True,  # Non-blocking API calls
            max_workers=4,
            request_timeout_seconds=30,
        ),
    )
    
    client = XRayClient(config=config)
    
    # All these calls are non-blocking and will be queued if API fails
    with client.run("Pipeline") as run:
        for i in range(10):
            run.add_step(f"Step {i}", inputs={"iteration": i})
    
    # Client will attempt to flush queue before closing
    client.close()


# =============================================================================
# Example 5: Detailed Candidate Tracking
# =============================================================================

def example_detailed_candidate_tracking():
    """Example showing detailed candidate acceptance/rejection tracking."""
    
    client = XRayClient()
    
    with client.run("Search Pipeline", user_id="user_123") as run:
        # Create step
        step = run.add_step(
            name="Search and Filter",
            step_type="search",
            inputs={"query": "laptop", "max_results": 100}
        )
        
        # Add all initial candidates
        all_candidates = [
            Candidate(id="item_1", score=0.95, data={"name": "MacBook Pro", "price": 2499}),
            Candidate(id="item_2", score=0.90, data={"name": "Dell XPS 15", "price": 1799}),
            Candidate(id="item_3", score=0.85, data={"name": "ThinkPad X1", "price": 1599}),
            Candidate(id="item_4", score=0.80, data={"name": "HP Spectre", "price": 1499}),
            Candidate(id="item_5", score=0.75, data={"name": "Surface Pro", "price": 999}),
        ]
        
        step.add_candidates(all_candidates, source="elasticsearch")
        
        # Apply price filter
        step.add_filter(
            filter_type="price_range",
            name="Budget Filter",
            config={"max_price": 2000},
            input_count=5,
            output_count=4,
            removed_ids=["item_1"],
            removed_reasons={"item_1": "Price $2499 exceeds budget of $2000"}
        )
        step.reject_candidate("item_1", "Over budget", "price_range")
        
        # Apply brand filter
        step.add_filter(
            filter_type="brand_preference",
            name="Preferred Brands",
            config={"preferred_brands": ["Dell", "Lenovo"]},
            input_count=4,
            output_count=2,
            removed_ids=["item_4", "item_5"],
            removed_reasons={
                "item_4": "HP not in preferred brands",
                "item_5": "Microsoft not in preferred brands"
            }
        )
        step.reject_candidate("item_4", "Not preferred brand", "brand_preference")
        step.reject_candidate("item_5", "Not preferred brand", "brand_preference")
        
        # Set reasoning for final selection
        step.set_reasoning(
            algorithm="multi_criteria_filter",
            explanation="Applied budget and brand preference filters to narrow down options",
            confidence=0.95,
            factors=[
                ReasoningFactor(name="budget_match", value=1.0, weight=0.6),
                ReasoningFactor(name="brand_preference", value=1.0, weight=0.4),
            ]
        )
        
        # Complete with final results
        step.complete(result_ids=["item_2", "item_3"], count=2)
    
    client.close()
    print("Detailed tracking complete!")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("X-Ray SDK Examples")
    print("=" * 50)
    
    print("\n1. Basic Usage Example:")
    print("-" * 30)
    # example_basic_usage()  # Uncomment to run
    
    print("\n2. Decorator Example:")
    print("-" * 30)
    # example_decorators()  # Uncomment to run
    
    print("\n3. Environment Config Example:")
    print("-" * 30)
    # example_env_config()  # Uncomment to run
    
    print("\n4. Failure Handling Example:")
    print("-" * 30)
    # example_failure_handling()  # Uncomment to run
    
    print("\n5. Detailed Candidate Tracking Example:")
    print("-" * 30)
    # example_detailed_candidate_tracking()  # Uncomment to run
    
    print("\nAll examples defined! Uncomment the ones you want to run.")
