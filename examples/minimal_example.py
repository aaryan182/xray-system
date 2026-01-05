"""
Minimal X-Ray SDK Example

This is the absolute minimum code needed to start tracking decisions
with X-Ray. Just 3 patterns to get immediate value.
"""

from xray_sdk import XRayClient

# =============================================================================
# PATTERN 1: Simplest Possible Usage (3 lines)
# =============================================================================

def pattern_1_simplest():
    """
    The absolute minimum: create client, start run, add step.
    """
    xray = XRayClient(base_url="http://localhost:8000/api/v1")
    
    with xray.run("My Pipeline") as run:
        run.add_step("Process Data", inputs={"query": "example"})


# =============================================================================
# PATTERN 2: Wrap Existing Code (Most Common)
# =============================================================================

def my_existing_function(input_data):
    """Your existing business logic - unchanged."""
    # Simulate processing
    return {"processed": True, "items": [1, 2, 3]}


def pattern_2_wrap_existing():
    """
    Add X-Ray tracking to existing code with minimal changes.
    Your existing code stays exactly the same.
    """
    xray = XRayClient(base_url="http://localhost:8000/api/v1")
    
    # Your existing input
    input_data = {"user_id": "user_123", "query": "headphones"}
    
    with xray.run("my_pipeline") as run:
        
        # Your existing code - UNCHANGED
        results = my_existing_function(input_data)
        
        # Add X-Ray tracking - ONE LINE
        run.add_step(
            name="my_function",
            inputs={"data": input_data},
            outputs={"results": results}
        )
    
    return results


# =============================================================================
# PATTERN 3: Track Multiple Steps
# =============================================================================

def fetch_candidates(query):
    """Mock: fetch items from database/API."""
    return [
        {"id": "item_1", "score": 0.9},
        {"id": "item_2", "score": 0.7},
        {"id": "item_3", "score": 0.5},
    ]


def filter_candidates(items, min_score):
    """Mock: filter by criteria."""
    return [i for i in items if i["score"] >= min_score]


def rank_candidates(items):
    """Mock: rank remaining items."""
    return sorted(items, key=lambda x: x["score"], reverse=True)


def pattern_3_multiple_steps():
    """
    Track a multi-step pipeline with candidate filtering.
    """
    xray = XRayClient(base_url="http://localhost:8000/api/v1")
    
    with xray.run("Search Pipeline", user_id="user_123") as run:
        
        # Step 1: Fetch candidates
        candidates = fetch_candidates("wireless headphones")
        step1 = run.add_step(
            name="Fetch Candidates",
            step_type="candidate_generation",
            inputs={"query": "wireless headphones"},
        )
        step1.add_candidates(candidates, source="database")
        step1.complete(result_ids=[c["id"] for c in candidates])
        
        # Step 2: Filter
        filtered = filter_candidates(candidates, min_score=0.6)
        step2 = run.add_step(
            name="Filter by Score",
            step_type="filtering",
            inputs={"min_score": 0.6, "input_count": len(candidates)},
        )
        # Track why items were rejected
        for c in candidates:
            if c["score"] < 0.6:
                step2.reject_candidate(c["id"], reason=f"Score {c['score']} < 0.6")
        step2.complete(result_ids=[c["id"] for c in filtered])
        
        # Step 3: Rank
        ranked = rank_candidates(filtered)
        step3 = run.add_step(
            name="Rank Results",
            step_type="ranking",
        )
        step3.add_candidates(ranked, source="filtered_results")
        step3.set_reasoning(
            algorithm="score_descending",
            explanation="Sorted by relevance score",
            confidence=0.95,
        )
        step3.complete(result_ids=[c["id"] for c in ranked])
    
    return ranked


# =============================================================================
# PATTERN 4: Using Context Manager (Recommended)
# =============================================================================

def pattern_4_context_manager():
    """
    Using context manager ensures run is always completed/failed properly.
    Even if an exception occurs, the run will be marked as failed.
    """
    xray = XRayClient(base_url="http://localhost:8000/api/v1")
    
    try:
        with xray.run(
            name="Recommendation Pipeline",
            user_id="user_456",
            session_id="session_789",
            tags=["production", "v1.0"],
        ) as run:
            
            # Step 1
            run.add_step("Load Data", inputs={"source": "database"})
            
            # Step 2
            run.add_step("Process", inputs={"algorithm": "v2"})
            
            # Step 3
            run.add_step("Output", outputs={"count": 10})
            
            # Run completes automatically when exiting 'with' block
            
    finally:
        xray.close()  # Clean up resources


# =============================================================================
# Run Examples
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("X-Ray SDK Minimal Examples")
    print("=" * 60)
    
    print("\n1. Simplest possible usage...")
    try:
        pattern_1_simplest()
        print("   Done - check /api/v1/runs for the new run")
    except Exception as e:
        print(f"   Error (is API running?): {e}")
    
    print("\n2. Wrapping existing code...")
    try:
        result = pattern_2_wrap_existing()
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Multiple steps with filtering...")
    try:
        ranked = pattern_3_multiple_steps()
        print(f"   Final ranked items: {[r['id'] for r in ranked]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Context manager pattern...")
    try:
        pattern_4_context_manager()
        print("   Done - run completed cleanly")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("View runs at: http://localhost:8000/api/v1/runs/query")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 60)
