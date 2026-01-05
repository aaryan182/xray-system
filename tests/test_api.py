"""
Integration tests for the X-Ray Debugging API.

Tests creating runs, adding steps, filtering, pagination, and error handling.
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from api.main import app
from api.database import Base, get_db

# Use a temporary SQLite database for testing
TEST_DATABASE_URL = "sqlite:///./test_xray.db"

engine = create_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override the dependency
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Create a clean database for each test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup after tests
    if os.path.exists("./test_xray.db"):
        pass # Keep it for debugging if needed, or delete it


def test_create_run():
    """Test creating a new run."""
    response = client.post(
        "/api/v1/runs",
        json={
            "name": "Standard Retrieval Run",
            "pipeline": "search_v1",
            "metadata": {"environment": "test"}
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Standard Retrieval Run"
    assert data["pipeline"] == "search_v1"
    assert "run_id" in data
    assert data["status"] == "running"


def test_add_step_to_run():
    """Test adding a step to an existing run."""
    # 1. Create run
    run_resp = client.post("/api/v1/runs", json={"name": "Step Test Run"})
    run_id = run_resp.json()["run_id"]
    
    # 2. Add step
    step_data = {
        "name": "Filter Low Scores",
        "step_type": "filter",
        "candidates": {"total_count": 100},
        "outputs": {"count": 75}
    }
    response = client.post(f"/api/v1/runs/{run_id}/steps", json=step_data)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Filter Low Scores"
    assert data["run_id"] == run_id
    assert data["step_index"] == 0


def test_query_runs_filters():
    """Test querying runs with various filters."""
    # 1. Create runs
    r1 = client.post("/api/v1/runs", json={"name": "Prod Run", "pipeline": "prod_p"}).json()
    r2 = client.post("/api/v1/runs", json={"name": "Dev Run", "pipeline": "dev_p"}).json()
    
    # 2. Update status for Dev Run
    client.patch(f"/api/v1/runs/{r2['run_id']}", json={"status": "completed"})
    
    # Filter by status
    resp = client.get("/api/v1/runs/query?status=completed")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["runs"][0]["name"] == "Dev Run"
    
    # Filter by pipeline
    resp = client.get("/api/v1/runs/query?pipeline=prod_p")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["runs"][0]["name"] == "Prod Run"


def test_rejection_rate_filter():
    """Test filtering by rejection rate computed in summary."""
    # 1. Create run
    run_resp = client.post("/api/v1/runs", json={"name": "High Reject Run"})
    run_id = run_resp.json()["run_id"]
    
    # 2. Add high rejection step (100 -> 10)
    client.post(f"/api/v1/runs/{run_id}/steps", json={
        "name": "Heavy Filter",
        "candidates": {"total_count": 100},
        "outputs": {"count": 10}
    })
    
    # 3. Add low rejection run
    low_run_resp = client.post("/api/v1/runs", json={"name": "Low Reject Run"})
    low_run_id = low_run_resp.json()["run_id"]
    client.post(f"/api/v1/runs/{low_run_id}/steps", json={
        "name": "Light Filter",
        "candidates": {"total_count": 100},
        "outputs": {"count": 90}
    })
    
    # 4. Compute summaries
    client.post(f"/api/v1/runs/{run_id}/compute-summary")
    client.post(f"/api/v1/runs/{low_run_id}/compute-summary")
    
    # 5. Query for rejection_rate > 0.8
    resp = client.get("/api/v1/runs/query?rejection_rate_min=0.8")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["runs"][0]["name"] == "High Reject Run"


def test_pagination():
    """Test pagination limit and offset."""
    # Create 5 runs
    for i in range(5):
        client.post("/api/v1/runs", json={"name": f"Run {i}"})
    
    # Page 1 (limit 2)
    resp = client.get("/api/v1/runs/query?limit=2&offset=0")
    assert len(resp.json()["runs"]) == 2
    assert resp.json()["total"] == 5
    assert resp.json()["has_more"] is True
    
    # Page 3 (limit 2, offset 4)
    resp = client.get("/api/v1/runs/query?limit=2&offset=4")
    assert len(resp.json()["runs"]) == 1
    assert resp.json()["has_more"] is False


def test_error_cases():
    """Test error handling for non-existent IDs and malformed data."""
    # Invalid Run ID
    resp = client.get("/api/v1/runs/run_invalid")
    assert resp.status_code == 404
    
    # Malformed Create Data (missing name)
    resp = client.post("/api/v1/runs", json={"pipeline": "missing_name"})
    assert resp.status_code == 422
    
    # Invalid Status
    resp = client.get("/api/v1/runs/query?status=invalid_status")
    assert resp.status_code == 422


def test_get_run_with_steps():
    """Test getting a run with all its steps."""
    # 1. Create run and steps
    run_resp = client.post("/api/v1/runs", json={"name": "Deep Run"})
    run_id = run_resp.json()["run_id"]
    client.post(f"/api/v1/runs/{run_id}/steps", json={"name": "S1"})
    client.post(f"/api/v1/runs/{run_id}/steps", json={"name": "S2"})
    
    # 2. Get run
    resp = client.get(f"/api/v1/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["steps"]) == 2
    assert data["steps"][0]["name"] == "S1"
    assert data["steps"][1]["name"] == "S2"
