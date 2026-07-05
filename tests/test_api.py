"""API integration and unit tests.

Uses FastAPI's TestClient to verify authentication flows, match predictions,
simulations, database integrations, and the frontend API Client.
"""

from __future__ import annotations

import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup test database configuration before importing config/database
test_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
test_db_path = test_db_file.name
test_db_file.close()

# Force environment overrides
os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"
os.environ["ENV"] = "development"
os.environ["DEBUG"] = "true"

from backend.main import app
from backend.database.database import Base, get_db
from backend.database.models import User
from frontend.services.api_client import APIClient

# Setup test database engine
engine = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Override database dependency in FastAPI app
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

# Create Client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    """Create all tables before testing, drop after completion."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    try:
        os.remove(test_db_path)
    except OSError:
        pass


def test_health_and_version():
    """Verify health and version endpoints are functional."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    res_version = client.get("/version")
    assert res_version.status_code == 200
    assert "model_version" in res_version.json()


def test_user_authentication_flow():
    """Test user registration, login, profile retrieval, and token validation."""
    # 1. Register a user
    reg_payload = {"username": "testuser", "email": "test@example.com", "password": "password123"}
    res_reg = client.post("/api/auth/register", json=reg_payload)
    assert res_reg.status_code == 201
    assert res_reg.json()["username"] == "testuser"
    assert "id" in res_reg.json()

    # 2. Login user
    login_payload = {"username": "testuser", "password": "password123"}
    res_login = client.post("/api/auth/login", json=login_payload)
    assert res_login.status_code == 200
    assert "access_token" in res_login.json()
    token = res_login.json()["access_token"]

    # 3. Retrieve user profile
    headers = {"Authorization": f"Bearer {token}"}
    res_prof = client.get("/api/auth/profile", headers=headers)
    assert res_prof.status_code == 200
    assert res_prof.json()["email"] == "test@example.com"

    # 4. Attempt profile access with invalid token
    bad_headers = {"Authorization": "Bearer badtoken123"}
    res_bad_prof = client.get("/api/auth/profile", headers=bad_headers)
    assert res_bad_prof.status_code == 401


def test_predict_endpoint():
    """Test match prediction outputs, probabilities, xG, and SHAP values."""
    payload = {
        "home_team": "Argentina",
        "away_team": "France",
        "tournament": "FIFA World Cup",
        "venue": "neutral",
        "match_date": "2026-06-25"
    }
    response = client.post("/api/predict/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["home_team"] == "Argentina"
    assert data["away_team"] == "France"
    assert "predicted_winner" in data
    assert "probabilities" in data
    assert "expected_goals" in data
    assert "shap_explanation" in data
    assert "home_xg" in data["expected_goals"]


def test_simulate_endpoint():
    """Test tournament Monte Carlo simulations (run count = 100)."""
    # Use small simulation count for testing speed
    payload = {"run_count": 100}
    response = client.post("/api/simulate/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["run_count"] == 100
    assert "champion_odds" in data
    assert "stage_probabilities" in data
    assert len(data["stage_probabilities"]) == 48  # 48 teams simulated


def test_frontend_api_client():
    """Test frontend API Client methods by mocking requests against the TestClient."""
    # Custom requests wrapper to redirect to TestClient
    class MockSession:
        def request(self, method, url, **kwargs):
            # Extract path relative to base url
            path = url.replace("http://testserver", "")
            return client.request(method, path, **kwargs)

    import requests
    # Patch requests.request
    original_request = requests.request
    requests.request = MockSession().request

    try:
        # Instantiate Client
        fc = APIClient(base_url="http://testserver")
        
        # Test auth connection flow
        login_res = fc.login("testuser", "password123")
        assert "access_token" in login_res
        
        # Test fetch profile
        prof = fc.get_profile()
        assert prof["username"] == "testuser"

        # Test predict wrapper
        pred = fc.predict("Argentina", "France")
        assert pred["home_team"] == "Argentina"

        # Test teams list
        teams = fc.get_teams()
        assert "Argentina" in teams

    finally:
        # Restore requests.request
        requests.request = original_request
