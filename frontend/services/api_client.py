"""API Client for the Streamlit frontend.

Coordinates all communication with the FastAPI backend, handling timeouts,
automatic request retries, token-based authorization headers, and error handling.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Any, List, Optional
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_client")


class APIClient:
    """HTTP client wrapper to fetch data from the FastAPI prediction server."""

    def __init__(self, base_url: Optional[str] = None):
        """Initialize the client, defaulting to local FastAPI port or env variable."""
        self.base_url = base_url or os.getenv("API_URL", "http://localhost:8000")
        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]
            
        self.token: Optional[str] = None
        self.timeout = 180.0  # High timeout for long-running Monte Carlo simulations

    def set_token(self, token: str) -> None:
        """Store the JWT access token in memory for authenticated requests."""
        self.token = token

    def clear_token(self) -> None:
        """Clear the JWT access token (logout)."""
        self.token = None

    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers, adding JWT bearer token if authenticated."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> Dict[str, Any]:
        """Perform an HTTP request with retries, backoff, and exception handling.

        Args:
            method: GET or POST.
            endpoint: URL path segment.
            json_data: Body payload.
            params: Query parameters.
            max_retries: Maximum attempts for transient network issues.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    params=params,
                    timeout=self.timeout,
                )
                
                # Check status codes
                if response.status_code == 401:
                    logger.warning("Unauthenticated request (401) to: %s", url)
                    self.clear_token()  # Token expired or invalid
                    raise requests.HTTPError("Session expired. Please log in again.", response=response)

                if response.status_code >= 500:
                    # Server errors can be retried
                    response.raise_for_status()
                    
                # Success
                try:
                    return response.json()
                except ValueError:
                    return {"detail": response.text}

            except (requests.ConnectionError, requests.Timeout) as e:
                attempt += 1
                logger.warning(
                    "Network error on attempt %d/%d requesting %s: %s. Retrying...",
                    attempt, max_retries, url, e
                )
                if attempt >= max_retries:
                    raise RuntimeError(f"Could not connect to backend prediction server after {max_retries} attempts.") from e
                time.sleep(backoff_factor * (2 ** (attempt - 1)))

            except requests.HTTPError as e:
                # Client errors (4xx) shouldn't be retried
                if e.response is not None and e.response.status_code < 500:
                    try:
                        err_detail = e.response.json().get("detail", str(e))
                    except Exception:
                        err_detail = e.response.text or str(e)
                    raise ValueError(err_detail) from e
                
                # Server error retrying
                attempt += 1
                if attempt >= max_retries:
                    raise RuntimeError("Internal Server Error from prediction server.") from e
                time.sleep(backoff_factor * (2 ** (attempt - 1)))

        raise RuntimeError("Request failed unexpectedly.")

    # --- Authentication Methods ---

    def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new account on the server."""
        payload = {"username": username, "email": email, "password": password}
        return self._request("POST", "/api/auth/register", json_data=payload)

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Log in and capture the JWT token."""
        payload = {"username": username, "password": password}
        result = self._request("POST", "/api/auth/login", json_data=payload)
        if "access_token" in result:
            self.set_token(result["access_token"])
        return result

    def get_profile(self) -> Dict[str, Any]:
        """Retrieve current user profile details."""
        return self._request("GET", "/api/auth/profile")

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Retrieve past predictions run by the authenticated user."""
        return self._request("GET", "/api/auth/prediction-history")  # type: ignore

    # --- Prediction & Simulation Methods ---

    def predict(
        self,
        home_team: str,
        away_team: str,
        tournament: str = "FIFA World Cup",
        venue: str = "neutral",
        match_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query match prediction probabilities and SHAP insights."""
        payload = {
            "home_team": home_team,
            "away_team": away_team,
            "tournament": tournament,
            "venue": venue,
            "match_date": match_date
        }
        return self._request("POST", "/api/predict/", json_data=payload)

    def simulate(self, run_count: int = 1000) -> Dict[str, Any]:
        """Trigger tournament Monte Carlo simulations."""
        payload = {"run_count": run_count}
        return self._request("POST", "/api/simulate/", json_data=payload)

    # --- Analytics & Statistics Methods ---

    def get_teams(self) -> List[str]:
        """Fetch list of all recognized national teams."""
        return self._request("GET", "/api/analytics/teams")  # type: ignore

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Fetch ELO rankings and detailed rolling statistics for a team."""
        return self._request("GET", f"/api/analytics/team/{team_name}")

    def get_global_rankings(self) -> Dict[str, Any]:
        """Fetch global ELO leaders, goal distributions, and stats summaries."""
        return self._request("GET", "/api/analytics/global")

    def get_historical_replay(self) -> Dict[str, Any]:
        """Fetch historical validation accuracy reports and prediction replay logs."""
        return self._request("GET", "/api/analytics/historical")

    def get_feature_importance(self) -> Dict[str, float]:
        """Fetch global feature importances derived from model parameters."""
        return self._request("GET", "/api/analytics/feature-importance")  # type: ignore

    def get_model_performance(self) -> Dict[str, Any]:
        """Fetch accuracy statistics, ECE, Brier Scores, and model validation details."""
        return self._request("GET", "/api/analytics/model-performance")

    # --- Live Fixtures Methods ---
    # Backed by the public worldcup2026 API (github.com/rezarahiminia/worldcup2026).
    # Third-party, unofficial, best-effort data -- callers should handle failures gracefully.

    def get_qualified_teams(self) -> List[str]:
        """Fetch the 48 real FIFA World Cup 2026 qualified teams (not the full historical ELO database)."""
        return self._request("GET", "/api/live/teams")  # type: ignore

    def get_fixtures(self) -> List[Dict[str, Any]]:
        """Fetch all known tournament fixtures (scheduled, live, and finished)."""
        return self._request("GET", "/api/live/fixtures")  # type: ignore

    def get_live_fixtures(self) -> List[Dict[str, Any]]:
        """Fetch only fixtures currently in progress."""
        return self._request("GET", "/api/live/fixtures/live")  # type: ignore

    def get_live_adjusted_prediction(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Fetch the pre-match prediction blended with the current live score, if the fixture is live."""
        return self._request("GET", f"/api/live/predict/{home_team}/{away_team}")
