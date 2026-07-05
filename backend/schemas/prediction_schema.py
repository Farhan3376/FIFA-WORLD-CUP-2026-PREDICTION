"""Pydantic schemas for Match Predictions and Simulations.

Validates input formats and outputs for predictions and tournament simulations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for predicting match outcomes."""
    home_team: str = Field(..., example="Argentina", description="Name of the home team")
    away_team: str = Field(..., example="France", description="Name of the away team")
    tournament: str = Field(default="FIFA World Cup", example="FIFA World Cup", description="Name of the tournament")
    venue: str = Field(default="neutral", example="neutral", description="Venue state: 'home', 'away', or 'neutral'")
    match_date: Optional[str] = Field(default=None, example="2026-06-25", description="Match date in YYYY-MM-DD format")


class Probabilities(BaseModel):
    """W/D/L probabilities structure."""
    home_win: float = Field(..., description="Probability of home win")
    draw: float = Field(..., description="Probability of a draw")
    away_win: float = Field(..., description="Probability of away win")


class ExpectedGoals(BaseModel):
    """Expected goals (xG) structure."""
    home_xg: float = Field(..., description="Expected goals for home team")
    away_xg: float = Field(..., description="Expected goals for away team")


class PredictionResponse(BaseModel):
    """Response schema containing match prediction analytics."""
    home_team: str
    away_team: str
    predicted_winner: str
    predicted_outcome: str
    probabilities: Probabilities
    confidence_score: float
    expected_goals: ExpectedGoals
    feature_importance: Dict[str, float]
    shap_explanation: Dict[str, Any]
    timestamp: datetime


class SimulationRequest(BaseModel):
    """Request schema to trigger Monte Carlo World Cup simulations."""
    run_count: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of simulations (100, 500, 1000, 5000, 10000)"
    )


class SimulationResponse(BaseModel):
    """Response schema containing tournament odds and results."""
    run_count: int
    champion_odds: Dict[str, float]
    stage_probabilities: List[Dict[str, Any]]
    most_likely_final: Optional[Dict[str, Any]] = None
    upsets: Optional[List[Dict[str, Any]]] = None
    sample_bracket: Optional[Dict[str, Any]] = None
    timestamp: datetime


class Fixture(BaseModel):
    """A single tournament fixture, sourced from the live fixtures API."""
    fixture_id: Optional[str] = None
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = Field(..., description="'scheduled', 'live', or 'finished'")
    group: Optional[str] = None
    matchday: Optional[str] = None
    kickoff_local: Optional[str] = None
    stage: Optional[str] = None


class LiveAdjustedPrediction(BaseModel):
    """Pre-match model prediction blended with a live in-play heuristic adjustment."""
    fixture: Fixture
    baseline_probabilities: Probabilities = Field(..., description="Pre-match model output, unadjusted")
    live_probabilities: Probabilities = Field(..., description="Heuristic-adjusted probabilities reflecting current score")
    is_heuristic: bool = Field(True, description="Always true: live_probabilities are a heuristic blend, not a re-run model")
    timestamp: datetime
