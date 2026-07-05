"""Live fixtures & in-play prediction API router.

Exposes real tournament fixtures and scores sourced from the public
worldcup2026 API (github.com/rezarahiminia/worldcup2026), plus a heuristic
live-adjusted prediction for matches currently in progress.

This data source is third-party and unofficial with no SLA -- endpoints
degrade to an empty list / 503 rather than crashing when it's unreachable.
"""

from __future__ import annotations

import logging
from typing import List
from fastapi import APIRouter, HTTPException, status

from backend.schemas.prediction_schema import Fixture, LiveAdjustedPrediction
from backend.services.live_fixtures_service import LiveFixturesService
from backend.services.prediction_service import PredictionService
from backend.schemas.prediction_schema import PredictionRequest

router = APIRouter()
logger = logging.getLogger("backend")


@router.get("/teams", response_model=List[str])
def get_qualified_teams():
    """Retrieve the 48 real FIFA World Cup 2026 qualified teams (not the full historical ELO database)."""
    return LiveFixturesService.get_qualified_teams()


@router.get("/fixtures", response_model=List[Fixture])
def get_fixtures():
    """Retrieve all known tournament fixtures (scheduled, live, and finished)."""
    return LiveFixturesService.get_fixtures()


@router.get("/fixtures/live", response_model=List[Fixture])
def get_live_fixtures():
    """Retrieve only fixtures currently in progress."""
    return LiveFixturesService.get_live_fixtures()


@router.get("/predict/{home_team}/{away_team}", response_model=LiveAdjustedPrediction)
def get_live_adjusted_prediction(home_team: str, away_team: str):
    """Return the pre-match model prediction blended with the current live score.

    If the fixture isn't currently live (not found, scheduled, or finished),
    this still returns the baseline prediction with live_probabilities equal
    to baseline_probabilities (no adjustment applied).
    """
    fixture = LiveFixturesService.get_fixture_by_teams(home_team, away_team)
    if not fixture:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No fixture found for {home_team} vs {away_team} in the live fixtures feed.",
        )

    baseline = PredictionService.predict_match(
        PredictionRequest(home_team=home_team, away_team=away_team)
    )
    baseline_probs = baseline["probabilities"]

    if fixture["status"] == "live" and fixture["home_score"] is not None and fixture["away_score"] is not None:
        live_probs = PredictionService.apply_live_score_adjustment(
            baseline_probs, fixture["home_score"], fixture["away_score"]
        )
    else:
        live_probs = baseline_probs

    return {
        "fixture": fixture,
        "baseline_probabilities": baseline_probs,
        "live_probabilities": live_probs,
        "is_heuristic": True,
        "timestamp": baseline["timestamp"],
    }
