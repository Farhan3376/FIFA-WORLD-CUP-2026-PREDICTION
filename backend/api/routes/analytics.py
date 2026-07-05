"""Analytics and statistics API router.

Exposes read-only endpoints to retrieve team databases, rankings,
historical evaluations, feature importances, and model calibration metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status

from backend.services.analytics_service import AnalyticsService

router = APIRouter()
logger = logging.getLogger("backend")


@router.get("/teams", response_model=List[str])
def get_teams():
    """Retrieve list of all teams currently registered in the ELO database."""
    return AnalyticsService.get_all_teams()


@router.get("/team/{team_name}", response_model=Dict[str, Any])
def get_team_stats(team_name: str):
    """Retrieve database stats, ELO, form index, and historical records for a team."""
    stats = AnalyticsService.get_team_details(team_name)
    if not stats.get("recognized", False):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team '{team_name}' was not found in the database."
        )
    return stats


@router.get("/global", response_model=Dict[str, Any])
def get_global_rankings():
    """Retrieve global database benchmarks, top ELO rankings, and goal distributions."""
    return AnalyticsService.get_global_analytics()


@router.get("/historical", response_model=Dict[str, Any])
def get_historical_reports():
    """Retrieve historical World Cup prediction replay logs and accuracy scores."""
    return AnalyticsService.get_historical_replay()


@router.get("/feature-importance", response_model=Dict[str, float])
def get_feature_importances():
    """Retrieve global feature importance weights computed from the LightGBM classifier."""
    return AnalyticsService.get_global_feature_importances()


@router.get("/model-performance", response_model=Dict[str, Any])
def get_model_performance():
    """Retrieve ML model training configurations, calibration logs, and sensitivity metrics."""
    return AnalyticsService.get_model_performance_details()
