"""Tournament simulation API router.

Exposes endpoints to trigger Monte Carlo simulations of the FIFA World Cup 2026
and record statistical summaries in database logs.
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import User, SimulationHistory
from backend.schemas.prediction_schema import SimulationRequest, SimulationResponse
from backend.services.simulation_service import SimulationService
from backend.api.routes.prediction import get_optional_user

router = APIRouter()
logger = logging.getLogger("backend")


@router.post("/", response_model=SimulationResponse)
async def simulate_tournament(
    request: SimulationRequest,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Execute Monte Carlo tournament simulation and compile advancement probabilities.

    Saves summary stats to user history if authenticated.
    """
    # 1. Run simulation service
    result = SimulationService.run_simulation(request)

    # 2. Persist to simulation history
    try:
        user_id = current_user.id if current_user else None
        db_simulation = SimulationHistory(
            user_id=user_id,
            run_count=result["run_count"],
            champion_odds=result["champion_odds"],
            stage_probabilities=result["stage_probabilities"],
        )
        db.add(db_simulation)
        db.commit()
        logger.info("Saved tournament simulation history to database (User ID: %s).", user_id)
    except Exception as e:
        logger.error("Failed to write simulation query to database: %s", e)
        db.rollback()

    return result
