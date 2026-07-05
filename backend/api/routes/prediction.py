"""Match prediction API router.

Exposes endpoints to query the LightGBM match prediction model,
explain outcomes using SHAP, and persist predictions in database logs.
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Header
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.database import get_db
from backend.database.models import User, PredictionHistory
from backend.schemas.prediction_schema import PredictionRequest, PredictionResponse
from backend.services.prediction_service import PredictionService

router = APIRouter()
logger = logging.getLogger("backend")


def get_optional_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Helper dependency to optionally authenticate a user if Bearer token is present.

    Allows anonymous API usage while logging history for logged-in sessions.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username:
            return db.query(User).filter(User.username == username).first()
    except JWTError:
        pass
    return None


@router.post("/", response_model=PredictionResponse)
async def predict_match(
    request: PredictionRequest,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Execute prediction inference and compute SHAP explanation.

    Persists query details to user history if authenticated.
    """
    # 1. Run prediction service
    result = PredictionService.predict_match(request)

    # 2. Persist to prediction history
    try:
        user_id = current_user.id if current_user else None
        db_prediction = PredictionHistory(
            user_id=user_id,
            home_team=result["home_team"],
            away_team=result["away_team"],
            tournament=request.tournament,
            venue=request.venue,
            match_date=request.match_date,
            predicted_winner=result["predicted_winner"],
            prob_home_win=result["probabilities"]["home_win"],
            prob_draw=result["probabilities"]["draw"],
            prob_away_win=result["probabilities"]["away_win"],
            confidence_score=result["confidence_score"],
            expected_goals=result["expected_goals"],
            feature_importance=result["feature_importance"],
            shap_explanation=result["shap_explanation"],
        )
        db.add(db_prediction)
        db.commit()
        logger.info("Saved prediction query to database (User ID: %s).", user_id)
    except Exception as e:
        logger.error("Failed to write prediction query to database: %s", e)
        # Rollback db session if write fails, but don't fail the prediction endpoint
        db.rollback()

    return result
