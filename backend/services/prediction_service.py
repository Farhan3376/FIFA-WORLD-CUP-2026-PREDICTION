"""Match prediction services layer.

Coordinates between the request inputs, ML pipelines, SHAP explainability service,
and returns structured prediction data.
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from backend.ml.model_loader import ml_loader
from backend.ml.shap_explainer import SHAPExplainerService
from backend.schemas.prediction_schema import PredictionRequest

logger = logging.getLogger("backend")

# How strongly the current score differential pulls probability mass toward
# the leading team. This is a fixed heuristic weight, not a learned parameter --
# the live fixtures API does not expose match-clock minutes elapsed, only a
# coarse status ("live"/"finished"/"notstarted"), so there's no reliable signal
# for "how much time is left" to weight this by. Capped well below 1.0 so the
# pre-match model's read of the two teams is never fully discarded.
LIVE_SCORE_ADJUSTMENT_WEIGHT = 0.12


class PredictionService:
    """Orchestrates match inference and SHAP explainability calculations."""

    @staticmethod
    def predict_match(request: PredictionRequest) -> Dict[str, Any]:
        """Perform match prediction and compute SHAP explanations.

        Args:
            request: PredictionRequest schema holding match parameters.

        Returns:
            Dictionary matching the structure of PredictionResponse.
        """
        logger.info("Executing prediction service for: %s vs %s", request.home_team, request.away_team)
        
        predictor = ml_loader.predictor

        # 1. Resolve date and tournament weights
        match_date = request.match_date
        if not match_date:
            match_date = datetime.date.today().strftime("%Y-%m-%d")

        try:
            dt = datetime.datetime.strptime(match_date, "%Y-%m-%d")
        except ValueError:
            dt = datetime.datetime.now()
            match_date = dt.strftime("%Y-%m-%d")

        # Resolve tournament importance
        tournament = request.tournament or "FIFA World Cup"
        tournament_importance = 1.0
        if "friendly" in tournament.lower():
            tournament_importance = 0.2
        elif "qual" in tournament.lower() or "euro" in tournament.lower() or "nations" in tournament.lower():
            tournament_importance = 0.6

        is_neutral = request.venue.lower() == "neutral"

        # 2. Retrieve rolling stats from team database
        home_stats = predictor.get_team_stats(request.home_team)
        away_stats = predictor.get_team_stats(request.away_team)

        # 3. Reconstruct match features
        elo_diff = home_stats["elo"] - away_stats["elo"]
        home_adv = 100 * (1 - int(is_neutral))
        elo_win_prob = 1.0 / (1.0 + 10.0 ** (- (elo_diff + home_adv) / 400.0))
        log_elo_diff = np.sign(elo_diff) * np.log1p(np.abs(elo_diff))

        raw_features = {
            "home_elo_before": home_stats["elo"],
            "away_elo_before": away_stats["elo"],
            "elo_diff": elo_diff,
            
            "home_overall_win_pct": home_stats["overall_win_pct"],
            "home_draw_pct": home_stats["draw_pct"],
            "home_loss_pct": home_stats["loss_pct"],
            "home_avg_goals_scored": home_stats["avg_goals_scored"],
            "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
            "home_avg_goal_diff": home_stats["avg_goal_diff"],
            "home_games_played": home_stats["games_played"],
            
            "away_overall_win_pct": away_stats["overall_win_pct"],
            "away_draw_pct": away_stats["draw_pct"],
            "away_loss_pct": away_stats["loss_pct"],
            "away_avg_goals_scored": away_stats["avg_goals_scored"],
            "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
            "away_avg_goal_diff": away_stats["avg_goal_diff"],
            "away_games_played": away_stats["games_played"],
            
            "home_home_win_pct": home_stats["home_home_win_pct"],
            "away_away_win_pct": away_stats["away_away_win_pct"],
            
            "home_form": home_stats["form"],
            "away_form": away_stats["form"],
            
            "home_rest_days": 7.0,
            "away_rest_days": 7.0,
            "is_neutral": float(int(is_neutral)),
            
            "tournament_importance": tournament_importance,
            "year": float(dt.year),
            "month": float(dt.month),
            
            "elo_win_prob": elo_win_prob,
            "log_elo_diff": log_elo_diff,
            "goal_avg_diff": home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
            "goal_conceded_avg_diff": home_stats["avg_goals_conceded"] - away_stats["avg_goals_conceded"],
            "form_diff": home_stats["form"] - away_stats["form"],
            "rest_days_diff": 0.0,
            "games_played_diff": float(home_stats["games_played"] - away_stats["games_played"]),
            "win_pct_diff": home_stats["overall_win_pct"] - away_stats["overall_win_pct"],
            "home_vs_away_field_pct": home_stats["home_home_win_pct"] - away_stats["away_away_win_pct"]
        }

        # 4. Pipeline preprocessing
        imputer_cols = list(predictor.imputer.feature_names_in_)
        df_raw = pd.DataFrame([raw_features])[imputer_cols]
        df_imputed = pd.DataFrame(predictor.imputer.transform(df_raw), columns=imputer_cols)
        
        df_scaled = df_imputed.copy()
        scaler_features = list(predictor.scaler.feature_names_in_)
        df_scaled[scaler_features] = predictor.scaler.transform(df_imputed[scaler_features])

        # Extract selected features
        df_model_input = df_scaled[predictor.selected_features]

        # 5. Model Inference
        probabilities = predictor.model.predict_proba(df_model_input)[0]
        prediction_class = int(np.argmax(probabilities))

        # Outcome mapping: 0 = home win, 1 = draw, 2 = away win
        class_mapping = {0: "home_win", 1: "draw", 2: "away_win"}
        predicted_outcome = class_mapping[prediction_class]

        # Resolve winner name and confidence
        if predicted_outcome == "home_win":
            winner = home_stats["name"]
        elif predicted_outcome == "away_win":
            winner = away_stats["name"]
        else:
            winner = "Draw"

        confidence_score = float(probabilities[prediction_class])

        # 6. Calculate expected goals (xG)
        global_avg_goals = 1.35
        home_base_xg = home_stats["avg_goals_scored"] * (away_stats["avg_goals_conceded"] / global_avg_goals)
        away_base_xg = away_stats["avg_goals_scored"] * (home_stats["avg_goals_conceded"] / global_avg_goals)
        elo_diff_adjust = elo_diff / 400.0
        
        home_xg = max(0.1, home_base_xg + elo_diff_adjust)
        away_xg = max(0.1, away_base_xg - elo_diff_adjust)

        # 7. Compute SHAP explanations
        shap_res = SHAPExplainerService.get_match_explanation(df_model_input, prediction_class)

        # Generate feature importance mapping (absolute SHAP contributions)
        feature_importance = {
            feat: abs(val) for feat, val in shap_res["contributions"].items()
        }

        # Format and return the result
        return {
            "home_team": home_stats["name"],
            "away_team": away_stats["name"],
            "predicted_winner": winner,
            "predicted_outcome": predicted_outcome,
            "probabilities": {
                "home_win": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "away_win": float(probabilities[2]),
            },
            "confidence_score": confidence_score,
            "expected_goals": {
                "home_xg": round(home_xg, 2),
                "away_xg": round(away_xg, 2),
            },
            "feature_importance": feature_importance,
            "shap_explanation": shap_res,
            "timestamp": datetime.datetime.utcnow(),
        }

    @staticmethod
    def apply_live_score_adjustment(
        baseline_probabilities: Dict[str, float],
        home_score: int,
        away_score: int,
    ) -> Dict[str, float]:
        """Blend a pre-match probability distribution with the current live score.

        This is a fixed-weight heuristic, not a re-run of the trained model: it shifts
        probability mass toward whichever side is currently leading (or toward "draw"
        if scores are level), proportional to the goal difference. It exists because
        the live fixtures API only exposes a coarse match status, not minutes elapsed,
        so there is no reliable basis for a time-weighted in-play model.

        Args:
            baseline_probabilities: dict with home_win/draw/away_win, summing to ~1.0.
            home_score: current home team goals.
            away_score: current away team goals.

        Returns:
            A new probabilities dict (home_win/draw/away_win), renormalized to sum to 1.0.
        """
        goal_diff = home_score - away_score
        # Diminishing marginal effect per additional goal, capped so a blowout
        # doesn't fully zero out the other two outcomes.
        pull = min(abs(goal_diff) * LIVE_SCORE_ADJUSTMENT_WEIGHT, 0.45)

        home_p = baseline_probabilities["home_win"]
        draw_p = baseline_probabilities["draw"]
        away_p = baseline_probabilities["away_win"]

        if goal_diff > 0:
            home_p += pull
            draw_p -= pull * 0.5
            away_p -= pull * 0.5
        elif goal_diff < 0:
            away_p += pull
            draw_p -= pull * 0.5
            home_p -= pull * 0.5
        else:
            # Level score: nudge toward draw, taking proportionally from both sides.
            nudge = LIVE_SCORE_ADJUSTMENT_WEIGHT * 0.5
            home_p -= nudge * home_p
            away_p -= nudge * away_p
            draw_p += nudge * (home_p + away_p)

        # Clip negatives (can occur in extreme blowouts) and renormalize to sum to 1.0.
        home_p, draw_p, away_p = max(home_p, 0.01), max(draw_p, 0.01), max(away_p, 0.01)
        total = home_p + draw_p + away_p
        return {
            "home_win": round(home_p / total, 4),
            "draw": round(draw_p / total, 4),
            "away_win": round(away_p / total, 4),
        }
