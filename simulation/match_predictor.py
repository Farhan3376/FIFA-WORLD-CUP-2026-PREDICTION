"""Phase 4 - Step 2: Match Prediction Engine.

This module provides the MatchPredictor class to load the serialized model,
preprocessors, and features, and perform match predictions with auto-engineered
features and expected goals estimations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    TRAINED_MODELS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
)

# Setup logger
logger = logging.getLogger("simulation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class MatchPredictor:
    """Core prediction engine for individual international football matches."""

    # Tournament importance mapping for feature engineering
    TOURNAMENT_IMPORTANCE: Dict[str, float] = {
        "fifa world cup": 1.00,
        "confederations cup": 0.85,
        "copa america": 0.85,
        "africa cup of nations": 0.85,
        "uefa euro": 0.85,
        "gold cup": 0.75,
        "afc asian cup": 0.75,
        "ofc nations cup": 0.75,
        "olympic games": 0.70,
        "fifa world cup qualification": 0.65,
        "copa america qualification": 0.55,
        "uefa euro qualification": 0.55,
        "african cup of nations qualification": 0.55,
        "friendly": 0.20,
    }
    DEFAULT_IMPORTANCE = 0.45

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        imputer_path: Optional[Path] = None,
        selector_path: Optional[Path] = None,
        team_db_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
    ):
        """Initialize the MatchPredictor and validate required files exist."""
        # Setup paths with defaults if not specified
        self.model_path = model_path or TRAINED_MODELS_DIR / "best_model.pkl"
        self.scaler_path = scaler_path or MODELS_DIR / "scaler.pkl"
        self.imputer_path = imputer_path or MODELS_DIR / "imputer.pkl"
        self.selector_path = selector_path or MODELS_DIR / "feature_selector.pkl"
        self.team_db_path = team_db_path or MODELS_DIR / "team_database.csv"
        
        # Optional encoder path (from config requirements)
        self.encoder_path = encoder_path or MODELS_DIR / "encoder.pkl"

        # Validate existence of critical files
        self._validate_files()

        # Load models and preprocessing objects
        logger.info("Loading prediction models and preprocessors...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.imputer = joblib.load(self.imputer_path)
        self.selected_features = joblib.load(self.selector_path)
        self.team_db = pd.read_csv(self.team_db_path, index_col="team")
        self.team_index_map = {t.lower(): t for t in self.team_db.index}

        # Load encoder if it exists, otherwise keep it as None
        self.encoder = None
        if self.encoder_path.is_file():
            try:
                self.encoder = joblib.load(self.encoder_path)
                logger.info("Loaded categorical encoder from %s", self.encoder_path)
            except Exception as e:
                logger.warning("Could not load encoder: %s. Proceeding without encoder.", e)

    def _validate_files(self) -> None:
        """Validate that all required model files exist, raising descriptive errors if missing."""
        missing = []
        for name, p in [
            ("Best Model", self.model_path),
            ("Scaler", self.scaler_path),
            ("Imputer", self.imputer_path),
            ("Feature Selector List", self.selector_path),
            ("Team Database", self.team_db_path),
        ]:
            if not p.is_file():
                missing.append(f"{name} (expected at: {p})")

        if missing:
            err_msg = "Critical prediction files are missing:\n" + "\n".join(missing)
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Retrieve latest team statistics, falling back to defaults if team is unrecognized."""
        cleaned_name = team_name.strip()
        
        # Apply alias mapping if applicable
        alias_mapping = {
            "united states": "USA",
            "czechia": "Czech Republic",
            "bosnia and herzzegovina": "Bosnia And Herzegovina",
            "bosnia and herzegovina": "Bosnia And Herzegovina",
            "curacao": "Curaçao",
        }
        mapped_name = alias_mapping.get(cleaned_name.lower(), cleaned_name)
        
        # Case-insensitive lookup using pre-computed map
        matched_team = self.team_index_map.get(mapped_name.lower(), None)

        if matched_team is not None:
            stats = self.team_db.loc[matched_team].to_dict()
            stats["name"] = matched_team
            stats["recognized"] = True
            return stats

        # Fallback default statistics for unrecognized teams
        logger.warning("Team '%s' not found in database. Using default fallback statistics.", team_name)
        return {
            "name": cleaned_name,
            "recognized": False,
            "elo": 1500.0,
            "overall_win_pct": 0.35,
            "draw_pct": 0.25,
            "loss_pct": 0.40,
            "avg_goals_scored": 1.0,
            "avg_goals_conceded": 1.2,
            "avg_goal_diff": -0.2,
            "games_played": 10,
            "form": 0.4,
            "home_home_win_pct": 0.40,
            "away_away_win_pct": 0.30,
        }

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
        tournament: str = "FIFA World Cup",
        venue: str = "neutral",
    ) -> Dict[str, Any]:
        """Predict match outcome between two teams, auto-engineering the feature set.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            match_date: Date string (YYYY-MM-DD), defaults to current date.
            tournament: Name of the tournament (determines weight).
            venue: 'home', 'away', or 'neutral'.

        Returns:
            Dictionary containing match prediction results and probability distribution.
        """
        # Parse match date
        if match_date is None:
            dt = pd.Timestamp.now()
        else:
            dt = pd.to_datetime(match_date)

        is_neutral = (venue.lower() == "neutral")
        
        # Resolve tournament weight
        tournament_key = tournament.lower().strip()
        tournament_importance = self.TOURNAMENT_IMPORTANCE.get(
            tournament_key, self.DEFAULT_IMPORTANCE
        )

        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        # 1. Feature Engineering (match the exact 36-feature set expected by the imputer)
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
            
            "home_rest_days": 7.0,  # assumption
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

        # 2. Convert to DataFrame and align columns
        imputer_cols = list(self.imputer.feature_names_in_)
        df_raw = pd.DataFrame([raw_features])[imputer_cols]

        # 3. Apply imputation and scaling
        df_imputed = pd.DataFrame(self.imputer.transform(df_raw), columns=imputer_cols)
        df_scaled = df_imputed.copy()
        scaler_features = list(self.scaler.feature_names_in_)
        df_scaled[scaler_features] = self.scaler.transform(df_imputed[scaler_features])

        # 4. Extract selected features
        df_model_input = df_scaled[self.selected_features]

        # 5. Predict match outcome
        probabilities = self.model.predict_proba(df_model_input)[0]
        prediction_class = int(np.argmax(probabilities))

        # Outcome: 0 = home win, 1 = draw, 2 = away win
        class_mapping = {0: "home_win", 1: "draw", 2: "away_win"}
        predicted_outcome = class_mapping[prediction_class]

        # Determine winner name, confidence, and margin
        if predicted_outcome == "home_win":
            winner = home_stats["name"]
            winning_prob = float(probabilities[0])
        elif predicted_outcome == "away_win":
            winner = away_stats["name"]
            winning_prob = float(probabilities[2])
        else:
            winner = "Draw"
            winning_prob = float(probabilities[1])

        # Confidence Score is the maximum predicted probability
        confidence_score = float(np.max(probabilities))

        # 6. Estimate Expected Goals (xG) using team goals scored/conceded and ELO adjustment
        # baseline: avg goals scored by team adjusted by opponent's defensive strength
        global_avg_goals = 1.35
        home_base_xg = home_stats["avg_goals_scored"] * (away_stats["avg_goals_conceded"] / global_avg_goals)
        away_base_xg = away_stats["avg_goals_scored"] * (home_stats["avg_goals_conceded"] / global_avg_goals)

        # Elo factor: adjust goals based on rating difference
        # E.g. every 100 Elo points diff shifts xG by ~0.15 goals
        elo_diff_adjust = elo_diff / 400.0
        home_xg = max(0.1, home_base_xg + elo_diff_adjust)
        away_xg = max(0.1, away_base_xg - elo_diff_adjust)

        # Round to 2 decimal places
        home_xg = round(home_xg, 2)
        away_xg = round(away_xg, 2)

        return {
            "matchup": f"{home_stats['name']} vs {away_stats['name']}",
            "home_team": home_stats["name"],
            "away_team": away_stats["name"],
            "match_date": dt.strftime("%Y-%m-%d"),
            "tournament": tournament,
            "is_neutral": is_neutral,
            "predicted_winner": winner,
            "predicted_outcome": predicted_outcome,
            "probabilities": {
                "home_win": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "away_win": float(probabilities[2])
            },
            "confidence_score": confidence_score,
            "expected_goals": {
                "home_xg": home_xg,
                "away_xg": away_xg
            }
        }

    @staticmethod
    def to_json(prediction: Dict[str, Any], pretty: bool = False) -> str:
        """Convert prediction result dictionary to JSON string."""
        indent = 4 if pretty else None
        return json.dumps(prediction, indent=indent)

    @staticmethod
    def to_csv(prediction: Dict[str, Any]) -> str:
        """Convert prediction result dictionary to a flat CSV string."""
        flat_dict = {
            "matchup": prediction["matchup"],
            "home_team": prediction["home_team"],
            "away_team": prediction["away_team"],
            "match_date": prediction["match_date"],
            "tournament": prediction["tournament"],
            "is_neutral": prediction["is_neutral"],
            "predicted_winner": prediction["predicted_winner"],
            "predicted_outcome": prediction["predicted_outcome"],
            "prob_home_win": prediction["probabilities"]["home_win"],
            "prob_draw": prediction["probabilities"]["draw"],
            "prob_away_win": prediction["probabilities"]["away_win"],
            "confidence_score": prediction["confidence_score"],
            "home_xg": prediction["expected_goals"]["home_xg"],
            "away_xg": prediction["expected_goals"]["away_xg"],
        }
        df = pd.DataFrame([flat_dict])
        return df.to_csv(index=False)
