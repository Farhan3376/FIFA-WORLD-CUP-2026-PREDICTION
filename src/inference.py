"""Phase 3 - Step 6: Inference Pipeline.

Provides the InferenceWrapper class to load the serialized LightGBM model,
feature scaler, and precalculate/load the latest team statistics database to
predict future matches.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from src.config import (
    TRAINED_MODELS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="training.log")


class InferenceWrapper:
    """Wrapper class for loading ML components and running match outcome predictions."""

    def __init__(self, team_db_path: Path | None = None):
        """Initialize the inference wrapper by loading serialized components."""
        self.model_path = TRAINED_MODELS_DIR / "best_model.pkl"
        self.scaler_path = MODELS_DIR / "scaler.pkl"
        self.imputer_path = MODELS_DIR / "imputer.pkl"
        self.selector_path = MODELS_DIR / "feature_selector.pkl"
        
        # Load models and preprocessing objects
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Run model_selection.py first.")
        if not self.scaler_path.is_file():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}.")
        if not self.imputer_path.is_file():
            raise FileNotFoundError(f"Imputer not found at {self.imputer_path}.")
        if not self.selector_path.is_file():
            raise FileNotFoundError(f"Feature selector not found at {self.selector_path}.")

        logger.info("Loading ML models and preprocessors for inference...")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.imputer = joblib.load(self.imputer_path)
        self.selected_features = joblib.load(self.selector_path)
        
        # Load or generate the latest team statistics database
        if team_db_path is None:
            self.team_db_path = MODELS_DIR / "team_database.csv"
        else:
            self.team_db_path = team_db_path
            
        self.team_db = self._load_or_generate_team_db()

    def _load_or_generate_team_db(self) -> pd.DataFrame:
        """Load the team stats database if it exists, otherwise generate it."""
        if self.team_db_path.is_file():
            logger.info("Loading precalculated team database from: %s", self.team_db_path)
            return pd.read_csv(self.team_db_path, index_col="team")
        
        logger.info("Team database not found. Generating from matches_merged.csv...")
        merged_path = PROJECT_ROOT / "data" / "interim" / "matches_merged.csv"
        if not merged_path.is_file():
            raise FileNotFoundError(f"Cannot generate team database; matches_merged.csv not found at {merged_path}")
        
        df = pd.read_csv(merged_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
        
        # Build team perspective rows to compute stats after their last match
        home = pd.DataFrame({
            "date": df["date"].values,
            "team": df["home_team"].values,
            "goals_scored": df["home_goals"].values,
            "goals_conceded": df["away_goals"].values,
            "result_code": df["result"].values,
            "is_home": True,
            "elo_after": df["home_elo_after"].values,
            "match_idx": df.index.values
        })
        home["is_win"] = (home["result_code"] == 0).astype(float)
        home["is_draw"] = (home["result_code"] == 1).astype(float)
        home["is_loss"] = (home["result_code"] == 2).astype(float)
        
        away = pd.DataFrame({
            "date": df["date"].values,
            "team": df["away_team"].values,
            "goals_scored": df["away_goals"].values,
            "goals_conceded": df["home_goals"].values,
            "result_code": df["result"].values,
            "is_home": False,
            "elo_after": df["away_elo_after"].values,
            "match_idx": df.index.values
        })
        away["is_win"] = (away["result_code"] == 2).astype(float)
        away["is_draw"] = (away["result_code"] == 1).astype(float)
        away["is_loss"] = (away["result_code"] == 0).astype(float)
        
        combined = pd.concat([home, away], ignore_index=True)
        combined = combined.sort_values(["team", "date", "match_idx"]).reset_index(drop=True)
        
        combined["goal_diff"] = combined["goals_scored"] - combined["goals_conceded"]
        combined["form_pts"] = combined["is_win"] * 3.0 + combined["is_draw"] * 1.0
        
        grp = combined.groupby("team", sort=False)
        
        combined["overall_win_pct"] = grp["is_win"].transform(lambda s: s.expanding().mean())
        combined["draw_pct"] = grp["is_draw"].transform(lambda s: s.expanding().mean())
        combined["loss_pct"] = grp["is_loss"].transform(lambda s: s.expanding().mean())
        combined["avg_goals_scored"] = grp["goals_scored"].transform(lambda s: s.expanding().mean())
        combined["avg_goals_conceded"] = grp["goals_conceded"].transform(lambda s: s.expanding().mean())
        combined["avg_goal_diff"] = grp["goal_diff"].transform(lambda s: s.expanding().mean())
        combined["games_played"] = grp["is_win"].transform(lambda s: s.expanding().count())
        combined["form"] = grp["form_pts"].transform(lambda s: s.rolling(5, min_periods=1).mean()) / 3.0
        
        home_mask = combined["is_home"]
        combined["venue_win_pct"] = np.nan
        
        home_view = combined[home_mask].copy()
        home_view["venue_win_pct"] = home_view.groupby("team")["is_win"].transform(lambda s: s.expanding().mean())
        away_view = combined[~home_mask].copy()
        away_view["venue_win_pct"] = away_view.groupby("team")["is_win"].transform(lambda s: s.expanding().mean())
        
        combined.loc[home_mask, "venue_win_pct"] = home_view["venue_win_pct"].values
        combined.loc[~home_mask, "venue_win_pct"] = away_view["venue_win_pct"].values
        
        latest_rows = combined.groupby("team").last().reset_index()
        
        home_stats = combined[combined["is_home"]].groupby("team").last()[["venue_win_pct"]].rename(columns={"venue_win_pct": "home_home_win_pct"})
        away_stats = combined[~combined["is_home"]].groupby("team").last()[["venue_win_pct"]].rename(columns={"venue_win_pct": "away_away_win_pct"})
        
        latest_stats = latest_rows.merge(home_stats, on="team", how="left")
        latest_stats = latest_stats.merge(away_stats, on="team", how="left")
        
        latest_stats["home_home_win_pct"] = latest_stats["home_home_win_pct"].fillna(latest_stats["overall_win_pct"])
        latest_stats["away_away_win_pct"] = latest_stats["away_away_win_pct"].fillna(latest_stats["overall_win_pct"])
        
        # Keep only required columns
        db_cols = [
            "team", "elo_after", "overall_win_pct", "draw_pct", "loss_pct",
            "avg_goals_scored", "avg_goals_conceded", "avg_goal_diff",
            "games_played", "form", "home_home_win_pct", "away_away_win_pct"
        ]
        team_db = latest_stats[db_cols].rename(columns={"elo_after": "elo"}).set_index("team")
        
        # Save to CSV for future usage
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        team_db.to_csv(self.team_db_path)
        logger.info("Saved newly generated team database to: %s", self.team_db_path)
        return team_db

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Look up a team's latest stats, falling back to defaults if unrecognized."""
        # Clean team name
        cleaned_name = team_name.strip()
        
        # Match case-insensitively
        matched_team = None
        for t in self.team_db.index:
            if t.lower() == cleaned_name.lower():
                matched_team = t
                break
                
        if matched_team is not None:
            stats = self.team_db.loc[matched_team].to_dict()
            stats["name"] = matched_team
            stats["recognized"] = True
            return stats
        
        # Default stats for unrecognized teams (e.g. newly qualified teams)
        logger.warning("Team '%s' not recognized in historical database. Using fallback baseline stats.", team_name)
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
            "away_away_win_pct": 0.30
        }

    def predict_match(
        self,
        home_team_name: str,
        away_team_name: str,
        is_neutral: bool = False,
        tournament_importance: float = 1.0,
        year: int = 2026,
        month: int = 6
    ) -> Dict[str, Any]:
        """Calculate features, scale them, and predict the match outcome."""
        home_stats = self.get_team_stats(home_team_name)
        away_stats = self.get_team_stats(away_team_name)
        
        # 1. Construct all 36 raw features in the exact order expected by the imputer
        raw_features = {
            "home_elo_before": home_stats["elo"],
            "away_elo_before": away_stats["elo"],
            "elo_diff": home_stats["elo"] - away_stats["elo"],
            
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
            
            "home_rest_days": 7.0,  # default rest days
            "away_rest_days": 7.0,
            "is_neutral": float(int(is_neutral)),
            
            "tournament_importance": tournament_importance,
            "year": float(year),
            "month": float(month),
            
            # Interactives
            "elo_win_prob": 1.0 / (1.0 + 10.0 ** (- ((home_stats["elo"] - away_stats["elo"]) + 100 * (1 - int(is_neutral))) / 400.0)),
            "log_elo_diff": np.sign(home_stats["elo"] - away_stats["elo"]) * np.log1p(np.abs(home_stats["elo"] - away_stats["elo"])),
            "goal_avg_diff": home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
            "goal_conceded_avg_diff": home_stats["avg_goals_conceded"] - away_stats["avg_goals_conceded"],
            "form_diff": home_stats["form"] - away_stats["form"],
            "rest_days_diff": 0.0,
            "games_played_diff": float(home_stats["games_played"] - away_stats["games_played"]),
            "win_pct_diff": home_stats["overall_win_pct"] - away_stats["overall_win_pct"],
            "home_vs_away_field_pct": home_stats["home_home_win_pct"] - away_stats["away_away_win_pct"]
        }
        
        # 2. Put into DataFrame with the exact column order expected by the imputer
        imputer_cols = list(self.imputer.feature_names_in_)
        df_raw = pd.DataFrame([raw_features])[imputer_cols]
        
        # 3. Apply Imputer & Scaler
        df_imputed = pd.DataFrame(self.imputer.transform(df_raw), columns=imputer_cols)
        df_scaled = df_imputed.copy()
        
        # Scale continuous variables
        scaler_features = list(self.scaler.feature_names_in_)
        df_scaled[scaler_features] = self.scaler.transform(df_imputed[scaler_features])
        
        # 4. Extract only the 15 selected features that the model expects
        df_model_input = df_scaled[self.selected_features]
        
        # 5. Predict probabilities
        probabilities = self.model.predict_proba(df_model_input)[0]
        prediction_class = int(np.argmax(probabilities))
        
        # Mapping: 0 = Home Win, 1 = Draw, 2 = Away Win
        class_mapping = {0: "home_win", 1: "draw", 2: "away_win"}
        predicted_outcome = class_mapping[prediction_class]
        
        # Determine winner name, confidence, and margin
        if predicted_outcome == "home_win":
            winner = home_stats["name"]
            probability = float(probabilities[0])
        elif predicted_outcome == "away_win":
            winner = away_stats["name"]
            probability = float(probabilities[2])
        else:
            winner = "Draw"
            probability = float(probabilities[1])
            
        # Confidence score (highest probability)
        confidence_score = float(np.max(probabilities))
        
        return {
            "matchup": f"{home_stats['name']} vs {away_stats['name']}",
            "home_team": {
                "name": home_stats["name"],
                "elo": float(home_stats["elo"]),
                "overall_win_pct": float(home_stats["overall_win_pct"]),
                "recognized": home_stats["recognized"]
            },
            "away_team": {
                "name": away_stats["name"],
                "elo": float(away_stats["elo"]),
                "overall_win_pct": float(away_stats["overall_win_pct"]),
                "recognized": away_stats["recognized"]
            },
            "is_neutral": is_neutral,
            "predicted_winner": winner,
            "predicted_outcome": predicted_outcome,
            "probabilities": {
                "home_win": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "away_win": float(probabilities[2])
            },
            "confidence_score": confidence_score
        }
