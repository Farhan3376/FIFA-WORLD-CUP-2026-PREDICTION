"""Phase 4 - Step 9: AI Power Rankings Engine.

This module provides the RankingEngine class to compute power rankings,
strength scores, attack/defense scores, form ratings, and predicted tournament finishes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from simulation.match_predictor import MatchPredictor

logger = logging.getLogger("simulation")


class RankingEngine:
    """Engine to calculate team strength ratings and generate AI Power Rankings."""

    def __init__(self, predictor: MatchPredictor):
        """Initialize the RankingEngine with a MatchPredictor instance."""
        self.predictor = predictor
        self.team_db = predictor.team_db

    def generate_power_rankings(
        self,
        save_path: Optional[Path] = None,
        predicted_finishes: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Calculate and generate the AI Power Rankings table.

        Args:
            save_path: Optional file path to save the rankings CSV.
            predicted_finishes: Optional dictionary mapping team names to their
                                predicted tournament finish (e.g., 'Champion', 'Finalist').

        Returns:
            DataFrame containing the computed rankings and scores.
        """
        logger.info("Generating AI Power Rankings for all teams...")
        
        # Determine min-max ranges for normalization to a [40, 99] scale
        min_elo = self.team_db["elo"].min()
        max_elo = self.team_db["elo"].max()
        elo_range = max_elo - min_elo if max_elo > min_elo else 1.0

        min_attack = self.team_db["avg_goals_scored"].min()
        max_attack = self.team_db["avg_goals_scored"].max()
        attack_range = max_attack - min_attack if max_attack > min_attack else 1.0

        # For defense, lower conceded is better
        min_defense = self.team_db["avg_goals_conceded"].min()
        max_defense = self.team_db["avg_goals_conceded"].max()
        defense_range = max_defense - min_defense if max_defense > min_defense else 1.0

        rows = []
        for team_name, stats in self.team_db.iterrows():
            # 1. Strength Score: [40, 99] based on ELO
            strength_score = 40.0 + 59.0 * ((stats["elo"] - min_elo) / elo_range)
            
            # 2. Attack Score: [40, 99] based on average goals scored
            attack_score = 40.0 + 59.0 * ((stats["avg_goals_scored"] - min_attack) / attack_range)
            
            # 3. Defense Score: [40, 99] based on average goals conceded (inverted)
            defense_score = 40.0 + 59.0 * (1.0 - ((stats["avg_goals_conceded"] - min_defense) / defense_range))
            
            # 4. Recent Form: [40, 99] based on the team's form index (0.0 to 1.0)
            form_score = 40.0 + 59.0 * stats["form"]

            # 5. Composite Power Rating: weighted average
            power_rating = (
                strength_score * 0.40 +
                attack_score * 0.25 +
                defense_score * 0.20 +
                form_score * 0.15
            )

            # Determine predicted tournament finish placeholder or actual mapping
            finish = "TBD"
            if predicted_finishes and team_name in predicted_finishes:
                finish = predicted_finishes[team_name]

            rows.append({
                "Team": team_name,
                "Elo": round(stats["elo"], 1),
                "Strength Score": round(strength_score, 1),
                "Attack Score": round(attack_score, 1),
                "Defense Score": round(defense_score, 1),
                "Recent Form": round(form_score, 1),
                "Power Rating": round(power_rating, 1),
                "Predicted Finish": finish
            })

        df_rankings = pd.DataFrame(rows)
        
        # Sort by overall Power Rating descending
        df_rankings = df_rankings.sort_values(by="PowerRating" if "PowerRating" in df_rankings.columns else "Power Rating", ascending=False).reset_index(drop=True)
        
        # If no predicted finishes provided, we can assign seed-based predicted finishes as a baseline
        if not predicted_finishes:
            for idx, row in df_rankings.iterrows():
                if idx == 0:
                    df_rankings.at[idx, "Predicted Finish"] = "Champion"
                elif idx == 1:
                    df_rankings.at[idx, "Predicted Finish"] = "Finalist"
                elif idx < 4:
                    df_rankings.at[idx, "Predicted Finish"] = "Semifinalist"
                elif idx < 8:
                    df_rankings.at[idx, "Predicted Finish"] = "Quarterfinalist"
                elif idx < 16:
                    df_rankings.at[idx, "Predicted Finish"] = "Round of 16"
                elif idx < 32:
                    df_rankings.at[idx, "Predicted Finish"] = "Round of 32"
                else:
                    df_rankings.at[idx, "Predicted Finish"] = "Group Stage"

        # Save to file
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_rankings.to_csv(save_path, index=False)
            logger.info("Saved AI Power Rankings to: %s", save_path)

        return df_rankings
