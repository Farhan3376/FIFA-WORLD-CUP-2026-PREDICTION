"""Phase 4 - Step 1.4: Batch Match Predictor.

This module provides the BatchPredictor class to run prediction pipelines on
large batches of matchups loaded from CSV files, exporting the results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd

from simulation.match_predictor import MatchPredictor

logger = logging.getLogger("simulation")


class BatchPredictor:
    """Handles batch predictions for list of matchups loaded from CSV/DataFrames."""

    def __init__(self, predictor: MatchPredictor):
        """Initialize with a MatchPredictor instance."""
        self.predictor = predictor

    def predict_batch(
        self,
        input_data: Union[pd.DataFrame, Path, str],
        output_csv_path: Optional[Path] = None,
        date_col: str = "date",
        home_col: str = "home_team",
        away_col: str = "away_team",
        tournament_col: str = "tournament",
        venue_col: str = "venue",
    ) -> pd.DataFrame:
        """Predict outcomes for a batch of matches.

        Args:
            input_data: A pandas DataFrame or path to a CSV file.
            output_csv_path: Optional path to save prediction results as CSV.
            date_col: Column name for match date.
            home_col: Column name for home team.
            away_col: Column name for away team.
            tournament_col: Column name for tournament.
            venue_col: Column name for venue ('home', 'away', or 'neutral').

        Returns:
            DataFrame containing all predictions and probabilities.
        """
        # Load data if path is provided
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)
            if not input_path.is_file():
                raise FileNotFoundError(f"Input batch file not found: {input_path}")
            df_input = pd.read_csv(input_path)
            logger.info("Loaded %d matches for batch prediction from %s", len(df_input), input_path)
        else:
            df_input = input_data.copy()
            logger.info("Processing %d matches for batch prediction from DataFrame", len(df_input))

        # Check required columns
        for col, name in [
            (home_col, "Home Team"),
            (away_col, "Away Team"),
        ]:
            if col not in df_input.columns:
                raise KeyError(f"Required column '{col}' ({name}) not found in input data.")

        results = []
        for idx, row in df_input.iterrows():
            home = row[home_col]
            away = row[away_col]
            
            # Resolve optional arguments
            match_date = str(row[date_col]) if date_col in df_input.columns else None
            tournament = str(row[tournament_col]) if tournament_col in df_input.columns else "FIFA World Cup"
            
            # Venue resolution: default to neutral
            venue = "neutral"
            if venue_col in df_input.columns:
                venue = str(row[venue_col])
            elif "neutral" in df_input.columns:
                # If there's a boolean 'neutral' column
                is_neutral_val = row["neutral"]
                if isinstance(is_neutral_val, (bool, int, float)) and bool(is_neutral_val):
                    venue = "neutral"
                elif str(is_neutral_val).lower().strip() in ["true", "1", "yes"]:
                    venue = "neutral"
                else:
                    venue = "home"

            try:
                pred = self.predictor.predict_match(
                    home_team=home,
                    away_team=away,
                    match_date=match_date,
                    tournament=tournament,
                    venue=venue,
                )
                results.append({
                    "date": pred["match_date"],
                    "home_team": pred["home_team"],
                    "away_team": pred["away_team"],
                    "tournament": pred["tournament"],
                    "is_neutral": pred["is_neutral"],
                    "predicted_winner": pred["predicted_winner"],
                    "predicted_outcome": pred["predicted_outcome"],
                    "prob_home_win": pred["probabilities"]["home_win"],
                    "prob_draw": pred["probabilities"]["draw"],
                    "prob_away_win": pred["probabilities"]["away_win"],
                    "confidence_score": pred["confidence_score"],
                    "home_xg": pred["expected_goals"]["home_xg"],
                    "away_xg": pred["expected_goals"]["away_xg"],
                })
            except Exception as e:
                logger.error("Error predicting match %s vs %s: %s", home, away, e)
                # Append fallback row
                results.append({
                    "date": match_date or "N/A",
                    "home_team": home,
                    "away_team": away,
                    "tournament": tournament,
                    "is_neutral": True,
                    "predicted_winner": "Error",
                    "predicted_outcome": "draw",
                    "prob_home_win": 0.3333,
                    "prob_draw": 0.3333,
                    "prob_away_win": 0.3333,
                    "confidence_score": 0.3333,
                    "home_xg": 0.0,
                    "away_xg": 0.0,
                })

        df_out = pd.DataFrame(results)

        if output_csv_path:
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(output_csv_path, index=False)
            logger.info("Saved batch predictions to: %s", output_csv_path)

        return df_out
