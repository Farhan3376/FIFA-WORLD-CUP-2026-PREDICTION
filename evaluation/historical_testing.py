"""Phase 4 - Step 4: Historical Tournament Replay and Validation.

This module replays the 2010, 2014, 2018, and 2022 FIFA World Cups by loading
historical matches, predicting outcomes using the trained model, and comparing
predictions against actual results to compute validation metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.config import (
    TRAINED_MODELS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger("evaluation")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HistoricalTester:
    """Validator class to test the trained model against historical World Cups (2010-2022)."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        selector_path: Optional[Path] = None,
        clean_matches_path: Optional[Path] = None,
        engineered_matches_path: Optional[Path] = None,
    ):
        """Initialize and validate that the required datasets and models exist."""
        self.model_path = model_path or TRAINED_MODELS_DIR / "best_model.pkl"
        self.selector_path = selector_path or MODELS_DIR / "feature_selector.pkl"
        self.clean_matches_path = clean_matches_path or PROJECT_ROOT / "data" / "interim" / "matches_clean.csv"
        self.engineered_matches_path = engineered_matches_path or PROJECT_ROOT / "data" / "processed" / "matches_engineered.csv"

        # Validate file existence
        for label, path in [
            ("Best Model", self.model_path),
            ("Feature Selector", self.selector_path),
            ("Clean Matches Dataset", self.clean_matches_path),
            ("Engineered Matches Dataset", self.engineered_matches_path),
        ]:
            if not path.is_file():
                raise FileNotFoundError(f"Required file missing: {label} (expected at {path})")

        # Load resources
        self.model = joblib.load(self.model_path)
        self.selected_features = joblib.load(self.selector_path)
        self.df_clean = pd.read_csv(self.clean_matches_path)
        self.df_engineered = pd.read_csv(self.engineered_matches_path)

        # Parse date and add year to clean matches
        self.df_clean["year"] = pd.to_datetime(self.df_clean["date"]).dt.year

    def run_historical_testing(
        self,
        years: List[int] = [2010, 2014, 2018, 2022],
        output_dir: Optional[Path] = None,
        report_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Replay historical tournaments, predict matches, and calculate evaluation metrics.

        Args:
            years: List of years to validate against (default: 2010, 2014, 2018, 2022).
            output_dir: Optional directory to save the predictions CSV.
            report_dir: Optional directory to save the validation report.

        Returns:
            Dictionary containing metrics and tournament accuracy summary.
        """
        logger.info("Starting Historical Replay for years: %s", years)

        # Filter clean matches for the specified historical World Cups
        # Note: Tournament is labeled as 'Fifa World Cup' in clean matches
        wc_mask = (self.df_clean["tournament"] == "Fifa World Cup") & (self.df_clean["year"].isin(years))
        wc_indices = self.df_clean[wc_mask].index

        if len(wc_indices) == 0:
            logger.warning("No matches found for years %s in dataset.", years)
            return {}

        df_wc_clean = self.df_clean.loc[wc_indices].copy()
        
        # Extract features corresponding to these historical matches
        X_historical = self.df_engineered.loc[wc_indices, self.selected_features]
        y_actual = df_wc_clean["result"].astype(int).values

        # Predict outcomes
        probs = self.model.predict_proba(X_historical)
        y_pred = np.argmax(probs, axis=1)

        # Map results to human readable names
        result_map = {0: "home_win", 1: "draw", 2: "away_win"}
        actual_labels = [result_map[val] for val in y_actual]
        pred_labels = [result_map[val] for val in y_pred]

        # Assemble prediction log
        df_predictions = df_wc_clean[[
            "date", "year", "home_team", "away_team", "home_goals", "away_goals"
        ]].copy()
        df_predictions["actual_result"] = actual_labels
        df_predictions["predicted_result"] = pred_labels
        df_predictions["prob_home_win"] = probs[:, 0]
        df_predictions["prob_draw"] = probs[:, 1]
        df_predictions["prob_away_win"] = probs[:, 2]
        df_predictions["correct"] = (y_actual == y_pred)

        # Compute metrics
        overall_accuracy = accuracy_score(y_actual, y_pred)
        clf_rep = classification_report(
            y_actual, y_pred, target_names=["home_win", "draw", "away_win"], output_dict=True, zero_division=0
        )
        conf_mat = confusion_matrix(y_actual, y_pred).tolist()

        # Calculate accuracy per tournament year
        accuracy_by_year = {}
        for y in years:
            year_mask = df_predictions["year"] == y
            if year_mask.sum() > 0:
                y_act_sub = y_actual[year_mask]
                y_pred_sub = y_pred[year_mask]
                accuracy_by_year[str(y)] = float(accuracy_score(y_act_sub, y_pred_sub))

        metrics = {
            "overall_accuracy": float(overall_accuracy),
            "accuracy_by_year": accuracy_by_year,
            "classification_report": clf_rep,
            "confusion_matrix": conf_mat,
            "total_matches_evaluated": len(df_predictions)
        }

        # Save predictions
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            pred_file = output_dir / "historical_predictions.csv"
            df_predictions.to_csv(pred_file, index=False)
            logger.info("Saved historical predictions to: %s", pred_file)

        # Save metrics and text report
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # JSON format
            with open(report_dir / "historical_metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
                
            # Text format report
            report_file = report_dir / "historical_replay_report.txt"
            with open(report_file, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("        HISTORICAL WORLD CUP REPLAY VALIDATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total Matches Replayed: {metrics['total_matches_evaluated']}\n")
                f.write(f"Overall Prediction Accuracy: {metrics['overall_accuracy']:.2%}\n\n")
                
                f.write("Accuracy by Tournament Year:\n")
                for yr, acc in accuracy_by_year.items():
                    f.write(f"  * {yr} World Cup: {acc:.2%}\n")
                f.write("\n")
                
                f.write("Classification Performance:\n")
                for outcome in ["home_win", "draw", "away_win"]:
                    prec = clf_rep[outcome]["precision"]
                    rec = clf_rep[outcome]["recall"]
                    f1 = clf_rep[outcome]["f1-score"]
                    f.write(f"  * {outcome:<10} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1-Score: {f1:.2%}\n")
                f.write("\n")
                
                f.write("Confusion Matrix (Rows = Actual, Columns = Predicted):\n")
                f.write(f"            home_win   draw    away_win\n")
                f.write(f"  home_win   {conf_mat[0][0]:<10} {conf_mat[0][1]:<7} {conf_mat[0][2]}\n")
                f.write(f"  draw       {conf_mat[1][0]:<10} {conf_mat[1][1]:<7} {conf_mat[1][2]}\n")
                f.write(f"  away_win   {conf_mat[2][0]:<10} {conf_mat[2][1]:<7} {conf_mat[2][2]}\n")
                
            logger.info("Saved historical validation report to: %s", report_file)

        return metrics
