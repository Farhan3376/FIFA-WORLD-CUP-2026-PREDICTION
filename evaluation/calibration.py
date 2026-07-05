"""Phase 4 - Step 5: Probability Calibration Engine.

This module provides the CalibrationEngine class to evaluate the reliability and
calibration of the predicted probabilities (Brier score, ECE, reliability tables).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger("evaluation")


class CalibrationEngine:
    """Engine to assess and report probability calibration accuracy (ECE, Brier Score)."""

    def __init__(self, predictions_csv_path: Optional[Path] = None):
        """Initialize and check that the historical predictions file exists."""
        self.predictions_csv_path = predictions_csv_path or PROJECT_ROOT / "outputs" / "predictions" / "historical_predictions.csv"
        
        if not self.predictions_csv_path.is_file():
            raise FileNotFoundError(
                f"Historical predictions CSV not found at: {self.predictions_csv_path}. "
                "Please run historical testing first to generate this file."
            )

        self.df_pred = pd.read_csv(self.predictions_csv_path)

    @staticmethod
    def calculate_ece(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Compute the Expected Calibration Error (ECE) and bin statistics.

        ECE = sum_b (|bin_b| / N) * |accuracy_b - confidence_b|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(confidences)
        bin_details = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin range
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            prop_in_bin = np.mean(in_bin)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                confidence_in_bin = np.mean(confidences[in_bin])
                
                # ECE summation term
                ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)
                
                bin_details.append({
                    "bin_idx": i,
                    "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
                    "count": int(bin_size),
                    "mean_confidence": float(confidence_in_bin),
                    "accuracy": float(accuracy_in_bin),
                    "deviation": float(accuracy_in_bin - confidence_in_bin)
                })
            else:
                bin_details.append({
                    "bin_idx": i,
                    "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f})",
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                    "deviation": 0.0
                })

        return float(ece), bin_details

    def run_calibration_analysis(
        self,
        n_bins: int = 10,
        report_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Perform probability calibration analysis (Brier score, ECE).

        Returns:
            Dictionary containing calibration metrics and reliability tables.
        """
        logger.info("Starting probability calibration analysis...")

        # 1. Parse prediction targets and probabilities
        # Map outcomes: home_win=0, draw=1, away_win=2
        outcome_map = {"home_win": 0, "draw": 1, "away_win": 2}
        
        y_true = self.df_pred["actual_result"].map(outcome_map).values
        
        probs = self.df_pred[[
            "prob_home_win", "prob_draw", "prob_away_win"
        ]].values

        # 2. Multi-class Brier Score calculation
        # Brier = (1 / N) * sum_i sum_c (p_ic - y_ic)^2
        n_classes = probs.shape[1]
        y_true_one_hot = np.zeros_like(probs)
        for i, val in enumerate(y_true):
            y_true_one_hot[i, val] = 1.0

        brier_score = np.mean(np.sum((probs - y_true_one_hot) ** 2, axis=1))

        # 3. Maximum-probability ECE (confidence vs accuracy of predicted outcome)
        y_pred = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)

        ece, bin_details = self.calculate_ece(confidences, accuracies, n_bins=n_bins)

        # 4. Class-specific Brier scores (e.g. draw vs non-draw calibration)
        class_brier = {}
        class_names = ["home_win", "draw", "away_win"]
        for c in range(n_classes):
            p_c = probs[:, c]
            y_c = y_true_one_hot[:, c]
            c_brier = np.mean((p_c - y_c) ** 2)
            class_brier[class_names[c]] = float(c_brier)

        results = {
            "overall_brier_score": float(brier_score),
            "expected_calibration_error": float(ece),
            "class_brier_scores": class_brier,
            "reliability_bins": bin_details
        }

        # Save calibration results and text report
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # JSON format
            with open(report_dir / "calibration_metrics.json", "w") as f:
                json.dump(results, f, indent=4)

            # Save bin details to CSV for plot generation
            df_bins = pd.DataFrame(bin_details)
            df_bins.to_csv(report_dir / "reliability_table.csv", index=False)

            # Text report
            report_file = report_dir / "probability_calibration_report.txt"
            with open(report_file, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("        PROBABILITY CALIBRATION ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Overall Brier Score (lower is better): {brier_score:.4f}\n")
                f.write(f"Expected Calibration Error (ECE):      {ece:.4f} ({ece:.2%})\n\n")
                
                f.write("Class-Specific Brier Scores:\n")
                for c_name, c_score in class_brier.items():
                    f.write(f"  * {c_name:<10}: {c_score:.4f}\n")
                f.write("\n")
                
                f.write("Reliability Table (predicted confidence vs empirical accuracy):\n")
                f.write(f"{'Bin Range':<15} | {'Count':<7} | {'Mean Confidence':<15} | {'Accuracy':<8} | {'Deviation':<9}\n")
                f.write("-" * 65 + "\n")
                for bin_d in bin_details:
                    f.write(
                        f"{bin_d['bin_range']:<15} | "
                        f"{bin_d['count']:<7} | "
                        f"{bin_d['mean_confidence']:<15.2%} | "
                        f"{bin_d['accuracy']:<8.2%} | "
                        f"{bin_d['deviation']:<+9.2%}\n"
                    )
            
            logger.info("Saved probability calibration report to: %s", report_file)
            logger.info("Saved reliability table CSV to: %s", report_dir / "reliability_table.csv")

        return results
