"""Phase 4 - Step 5 (cont): Sensitivity and Stability Testing.

This module provides the SensitivityAnalyzer class to test model behavior
under simulated perturbations of ELO and team form.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from simulation.match_predictor import MatchPredictor

logger = logging.getLogger("evaluation")


class SensitivityAnalyzer:
    """Performs sensitivity testing on match predictions by perturbing ELO and Form."""

    def __init__(self, predictor: MatchPredictor):
        """Initialize the analyzer with a MatchPredictor instance."""
        self.predictor = predictor

    def perturb_match_prediction(
        self,
        home_team: str,
        away_team: str,
        elo_shift: float = 0.0,
        form_shift: float = 0.0,
    ) -> Dict[str, float]:
        """Predict match probabilities with shifted ELO or Form values.

        This method temporarily overrides stats returned by get_team_stats
        to observe prediction sensitivity.
        """
        # Get baseline stats
        home_stats = self.predictor.get_team_stats(home_team).copy()
        away_stats = self.predictor.get_team_stats(away_team).copy()

        # Apply shifts
        home_stats["elo"] += elo_shift
        home_stats["form"] = max(0.0, min(1.0, home_stats["form"] + form_shift))

        # Re-run prediction steps using modified stats
        is_neutral = True
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
            
            "tournament_importance": 1.0,
            "year": 2026.0,
            "month": 6.0,
            
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

        # Align, scale, and predict
        imputer_cols = list(self.predictor.imputer.feature_names_in_)
        df_raw = pd.DataFrame([raw_features])[imputer_cols]
        df_imputed = pd.DataFrame(self.predictor.imputer.transform(df_raw), columns=imputer_cols)
        
        df_scaled = df_imputed.copy()
        scaler_features = list(self.predictor.scaler.feature_names_in_)
        df_scaled[scaler_features] = self.predictor.scaler.transform(df_imputed[scaler_features])

        df_model_input = df_scaled[self.predictor.selected_features]
        probabilities = self.predictor.model.predict_proba(df_model_input)[0]

        return {
            "home_win": float(probabilities[0]),
            "draw": float(probabilities[1]),
            "away_win": float(probabilities[2])
        }

    def run_sensitivity_suite(
        self,
        matchups: List[Tuple[str, str]] = [
            ("Argentina", "Brazil"),
            ("England", "Croatia"),
            ("USA", "Mexico"),
            ("Germany", "Spain"),
            ("France", "Senegal")
        ],
        report_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Test sensitivity for standard matchups under varying ELO and Form.

        Args:
            matchups: List of (home, away) team tuples.
            report_dir: Directory path to save output reports.
        """
        logger.info("Starting Sensitivity Suite for %d matchups...", len(matchups))

        elo_shifts = [-150, -100, -50, 0, 50, 100, 150]
        form_shifts = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

        sensitivity_records = []
        elo_slopes = []
        form_slopes = []

        for home, away in matchups:
            # Baseline probabilities
            base_pred = self.perturb_match_prediction(home, away, elo_shift=0.0, form_shift=0.0)
            base_p_home = base_pred["home_win"]

            # 1. Elo Sensitivity Test
            elo_probs = []
            for shift in elo_shifts:
                p = self.perturb_match_prediction(home, away, elo_shift=shift, form_shift=0.0)
                p_home = p["home_win"]
                elo_probs.append(p_home)
                
                sensitivity_records.append({
                    "matchup": f"{home} vs {away}",
                    "perturbation_type": "elo",
                    "shift_value": shift,
                    "prob_home_win": p_home,
                    "prob_draw": p["draw"],
                    "prob_away_win": p["away_win"],
                    "delta_win_prob": p_home - base_p_home
                })

            # Calculate ELO slope (change in probability per 100 Elo points)
            # Linear regression slope of win probability vs elo shift
            slope_elo = np.polyfit(elo_shifts, elo_probs, 1)[0] * 100.0
            elo_slopes.append(slope_elo)

            # 2. Form Sensitivity Test
            form_probs = []
            for shift in form_shifts:
                p = self.perturb_match_prediction(home, away, elo_shift=0.0, form_shift=shift)
                p_home = p["home_win"]
                form_probs.append(p_home)

                sensitivity_records.append({
                    "matchup": f"{home} vs {away}",
                    "perturbation_type": "form",
                    "shift_value": shift,
                    "prob_home_win": p_home,
                    "prob_draw": p["draw"],
                    "prob_away_win": p["away_win"],
                    "delta_win_prob": p_home - base_p_home
                })

            # Calculate Form slope (change in probability per 0.1 form units)
            slope_form = np.polyfit(form_shifts, form_probs, 1)[0] * 0.1
            form_slopes.append(slope_form)

        df_sensitivity = pd.DataFrame(sensitivity_records)
        
        # Summary metrics
        avg_elo_sensitivity = float(np.mean(np.abs(elo_slopes)))
        avg_form_sensitivity = float(np.mean(np.abs(form_slopes)))

        summary = {
            "avg_win_prob_shift_per_100_elo": avg_elo_sensitivity,
            "avg_win_prob_shift_per_0_1_form": avg_form_sensitivity,
            "elo_sensitivity_by_matchup": {
                f"{m[0]} vs {m[1]}": float(slope) for m, slope in zip(matchups, elo_slopes)
            },
            "form_sensitivity_by_matchup": {
                f"{m[0]} vs {m[1]}": float(slope) for m, slope in zip(matchups, form_slopes)
            }
        }

        # Save outputs
        if report_dir:
            report_dir.mkdir(parents=True, exist_ok=True)
            df_sensitivity.to_csv(report_dir / "sensitivity_results.csv", index=False)
            
            with open(report_dir / "sensitivity_metrics.json", "w") as f:
                json.dump(summary, f, indent=4)

            # Text report
            report_file = report_dir / "sensitivity_report.txt"
            with open(report_file, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("        MODEL SENSITIVITY & STABILITY ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Average Win Probability Shift per 100 Elo points: {avg_elo_sensitivity:.2%}\n")
                f.write(f"Average Win Probability Shift per 0.1 Form points: {avg_form_sensitivity:.2%}\n\n")
                
                f.write("Elo Sensitivity By Matchup (Win Prob shift per 100 Elo):\n")
                for match, slope in summary["elo_sensitivity_by_matchup"].items():
                    f.write(f"  * {match:<25}: {slope:+.2%}\n")
                f.write("\n")
                
                f.write("Form Sensitivity By Matchup (Win Prob shift per 0.1 Form):\n")
                for match, slope in summary["form_sensitivity_by_matchup"].items():
                    f.write(f"  * {match:<25}: {slope:+.2%}\n")
                
            logger.info("Saved sensitivity analysis CSV to: %s", report_dir / "sensitivity_results.csv")
            logger.info("Saved sensitivity report to: %s", report_file)

        return {
            "detailed_records": df_sensitivity,
            "summary": summary
        }
