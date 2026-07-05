#!/usr/bin/env python3
"""FIFA World Cup 2026 Prediction and Simulation Runner.

This script orchestrates the complete prediction, simulation, validation,
and analytics workflow for the Phase 4 World Cup engine.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from simulation.match_predictor import MatchPredictor
from simulation.probability_engine import ProbabilityEngine
from simulation.ranking_engine import RankingEngine
from simulation.tournament_simulator import TournamentSimulator
from simulation.monte_carlo import MonteCarloSimulator
from evaluation.historical_testing import HistoricalTester
from evaluation.calibration import CalibrationEngine
from evaluation.prediction_analysis import SensitivityAnalyzer

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="World Cup 2026 Simulation and Validation Engine")
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations to run (default: 1000)"
    )
    return parser.parse_args()


def main():
    """Main execution orchestrator."""
    args = parse_args()

    # Define output directories
    output_dir = PROJECT_ROOT / "outputs" / "simulations"
    pred_dir = PROJECT_ROOT / "outputs" / "predictions"
    report_dir = PROJECT_ROOT / "reports" / "simulation"

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("      FIFA WORLD CUP 2026 PREDICTION & SIMULATION PIPELINE")
    logger.info("=" * 70)

    try:
        # 1. Initialize prediction and simulation components
        logger.info("Initializing prediction engines and loading models...")
        predictor = MatchPredictor()
        prob_engine = ProbabilityEngine(predictor)
        ranking_engine = RankingEngine(predictor)
        tournament_sim = TournamentSimulator(predictor, prob_engine)
        mc_simulator = MonteCarloSimulator(tournament_sim)

        # 2. Generate and save AI Power Rankings
        logger.info("Step 1: Generating AI Power Rankings...")
        power_rankings_csv = output_dir / "power_rankings.csv"
        df_rankings = ranking_engine.generate_power_rankings(save_path=power_rankings_csv)
        
        # 3. Run Monte Carlo Simulation
        logger.info("Step 2: Running Monte Carlo Simulation (%d runs)...", args.simulations)
        mc_results = mc_simulator.run_monte_carlo(
            n_simulations=args.simulations,
            output_dir=output_dir
        )
        
        # Retrieve the champion probabilities from the simulation to update the power rankings Predicted Finish
        df_champs = mc_results["champion_probabilities"]
        # Map team to champion winning probability percentage
        champ_pct_map = dict(zip(df_champs["Team"], df_champs["Champion"]))
        
        # Regene power rankings with actual predicted finishes from the Monte Carlo
        predicted_finishes = {}
        df_probs = mc_results["team_probabilities"]
        for _, row in df_probs.iterrows():
            team = row["Team"]
            if row["Champion"] > 0.05:
                finish = f"Contender (Win Prob: {row['Champion']:.1%})"
            elif row["Finalist"] > 0.15:
                finish = "Likely Finalist"
            elif row["Semifinal ="] if "Semifinal =" in df_probs.columns else row.get("Semifinal", 0) > 0.25:
                finish = "Likely Semifinalist"
            elif row["Quarterfinal"] > 0.40:
                finish = "Likely Quarterfinalist"
            elif row["Round of 16"] > 0.60:
                finish = "Likely Round of 16"
            elif row["Round of 32"] > 0.80:
                finish = "Likely Round of 32"
            else:
                finish = "Group Stage"
            predicted_finishes[team] = finish
            
        # Re-save power rankings updated with MC simulation results
        ranking_engine.generate_power_rankings(
            save_path=power_rankings_csv,
            predicted_finishes=predicted_finishes
        )

        # 4. Run Historical Replay Testing
        logger.info("Step 3: Replaying Historical World Cups (2010 - 2022)...")
        tester = HistoricalTester()
        historical_metrics = tester.run_historical_testing(
            years=[2010, 2014, 2018, 2022],
            output_dir=pred_dir,
            report_dir=report_dir
        )

        # 5. Run Probability Calibration Analysis
        logger.info("Step 4: Running Calibration Assessment...")
        calibrator = CalibrationEngine(predictions_csv_path=pred_dir / "historical_predictions.csv")
        calibration_metrics = calibrator.run_calibration_analysis(report_dir=report_dir)

        # 6. Run Sensitivity & Stability Testing
        logger.info("Step 5: Running Input Sensitivity Tests (ELO / Form perturbations)...")
        analyzer = SensitivityAnalyzer(predictor)
        sensitivity_results = analyzer.run_sensitivity_suite(report_dir=report_dir)

        # Print Executive Summary to Terminal
        print("\n" + "=" * 70)
        print("                   EXECUTIVE SIMULATION SUMMARY")
        print("=" * 70)
        print(f"Top 5 AI Power Ranked Teams (ELO + Stats + Form):")
        for idx, row in df_rankings.head(5).iterrows():
            print(f"  {idx+1}. {row['Team']:<15} | Power Rating: {row['Power Rating']:.1f} | Elo: {row['Elo']}")
            
        print(f"\nTop 5 Monte Carlo World Cup 2026 Champion Favorites (Odds):")
        for idx, row in df_champs.head(5).iterrows():
            print(f"  {idx+1}. {row['Team']:<15} | Champion Odds: {row['Champion']:.2%}")

        print(f"\nModel Validation Performance:")
        print(f"  * Historical World Cups Accuracy: {historical_metrics['overall_accuracy']:.2%}")
        print(f"  * Expected Calibration Error:    {calibration_metrics['expected_calibration_error']:.2%}")
        print(f"  * Brier Score (probability error): {calibration_metrics['overall_brier_score']:.4f}")
        
        print(f"\nSensitivity / Stability Indexes:")
        print(f"  * Win Probability shift per 100 Elo points: {sensitivity_results['summary']['avg_win_prob_shift_per_100_elo']:.2%}")
        print(f"  * Win Probability shift per 0.1 Form units: {sensitivity_results['summary']['avg_win_prob_shift_per_0_1_form']:.2%}")
        print("=" * 70)
        logger.info("All pipeline steps completed successfully! Outputs saved to outputs/ and reports/.")

    except Exception as e:
        logger.error("An error occurred during pipeline execution: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
