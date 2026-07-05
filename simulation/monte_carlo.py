"""Phase 4 - Step 6 & 8: Monte Carlo Simulation Engine.

This module provides the MonteCarloSimulator class to run tournament simulations
repeatedly, calculate progression probabilities, and save probability lookup tables.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from simulation.tournament_simulator import TournamentSimulator

logger = logging.getLogger("simulation")


class MonteCarloSimulator:
    """Orchestrates multi-run Monte Carlo simulations for the FIFA World Cup 2026."""

    def __init__(self, tournament_sim: TournamentSimulator):
        """Initialize with a TournamentSimulator instance."""
        self.sim = tournament_sim
        self.all_teams = []
        for teams in self.sim.bracket_gen.GROUPS.values():
            self.all_teams.extend(teams)

    def run_monte_carlo(
        self,
        n_simulations: int = 1000,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run repeated tournament simulations and compile progression probabilities.

        Args:
            n_simulations: Number of tournament simulations to run (e.g., 1000, 5000, 10000).
            output_dir: Directory path to save output CSV files.

        Returns:
            Dictionary containing DataFrames for team advancement probabilities and champion odds.
        """
        logger.info("=" * 60)
        logger.info("Starting Monte Carlo Simulation: %d runs", n_simulations)
        logger.info("=" * 60)

        # Initialize counters for each stage progression
        counts = {
            t: {
                "group_stage": n_simulations,  # all 48 teams start in group stage
                "round_of_32": 0,
                "round_of_16": 0,
                "quarterfinals": 0,
                "semifinals": 0,
                "finalist": 0,
                "champion": 0,
            }
            for t in self.all_teams
        }

        # Track champions frequency for simulation_results logging
        champion_counts: Dict[str, int] = {t: 0 for t in self.all_teams}

        # Track how often each unordered pair of teams meets in the final,
        # so the API can report the single most common real final matchup
        # (rather than just each team's independent finalist odds).
        finalist_pair_counts: Dict[Tuple[str, str], int] = {}

        # Keep the last run's full knockout bracket as a real, representative
        # sample tournament path (Round of 32 -> Final) to show in the UI.
        last_bracket: Optional[Dict[str, Any]] = None
        last_placements: Optional[Dict[str, str]] = None

        for run in range(1, n_simulations + 1):
            if run % max(1, n_simulations // 10) == 0 or run == 1:
                logger.info("Simulation Run %d / %d...", run, n_simulations)

            res = self.sim.simulate_tournament()
            placements = res["placements"]
            knockout = res["knockout"]
            last_bracket = knockout
            last_placements = placements

            # 1. Champion
            champ = placements["Champion"]
            counts[champ]["champion"] += 1
            champion_counts[champ] += 1

            # 2. Finalists (both finalist teams)
            counts[placements["Champion"]]["finalist"] += 1
            counts[placements["Runner-up"]]["finalist"] += 1

            pair_key = tuple(sorted((placements["Champion"], placements["Runner-up"])))
            finalist_pair_counts[pair_key] = finalist_pair_counts.get(pair_key, 0) + 1

            # 3. Semifinalists (top 4 teams)
            counts[placements["Champion"]]["semifinals"] += 1
            counts[placements["Runner-up"]]["semifinals"] += 1
            counts[placements["Third-place"]]["semifinals"] += 1
            counts[placements["Fourth-place"]]["semifinals"] += 1

            # 4. Quarterfinalists (8 teams in QF matches)
            for match in knockout["Quarterfinals"]:
                counts[match["home_team"]]["quarterfinals"] += 1
                counts[match["away_team"]]["quarterfinals"] += 1

            # 5. Round of 16 (16 teams in R16 matches)
            for match in knockout["Round of 16"]:
                counts[match["home_team"]]["round_of_16"] += 1
                counts[match["away_team"]]["round_of_16"] += 1

            # 6. Round of 32 (32 teams in R32 matches)
            for match in knockout["Round of 32"]:
                counts[match["home_team"]]["round_of_32"] += 1
                counts[match["away_team"]]["round_of_32"] += 1

        # Calculate probabilities (frequency / total simulations)
        prob_rows = []
        for team, stages in counts.items():
            prob_rows.append({
                "Team": team,
                "Group Stage": round(stages["group_stage"] / n_simulations, 4),
                "Round of 32": round(stages["round_of_32"] / n_simulations, 4),
                "Round of 16": round(stages["round_of_16"] / n_simulations, 4),
                "Quarterfinal": round(stages["quarterfinals"] / n_simulations, 4),
                "Semifinal": round(stages["semifinals"] / n_simulations, 4),
                "Finalist": round(stages["finalist"] / n_simulations, 4),
                "Champion": round(stages["champion"] / n_simulations, 4),
            })

        df_probs = pd.DataFrame(prob_rows)
        # Sort by Champion probability descending
        df_probs = df_probs.sort_values(by="Champion", ascending=False).reset_index(drop=True)

        # Champion probabilities table (subset with non-zero champion odds)
        df_champs = df_probs[["Team", "Champion"]].copy()
        df_champs = df_champs[df_champs["Champion"] > 0].reset_index(drop=True)

        # Save to output files if directory is provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save files as requested by Step 14
            df_probs.to_csv(output_dir / "team_probabilities.csv", index=False)
            df_champs.to_csv(output_dir / "champion_probabilities.csv", index=False)
            
            # Save simulation results summary
            sim_summary = []
            for team, count in champion_counts.items():
                if count > 0:
                    sim_summary.append({
                        "Team": team,
                        "Champion Runs": count,
                        "Winning Odds (%)": round((count / n_simulations) * 100, 2)
                    })
            df_summary = pd.DataFrame(sim_summary).sort_values(by="Champion Runs", ascending=False).reset_index(drop=True)
            df_summary.to_csv(output_dir / "simulation_results.csv", index=False)
            
            logger.info("Saved team probabilities to: %s", output_dir / "team_probabilities.csv")
            logger.info("Saved champion probabilities to: %s", output_dir / "champion_probabilities.csv")
            logger.info("Saved simulation summary to: %s", output_dir / "simulation_results.csv")

        # Most common real final matchup across all runs (unordered team pair)
        most_likely_final = None
        if finalist_pair_counts:
            best_pair, best_count = max(finalist_pair_counts.items(), key=lambda kv: kv[1])
            most_likely_final = {
                "team_a": best_pair[0],
                "team_b": best_pair[1],
                "probability": round(best_count / n_simulations, 4),
            }

        return {
            "team_probabilities": df_probs,
            "champion_probabilities": df_champs,
            "most_likely_final": most_likely_final,
            "sample_bracket": last_bracket,
            "sample_placements": last_placements,
        }
