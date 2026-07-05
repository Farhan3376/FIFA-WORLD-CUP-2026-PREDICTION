"""Phase 4 - Step 7: Tournament Simulator.

This module provides the TournamentSimulator class to simulate the complete
FIFA World Cup 2026. It handles the group stage standing updates, third-place
advancement sorting, and knockout round progressions.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from simulation.match_predictor import MatchPredictor
from simulation.probability_engine import ProbabilityEngine
from simulation.bracket_generator import BracketGenerator

logger = logging.getLogger("simulation")


class TournamentSimulator:
    """Simulates the FIFA World Cup 2026 group stage and knockout rounds."""

    def __init__(self, predictor: MatchPredictor, prob_engine: ProbabilityEngine):
        """Initialize the simulator with predictor and probability engines."""
        self.predictor = predictor
        self.prob_engine = prob_engine
        self.bracket_gen = BracketGenerator()

    def simulate_match_outcome(
        self,
        home_team: str,
        away_team: str,
        stage: str = "group",
        match_date: str = "2026-06-15",
    ) -> Dict[str, Any]:
        """Simulate a match outcome and sample goals using Poisson distributions.

        Args:
            home_team: Name of the home team.
            away_team: Name of the away team.
            stage: 'group' or 'knockout'.
            match_date: Date string of the match.

        Returns:
            Dictionary containing match results (score, winner, outcome).
        """
        if stage == "group":
            pred = self.prob_engine.predict_group_stage(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
            )
            outcomes = ["home_win", "draw", "away_win"]
            probs = [
                pred["probabilities"]["home_win"],
                pred["probabilities"]["draw"],
                pred["probabilities"]["away_win"],
            ]
        else:
            pred = self.prob_engine.predict_knockout_stage(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
            )
            outcomes = ["home_win", "away_win"]
            probs = [
                pred["probabilities"]["home_win"],
                pred["probabilities"]["away_win"],
            ]

        # Normalize probabilities to ensure they sum to exactly 1.0
        probs = np.array(probs)
        probs /= probs.sum()

        # 1. Sample the match outcome based on predicted probabilities
        outcome = np.random.choice(outcomes, p=probs)

        # 2. Sample goals using Poisson distribution from predicted xG
        home_xg = pred["expected_goals"]["home_xg"]
        away_xg = pred["expected_goals"]["away_xg"]

        # Ensure goal sampling aligns with the simulated outcome (resample if necessary)
        home_goals = 0
        away_goals = 0
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            hg = int(np.random.poisson(home_xg))
            ag = int(np.random.poisson(away_xg))
            
            if outcome == "home_win" and hg > ag:
                home_goals, away_goals = hg, ag
                break
            elif outcome == "away_win" and ag > hg:
                home_goals, away_goals = hg, ag
                break
            elif outcome == "draw" and hg == ag:
                home_goals, away_goals = hg, ag
                break
            attempt += 1
            
        # Fallback in case of resampling limit
        if attempt == max_attempts:
            if outcome == "home_win":
                home_goals, away_goals = 1, 0
            elif outcome == "away_win":
                home_goals, away_goals = 0, 1
            else:
                home_goals, away_goals = 1, 1

        # Determine winner name
        if outcome == "home_win":
            winner = home_team
        elif outcome == "away_win":
            winner = away_team
        else:
            winner = "Draw"

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "outcome": outcome,
            "winner": winner,
            "probabilities": pred["probabilities"],
        }

    def simulate_group_stage(self) -> Dict[str, pd.DataFrame]:
        """Simulate all group stage matches and calculate standings tables.

        Returns:
            Dictionary mapping Group Name to standings DataFrame.
        """
        group_standings = {}

        for group_name, teams in self.bracket_gen.GROUPS.items():
            # Initialize standings stats for the group
            standings = {
                t: {
                    "team": t,
                    "points": 0,
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "gf": 0,  # goals for
                    "ga": 0,  # goals against
                    "gd": 0,  # goal difference
                    "elo": self.predictor.get_team_stats(t)["elo"]
                }
                for t in teams
            }

            # Generate and simulate all matches
            matches = self.bracket_gen.get_group_matches(teams)
            for home, away in matches:
                res = self.simulate_match_outcome(home, away, stage="group")
                
                # Update stats
                standings[home]["played"] += 1
                standings[away]["played"] += 1
                standings[home]["gf"] += res["home_goals"]
                standings[home]["ga"] += res["away_goals"]
                standings[away]["gf"] += res["away_goals"]
                standings[away]["ga"] += res["home_goals"]

                if res["outcome"] == "home_win":
                    standings[home]["points"] += 3
                    standings[home]["wins"] += 1
                    standings[away]["losses"] += 1
                elif res["outcome"] == "away_win":
                    standings[away]["points"] += 3
                    standings[away]["wins"] += 1
                    standings[home]["losses"] += 1
                else:
                    standings[home]["points"] += 1
                    standings[away]["points"] += 1
                    standings[home]["draws"] += 1
                    standings[away]["draws"] += 1

            # Recalculate goal differences
            for t in teams:
                standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]

            # Sort standings based on tie-breakers: Points, GD, GF, ELO (fallback)
            df_standings = pd.DataFrame(standings.values())
            df_standings = df_standings.sort_values(
                by=["points", "gd", "gf", "elo"],
                ascending=[False, False, False, False]
            ).reset_index(drop=True)
            
            group_standings[group_name] = df_standings

        return group_standings

    def determine_advancing_teams(
        self,
        group_standings: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
        """Identify which teams qualify for the Round of 32 from standings.

        Returns:
            Tuple of:
            - Dict mapping Group Name to Winner team name
            - Dict mapping Group Name to Runner-up team name
            - List of the 8 best 3rd-placed team names
        """
        winners = {}
        runners = {}
        thirds = []

        for group_name, df in group_standings.items():
            winners[group_name] = df.loc[0, "team"]
            runners[group_name] = df.loc[1, "team"]
            
            # Save 3rd place team stats for wildcard comparisons
            third_place_team = df.loc[2].to_dict()
            third_place_team["group"] = group_name
            thirds.append(third_place_team)

        # Sort the 12 third-placed teams to find the top 8
        df_thirds = pd.DataFrame(thirds)
        df_thirds = df_thirds.sort_values(
            by=["points", "gd", "gf", "elo"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)

        best_thirds = df_thirds.loc[:7, "team"].tolist()
        return winners, runners, best_thirds

    def simulate_knockout_stage(
        self,
        winners: Dict[str, str],
        runners: Dict[str, str],
        best_thirds: List[str]
    ) -> Dict[str, Any]:
        """Simulate all knockout rounds to determine the tournament champion.

        Returns:
            Dictionary containing knockout bracket matches and placements.
        """
        bracket_history = {}

        # 1. Round of 32
        r32_pairings = self.bracket_gen.get_round_of_32_pairings(winners, runners, best_thirds)
        r32_winners = []
        r32_matches = []
        for home, away in r32_pairings:
            res = self.simulate_match_outcome(home, away, stage="knockout", match_date="2026-06-30")
            r32_winners.append(res["winner"])
            r32_matches.append(res)
        bracket_history["Round of 32"] = r32_matches

        # 2. Round of 16
        r16_pairings = self.bracket_gen.get_next_round_pairings(r32_winners)
        r16_winners = []
        r16_matches = []
        for home, away in r16_pairings:
            res = self.simulate_match_outcome(home, away, stage="knockout", match_date="2026-07-04")
            r16_winners.append(res["winner"])
            r16_matches.append(res)
        bracket_history["Round of 16"] = r16_matches

        # 3. Quarterfinals
        qf_pairings = self.bracket_gen.get_next_round_pairings(r16_winners)
        qf_winners = []
        qf_matches = []
        for home, away in qf_pairings:
            res = self.simulate_match_outcome(home, away, stage="knockout", match_date="2026-07-09")
            qf_winners.append(res["winner"])
            qf_matches.append(res)
        bracket_history["Quarterfinals"] = qf_matches

        # 4. Semifinals
        sf_pairings = self.bracket_gen.get_next_round_pairings(qf_winners)
        sf_winners = []
        sf_losers = []
        sf_matches = []
        for home, away in sf_pairings:
            res = self.simulate_match_outcome(home, away, stage="knockout", match_date="2026-07-14")
            sf_winners.append(res["winner"])
            sf_losers.append(home if res["winner"] == away else away)
            sf_matches.append(res)
        bracket_history["Semifinals"] = sf_matches

        # 5. Third-place Play-off
        third_place_res = self.simulate_match_outcome(
            sf_losers[0], sf_losers[1], stage="knockout", match_date="2026-07-18"
        )
        third_place_winner = third_place_res["winner"]
        fourth_place = sf_losers[1] if third_place_winner == sf_losers[0] else sf_losers[0]
        bracket_history["Third-place Play-off"] = third_place_res

        # 6. Final
        final_res = self.simulate_match_outcome(
            sf_winners[0], sf_winners[1], stage="knockout", match_date="2026-07-19"
        )
        champion = final_res["winner"]
        runner_up = sf_winners[1] if champion == sf_winners[0] else sf_winners[0]
        bracket_history["Final"] = final_res

        # Compile final placements
        placements = {
            "Champion": champion,
            "Runner-up": runner_up,
            "Third-place": third_place_winner,
            "Fourth-place": fourth_place,
        }

        return {
            "bracket": bracket_history,
            "placements": placements
        }

    def simulate_tournament(self) -> Dict[str, Any]:
        """Simulate a complete FIFA World Cup 2026 tournament.

        Returns:
            Dictionary containing group standings, knockout progression, and placements.
        """
        logger.info("Simulating complete FIFA World Cup 2026...")
        
        # Simulate group stage
        standings = self.simulate_group_stage()
        
        # Advance teams
        winners, runners, best_thirds = self.determine_advancing_teams(standings)
        
        # Simulate knockouts
        knockout_results = self.simulate_knockout_stage(winners, runners, best_thirds)
        
        return {
            "group_standings": standings,
            "knockout": knockout_results["bracket"],
            "placements": knockout_results["placements"]
        }
