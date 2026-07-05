"""Tournament simulation services layer.

Coordinates between the API request, MonteCarloSimulator, and compiles
progression/advancement odds for all 48 teams.
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, Any, List

from backend.ml.model_loader import ml_loader
from backend.schemas.prediction_schema import SimulationRequest
from simulation.monte_carlo import MonteCarloSimulator

logger = logging.getLogger("backend")


class SimulationService:
    """Orchestrates tournament simulation tasks and formats outputs."""

    @staticmethod
    def run_simulation(request: SimulationRequest) -> Dict[str, Any]:
        """Execute a Monte Carlo simulation of the FIFA World Cup 2026.

        Args:
            request: SimulationRequest schema containing simulation count.

        Returns:
            Dictionary matching the structure of SimulationResponse.
        """
        logger.info("Executing simulation service: %d tournament runs requested.", request.run_count)

        # 1. Initialize simulator wrappers
        predictor = ml_loader.predictor
        prob_engine = ml_loader.prob_engine
        
        # Instantiate TournamentSimulator and MonteCarloSimulator
        from simulation.tournament_simulator import TournamentSimulator
        tournament_sim = TournamentSimulator(predictor, prob_engine)
        mc_simulator = MonteCarloSimulator(tournament_sim)

        # 2. Execute simulations
        mc_results = mc_simulator.run_monte_carlo(n_simulations=request.run_count)
        
        df_probs = mc_results["team_probabilities"]
        df_champs = mc_results["champion_probabilities"]

        # 3. Format outputs
        # Convert champion odds into a direct team -> odds dictionary
        champion_odds = {}
        for _, row in df_champs.iterrows():
            if row["Champion"] > 0:
                champion_odds[row["Team"]] = float(row["Champion"])

        # Convert full stage probabilities into a list of dictionaries
        stage_probs = []
        for _, row in df_probs.iterrows():
            stage_probs.append({
                "team": row["Team"],
                "group_stage": float(row["Group Stage"]),
                "round_of_32": float(row["Round of 32"]),
                "round_of_16": float(row["Round of 16"]),
                "quarterfinals": float(row["Quarterfinal"]),
                "semifinals": float(row["Semifinal"]),
                "finalist": float(row["Finalist"]),
                "champion": float(row["Champion"]),
            })

        # 4. Real ELO-rank-vs-simulated-odds upset metric: compare each team's
        # rank by champion probability to its rank by raw ELO. A team ranked
        # much higher on simulated odds than on ELO is a genuine model-derived
        # overperformer, not a scripted example.
        elo_by_team = {}
        for row in stage_probs:
            try:
                elo_by_team[row["team"]] = float(predictor.get_team_stats(row["team"]).get("elo", 0.0))
            except Exception:
                elo_by_team[row["team"]] = 0.0

        elo_ranked = sorted(elo_by_team.items(), key=lambda kv: kv[1], reverse=True)
        elo_rank = {team: idx + 1 for idx, (team, _) in enumerate(elo_ranked)}

        sim_ranked = sorted(stage_probs, key=lambda r: r["champion"], reverse=True)
        sim_rank = {row["team"]: idx + 1 for idx, row in enumerate(sim_ranked)}

        upsets = []
        for row in stage_probs:
            team = row["team"]
            rank_delta = elo_rank.get(team, 0) - sim_rank.get(team, 0)
            # Only surface teams with a real, non-trivial shot at the semifinal --
            # otherwise low-ELO teams "beating" other near-zero-odds teams would
            # register as a rank delta despite having no meaningful upset chance.
            if rank_delta > 0 and row["semifinals"] >= 0.01:
                upsets.append({
                    "team": team,
                    "elo_rank": elo_rank.get(team, 0),
                    "simulated_rank": sim_rank.get(team, 0),
                    "rank_delta": rank_delta,
                    "champion_probability": row["champion"],
                    "semifinal_probability": row["semifinals"],
                })
        upsets.sort(key=lambda u: u["rank_delta"], reverse=True)

        most_likely_final = mc_results.get("most_likely_final")
        sample_bracket = SimulationService._format_sample_bracket(
            mc_results.get("sample_bracket"), mc_results.get("sample_placements")
        )

        return {
            "run_count": request.run_count,
            "champion_odds": champion_odds,
            "stage_probabilities": stage_probs,
            "most_likely_final": most_likely_final,
            "upsets": upsets[:10],
            "sample_bracket": sample_bracket,
            "timestamp": datetime.datetime.utcnow(),
        }

    @staticmethod
    def _format_sample_bracket(bracket: Any, placements: Any) -> Dict[str, Any] | None:
        """Flatten one real simulated knockout bracket into a JSON-friendly shape.

        Returns Round of 32 -> Final matches (each with the winner and its
        real simulated win probability) plus the final placements, sourced
        from the last Monte Carlo run rather than any scripted example.
        """
        if not bracket:
            return None

        def _match_summary(match: Dict[str, Any]) -> Dict[str, Any]:
            winner = match["winner"]
            probs = match.get("probabilities", {})
            win_prob = probs.get("home_win") if winner == match["home_team"] else probs.get("away_win")
            return {
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "winner": winner,
                "win_probability": round(float(win_prob), 4) if win_prob is not None else None,
            }

        rounds = {}
        for stage in ("Round of 32", "Round of 16", "Quarterfinals", "Semifinals"):
            matches = bracket.get(stage, [])
            rounds[stage] = [_match_summary(m) for m in matches]

        final_match = bracket.get("Final")
        if final_match:
            rounds["Final"] = [_match_summary(final_match)]

        return {"rounds": rounds, "placements": placements}
