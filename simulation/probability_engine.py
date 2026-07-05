"""Phase 4 - Step 2 (cont): Probability Engine.

This module provides the ProbabilityEngine class to handle probability distributions
for group stage (allowing draws) and knockout stage (no draws, requires extra time
and penalty resolution) matches.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from simulation.match_predictor import MatchPredictor

logger = logging.getLogger("simulation")


class ProbabilityEngine:
    """Engine to process and refine match outcome probabilities for different tournament stages."""

    def __init__(self, predictor: MatchPredictor):
        """Initialize the ProbabilityEngine with a MatchPredictor instance."""
        self.predictor = predictor
        self._cache = {}

    def predict_group_stage(
        self,
        home_team: str,
        away_team: str,
        match_date: str = "2026-06-15",
        tournament: str = "FIFA World Cup",
        venue: str = "neutral",
    ) -> Dict[str, Any]:
        """Predict a group stage match where draws are valid outcomes.

        Returns:
            The raw match predictor result dictionary.
        """
        key = ("group", home_team, away_team, match_date, tournament, venue)
        if key in self._cache:
            return self._cache[key]

        res = self.predictor.predict_match(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            tournament=tournament,
            venue=venue,
        )
        self._cache[key] = res
        return res

    def predict_knockout_stage(
        self,
        home_team: str,
        away_team: str,
        match_date: str = "2026-06-30",
        tournament: str = "FIFA World Cup",
        venue: str = "neutral",
    ) -> Dict[str, Any]:
        """Predict a knockout stage match where a winner must be decided (draw is resolved).

        This method retrieves prediction probabilities and, if a draw is predicted,
        resolves it using a pro-rata scaling method modified by team ELO and form
        to simulate extra time / penalty shootout outcomes.

        Returns:
            Dictionary containing matchup prediction with resolved knockout probabilities.
        """
        key = ("knockout", home_team, away_team, match_date, tournament, venue)
        if key in self._cache:
            return self._cache[key]

        # Get raw prediction
        prediction = self.predictor.predict_match(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            tournament=tournament,
            venue=venue,
        )

        probs = prediction["probabilities"]
        p_home = probs["home_win"]
        p_draw = probs["draw"]
        p_away = probs["away_win"]

        # Pro-rata baseline distribution of draw probability
        p_sum = p_home + p_away
        if p_sum > 0:
            p_home_ko = p_home / p_sum
            p_away_ko = p_away / p_sum
        else:
            p_home_ko = 0.5
            p_away_ko = 0.5

        # Incorporate subtle team skill adjustment for extra time / penalties
        # Teams with higher ELO and better form have a slight advantage in resolving draws
        home_stats = self.predictor.get_team_stats(home_team)
        away_stats = self.predictor.get_team_stats(away_team)

        elo_diff = home_stats["elo"] - away_stats["elo"]
        form_diff = home_stats["form"] - away_stats["form"]

        # Normalize ELO difference to a small probability shift [-0.05, 0.05]
        elo_shift = elo_diff / 800.0
        elo_shift = max(-0.05, min(0.05, elo_shift))

        # Normalize form difference to a small shift [-0.02, 0.02]
        form_shift = form_diff * 0.05
        form_shift = max(-0.02, min(0.02, form_shift))

        # Adjust knockout probabilities
        p_home_ko += (elo_shift + form_shift)
        p_home_ko = max(0.05, min(0.95, p_home_ko))
        p_away_ko = 1.0 - p_home_ko

        # Determine predicted knockout winner
        if p_home_ko >= p_away_ko:
            winner = home_stats["name"]
            predicted_outcome = "home_win"
            confidence = p_home_ko
        else:
            winner = away_stats["name"]
            predicted_outcome = "away_win"
            confidence = p_away_ko

        # Format resolved knockout result
        knockout_prediction = prediction.copy()
        knockout_prediction["predicted_winner"] = winner
        knockout_prediction["predicted_outcome"] = predicted_outcome
        knockout_prediction["probabilities"] = {
            "home_win": p_home_ko,
            "draw": 0.0,  # Draws are impossible in knockout stage
            "away_win": p_away_ko,
        }
        knockout_prediction["confidence_score"] = confidence
        
        # Include metadata about draw resolution
        knockout_prediction["draw_resolved"] = True
        knockout_prediction["raw_probabilities"] = probs

        self._cache[key] = knockout_prediction
        return knockout_prediction

