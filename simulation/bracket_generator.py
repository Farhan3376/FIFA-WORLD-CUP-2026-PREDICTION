"""Phase 4 - Step 2 (cont): Tournament Bracket Generator.

This module defines the 48-team FIFA World Cup 2026 group structure and
implements the logic to pair teams for the Round of 32 based on group standings
and best 3rd-placed teams.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("simulation")


class BracketGenerator:
    """Generates groups and knockout stage pairings for the FIFA World Cup 2026."""

    # User-specified groups for the 48-team tournament (12 groups of 4 teams each)
    GROUPS: Dict[str, List[str]] = {
        "Group A": ["Mexico", "South Africa", "South Korea", "Czechia"],
        "Group B": ["Canada", "Bosnia and Herzzegovina", "Qatar", "Switzerland"],
        "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "Group D": ["United States", "Paraguay", "Australia", "Turkey"],
        "Group E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
        "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "Group H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
        "Group I": ["France", "Senegal", "Iraq", "Norway"],
        "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "Group L": ["England", "Croatia", "Ghana", "Panama"],
    }

    @staticmethod
    def get_group_matches(group_teams: List[str]) -> List[Tuple[str, str]]:
        """Generate all 6 round-robin matches for a group of 4 teams."""
        matches = []
        for i in range(len(group_teams)):
            for j in range(i + 1, len(group_teams)):
                matches.append((group_teams[i], group_teams[j]))
        return matches

    @staticmethod
    def get_round_of_32_pairings(
        winners: Dict[str, str],
        runners: Dict[str, str],
        best_thirds: List[str],
    ) -> List[Tuple[str, str]]:
        """Pair the 32 advancing teams into 16 knockout matches.

        The 32 teams are composed of:
        - 12 group winners (keys: Group A to L)
        - 12 group runners-up (keys: Group A to L)
        - 8 best 3rd-placed teams (ordered list of 8 team names)

        Pairing rules:
        - 8 group winners face the 8 best 3rd-placed teams.
        - 4 group winners face 4 runners-up.
        - 8 runners-up face each other in 4 matches.
        """
        # Ensure we have exactly 8 third-place teams
        if len(best_thirds) < 8:
            logger.warning("Fewer than 8 third-placed teams provided. Padding with default.")
            # Pad if necessary
            best_thirds = list(best_thirds) + ["Fallback"] * (8 - len(best_thirds))

        pairings = [
            # 8 Group Winners vs 8 Best 3rd-placed teams
            (winners["Group A"], best_thirds[0]),
            (winners["Group D"], best_thirds[1]),
            (winners["Group G"], best_thirds[2]),
            (winners["Group J"], best_thirds[3]),
            (winners["Group B"], best_thirds[4]),
            (winners["Group C"], best_thirds[5]),
            (winners["Group E"], best_thirds[6]),
            (winners["Group F"], best_thirds[7]),

            # 4 Group Winners vs 4 Runners-up
            (winners["Group H"], runners["Group A"]),
            (winners["Group I"], runners["Group D"]),
            (winners["Group K"], runners["Group G"]),
            (winners["Group L"], runners["Group J"]),

            # 8 Runners-up vs each other
            (runners["Group B"], runners["Group C"]),
            (runners["Group E"], runners["Group F"]),
            (runners["Group H"], runners["Group I"]),
            (runners["Group K"], runners["Group L"]),
        ]
        return pairings

    @staticmethod
    def get_next_round_pairings(winners_previous_round: List[str]) -> List[Tuple[str, str]]:
        """Pair the winners of the previous knockout round for the next round.

        Args:
            winners_previous_round: Ordered list of winning teams from the previous round matches.

        Returns:
            List of matchup tuples for the next round.
        """
        pairings = []
        for i in range(0, len(winners_previous_round), 2):
            if i + 1 < len(winners_previous_round):
                pairings.append((winners_previous_round[i], winners_previous_round[i + 1]))
        return pairings
