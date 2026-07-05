"""Shared team-stats and model-explanation data helpers.

Centralizes fetch logic used by both the Home page's Quick Match Predictor
and the Match Prediction page, so both pages read the same real backend
data the same way rather than duplicating fetch/cache logic per page.
"""

from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_team_snapshot(team: str) -> dict | None:
    """Fetch ELO and rolling stats for a single team."""
    try:
        return st.session_state.api_client.get_team_stats(team)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_elo_rank_map() -> dict[str, int]:
    """Build a global ELO-based rank (1 = highest ELO) across all teams.

    This is explicitly an ELO-derived ranking, not the official FIFA World
    Ranking (which the backend does not provide) -- labeled as "ELO Rank"
    everywhere it's displayed so it's never mistaken for the real thing.
    """
    try:
        data = st.session_state.api_client.get_global_rankings()
        top_elo = data.get("top_elo_teams", [])
        return {row["team"]: idx + 1 for idx, row in enumerate(top_elo)}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_feature_importances(limit: int = 4) -> list[tuple[str, float]]:
    """Fetch the model's top global feature weights."""
    try:
        importances = st.session_state.api_client.get_feature_importance()
        ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
    except Exception:
        return []


FEATURE_LABELS = {
    "elo_diff": "ELO Difference",
    "home_elo_before": "Home ELO Rating",
    "away_elo_before": "Away ELO Rating",
    "form_diff": "Team Form",
    "goal_avg_diff": "Goal Difference",
    "elo_win_prob": "FIFA Ranking Gap",
    "win_pct_diff": "Win Percentage Gap",
    "home_avg_goal_diff": "Home Goal Difference",
    "away_avg_goal_diff": "Away Goal Difference",
    "goal_conceded_avg_diff": "Defensive Difference",
    "home_draw_pct": "Home Draw Tendency",
    "away_draw_pct": "Away Draw Tendency",
    "home_avg_goals_scored": "Home Attack Output",
    "away_avg_goals_scored": "Away Attack Output",
    "away_games_played": "Away Match Experience",
    "home_games_played": "Home Match Experience",
    "away_away_win_pct": "Away Win Tendency (Traveling)",
    "home_home_win_pct": "Home Win Tendency (Hosting)",
    "games_played_diff": "Match Experience Gap",
}


def humanize_feature(name: str) -> str:
    """Map a raw model feature name to a human-readable label, falling back to a cleaned version."""
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


FALLBACK_QUALIFIED_TEAMS = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bosnia and Herzegovina",
    "Brazil", "Canada", "Cape Verde", "Colombia", "Croatia", "Curaçao", "Czech Republic",
    "Democratic Republic of the Congo", "Ecuador", "Egypt", "England", "France", "Germany",
    "Ghana", "Iran", "Italy", "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco",
    "Netherlands", "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar",
    "Saudi Arabia", "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Switzerland", "Tunisia", "USA", "Uruguay", "Uzbekistan",
]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tournament_stage_probs(run_count: int = 500) -> dict[str, dict] | None:
    """Run a Monte Carlo simulation and index stage probabilities by team name.

    Reuses the real Tournament Simulator output rather than inventing a
    separate calculation -- returns None if the backend is unavailable.
    """
    try:
        result = st.session_state.api_client.simulate(run_count=run_count)
        return {s["team"]: s for s in result.get("stage_probabilities", [])}
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_qualified_teams() -> list[str]:
    """Load the 48 real FIFA World Cup 2026 qualified teams (not the full ~336-team
    historical ELO database), with a hardcoded fallback if the live fixtures API
    is unreachable.
    """
    try:
        qualified = st.session_state.api_client.get_qualified_teams()
        if qualified:
            return qualified
    except Exception:
        pass
    return FALLBACK_QUALIFIED_TEAMS
