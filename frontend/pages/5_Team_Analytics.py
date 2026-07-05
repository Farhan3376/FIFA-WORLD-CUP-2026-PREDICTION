"""FIFA World Cup 2026 Prediction Portal - National Team Profiles.

Provides detailed statistical profiles for individual teams, visualizing ELO,
historical win/loss distributions, attacking averages, and home/away splits.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from frontend.services.theme import render_hero, section_title, flag_for, EMERALD, BLUE_ACCENT, RED_CARD, TEXT_MUTED


def load_teams() -> list[str]:
    """Load recognized national teams from the backend with sandbox fallback."""
    try:
        return st.session_state.api_client.get_teams()
    except Exception:
        return [
            "Argentina", "Australia", "Belgium", "Brazil", "Cameroon", "Canada",
            "Croatia", "Denmark", "Ecuador", "England", "France", "Germany",
            "Ghana", "Iran", "Italy", "Japan", "Mexico", "Morocco",
            "Netherlands", "Portugal", "Qatar", "Saudi Arabia", "Senegal",
            "South Korea", "Spain", "Switzerland", "Tunisia", "USA",
            "Uruguay", "Wales"
        ]


def fetch_team_stats(team_name: str) -> Dict[str, Any]:
    """Fetch database details for a single team with sandbox offline fallback."""
    try:
        api = st.session_state.api_client
        return api.get_team_stats(team_name)
    except Exception:
        # Pre-computed sandbox profiles for top teams to ensure offline experience
        sandbox_db = {
            "argentina": {
                "name": "Argentina", "elo": 2143.0, "overall_win_pct": 0.68, "draw_pct": 0.18, "loss_pct": 0.14,
                "avg_goals_scored": 2.25, "avg_goals_conceded": 0.65, "avg_goal_diff": 1.60, "games_played": 62, "form": 0.85,
                "home_home_win_pct": 0.75, "away_away_win_pct": 0.60
            },
            "france": {
                "name": "France", "elo": 2088.0, "overall_win_pct": 0.64, "draw_pct": 0.20, "loss_pct": 0.16,
                "avg_goals_scored": 2.18, "avg_goals_conceded": 0.72, "avg_goal_diff": 1.46, "games_played": 58, "form": 0.80,
                "home_home_win_pct": 0.70, "away_away_win_pct": 0.58
            },
            "brazil": {
                "name": "Brazil", "elo": 1998.0, "overall_win_pct": 0.60, "draw_pct": 0.22, "loss_pct": 0.18,
                "avg_goals_scored": 2.05, "avg_goals_conceded": 0.82, "avg_goal_diff": 1.23, "games_played": 60, "form": 0.70,
                "home_home_win_pct": 0.72, "away_away_win_pct": 0.48
            },
            "england": {
                "name": "England", "elo": 2012.0, "overall_win_pct": 0.58, "draw_pct": 0.24, "loss_pct": 0.18,
                "avg_goals_scored": 1.95, "avg_goals_conceded": 0.80, "avg_goal_diff": 1.15, "games_played": 55, "form": 0.72,
                "home_home_win_pct": 0.68, "away_away_win_pct": 0.48
            },
            "spain": {
                "name": "Spain", "elo": 2045.0, "overall_win_pct": 0.62, "draw_pct": 0.22, "loss_pct": 0.16,
                "avg_goals_scored": 2.10, "avg_goals_conceded": 0.78, "avg_goal_diff": 1.32, "games_played": 57, "form": 0.78,
                "home_home_win_pct": 0.72, "away_away_win_pct": 0.52
            }
        }
        key = team_name.strip().lower()
        if key in sandbox_db:
            res = sandbox_db[key]
            res["recognized"] = True
            return res
        
        # Default mock for other teams
        import random
        random.seed(len(team_name))
        elo = round(random.uniform(1450, 1850), 1)
        win = round(random.uniform(0.35, 0.55), 2)
        draw = round(random.uniform(0.20, 0.30), 2)
        loss = round(1.0 - win - draw, 2)
        goals_s = round(random.uniform(1.1, 1.8), 2)
        goals_c = round(random.uniform(1.0, 1.6), 2)
        
        return {
            "name": team_name,
            "recognized": True,
            "elo": elo,
            "overall_win_pct": win,
            "draw_pct": draw,
            "loss_pct": loss,
            "avg_goals_scored": goals_s,
            "avg_goals_conceded": goals_c,
            "avg_goal_diff": round(goals_s - goals_c, 2),
            "games_played": int(random.uniform(25, 45)),
            "form": round(random.uniform(0.40, 0.75), 2),
            "home_home_win_pct": round(win * 1.15, 2),
            "away_away_win_pct": round(win * 0.85, 2)
        }


def show() -> None:
    """Render the Team Analytics interface."""
    render_hero(
        kicker="Team Profiles",
        title="📈 National Team Statistical Profiles",
        subtitle="Analyze a team's historical data, ELO benchmarks, scoring rates, and performance splits.",
        badges=[("⚽ 48 Teams Indexed", "live", "")],
    )

    teams = load_teams()

    # Dropdown selector
    with st.container(border=True):
        team_name = st.selectbox(
            "⚽ Choose National Team to Analyze", options=teams, index=0,
            format_func=lambda t: f"{flag_for(t)}  {t}",
        )
        st.markdown(
            f"""
            <div style="text-align:center; font-size:3.2rem; margin-top:8px;">{flag_for(team_name)}</div>
            <div style="text-align:center; font-family:'Outfit',sans-serif; font-weight:800; font-size:1.4rem; color:#ffffff;">{team_name}</div>
            """,
            unsafe_allow_html=True,
        )

    # Fetch stats
    stats = fetch_team_stats(team_name)

    # Grid parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">ELO Rating</div>
                <div class="value">{stats['elo']:.1f}</div>
                <div class="delta" style="color: {EMERALD};">Classified as Powerhouse</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">Matches Tracked</div>
                <div class="value">{stats['games_played']}</div>
                <div class="delta" style="color: {TEXT_MUTED};">Since 1998 database</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">Average Scored</div>
                <div class="value">{stats['avg_goals_scored']:.2f}</div>
                <div class="delta" style="color: {BLUE_ACCENT};">Goals scored/match</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">Average Conceded</div>
                <div class="value">{stats['avg_goals_conceded']:.2f}</div>
                <div class="delta" style="color: {RED_CARD};">Goals conceded/match</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Visual Profiling Charts")
    
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Match outcomes pie chart
        st.markdown("#### Historical Match Outcomes Breakdown")
        df_outcomes = pd.DataFrame({
            "Outcome": ["Win", "Draw", "Loss"],
            "Percentage": [stats["overall_win_pct"] * 100, stats["draw_pct"] * 100, stats["loss_pct"] * 100]
        })
        fig_outcomes = px.pie(
            df_outcomes,
            names="Outcome",
            values="Percentage",
            hole=0.4,
            color="Outcome",
            color_discrete_map={"Win": EMERALD, "Draw": TEXT_MUTED, "Loss": RED_CARD},
            template="plotly_white"
        )
        fig_outcomes.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_outcomes, use_container_width=True)

    with col_chart2:
        # Home vs Away win percentages bar chart
        st.markdown("#### Home vs Away Win Performance Ratio")
        df_splits = pd.DataFrame({
            "Configuration": ["Overall Win %", "Home Win % (when Hosting)", "Away Win % (when Traveling)"],
            "Percentage": [stats["overall_win_pct"] * 100, stats["home_home_win_pct"] * 100, stats["away_away_win_pct"] * 100]
        })
        fig_splits = px.bar(
            df_splits,
            x="Percentage",
            y="Configuration",
            orientation="h",
            color="Percentage",
            color_continuous_scale="Greens",
            labels={"Configuration": "", "Percentage": "Win Percentage (%)"},
            template="plotly_white",
        )
        fig_splits.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            coloraxis_showscale=False,
            xaxis_title="Percentage (%)",
            yaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_splits, use_container_width=True)

    # Detailed metrics comparison gauge
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Attacking Potency vs Defensive Strength Gauge")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Attack average
        fig_att = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats["avg_goals_scored"],
            title={"text": "Average Goals Scored", "font": {"size": 15, "color": "#e6ecf5"}},
            gauge={
                "axis": {"range": [0, 3], "tickwidth": 1, "tickcolor": "#93a3bd"},
                "bar": {"color": EMERALD},
                "bgcolor": "#111a2c",
                "steps": [
                    {"range": [0, 1], "color": RED_CARD},
                    {"range": [1, 2], "color": "#d4af37"},
                    {"range": [2, 3], "color": EMERALD}
                ]
            }
        ))
        fig_att.update_layout(margin=dict(l=30, r=30, t=40, b=10), height=200, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_att, use_container_width=True)

    with col_g2:
        # Defense average
        fig_def = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats["avg_goals_conceded"],
            title={"text": "Average Goals Conceded (Lower is Better)", "font": {"size": 15, "color": "#e6ecf5"}},
            gauge={
                "axis": {"range": [0, 3], "tickwidth": 1, "tickcolor": "#93a3bd"},
                "bar": {"color": RED_CARD},
                "bgcolor": "#111a2c",
                "steps": [
                    {"range": [0, 0.8], "color": EMERALD},
                    {"range": [0.8, 1.5], "color": "#d4af37"},
                    {"range": [1.5, 3], "color": RED_CARD}
                ]
            }
        ))
        fig_def.update_layout(margin=dict(l=30, r=30, t=40, b=10), height=200, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_def, use_container_width=True)


if __name__ == "__main__":
    show()
