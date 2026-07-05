"""FIFA World Cup 2026 Prediction Portal - Match Predictor.

FIFA Match Centre-style prediction experience: a stadium-toned match hero
with ELO rank/rating badges, a unified three-ring win-probability card,
a team comparison (radar + metric bars), and an AI explanation section
with human-readable SHAP factors and generated match insights.

Head-to-Head history is intentionally omitted -- the backend has no
per-match historical query between two specific teams, only aggregate
stats, so that section is not fabricated here.
"""

from __future__ import annotations

import json
import math
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

from frontend.services.theme import (
    section_title, flag_for,
    EMERALD, GOLD, TEXT_MUTED, WINE, FIFA_BLUE, FIFA_BLUE_LIGHT,
)
from frontend.services.team_data import (
    fetch_team_snapshot, fetch_elo_rank_map, humanize_feature, load_qualified_teams,
)


@st.cache_data(ttl=25, show_spinner=False)
def load_fixtures() -> list[dict]:
    """Fetch tournament fixtures from the live fixtures API, cached briefly to avoid hammering it.

    Returns an empty list on failure so the fixture picker can degrade to
    manual team selection only.
    """
    try:
        return st.session_state.api_client.get_fixtures()
    except Exception:
        return []


def get_mock_prediction(home: str, away: str) -> dict:
    """Generate professional mock prediction for sandbox offline execution."""
    import random
    random.seed(len(home) + len(away))
    h_win = round(random.uniform(0.35, 0.60), 3)
    draw = round(random.uniform(0.15, 0.30), 3)
    a_win = round(1.0 - h_win - draw, 3)

    winner = home if h_win > a_win else away
    if draw > h_win and draw > a_win:
        winner = "Draw"

    return {
        "home_team": home,
        "away_team": away,
        "predicted_winner": winner,
        "probabilities": {
            "home_win": h_win,
            "draw": draw,
            "away_win": a_win
        },
        "confidence_score": round(max(h_win, a_win) * 1.2, 3),
        "expected_goals": {
            "home_xg": round(random.uniform(1.2, 2.5), 2),
            "away_xg": round(random.uniform(0.8, 1.8), 2)
        },
        "shap_explanation": {
            "predicted_outcome": winner,
            "base_value": 0.33,
            "contributions": {
                "home_elo_diff": round(random.uniform(-0.15, 0.25), 3),
                "away_recent_form": round(random.uniform(-0.08, 0.12), 3),
                "neutral_venue_impact": round(random.uniform(-0.05, 0.05), 3),
                "head_to_head_record": round(random.uniform(-0.10, 0.15), 3),
                "tournament_weight": round(random.uniform(-0.02, 0.08), 3)
            },
            "total_impact": 0.5
        }
    }


def render_match_hero(home_team: str, away_team: str, rank_map: dict[str, int]) -> None:
    """Render the stadium-toned match hero: large flags, names, ELO rank/rating badges."""
    home_stats = fetch_team_snapshot(home_team)
    away_stats = fetch_team_snapshot(away_team)
    home_elo = home_stats.get("elo") if home_stats else None
    away_elo = away_stats.get("elo") if away_stats else None
    home_rank = rank_map.get(home_team)
    away_rank = rank_map.get(away_team)

    def _team_stats_html(rank: int | None, elo: float | None) -> str:
        badge = f'<span class="mp-stat-badge">ELO RANK #{rank}</span>' if rank else ""
        elo_line = f'<span class="mp-stat-line">ELO Rating <b>{elo:.0f}</b></span>' if elo is not None else ""
        return f'{badge}{elo_line}'

    st.markdown(
        f'<div class="mp-hero">'
        f'<div class="mp-kicker">FIFA World Cup 2026 · Match Centre</div>'
        f'<div class="mp-matchup">'
        f'<div class="mp-team">'
        f'<span class="flag">{flag_for(home_team)}</span>'
        f'<div class="name">{home_team}</div>'
        f'<div class="mp-team-stats" style="--qc-color:{EMERALD};">{_team_stats_html(home_rank, home_elo)}</div>'
        f'</div>'
        f'<div class="mp-vs">VS</div>'
        f'<div class="mp-team">'
        f'<span class="flag">{flag_for(away_team)}</span>'
        f'<div class="name">{away_team}</div>'
        f'<div class="mp-team-stats" style="--qc-color:{FIFA_BLUE_LIGHT};">{_team_stats_html(away_rank, away_elo)}</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_win_probability(home_team: str, away_team: str, probs: dict, confidence: float) -> None:
    """Render the unified three-ring win-probability card with a confidence bar beneath."""
    circumference = 2 * math.pi * 62
    outcomes = [
        (f"{home_team} Win", probs["home_win"], EMERALD),
        ("Draw", probs["draw"], GOLD),
        (f"{away_team} Win", probs["away_win"], FIFA_BLUE_LIGHT),
    ]

    ring_cards = []
    for label, pct, color in outcomes:
        offset = circumference * (1 - pct)
        ring_cards.append(
            f'<div class="mp-wp-ring-card">'
            f'<div class="mp-ring-wrap">'
            f'<svg width="140" height="140" viewBox="0 0 140 140">'
            f'<circle class="mp-ring-bg" cx="70" cy="70" r="62"></circle>'
            f'<circle class="mp-ring-fill" cx="70" cy="70" r="62" stroke="{color}" '
            f'stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"></circle>'
            f'</svg>'
            f'<div class="mp-ring-value">{pct*100:.0f}%</div>'
            f'</div>'
            f'<div class="mp-ring-label">{label}</div>'
            f'</div>'
        )

    conf_pct = max(0.0, min(1.0, confidence)) * 100
    html = (
        f'<div class="mp-wp-card">'
        f'<div class="mp-wp-title">Win Probability</div>'
        f'<div class="mp-wp-grid">{"".join(ring_cards)}</div>'
        f'<div class="mp-conf-row">'
        f'<span class="mp-conf-label">Prediction Confidence</span>'
        f'<div class="mp-conf-track"><div class="mp-conf-fill" style="width:{conf_pct:.0f}%;"></div></div>'
        f'<span class="mp-conf-value">{conf_pct:.0f}%</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


RADAR_AXES = [
    ("Attack", "avg_goals_scored", False),
    ("Form", "form", False),
    ("ELO", "elo", False),
    ("Win %", "overall_win_pct", False),
    ("Defense", "avg_goals_conceded", True),  # inverted: lower is better
]


def _normalize_radar_value(value: float, axis_key: str, invert: bool, all_values: dict[str, list[float]]) -> float:
    """Normalize a stat onto a 0-1 radar scale relative to both teams' values on this axis."""
    values = all_values.get(axis_key, [value])
    lo, hi = min(values), max(values)
    if hi == lo:
        return 0.7
    norm = (value - lo) / (hi - lo)
    return (1 - norm) if invert else norm


def render_team_comparison(home_team: str, away_team: str) -> None:
    """Render the team comparison: radar chart + horizontal metric bars, using real stats."""
    section_title("Team Comparison")
    home_stats = fetch_team_snapshot(home_team) or {}
    away_stats = fetch_team_snapshot(away_team) or {}

    col_radar, col_bars = st.columns([1, 1.1])

    with col_radar:
        st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
        st.markdown('<div class="mp-cmp-title">Statistical Radar</div>', unsafe_allow_html=True)

        # Build normalized radar values relative to both teams for each axis
        all_values = {
            key: [home_stats.get(key, 0.0), away_stats.get(key, 0.0)]
            for _, key, _ in RADAR_AXES
        }
        home_norm = [
            _normalize_radar_value(home_stats.get(key, 0.0), key, invert, all_values)
            for _, key, invert in RADAR_AXES
        ]
        away_norm = [
            _normalize_radar_value(away_stats.get(key, 0.0), key, invert, all_values)
            for _, key, invert in RADAR_AXES
        ]

        categories = [label for label, _, _ in RADAR_AXES] + [RADAR_AXES[0][0]]
        home_r = home_norm + [home_norm[0]]
        away_r = away_norm + [away_norm[0]]

        fig = px.line_polar(
            r=home_r, theta=categories, line_close=True,
        )
        fig.data[0].update(name=home_team, line_color=EMERALD, fill="toself", fillcolor="rgba(31,174,110,0.18)")
        fig.add_trace(go.Scatterpolar(
            r=away_r, theta=categories, name=away_team,
            line_color=FIFA_BLUE_LIGHT, fill="toself", fillcolor="rgba(61,139,255,0.15)",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#E1E6EE"),
                angularaxis=dict(gridcolor="#E1E6EE"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=20, b=20),
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bars:
        st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
        st.markdown('<div class="mp-cmp-title">Head-to-Head Metrics</div>', unsafe_allow_html=True)

        metrics = [
            ("Avg. Goals Scored", "avg_goals_scored", "{:.2f}", False),
            ("Avg. Goals Conceded (lower better)", "avg_goals_conceded", "{:.2f}", True),
            ("Overall Win %", "overall_win_pct", "{:.1%}", False),
            ("ELO Rating", "elo", "{:.0f}", False),
            ("Recent Form Index", "form", "{:.2f}", False),
        ]
        bar_rows = []
        for label, key, fmt, invert in metrics:
            home_val = home_stats.get(key, 0.0)
            away_val = away_stats.get(key, 0.0)
            max_val = max(home_val, away_val) or 1.0
            home_width = max(4, (home_val / max_val) * 100) if not invert else max(4, (1 - home_val / max_val) * 100 + 20)
            away_width = max(4, (away_val / max_val) * 100) if not invert else max(4, (1 - away_val / max_val) * 100 + 20)
            bar_rows.append(
                f'<div class="mp-bar-row">'
                f'<div class="mp-bar-metric">{label}</div>'
                f'<span class="mp-bar-val home">{fmt.format(home_val)}</span>'
                f'<div class="mp-track-pair">'
                f'<div class="mp-track-home"><div class="mp-fill-home" style="width:{home_width:.0f}%;"></div></div>'
                f'<div class="mp-track-away"><div class="mp-fill-away" style="width:{away_width:.0f}%;"></div></div>'
                f'</div>'
                f'<span class="mp-bar-val away">{fmt.format(away_val)}</span>'
                f'</div>'
            )
        st.markdown("".join(bar_rows), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_ai_explanation(home_team: str, away_team: str, result: dict) -> None:
    """Render the AI explanation: humanized SHAP factor bars + generated match insight cards."""
    section_title("AI Explanation")

    shap_data = result.get("shap_explanation", {})
    contributions = shap_data.get("contributions", shap_data) if shap_data else {}
    clean_contributions = {
        k: v for k, v in contributions.items()
        if k not in ["predicted_outcome", "base_value", "total_impact", "warning"]
    }

    col_factors, col_insights = st.columns([1.4, 1])

    with col_factors:
        st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
        st.markdown('<div class="mp-cmp-title">Top Prediction Factors</div>', unsafe_allow_html=True)
        if clean_contributions:
            ranked = sorted(clean_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            max_abs = max(abs(v) for _, v in ranked) or 1.0
            colors = [FIFA_BLUE, FIFA_BLUE_LIGHT, GOLD, EMERALD, WINE]
            rows = []
            for i, (feat, val) in enumerate(ranked):
                width = max(6, (abs(val) / max_abs) * 100)
                rows.append(
                    f'<div class="mp-factor-row">'
                    f'<span class="mp-factor-name">{humanize_feature(feat)}</span>'
                    f'<div class="mp-factor-track"><div class="mp-factor-fill" style="width:{width:.0f}%; background:{colors[i % len(colors)]};"></div></div>'
                    f'<span class="mp-factor-pct">{abs(val)*100:.1f}%</span>'
                    f'</div>'
                )
            st.markdown("".join(rows), unsafe_allow_html=True)
        else:
            st.info("SHAP values not populated in this prediction response.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_insights:
        st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
        st.markdown('<div class="mp-cmp-title">Match Insights</div>', unsafe_allow_html=True)

        home_stats = fetch_team_snapshot(home_team) or {}
        away_stats = fetch_team_snapshot(away_team) or {}
        insights = []

        home_form, away_form = home_stats.get("form"), away_stats.get("form")
        if home_form is not None and away_form is not None and abs(home_form - away_form) > 0.05:
            leader, trailer = (home_team, away_team) if home_form > away_form else (away_team, home_team)
            insights.append(f"{leader} enters with a stronger recent form index ({max(home_form, away_form):.2f} vs {min(home_form, away_form):.2f}).")

        home_conceded, away_conceded = home_stats.get("avg_goals_conceded"), away_stats.get("avg_goals_conceded")
        if home_conceded is not None and away_conceded is not None and abs(home_conceded - away_conceded) > 0.05:
            better, worse = (home_team, away_team) if home_conceded < away_conceded else (away_team, home_team)
            insights.append(f"{better} shows a tighter defensive record, conceding fewer goals per match than {worse}.")

        home_elo, away_elo = home_stats.get("elo"), away_stats.get("elo")
        if home_elo is not None and away_elo is not None:
            gap = abs(home_elo - away_elo)
            if gap < 60:
                insights.append(f"A {gap:.0f}-point ELO gap suggests a closely contested match with no dominant favorite.")
            else:
                favored = home_team if home_elo > away_elo else away_team
                insights.append(f"ELO ratings give {favored} a clear {gap:.0f}-point edge heading into this match.")

        if not insights:
            insights.append("Not enough statistical separation between these teams to generate a confident insight.")

        st.markdown("".join(f'<div class="mp-insight-card">{s}</div>' for s in insights), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_fixture_card(home_team: str, away_team: str, match_date, venue: str) -> None:
    """Render an official-style fixture card showing the selected matchup."""
    st.markdown(
        f"""
        <div class="match-card">
            <div class="match-teams">
                <div class="match-team">
                    <span class="flag">{flag_for(home_team)}</span>
                    <span class="name">{home_team}</span>
                </div>
                <div class="match-vs">VS</div>
                <div class="match-team">
                    <span class="flag">{flag_for(away_team)}</span>
                    <span class="name">{away_team}</span>
                </div>
            </div>
            <div style="text-align:center; margin-top:16px; color:{TEXT_MUTED}; font-size:0.85rem;">
                📅 {match_date} &nbsp;•&nbsp; 🏟️ {venue.title()} Venue
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show() -> None:
    """Render the Match Prediction interface."""
    section_title("Match Predictor")
    st.caption("Query the LightGBM classifier to analyze ELO differences, historical performance, and venue factors for any international fixture.")

    teams = load_qualified_teams()

    # Live fixture quick-select (backed by the worldcup2026 live fixtures API)
    fixtures = load_fixtures()

    def _format_fixture_label(f: dict) -> str:
        icon = {"live": "🔴", "finished": "✅", "scheduled": "🗓️"}.get(f["status"], "")
        label = f"{icon} {f['home_team']} vs {f['away_team']}"
        if f["status"] != "scheduled" and f["home_score"] is not None:
            label += f"  ({f['home_score']}-{f['away_score']})"
        return label

    if fixtures:
        fixture_options = ["— Manual team selection —"] + [_format_fixture_label(f) for f in fixtures]
        col_pick, col_refresh = st.columns([5, 1])
        with col_pick:
            fixture_choice = st.selectbox(
                "🌍 Real Fixture (auto-fills teams below)", options=fixture_options,
                label_visibility="collapsed",
            )
        with col_refresh:
            if st.button("🔄 Refresh", width="stretch"):
                load_fixtures.clear()
                st.rerun()

        selected_fixture = None
        if fixture_choice != fixture_options[0]:
            selected_fixture = fixtures[fixture_options.index(fixture_choice) - 1]
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        selected_fixture = None
        st.info("🔌 Live fixtures feed unavailable right now — select teams manually below.")

    # Form inputs
    with st.container(border=True):
        st.markdown("#### ⚙️ Match Parameters")

        default_home_idx = teams.index(selected_fixture["home_team"]) if selected_fixture and selected_fixture["home_team"] in teams else 0
        default_away_idx = teams.index(selected_fixture["away_team"]) if selected_fixture and selected_fixture["away_team"] in teams else min(10, len(teams) - 1)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            home_team = st.selectbox(
                "🏠 Home / Seed 1 Team", options=teams, index=default_home_idx,
                format_func=lambda t: f"{flag_for(t)}  {t}",
            )
        with col_t2:
            away_team = st.selectbox(
                "✈️ Away / Seed 2 Team", options=teams, index=default_away_idx,
                format_func=lambda t: f"{flag_for(t)}  {t}",
            )

        if selected_fixture and selected_fixture["status"] == "live":
            st.markdown(
                f"""
                <div style="background:{WINE}; color:#fff; border-radius:10px; padding:10px 16px; margin-bottom:14px; display:flex; align-items:center; justify-content:space-between;">
                    <span style="font-weight:700;">🔴 LIVE — {selected_fixture['home_team']} {selected_fixture['home_score']} - {selected_fixture['away_score']} {selected_fixture['away_team']}</span>
                    <span style="font-size:0.8rem; opacity:0.85;">Predicted odds below include a live-score adjustment</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            tournament = st.selectbox(
                "🏆 Match Tournament Type",
                ["FIFA World Cup", "Friendly", "UEFA Euro", "Copa América", "AFC Asian Cup", "CONCACAF Gold Cup", "WC Qualification"]
            )
        with col_p2:
            venue = st.selectbox("🏟️ Venue Configuration", ["neutral", "home", "away"])
        with col_p3:
            match_date = st.date_input("📅 Date of Match", value=date(2026, 6, 11))

        # Warning for identical teams
        if home_team == away_team:
            st.error("⚠️ Home and Away teams must be different. Please select distinct nations.")
            predict_disabled = True
        else:
            predict_disabled = False

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="mp-predict-btn-wrap">', unsafe_allow_html=True)
        predict_btn = st.button("⚽ Generate Prediction", width="stretch", disabled=predict_disabled, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    # Actions when Predict button clicked
    if predict_btn and not predict_disabled:
        live_adjustment_applied = False
        with st.spinner("Analyzing historical patterns & computing SHAP feature contributions..."):
            try:
                api = st.session_state.api_client
                result = api.predict(
                    home_team=home_team,
                    away_team=away_team,
                    tournament=tournament,
                    venue=venue,
                    match_date=str(match_date)
                )
            except Exception as e:
                st.warning(f"Backend connection failed ({e}). Simulating using sandbox model engine.")
                result = get_mock_prediction(home_team, away_team)

            # If this fixture is currently live, blend in the live-score heuristic adjustment
            if selected_fixture and selected_fixture["status"] == "live":
                try:
                    live_result = api.get_live_adjusted_prediction(home_team, away_team)
                    result["baseline_probabilities"] = live_result["baseline_probabilities"]
                    result["probabilities"] = live_result["live_probabilities"]
                    live_adjustment_applied = True
                except Exception as e:
                    st.info(f"Live-score adjustment unavailable ({e}); showing pre-match odds only.")

        st.markdown("<br>", unsafe_allow_html=True)

        if live_adjustment_applied:
            st.caption(
                "⚡ Odds below include a heuristic live-score adjustment based on the current "
                f"scoreline ({selected_fixture['home_score']}-{selected_fixture['away_score']}). "
                "This is a fixed-weight blend, not a re-run of the trained model."
            )

        rank_map = fetch_elo_rank_map()
        render_match_hero(home_team, away_team, rank_map)

        probs = result["probabilities"]
        confidence = result.get("confidence_score", 0.5)
        render_win_probability(home_team, away_team, probs, confidence)

        st.markdown("<br>", unsafe_allow_html=True)
        render_team_comparison(home_team, away_team)

        st.markdown("<br>", unsafe_allow_html=True)
        render_ai_explanation(home_team, away_team, result)

        # Data export buttons
        st.markdown("<br>", unsafe_allow_html=True)
        export_col1, export_col2, _ = st.columns([1, 1, 2])

        winner = result["predicted_winner"]
        json_download = json.dumps(result, indent=2)
        csv_df = pd.DataFrame([
            {
                "home_team": home_team,
                "away_team": away_team,
                "predicted_winner": winner,
                "prob_home": probs["home_win"],
                "prob_draw": probs["draw"],
                "prob_away": probs["away_win"],
                "confidence": confidence,
                "xg_home": result['expected_goals']['home_xg'],
                "xg_away": result['expected_goals']['away_xg']
            }
        ])
        csv_download = csv_df.to_csv(index=False)

        with export_col1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json_download,
                file_name=f"{home_team}_{away_team}_prediction.json",
                mime="application/json",
                width="stretch"
            )
        with export_col2:
            st.download_button(
                label="📊 Download CSV Summary",
                data=csv_download,
                file_name=f"{home_team}_{away_team}_prediction.csv",
                mime="text/csv",
                width="stretch"
            )


if __name__ == "__main__":
    show()
