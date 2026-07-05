"""FIFA World Cup 2026 Prediction Portal - Teams & Stats Comparison Centre.

FIFA-style team comparison experience: a stadium-toned comparison hero with
ELO rank/rating badges, key-stat comparison cards, a statistical radar,
form index, offense-vs-defense split bars, real tournament-strength odds
(reusing the Tournament Simulator's own output), AI-generated insights, and
a final "Who Has The Edge" verdict card.

Head-to-Head history is intentionally omitted -- the backend has no
per-match query between two specific teams, only aggregate stats, so that
section is not fabricated here (same decision made on the Match Prediction
page).
"""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from frontend.services.theme import section_title, flag_for, EMERALD, FIFA_BLUE_LIGHT, GOLD, WINE, FIFA_BLUE
from frontend.services.team_data import (
    fetch_team_snapshot, fetch_elo_rank_map, fetch_tournament_stage_probs, load_qualified_teams,
)


def render_comparison_hero(team_a: str, team_b: str, rank_map: dict[str, int]) -> None:
    """Render the stadium-toned comparison hero: large flags, names, ELO rank/rating badges."""
    stats_a = fetch_team_snapshot(team_a)
    stats_b = fetch_team_snapshot(team_b)
    elo_a = stats_a.get("elo") if stats_a else None
    elo_b = stats_b.get("elo") if stats_b else None
    rank_a = rank_map.get(team_a)
    rank_b = rank_map.get(team_b)

    def _team_stats_html(rank: int | None, elo: float | None) -> str:
        badge = f'<span class="mp-stat-badge">ELO RANK #{rank}</span>' if rank else ""
        elo_line = f'<span class="mp-stat-line">ELO Rating <b>{elo:.0f}</b></span>' if elo is not None else ""
        return f'{badge}{elo_line}'

    st.markdown(
        f'<div class="mp-hero">'
        f'<div class="mp-kicker">FIFA World Cup 2026 · Team Comparison Centre</div>'
        f'<div class="mp-matchup">'
        f'<div class="mp-team">'
        f'<span class="flag">{flag_for(team_a)}</span>'
        f'<div class="name">{team_a}</div>'
        f'<div class="mp-team-stats" style="--qc-color:{EMERALD};">{_team_stats_html(rank_a, elo_a)}</div>'
        f'</div>'
        f'<div class="mp-vs">VS</div>'
        f'<div class="mp-team">'
        f'<span class="flag">{flag_for(team_b)}</span>'
        f'<div class="name">{team_b}</div>'
        f'<div class="mp-team-stats" style="--qc-color:{FIFA_BLUE_LIGHT};">{_team_stats_html(rank_b, elo_b)}</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    return stats_a, stats_b


KEY_STAT_METRICS = [
    ("ELO Rating", "elo", "{:.0f}", False),
    ("Win Rate", "overall_win_pct", "{:.1%}", False),
    ("Goals / Match", "avg_goals_scored", "{:.2f}", False),
    ("Conceded / Match", "avg_goals_conceded", "{:.2f}", True),
    ("Draw %", "draw_pct", "{:.1%}", None),
    ("Recent Form Index", "form", "{:.2f}", False),
    ("Home Win % (Hosting)", "home_home_win_pct", "{:.1%}", False),
    ("Matches Played", "games_played", "{:.0f}", None),
]


def render_key_stats(team_a: str, team_b: str, stats_a: dict, stats_b: dict) -> None:
    """Render the 8-card key-stats comparison grid, leader's top edge highlighted."""
    section_title("Key Stats Comparison")
    cards = []
    for label, key, fmt, invert in KEY_STAT_METRICS:
        val_a = stats_a.get(key, 0.0)
        val_b = stats_b.get(key, 0.0)
        if invert is None:
            leader_cls, cls_a, cls_b = "", "", ""
        else:
            a_wins = (val_a < val_b) if invert else (val_a > val_b)
            b_wins = (val_a > val_b) if invert else (val_a < val_b)
            leader_cls = " leader-home" if a_wins else (" leader-away" if b_wins else "")
            cls_a = " leader" if a_wins else ""
            cls_b = " leader-blue" if b_wins else ""
        cards.append(
            f'<div class="ks-card{leader_cls}">'
            f'<div class="ks-metric-label">{label}</div>'
            f'<div class="ks-values">'
            f'<span class="ks-val{cls_a}">{fmt.format(val_a)}</span>'
            f'<span class="ks-sep">–</span>'
            f'<span class="ks-val{cls_b}">{fmt.format(val_b)}</span>'
            f'</div>'
            f'</div>'
        )
    st.markdown(f'<div class="ks-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


RADAR_AXES = [
    ("Attack", "avg_goals_scored", False),
    ("Form", "form", False),
    ("ELO", "elo", False),
    ("Win %", "overall_win_pct", False),
    ("Defense", "avg_goals_conceded", True),
]


def _normalize_radar_value(value: float, axis_key: str, invert: bool, all_values: dict[str, list[float]]) -> float:
    """Normalize a stat onto a 0-1 radar scale relative to both teams' values on this axis."""
    values = all_values.get(axis_key, [value])
    lo, hi = min(values), max(values)
    if hi == lo:
        return 0.7
    norm = (value - lo) / (hi - lo)
    return (1 - norm) if invert else norm


def render_radar(team_a: str, team_b: str, stats_a: dict, stats_b: dict) -> None:
    """Render the team-strength radar: 5 real axes, normalized relative to both teams."""
    section_title("Team Strength Radar")
    st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
    st.markdown('<div class="mp-cmp-title">Statistical Radar</div>', unsafe_allow_html=True)

    all_values = {key: [stats_a.get(key, 0.0), stats_b.get(key, 0.0)] for _, key, _ in RADAR_AXES}
    norm_a = [_normalize_radar_value(stats_a.get(key, 0.0), key, invert, all_values) for _, key, invert in RADAR_AXES]
    norm_b = [_normalize_radar_value(stats_b.get(key, 0.0), key, invert, all_values) for _, key, invert in RADAR_AXES]

    categories = [label for label, _, _ in RADAR_AXES] + [RADAR_AXES[0][0]]
    r_a = norm_a + [norm_a[0]]
    r_b = norm_b + [norm_b[0]]

    fig = px.line_polar(r=r_a, theta=categories, line_close=True)
    fig.data[0].update(name=team_a, line_color=EMERALD, fill="toself", fillcolor="rgba(31,174,110,0.18)")
    fig.add_trace(go.Scatterpolar(
        r=r_b, theta=categories, name=team_b,
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
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_form_index(team_a: str, team_b: str, stats_a: dict, stats_b: dict) -> None:
    """Render the Recent Form section using the real aggregate form score per team.

    Labeled "Form Index," not "Last 10 Matches" -- the backend has no literal
    match-by-match result sequence, only a single 0-1 form score.
    """
    section_title("Recent Form")
    form_a = stats_a.get("form", 0.0)
    form_b = stats_b.get("form", 0.0)

    def _form_row(team: str, form: float, color: str) -> str:
        pct = max(0.0, min(1.0, form)) * 100
        return (
            f'<div class="form-row">'
            f'<div class="form-team-label">{flag_for(team)} {team}</div>'
            f'<div class="form-track"><div class="form-fill" style="width:{pct:.0f}%; background:{color};"></div></div>'
            f'<div class="form-rating" style="color:{color};">{form:.2f}</div>'
            f'</div>'
        )

    st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
    st.markdown('<div class="mp-cmp-title">Form Index</div>', unsafe_allow_html=True)
    st.markdown(_form_row(team_a, form_a, EMERALD) + _form_row(team_b, form_b, FIFA_BLUE_LIGHT), unsafe_allow_html=True)
    st.caption("Aggregate rolling form score (0–1) from the team database — not a literal match-by-match log.")
    st.markdown('</div>', unsafe_allow_html=True)


OFFENSE_DEFENSE_METRICS = [
    ("Avg. Goals Scored", "avg_goals_scored", "{:.2f}", False),
    ("Avg. Goals Conceded (lower better)", "avg_goals_conceded", "{:.2f}", True),
    ("Avg. Goal Difference", "avg_goal_diff", "{:+.2f}", False),
]


def render_offense_defense(stats_a: dict, stats_b: dict) -> None:
    """Render mirrored split-bars for offense/defense metrics -- no legend required to read."""
    section_title("Offensive vs. Defensive Analysis")
    st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)

    rows = []
    for label, key, fmt, invert in OFFENSE_DEFENSE_METRICS:
        val_a = stats_a.get(key, 0.0)
        val_b = stats_b.get(key, 0.0)
        max_val = max(abs(val_a), abs(val_b)) or 1.0
        width_a = max(4, (abs(val_a) / max_val) * 100)
        width_b = max(4, (abs(val_b) / max_val) * 100)
        rows.append(
            f'<div class="mp-bar-row">'
            f'<div class="mp-bar-metric">{label}</div>'
            f'<span class="mp-bar-val home">{fmt.format(val_a)}</span>'
            f'<div class="mp-track-pair">'
            f'<div class="mp-track-home"><div class="mp-fill-home" style="width:{width_a:.0f}%;"></div></div>'
            f'<div class="mp-track-away"><div class="mp-fill-away" style="width:{width_b:.0f}%;"></div></div>'
            f'</div>'
            f'<span class="mp-bar-val away">{fmt.format(val_b)}</span>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_tournament_strength(team_a: str, team_b: str) -> None:
    """Render Tournament Strength using the real Tournament Simulator's stage_probabilities."""
    section_title("Tournament Strength")
    stage_probs = fetch_tournament_stage_probs(run_count=500)

    st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
    if not stage_probs or team_a not in stage_probs or team_b not in stage_probs:
        st.info("🔌 Tournament simulation data unavailable right now — run a full simulation on the Tournament Simulator page.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    sp_a, sp_b = stage_probs[team_a], stage_probs[team_b]
    rows_def = [
        ("Champion Probability", "champion"),
        ("Semi-Final Probability", "semifinals"),
        ("Quarter-Final Probability", "quarterfinals"),
        ("Group Stage Exit Risk", None),  # computed as 1 - group_stage
    ]
    rows_html = []
    for label, key in rows_def:
        if key is None:
            val_a = 1 - sp_a.get("group_stage", 1.0)
            val_b = 1 - sp_b.get("group_stage", 1.0)
            color = "#D4536E"
        else:
            val_a = sp_a.get(key, 0.0)
            val_b = sp_b.get(key, 0.0)
            color = EMERALD
        rows_html.append(
            f'<div class="ts-row">'
            f'<div class="ts-label">{label}</div>'
            f'<div class="ts-bar-wrap"><div class="ts-bar-track"><div class="ts-bar-fill" style="width:{val_a*100:.0f}%; background:{color if key else "#D4536E"};"></div></div>'
            f'<span class="ts-bar-pct" style="color:{color if key else "#D4536E"};">{val_a*100:.1f}%</span></div>'
            f'<div class="ts-bar-wrap"><div class="ts-bar-track"><div class="ts-bar-fill" style="width:{val_b*100:.0f}%; background:{FIFA_BLUE_LIGHT if key else "#D4536E"};"></div></div>'
            f'<span class="ts-bar-pct" style="color:{FIFA_BLUE_LIGHT if key else "#D4536E"};">{val_b*100:.1f}%</span></div>'
            f'</div>'
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)
    st.caption("Sourced from a 500-run Monte Carlo tournament simulation, filtered to these two teams.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_ai_insights_and_edge(team_a: str, team_b: str, stats_a: dict, stats_b: dict) -> None:
    """Render generated Match Insights plus the final Who Has The Edge verdict card."""
    section_title("AI Insights")

    insights = []
    form_a, form_b = stats_a.get("form"), stats_b.get("form")
    if form_a is not None and form_b is not None and abs(form_a - form_b) > 0.05:
        leader = team_a if form_a > form_b else team_b
        insights.append(f"{leader} currently possesses a stronger recent form index, suggesting better momentum heading into this matchup.")

    goals_a, goals_b = stats_a.get("avg_goals_scored"), stats_b.get("avg_goals_scored")
    if goals_a is not None and goals_b is not None and abs(goals_a - goals_b) > 0.05:
        leader = team_a if goals_a > goals_b else team_b
        insights.append(f"{leader} demonstrates stronger attacking metrics, averaging {max(goals_a, goals_b):.2f} goals per match versus {min(goals_a, goals_b):.2f}.")

    conceded_a, conceded_b = stats_a.get("avg_goals_conceded"), stats_b.get("avg_goals_conceded")
    if conceded_a is not None and conceded_b is not None and abs(conceded_a - conceded_b) > 0.05:
        leader = team_a if conceded_a < conceded_b else team_b
        insights.append(f"{leader} demonstrates superior defensive consistency, conceding fewer goals per match on average.")

    elo_a, elo_b = stats_a.get("elo"), stats_b.get("elo")
    winning_factors = []
    edge_team, edge_stats, other_stats = team_a, stats_a, stats_b
    if elo_a is not None and elo_b is not None:
        gap = abs(elo_a - elo_b)
        if gap < 60:
            insights.append(f"ELO ratings indicate a closely balanced matchup, with only a {gap:.0f}-point gap separating the two sides.")
        else:
            favored = team_a if elo_a > elo_b else team_b
            insights.append(f"ELO ratings give {favored} a clear {gap:.0f}-point edge heading into this comparison.")
        edge_team = team_a if elo_a > elo_b else team_b
        edge_stats, other_stats = (stats_a, stats_b) if edge_team == team_a else (stats_b, stats_a)
        winning_factors.append(f"Higher ELO Rating ({edge_stats.get('elo', 0):.0f} vs {other_stats.get('elo', 0):.0f})")

    if not insights:
        insights.append("Not enough statistical separation between these teams to generate a confident insight.")

    st.markdown('<div class="mp-cmp-card">', unsafe_allow_html=True)
    st.markdown('<div class="mp-cmp-title">Match Insights</div>', unsafe_allow_html=True)
    st.markdown("".join(f'<div class="mp-insight-card">{s}</div>' for s in insights), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Build the remaining winning factors for the edge card
    edge_form, other_form = (stats_a.get("form"), stats_b.get("form")) if edge_team == team_a else (stats_b.get("form"), stats_a.get("form"))
    if edge_form is not None and other_form is not None and edge_form > other_form:
        winning_factors.append(f"Better Recent Form Index ({edge_form:.2f} vs {other_form:.2f})")

    edge_goals, other_goals = (stats_a.get("avg_goals_scored"), stats_b.get("avg_goals_scored")) if edge_team == team_a else (stats_b.get("avg_goals_scored"), stats_a.get("avg_goals_scored"))
    if edge_goals is not None and other_goals is not None and edge_goals > other_goals:
        winning_factors.append(f"Higher Goal Production ({edge_goals:.2f} vs {other_goals:.2f} per match)")

    # Confidence reuses the real prediction model's confidence for this exact matchup
    confidence = None
    try:
        pred = st.session_state.api_client.predict(home_team=team_a, away_team=team_b)
        confidence = pred.get("confidence_score")
    except Exception:
        pass

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Who Has The Edge?")
    factors_html = "".join(f'<div class="edge-factor"><span class="check">✓</span> {f}</div>' for f in winning_factors) or '<div class="edge-factor">No single team shows a clear statistical edge.</div>'
    conf_html = ""
    if confidence is not None:
        conf_pct = max(0.0, min(1.0, confidence)) * 100
        conf_html = (
            f'<div class="edge-confidence">'
            f'<span class="edge-conf-label">Prediction Confidence</span>'
            f'<div class="edge-conf-track"><div class="edge-conf-fill" style="width:{conf_pct:.0f}%;"></div></div>'
            f'<span class="edge-conf-value">{conf_pct:.0f}%</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="edge-card">'
        f'<div class="edge-label">Overall Advantage</div>'
        f'<div class="edge-team"><span class="flag">{flag_for(edge_team)}</span><span class="name">{edge_team}</span></div>'
        f'<div class="edge-factors">{factors_html}</div>'
        f'{conf_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def show() -> None:
    """Render the Teams & Stats Comparison Centre."""
    section_title("Teams & Stats")
    st.caption("Compare any two FIFA World Cup 2026 teams across ELO, form, attacking/defensive metrics, and real tournament projections.")

    teams = load_qualified_teams()

    with st.container(border=True):
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            team_a = st.selectbox(
                "⚔️ Select First Team", options=teams, index=0,
                format_func=lambda t: f"{flag_for(t)}  {t}",
            )
        with col_sel2:
            team_b = st.selectbox(
                "🛡️ Select Second Team", options=teams, index=min(1, len(teams) - 1),
                format_func=lambda t: f"{flag_for(t)}  {t}",
            )

    if team_a == team_b:
        st.warning("⚠️ For a meaningful comparison, please choose two distinct national teams.")
        return

    rank_map = fetch_elo_rank_map()
    stats_a, stats_b = render_comparison_hero(team_a, team_b, rank_map)
    stats_a = stats_a or {}
    stats_b = stats_b or {}

    st.markdown("<br>", unsafe_allow_html=True)
    render_key_stats(team_a, team_b, stats_a, stats_b)

    st.markdown("<br>", unsafe_allow_html=True)
    render_radar(team_a, team_b, stats_a, stats_b)

    st.markdown("<br>", unsafe_allow_html=True)
    render_form_index(team_a, team_b, stats_a, stats_b)

    st.markdown("<br>", unsafe_allow_html=True)
    render_offense_defense(stats_a, stats_b)

    st.markdown("<br>", unsafe_allow_html=True)
    render_tournament_strength(team_a, team_b)

    st.markdown("<br>", unsafe_allow_html=True)
    render_ai_insights_and_edge(team_a, team_b, stats_a, stats_b)


if __name__ == "__main__":
    show()
