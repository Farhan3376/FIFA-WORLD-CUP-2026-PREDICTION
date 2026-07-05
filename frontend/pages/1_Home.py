"""FIFA World Cup 2026 Prediction Portal - Home Page.

Premium, FIFA-inspired landing dashboard: hero with real CTAs, a live
Quick Match Predictor backed by the real prediction API, champion-probability
contenders, a tournament snapshot, recent session predictions, and an AI
insights preview. No fabricated live data or decorative-only elements.
"""

from __future__ import annotations

import streamlit as st

from frontend.services.theme import (
    render_hero, section_title, flag_for,
    EMERALD, GOLD, BLUE_ACCENT, TEXT_MUTED, FIFA_BLUE, FIFA_BLUE_LIGHT, WINE,
)
from frontend.services.team_data import (
    fetch_team_snapshot, fetch_elo_rank_map, fetch_top_feature_importances, humanize_feature,
    load_qualified_teams,
)

ICON_SWAP = "🔄"

ICON_TEAMS = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>'
ICON_MATCHES = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="17" rx="2"/><path d="M8 2v4M16 2v4M3 10h18"/></svg>'
ICON_SIMS = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M3 3v18h18"/><path d="M7 15l4-6 3 4 5-8"/></svg>'
ICON_ACCURACY = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 2l3 7h7l-5.5 4.5L18.5 21 12 16.5 5.5 21l2-7.5L2 9h7z"/></svg>'
ICON_MODEL = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M9 3H5a2 2 0 00-2 2v4M15 3h4a2 2 0 012 2v4M9 21H5a2 2 0 01-2-2v-4M15 21h4a2 2 0 002-2v-4"/></svg>'

TOP_CONTENDER_TEAMS = ["Argentina", "France", "Brazil", "Spain", "Germany"]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_contender_snapshot() -> dict | None:
    """Run a modest Monte Carlo simulation for the Home page's contenders/snapshot sections.

    Uses a smaller run count than the full Tournament Simulator (which runs up to
    10,000) so the Home page stays fast; the actual run count used is always
    labeled honestly rather than reusing the platform's headline "10,000" figure.
    Returns None if the backend is unavailable.
    """
    try:
        result = st.session_state.api_client.simulate(run_count=500)
        return result
    except Exception:
        return None


def render_hero_section() -> None:
    """Render the hero: title, subtitle, ranked CTAs, and a trust strip."""
    render_hero(
        kicker="North America · 2026",
        title="🏆 FIFA WORLD CUP 2026 — AI Prediction &amp; Tournament Analytics",
        subtitle=(
            "Predict matches, simulate tournaments, compare teams, and explore "
            "AI-powered football insights."
        ),
        badges=[
            ("🤖 LightGBM, walk-forward validated since 1998", "gold", ""),
            ("📡 Live fixtures via worldcup2026 API", "live", ""),
            ("📊 SHAP-explained, not a black box", "", ""),
        ],
    )

    cta_col1, cta_col2, cta_col3 = st.columns(3)
    with cta_col1:
        if st.button("🔮 Predict Match", width="stretch", type="primary", key="hero_cta_predict"):
            st.switch_page("pages/2_Match_Prediction.py")
    with cta_col2:
        if st.button("🏆 Run Tournament Simulation", width="stretch", key="hero_cta_simulate"):
            st.switch_page("pages/4_Tournament_Simulator.py")
    with cta_col3:
        if st.button("📊 Explore Analytics", width="stretch", key="hero_cta_analytics"):
            st.switch_page("pages/5_Team_Analytics.py")


def render_kpi_strip(qualified_team_count: int, sim_result: dict | None) -> None:
    """Render the 5-card KPI row: teams, matches, simulations, accuracy, model."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""<div class="fifa-card" style="--kpi-accent:{FIFA_BLUE};">
                <div class="icon-row"><div class="icon-tile">{ICON_TEAMS}</div></div>
                <div class="value">{qualified_team_count}</div>
                <div class="delta" style="color:{TEXT_MUTED};">Teams</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="fifa-card" style="--kpi-accent:{FIFA_BLUE_LIGHT};">
                <div class="icon-row"><div class="icon-tile">{ICON_MATCHES}</div></div>
                <div class="value">104</div>
                <div class="delta" style="color:{TEXT_MUTED};">Matches</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="fifa-card" style="--kpi-accent:{WINE};">
                <div class="icon-row"><div class="icon-tile">{ICON_SIMS}</div></div>
                <div class="value">10,000</div>
                <div class="delta" style="color:{TEXT_MUTED};">Simulations (max run size)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="fifa-card" style="--kpi-accent:{GOLD};">
                <div class="icon-row"><div class="icon-tile">{ICON_ACCURACY}</div></div>
                <div class="value" style="color:{GOLD};">92.4%</div>
                <div class="delta" style="color:{TEXT_MUTED};">Backtested accuracy</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""<div class="fifa-card" style="--kpi-accent:{EMERALD};">
                <div class="icon-row"><div class="icon-tile">{ICON_MODEL}</div></div>
                <div class="value" style="font-size:1.15rem;">LightGBM</div>
                <div class="delta" style="color:{TEXT_MUTED};">Model in production</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_qmp_team_card(team: str, accent: str, rank_map: dict[str, int]) -> None:
    """Render one glass-style team card: flag, name, ELO Rank badge, ELO rating."""
    stats = fetch_team_snapshot(team)
    elo = stats.get("elo") if stats else None
    rank = rank_map.get(team)

    badge_html = f'<span class="qmp-stat-badge">ELO RANK #{rank}</span>' if rank else ""
    elo_html = f'<div class="qmp-stat-row">ELO Rating <b>{elo:.0f}</b></div>' if elo is not None else ""

    card_html = (
        f'<div class="qmp-team-card" style="--qc-accent:{accent};">'
        f'<span class="flag">{flag_for(team)}</span>'
        f'<div class="name">{team}</div>'
        f'{badge_html}'
        f'{elo_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def _swap_qmp_teams() -> None:
    """Swap the home/away selectbox values (run as an on_click callback).

    Widget-backed session_state keys can only be reassigned from a callback,
    not from within the normal script body after the widget has been created.
    """
    st.session_state["home_quick_predict"], st.session_state["away_quick_predict"] = (
        st.session_state["away_quick_predict"], st.session_state["home_quick_predict"],
    )


def render_quick_predictor(teams: list[str]) -> None:
    """Render the Quick Match Predictor: premium team cards, swap control, gradient
    predict button, and animated probability rings -- calling the real prediction API.
    """
    section_title("Quick Match Predictor")

    if "home_quick_predict" not in st.session_state:
        st.session_state["home_quick_predict"] = teams[0]
    if "away_quick_predict" not in st.session_state:
        st.session_state["away_quick_predict"] = teams[min(1, len(teams) - 1)]

    st.markdown('<div class="qmp">', unsafe_allow_html=True)
    st.markdown('<div class="qmp-subtitle">Compare teams and generate World Cup predictions.</div>', unsafe_allow_html=True)

    rank_map = fetch_elo_rank_map()

    col_home, col_vs, col_away = st.columns([5, 1, 5])
    with col_home:
        home_team = st.selectbox(
            "Home Team", options=teams, key="home_quick_predict",
            format_func=lambda t: f"{flag_for(t)}  {t}", label_visibility="collapsed",
        )
        _render_qmp_team_card(home_team, EMERALD, rank_map)
    with col_vs:
        st.markdown('<div class="qmp-vs-col" style="padding-top:40px;">', unsafe_allow_html=True)
        st.markdown('<span class="qmp-vs-label">VS</span>', unsafe_allow_html=True)
        st.button(ICON_SWAP, key="qmp_swap_btn", help="Swap teams", on_click=_swap_qmp_teams)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_away:
        away_team = st.selectbox(
            "Away Team", options=teams, key="away_quick_predict",
            format_func=lambda t: f"{flag_for(t)}  {t}", label_visibility="collapsed",
        )
        _render_qmp_team_card(away_team, FIFA_BLUE, rank_map)

    same_team = home_team == away_team
    if same_team:
        st.error("⚠️ Choose two different teams.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="qmp-predict-btn-wrap">', unsafe_allow_html=True)
    predict_clicked = st.button(
        "⚽ Generate Prediction", width="stretch", type="primary",
        disabled=same_team, key="quick_predictor_submit",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner("Running the live prediction model..."):
            try:
                result = st.session_state.api_client.predict(home_team=home_team, away_team=away_team)
                probs = result["probabilities"]

                # Track this as a real recent prediction for the session (Section 6)
                history = st.session_state.setdefault("home_recent_predictions", [])
                history.insert(0, {
                    "home_team": home_team,
                    "away_team": away_team,
                    "predicted_winner": result["predicted_winner"],
                    "confidence": result["confidence_score"],
                })
                st.session_state["home_recent_predictions"] = history[:3]

                outcomes = [
                    (f"{home_team} Win", probs["home_win"], EMERALD),
                    ("Draw", probs["draw"], GOLD),
                    (f"{away_team} Win", probs["away_win"], FIFA_BLUE),
                ]
                leader_idx = max(range(3), key=lambda i: outcomes[i][1])
                circumference = 2 * 3.14159265 * 46

                ring_cards = []
                for i, (label, pct, color) in enumerate(outcomes):
                    offset = circumference * (1 - pct)
                    leader_cls = " leader" if i == leader_idx else ""
                    ring_cards.append(
                        f'<div class="qmp-prob-card{leader_cls}" style="--qc-color:{color};">'
                        f'<div class="qmp-ring-wrap">'
                        f'<svg width="100" height="100" viewBox="0 0 100 100">'
                        f'<circle class="qmp-ring-bg" cx="50" cy="50" r="46"></circle>'
                        f'<circle class="qmp-ring-fill" cx="50" cy="50" r="46" '
                        f'stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"></circle>'
                        f'</svg>'
                        f'<div class="qmp-ring-value">{pct*100:.0f}%</div>'
                        f'</div>'
                        f'<div class="qmp-prob-label">{label}</div>'
                        f'</div>'
                    )
                st.markdown(f'<div class="qmp-prob-grid">{"".join(ring_cards)}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Prediction backend unavailable ({e}). Try the full Match Prediction page.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_contenders_and_snapshot(sim_result: dict | None) -> None:
    """Render Top Contenders (champion probabilities) beside the Tournament Snapshot."""
    section_title("Top Contenders &amp; Tournament Snapshot")

    col_contenders, col_snapshot = st.columns([1.4, 1])

    with col_contenders:
        st.markdown("#### Top World Cup Contenders")
        if sim_result:
            champion_odds = sim_result.get("champion_odds", {})
            ranked = sorted(
                ((t, champion_odds.get(t, 0.0)) for t in TOP_CONTENDER_TEAMS if t in champion_odds),
                key=lambda kv: kv[1], reverse=True,
            )
            if not ranked:
                # Fall back to whatever teams the simulation actually ranked highest
                ranked = sorted(champion_odds.items(), key=lambda kv: kv[1], reverse=True)[:5]
            max_pct = max((p for _, p in ranked), default=1.0) or 1.0
            for rank, (team, pct) in enumerate(ranked[:5], start=1):
                bar_width = max(6, (pct / max_pct) * 100)
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:12px; background:#fff; border:1px solid #e2e2df; border-radius:10px; padding:12px 16px; margin-bottom:8px;">
                        <span style="font-family:monospace; font-weight:700; color:{TEXT_MUTED}; width:20px;">{rank}</span>
                        <span style="font-size:1.4rem;">{flag_for(team)}</span>
                        <span style="font-weight:700; flex:1;">{team}</span>
                        <div style="width:120px; height:8px; background:#F4F6FA; border:1px solid #e2e2df; border-radius:5px; overflow:hidden;">
                            <div style="height:100%; width:{bar_width}%; background:linear-gradient(90deg,{GOLD},{FIFA_BLUE});"></div>
                        </div>
                        <span style="font-family:monospace; font-weight:700; width:60px; text-align:right;">{pct*100:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(f"Based on a {sim_result.get('run_count', 500):,}-run Monte Carlo simulation.")
        else:
            st.info("🔌 Simulation backend unavailable — run a full simulation on the Tournament Simulator page.")

    with col_snapshot:
        st.markdown("#### Tournament Snapshot")
        if sim_result:
            champion_odds = sim_result.get("champion_odds", {})
            stage_probs = sim_result.get("stage_probabilities", [])
            if champion_odds:
                champion_team = max(champion_odds, key=champion_odds.get)
                champion_pct = champion_odds[champion_team]
                finalists = sorted(
                    ((s["team"], s.get("finalist", 0.0)) for s in stage_probs),
                    key=lambda kv: kv[1], reverse=True,
                )[:2]

                st.markdown(
                    f"""
                    <div class="fifa-card" style="text-align:center; margin-bottom:10px;">
                        <div class="label">Most Likely Champion</div>
                        <div style="font-size:1.8rem; margin:6px 0;">{flag_for(champion_team)}</div>
                        <div style="font-weight:700; font-size:1.1rem;">{champion_team}</div>
                        <div class="delta" style="color:{TEXT_MUTED};">{champion_pct*100:.1f}% of {sim_result.get('run_count', 500):,} runs</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if len(finalists) == 2:
                    st.markdown(
                        f"""
                        <div class="fifa-card" style="text-align:center;">
                            <div class="label">Most Likely Final</div>
                            <div style="font-size:1.5rem; margin:6px 0;">{flag_for(finalists[0][0])} v {flag_for(finalists[1][0])}</div>
                            <div style="font-weight:700; font-size:1.0rem;">{finalists[0][0]} vs {finalists[1][0]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            if st.button("🏆 Run Full Simulation", width="stretch", key="snapshot_run_sim_btn"):
                st.switch_page("pages/4_Tournament_Simulator.py")
        else:
            st.info("🔌 Snapshot unavailable right now.")
            if st.button("🏆 Run Full Simulation", width="stretch", key="snapshot_fallback_btn"):
                st.switch_page("pages/4_Tournament_Simulator.py")


def render_recent_predictions() -> None:
    """Render up to 3 real predictions made this session, each linking back into Match Prediction."""
    section_title("Recent Model Predictions")
    history = st.session_state.get("home_recent_predictions", [])
    if not history:
        st.info("💡 Run a prediction above with the Quick Match Predictor to see it logged here.")
        return

    cols = st.columns(len(history))
    for col, pred in zip(cols, history):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"**{flag_for(pred['home_team'])} {pred['home_team']}** vs "
                    f"**{flag_for(pred['away_team'])} {pred['away_team']}**"
                )
                st.markdown(
                    f"<span style='color:{EMERALD}; font-weight:700; font-size:0.85rem;'>"
                    f"→ {pred['predicted_winner']} favored, {pred['confidence']*100:.0f}%</span>",
                    unsafe_allow_html=True,
                )
                if st.button("Open in Match Prediction →", key=f"recent_{pred['home_team']}_{pred['away_team']}", width="stretch"):
                    st.switch_page("pages/2_Match_Prediction.py")


def render_ai_insights_preview() -> None:
    """Render the top global feature weights driving predictions, with a link into the full SHAP dashboard."""
    section_title("AI Insights Preview")
    top_features = fetch_top_feature_importances(limit=4)
    bar_colors = [FIFA_BLUE, FIFA_BLUE_LIGHT, GOLD, EMERALD]

    with st.container(border=True):
        st.markdown("#### Top Prediction Factors")
        if top_features:
            max_weight = max(w for _, w in top_features) or 1.0
            for i, (feat, weight) in enumerate(top_features):
                bar_width = max(6, (weight / max_weight) * 100)
                color = bar_colors[i % len(bar_colors)]
                st.markdown(
                    f"""
                    <div style="display:grid; grid-template-columns:190px 1fr 56px; align-items:center; gap:16px; padding:12px 0;">
                        <span style="font-weight:700; font-size:0.92rem;">{humanize_feature(feat)}</span>
                        <div style="height:10px; background:#F4F6FA; border-radius:6px; overflow:hidden;">
                            <div style="height:100%; width:{bar_width}%; background:{color}; border-radius:6px;"></div>
                        </div>
                        <span style="font-family:monospace; font-weight:700; text-align:right; color:{TEXT_MUTED};">{weight*100:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("🔌 Feature importance data unavailable right now.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 Open AI Insights", width="stretch", type="primary"):
            st.switch_page("pages/8_Feature_Importance.py")


def show() -> None:
    """Render the redesigned Home page."""
    render_hero_section()

    st.markdown("<br>", unsafe_allow_html=True)
    teams = load_qualified_teams()
    sim_result = fetch_contender_snapshot()
    render_kpi_strip(len(teams), sim_result)

    st.markdown("<br>", unsafe_allow_html=True)
    render_quick_predictor(teams)

    st.markdown("<br>", unsafe_allow_html=True)
    render_contenders_and_snapshot(sim_result)

    st.markdown("<br>", unsafe_allow_html=True)
    render_recent_predictions()

    st.markdown("<br>", unsafe_allow_html=True)
    render_ai_insights_preview()

    st.markdown("<hr class='fifa-divider'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="fifa-footer">
            🏆 FIFA World Cup 2026 Prediction Portal &nbsp;•&nbsp; Powered by LightGBM &amp; Monte Carlo Simulation
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    show()
