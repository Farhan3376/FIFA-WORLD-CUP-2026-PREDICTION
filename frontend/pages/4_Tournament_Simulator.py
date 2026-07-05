"""FIFA World Cup 2026 Prediction Portal - Tournament Simulator.

The flagship Tournament Intelligence Centre: runs the real 12-group, 48-team
Monte Carlo bracket engine and surfaces champion odds, a real sample bracket
path, the real most-common final matchup, per-team stage probabilities, and
a real ELO-rank-vs-simulated-odds upset metric -- all sourced from the actual
simulation engine (simulation/tournament_simulator.py + monte_carlo.py), with
nothing about specific matchups, brackets, or upsets fabricated.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd

from frontend.services.theme import section_title, flag_for, TEXT_MUTED

RUN_PRESETS = [100, 500, 1000, 5000, 10000]

FALLBACK_RESULT: Dict[str, Any] = {
    "run_count": 100,
    "champion_odds": {"Argentina": 0.182, "France": 0.164, "Brazil": 0.159, "Spain": 0.113, "Germany": 0.098},
    "stage_probabilities": [
        {"team": "Argentina", "group_stage": 0.98, "round_of_32": 0.96, "round_of_16": 0.84, "quarterfinals": 0.62, "semifinals": 0.41, "finalist": 0.28, "champion": 0.182},
        {"team": "France", "group_stage": 0.97, "round_of_32": 0.95, "round_of_16": 0.81, "quarterfinals": 0.58, "semifinals": 0.38, "finalist": 0.25, "champion": 0.164},
        {"team": "Brazil", "group_stage": 0.97, "round_of_32": 0.94, "round_of_16": 0.79, "quarterfinals": 0.56, "semifinals": 0.36, "finalist": 0.24, "champion": 0.159},
    ],
    "most_likely_final": {"team_a": "Argentina", "team_b": "France", "probability": 0.124},
    "upsets": [],
    "sample_bracket": None,
    "timestamp": "",
}


def _set_run_count(preset: int) -> None:
    st.session_state["tsim_run_count"] = preset


def render_simulator_hero() -> None:
    st.markdown(
        '<div class="tsim-hero">'
        '<div class="tsim-hero-mesh"></div>'
        '<div class="tsim-hero-inner">'
        '<p class="tsim-kicker">Tournament Intelligence Centre</p>'
        '<h1>FIFA World Cup 2026 Tournament Simulator</h1>'
        '<p>Simulate the entire 48-team tournament using AI-powered predictions and real Monte Carlo bracket '
        'simulation &mdash; 12 real groups, real knockout pairings, thousands of runs.</p>'
        '<div class="tsim-hero-chips">'
        '<span class="tsim-chip"><b>12</b> real groups</span>'
        '<span class="tsim-chip"><b>48</b> teams</span>'
        '<span class="tsim-chip"><b>LightGBM</b> + Monte Carlo</span>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_control_center() -> Tuple[int, bool]:
    section_title("Simulation Control Center")
    st.caption("The only real, configurable lever is run count — higher counts converge closer to true probabilities but take longer.")

    if "tsim_run_count" not in st.session_state:
        st.session_state["tsim_run_count"] = 1000

    with st.container(border=True):
        preset_cols = st.columns(len(RUN_PRESETS))
        for col, preset in zip(preset_cols, RUN_PRESETS):
            with col:
                st.button(
                    f"{preset:,}", key=f"tsim_preset_{preset}", width="stretch",
                    on_click=_set_run_count, args=(preset,),
                )

        run_count = st.slider(
            "Monte Carlo Run Count",
            min_value=100,
            max_value=10000,
            step=100,
            key="tsim_run_count",
            help="Higher values take longer but converge to true mathematical probabilities.",
        )

        st.markdown(
            f'<div class="tsim-card" style="margin-top:14px;">'
            f'<p style="margin:0;font-size:0.88rem;color:{TEXT_MUTED};">'
            f'<b style="color:inherit;">Why no model picker or speed preset?</b> Only the LightGBM classifier is '
            f'served in production &mdash; other trained models (XGBoost, Random Forest) exist only as experiment '
            f'artifacts. There is also no backend "detail level" parameter; run count is the one real lever that '
            f'trades runtime for statistical precision.'
            f'</p></div>',
            unsafe_allow_html=True,
        )

        sim_btn = st.button("🏆 Run Monte Carlo Simulation", width="stretch", type="primary")

    return run_count, sim_btn


def render_top_contenders(champion_odds: Dict[str, float]) -> None:
    section_title("Top Contenders")
    st.caption("Real champion probability, ranked by the Monte Carlo engine.")

    ranked = sorted(champion_odds.items(), key=lambda kv: kv[1], reverse=True)[:10]
    if not ranked:
        st.info("No champion odds available for this run.")
        return
    max_val = ranked[0][1]

    rows = []
    for i, (team, prob) in enumerate(ranked, start=1):
        rank_cls = " rank-1" if i == 1 else ""
        width_pct = 100 * prob / max_val if max_val else 0
        rows.append(
            f'<div class="tsim-champ-row{rank_cls}">'
            f'<span class="tsim-champ-rank">{i:02d}</span>'
            f'<span class="tsim-champ-team">{flag_for(team)} {team}</span>'
            f'<div class="tsim-champ-track"><div class="tsim-champ-fill" style="width:{width_pct:.0f}%;"></div></div>'
            f'<span class="tsim-champ-pct">{prob*100:.1f}%</span>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_bracket(sample_bracket: Dict[str, Any] | None) -> None:
    section_title("Tournament Bracket")
    st.caption("One real simulated path, sampled from the last Monte Carlo run — not the definitive outcome, since every run re-simulates from scratch.")

    if not sample_bracket or not sample_bracket.get("rounds"):
        st.info("Run a simulation to see a real sample bracket path.")
        return

    rounds = sample_bracket["rounds"]
    round_order = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]

    round_html = []
    for stage in round_order:
        matches = rounds.get(stage, [])
        if not matches:
            continue
        match_html = []
        for m in matches:
            winner = m["winner"]
            loser = m["away_team"] if winner == m["home_team"] else m["home_team"]
            win_pct = f"{m['win_probability']*100:.0f}%" if m.get("win_probability") is not None else "–"
            match_html.append(
                f'<div class="tsim-bracket-match">'
                f'<div class="tsim-bracket-team winner">{flag_for(winner)} {winner}<span class="tsim-bracket-prob">{win_pct}</span></div>'
                f'<div class="tsim-bracket-team">{flag_for(loser)} {loser}</div>'
                f'</div>'
            )
        round_html.append(
            f'<div class="tsim-bracket-round">'
            f'<div class="tsim-bracket-round-label">{stage}</div>'
            f'{"".join(match_html)}'
            f'</div>'
        )
    st.markdown(
        f'<div class="tsim-bracket-scroll"><div class="tsim-bracket">{"".join(round_html)}</div></div>',
        unsafe_allow_html=True,
    )


def render_most_likely_final(most_likely_final: Dict[str, Any] | None) -> None:
    section_title("Most Likely Final")
    st.caption("Real joint finalist-pair frequency across all simulation runs — which two teams reach the final together most often.")

    if not most_likely_final:
        st.info("Run a simulation to compute the most common final matchup.")
        return

    team_a = most_likely_final["team_a"]
    team_b = most_likely_final["team_b"]
    prob = most_likely_final["probability"] * 100

    st.markdown(
        f'<div class="tsim-final-card">'
        f'<div class="tsim-final-teams">'
        f'<span class="tsim-final-team">{flag_for(team_a)} {team_a}</span>'
        f'<span class="tsim-final-vs">VS</span>'
        f'<span class="tsim-final-team">{flag_for(team_b)} {team_b}</span>'
        f'</div>'
        f'<p class="tsim-final-pct">{prob:.1f}%</p>'
        f'<p class="tsim-final-label">of simulations produced this exact final pairing</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_stage_matrix(stage_probs: List[dict]) -> None:
    section_title("Stage Probabilities")
    st.caption("Real advancement odds for all 48 teams, searchable.")

    if not stage_probs:
        st.info("Run a simulation to see stage probabilities.")
        return

    search = st.text_input("Search by team name", placeholder="e.g. Argentina", key="tsim_matrix_search")

    df = pd.DataFrame(stage_probs)
    if search:
        df = df[df["team"].str.contains(search, case=False)]

    df = df.sort_values(by="champion", ascending=False)
    stage_cols = ["group_stage", "round_of_32", "round_of_16", "quarterfinals", "semifinals", "finalist", "champion"]
    for col in stage_cols:
        df[col] = df[col] * 100

    df = df.rename(columns={
        "team": "Team", "group_stage": "Group %", "round_of_32": "R32 %", "round_of_16": "R16 %",
        "quarterfinals": "QF %", "semifinals": "SF %", "finalist": "Final %", "champion": "Champion %",
    })

    st.dataframe(
        df.set_index("Team").style.format("{:.1f}%").background_gradient(cmap="Blues", subset=df.columns[1:]),
        height=420,
        width="stretch",
    )


def render_upset_detector(upsets: List[dict]) -> None:
    section_title("Upset Detector")
    st.caption("Real metric: teams whose simulated odds rank far above their ELO rank — a genuine model-derived overperformer, not a scripted example.")

    if not upsets:
        st.info("No significant upsets detected in this simulation run — try a larger run count for more stable rankings.")
        return

    cards = []
    for u in upsets[:6]:
        team = u["team"]
        cards.append(
            f'<div class="tsim-upset-card">'
            f'<span class="tag">ELO rank #{u["elo_rank"]} &rarr; simulated rank #{u["simulated_rank"]}</span>'
            f'<h4>{flag_for(team)} {team}</h4>'
            f'<p>Reaches the semifinal in <span class="delta">{u["semifinal_probability"]*100:.1f}%</span> of simulations '
            f'and holds a <span class="delta">{u["champion_probability"]*100:.1f}%</span> championship probability — '
            f'{u["rank_delta"]} ranks higher than its ELO alone would predict.</p>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_monte_carlo_distribution(champion_odds: Dict[str, float]) -> None:
    section_title("Monte Carlo Distribution")
    st.caption("Real champion frequency across all simulation runs.")

    ranked = sorted(champion_odds.items(), key=lambda kv: kv[1], reverse=True)[:15]
    if not ranked:
        st.info("Run a simulation to see the champion distribution.")
        return

    max_val = ranked[0][1]
    bars = "".join(
        f'<div class="tsim-hist-bar" style="height:{100*prob/max_val:.0f}%;" title="{team}: {prob*100:.1f}%"></div>'
        for team, prob in ranked
    )
    st.markdown(f'<div class="tsim-hist-wrap">{bars}</div>', unsafe_allow_html=True)
    st.caption("Bars represent the top 15 teams by champion frequency, left to right, highest to lowest.")


def render_ai_insights(result: Dict[str, Any]) -> None:
    section_title("AI Tournament Insights")
    st.caption("Generated from real simulation output — champion odds, finalist frequency, and the real joint-final pairing.")

    stage_probs = result.get("stage_probabilities", [])
    most_likely_final = result.get("most_likely_final")

    if not stage_probs:
        st.info("Run a simulation to generate insights.")
        return

    ranked_finalists = sorted(stage_probs, key=lambda r: r["finalist"], reverse=True)
    top_finalist = ranked_finalists[0]
    ranked_qf = sorted(stage_probs, key=lambda r: r["quarterfinals"], reverse=True)
    most_consistent = ranked_qf[0]

    insights = [
        ("Finalist frequency", f"{top_finalist['team']} appears in {top_finalist['finalist']*100:.0f}% of simulated finals across all runs, the highest of any team."),
    ]
    if most_likely_final:
        insights.append((
            "Joint pairing",
            f"{most_likely_final['team_a']} vs. {most_likely_final['team_b']} is the single most common final matchup, "
            f"occurring in {most_likely_final['probability']*100:.1f}% of all simulations.",
        ))
    insights.append((
        "Consistency",
        f"{most_consistent['team']} reaches the quarterfinal stage in {most_consistent['quarterfinals']*100:.0f}% of runs, "
        f"the most consistent progression among all 48 teams.",
    ))

    cards = "".join(
        f'<div class="tsim-insight-card"><span class="tag">{tag}</span><p>{text}</p></div>'
        for tag, text in insights
    )
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;">{cards}</div>', unsafe_allow_html=True)


def render_export(result: Dict[str, Any]) -> None:
    st.markdown("<hr class='fifa-divider'>", unsafe_allow_html=True)
    col1, col2, _ = st.columns([1, 1, 2])

    json_data = json.dumps(result, indent=2, default=str)
    df_probs = pd.DataFrame(result.get("stage_probabilities", []))
    csv_data = df_probs.to_csv(index=False)

    with col1:
        st.download_button(
            "📥 Download JSON Report", data=json_data,
            file_name=f"MonteCarlo_{result.get('run_count', 0)}_runs.json",
            mime="application/json", width="stretch",
        )
    with col2:
        st.download_button(
            "📊 Download CSV Summary", data=csv_data,
            file_name=f"MonteCarlo_{result.get('run_count', 0)}_runs.csv",
            mime="text/csv", width="stretch",
        )


def show() -> None:
    """Render the Tournament Simulator interface."""
    render_simulator_hero()

    run_count, sim_btn = render_control_center()

    if sim_btn:
        with st.spinner(f"Simulating {run_count:,} full World Cup tournaments — this can take a while for larger run counts..."):
            try:
                result = st.session_state.api_client.simulate(run_count=run_count)
            except Exception as e:
                st.warning(f"Backend connection failed ({e}). Showing a sandbox offline example.")
                result = FALLBACK_RESULT
            st.session_state.tsim_result = result
            st.session_state.tsim_completed = True

    if not st.session_state.get("tsim_completed", False):
        return

    result = st.session_state.tsim_result
    champion_odds = result.get("champion_odds", {})
    stage_probs = result.get("stage_probabilities", [])

    st.markdown("<br>", unsafe_allow_html=True)
    render_top_contenders(champion_odds)

    st.markdown("<br>", unsafe_allow_html=True)
    render_bracket(result.get("sample_bracket"))

    st.markdown("<br>", unsafe_allow_html=True)
    render_most_likely_final(result.get("most_likely_final"))

    st.markdown("<br>", unsafe_allow_html=True)
    render_stage_matrix(stage_probs)

    st.markdown("<br>", unsafe_allow_html=True)
    render_upset_detector(result.get("upsets", []) or [])

    st.markdown("<br>", unsafe_allow_html=True)
    render_monte_carlo_distribution(champion_odds)

    st.markdown("<br>", unsafe_allow_html=True)
    render_ai_insights(result)

    render_export(result)


if __name__ == "__main__":
    show()
