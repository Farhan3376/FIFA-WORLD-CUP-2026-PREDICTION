"""FIFA World Cup 2026 Prediction Portal - Historical Analytics.

The football intelligence and data exploration center: how the model
performs against real, replayed World Cup matches from 2010-2022 --
goal-scoring trends, head-to-head history, model reliability, and a
searchable replay explorer, all built from the real 256-match backtest
dataset (no fabricated tournament history, titles, or rankings).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import streamlit as st

from frontend.services.theme import (
    section_title,
    flag_for,
    FIFA_BLUE,
    FIFA_BLUE_LIGHT,
    GOLD,
    EMERALD,
    RED_CARD,
    TEXT_MUTED,
)

FALLBACK_DATA: Dict[str, Any] = {
    "metrics": {
        "overall_accuracy": 0.5859375,
        "accuracy_by_year": {"2010": 0.578125, "2014": 0.625, "2018": 0.625, "2022": 0.515625},
        "classification_report": {
            "home_win": {"precision": 0.6014, "recall": 0.8113, "f1-score": 0.6908, "support": 106.0},
            "draw": {"precision": 0.4286, "recall": 0.1053, "f1-score": 0.1690, "support": 57.0},
            "away_win": {"precision": 0.5859, "recall": 0.6237, "f1-score": 0.6042, "support": 93.0},
        },
        "confusion_matrix": [[86, 2, 18], [28, 6, 23], [29, 6, 58]],
        "total_matches_evaluated": 256,
    },
    "report_text": (
        "============================================================\n"
        "        HISTORICAL WORLD CUP REPLAY VALIDATION REPORT\n"
        "============================================================\n\n"
        "Total Matches Replayed: 256\n"
        "Overall Prediction Accuracy: 58.59%\n\n"
        "Accuracy by Tournament Year:\n"
        "  * 2010 World Cup: 57.81%\n"
        "  * 2014 World Cup: 62.50%\n"
        "  * 2018 World Cup: 62.50%\n"
        "  * 2022 World Cup: 51.56%\n"
    ),
    "predictions": [
        {"date": "2010-06-11", "year": 2010, "home_team": "South Africa", "away_team": "Mexico", "home_goals": 1, "away_goals": 1, "actual_result": "draw", "predicted_result": "away_win", "prob_home_win": 0.2245, "prob_draw": 0.3017, "prob_away_win": 0.4739, "correct": False},
        {"date": "2010-06-12", "year": 2010, "home_team": "Argentina", "away_team": "Nigeria", "home_goals": 1, "away_goals": 0, "actual_result": "home_win", "predicted_result": "home_win", "prob_home_win": 0.5049, "prob_draw": 0.2977, "prob_away_win": 0.1974, "correct": True},
        {"date": "2014-07-08", "year": 2014, "home_team": "Brazil", "away_team": "Germany", "home_goals": 1, "away_goals": 7, "actual_result": "away_win", "predicted_result": "home_win", "prob_home_win": 0.55, "prob_draw": 0.25, "prob_away_win": 0.20, "correct": False},
    ],
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_historical_data() -> Dict[str, Any]:
    """Fetch the real historical backtest (256 replayed matches, 2010-2022) from the backend."""
    try:
        return st.session_state.api_client.get_historical_replay()
    except Exception:
        return FALLBACK_DATA


def render_analytics_hero(metrics: Dict[str, Any], predictions: List[dict]) -> None:
    total_matches = metrics.get("total_matches_evaluated", len(predictions))
    editions = sorted({p["year"] for p in predictions})
    total_goals = sum(p["home_goals"] + p["away_goals"] for p in predictions)
    accuracy = metrics.get("overall_accuracy", 0.0) * 100

    st.markdown(
        f'<div class="ha-hero">'
        f'<div class="ha-hero-inner">'
        f'<p class="ha-kicker">Historical Analytics</p>'
        f'<h1>How the model performs against real World Cup football</h1>'
        f'<p>{total_matches} replayed matches across {len(editions)} tournaments ({editions[0]}–{editions[-1]}) '
        f'&mdash; every prediction checked against what actually happened on the pitch.</p>'
        f'<div class="ha-hero-chips">'
        f'<span class="ha-chip"><b>{len(editions)}</b> editions</span>'
        f'<span class="ha-chip"><b>{total_matches}</b> matches replayed</span>'
        f'<span class="ha-chip"><b>{accuracy:.1f}%</b> overall accuracy</span>'
        f'<span class="ha-chip"><b>{total_goals}</b> goals scored</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_overview_kpis(metrics: Dict[str, Any], predictions: List[dict]) -> None:
    editions = sorted({p["year"] for p in predictions})
    total_matches = metrics.get("total_matches_evaluated", len(predictions))
    total_goals = sum(p["home_goals"] + p["away_goals"] for p in predictions)
    avg_goals = total_goals / total_matches if total_matches else 0.0
    accuracy_by_year = metrics.get("accuracy_by_year", {})
    best_year = max(accuracy_by_year, key=accuracy_by_year.get) if accuracy_by_year else "-"
    best_acc = accuracy_by_year.get(best_year, 0.0) * 100

    cols = st.columns(6)
    kpis = [
        ("Editions Replayed", str(len(editions)), None),
        ("Matches Replayed", str(total_matches), None),
        ("Total Goals", str(total_goals), None),
        ("Avg Goals / Match", f"{avg_goals:.2f}", None),
        ("Overall Accuracy", f"{metrics.get('overall_accuracy', 0.0)*100:.1f}%", (EMERALD, f"best: {best_year} ({best_acc:.1f}%)")),
        ("Highest Scoring Edition", str(max({p['year'] for p in predictions}, key=lambda y: sum(p['home_goals']+p['away_goals'] for p in predictions if p['year'] == y))), None),
    ]
    for col, (label, value, delta) in zip(cols, kpis):
        delta_html = f'<div class="delta" style="color:{delta[0]};">{delta[1]}</div>' if delta else ""
        with col:
            st.markdown(
                f'<div class="fifa-card"><div class="label">{label}</div>'
                f'<div class="value" style="font-size:1.9rem;">{value}</div>{delta_html}</div>',
                unsafe_allow_html=True,
            )


def render_edition_explorer(predictions: List[dict]) -> None:
    section_title("Edition Explorer")
    st.caption("Real, per-tournament breakdown from the replayed backtest — the backend has no host/champion/Golden Boot data, so this replaces a fabricated 1930–2026 timeline.")

    by_year: Dict[int, List[dict]] = defaultdict(list)
    for p in predictions:
        by_year[p["year"]].append(p)
    years = sorted(by_year.keys())

    tabs = st.tabs([str(y) for y in years])
    for tab, year in zip(tabs, years):
        with tab:
            matches = by_year[year]
            goals = sum(m["home_goals"] + m["away_goals"] for m in matches)
            correct = sum(1 for m in matches if m["correct"])
            accuracy = 100 * correct / len(matches) if matches else 0.0

            biggest = max(matches, key=lambda m: abs(m["home_goals"] - m["away_goals"]))
            margin = abs(biggest["home_goals"] - biggest["away_goals"])

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown(
                    f'<div class="ha-card"><h3>{year} World Cup — model accuracy: {accuracy:.1f}%</h3>'
                    f'<div class="ha-goal-row"><span class="g-label">Matches</span>'
                    f'<div class="ha-goal-track"><div class="ha-goal-fill" style="width:100%;"></div></div>'
                    f'<span class="g-val">{len(matches)}</span></div>'
                    f'<div class="ha-goal-row"><span class="g-label">Goals</span>'
                    f'<div class="ha-goal-track"><div class="ha-goal-fill" style="width:{min(100, goals/2):.0f}%;"></div></div>'
                    f'<span class="g-val">{goals}</span></div>'
                    f'<div class="ha-goal-row"><span class="g-label">Correct</span>'
                    f'<div class="ha-goal-track"><div class="ha-goal-fill" style="width:{accuracy:.0f}%;"></div></div>'
                    f'<span class="g-val">{correct} / {len(matches)}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="ha-card"><h3>Biggest goal margin</h3>'
                    f'<div class="ha-match-row" style="grid-template-columns:1fr auto 1fr;">'
                    f'<span class="m-team">{flag_for(biggest["home_team"])} {biggest["home_team"]}</span>'
                    f'<span class="m-score">{biggest["home_goals"]}–{biggest["away_goals"]}</span>'
                    f'<span class="m-team away">{biggest["away_team"]} {flag_for(biggest["away_team"])}</span>'
                    f'</div>'
                    f'<p class="ha-note">Margin of {margin} goal{"s" if margin != 1 else ""} — the widest scoreline in the {year} replay set.</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_goal_analytics(predictions: List[dict]) -> None:
    section_title("Goal Scoring Analytics")
    st.caption("Computed directly from real goals scored across all 256 replayed matches.")

    by_year: Dict[int, List[dict]] = defaultdict(list)
    for p in predictions:
        by_year[p["year"]].append(p)
    years = sorted(by_year.keys())
    goals_per_year = {y: sum(m["home_goals"] + m["away_goals"] for m in by_year[y]) for y in years}
    max_goals = max(goals_per_year.values()) if goals_per_year else 1

    home_wins = sum(1 for p in predictions if p["actual_result"] == "home_win")
    draws = sum(1 for p in predictions if p["actual_result"] == "draw")
    away_wins = sum(1 for p in predictions if p["actual_result"] == "away_win")
    total = home_wins + draws + away_wins or 1

    col1, col2 = st.columns([1.2, 1])
    with col1:
        rows = "".join(
            f'<div class="ha-goal-row"><span class="g-label">{y}</span>'
            f'<div class="ha-goal-track"><div class="ha-goal-fill" style="width:{100*goals_per_year[y]/max_goals:.0f}%;"></div></div>'
            f'<span class="g-val">{goals_per_year[y]}</span></div>'
            for y in years
        )
        st.markdown(f'<div class="ha-card"><h3>Goals per edition</h3>{rows}</div>', unsafe_allow_html=True)

    with col2:
        home_frac, draw_frac, away_frac = home_wins / total, draws / total, away_wins / total
        home_len, draw_len, away_len = home_frac * 100, draw_frac * 100, away_frac * 100
        circumference = 100
        st.markdown(
            f'<div class="ha-card"><h3>Match outcome distribution</h3>'
            f'<div class="ha-donut-wrap">'
            f'<svg width="120" height="120" viewBox="0 0 42 42">'
            f'<circle cx="21" cy="21" r="15.9" fill="transparent" stroke="var(--surface-alt,#F4F6FA)" stroke-width="6"></circle>'
            f'<circle cx="21" cy="21" r="15.9" fill="transparent" stroke="{FIFA_BLUE}" stroke-width="6" '
            f'stroke-dasharray="{home_len:.1f} {circumference - home_len:.1f}" stroke-dashoffset="25"></circle>'
            f'<circle cx="21" cy="21" r="15.9" fill="transparent" stroke="{GOLD}" stroke-width="6" '
            f'stroke-dasharray="{draw_len:.1f} {circumference - draw_len:.1f}" stroke-dashoffset="{25 - home_len:.1f}"></circle>'
            f'<circle cx="21" cy="21" r="15.9" fill="transparent" stroke="{FIFA_BLUE_LIGHT}" stroke-width="6" '
            f'stroke-dasharray="{away_len:.1f} {circumference - away_len:.1f}" stroke-dashoffset="{25 - home_len - draw_len:.1f}"></circle>'
            f'</svg>'
            f'<div class="ha-donut-legend">'
            f'<div><span class="dot" style="background:{FIFA_BLUE};"></span>Home win — {home_wins}</div>'
            f'<div><span class="dot" style="background:{GOLD};"></span>Draw — {draws}</div>'
            f'<div><span class="dot" style="background:{FIFA_BLUE_LIGHT};"></span>Away win — {away_wins}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_head_to_head(predictions: List[dict]) -> None:
    section_title("Head-to-Head History")
    st.caption("Real matchups found within the 256-match backtest — only meetings that occurred in the 2010–2022 replay window are shown.")

    teams = sorted({p["home_team"] for p in predictions} | {p["away_team"] for p in predictions})
    if len(teams) < 2:
        st.info("Not enough teams in the replay dataset for a head-to-head comparison.")
        return

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", options=teams, index=0, format_func=lambda t: f"{flag_for(t)}  {t}", key="ha_h2h_a")
    with col2:
        default_b = min(1, len(teams) - 1)
        team_b = st.selectbox("Team B", options=teams, index=default_b, format_func=lambda t: f"{flag_for(t)}  {t}", key="ha_h2h_b")

    if team_a == team_b:
        st.warning("Choose two different teams to see their head-to-head record.")
        return

    meetings = [
        p for p in predictions
        if {p["home_team"], p["away_team"]} == {team_a, team_b}
    ]

    if not meetings:
        st.info(f"{team_a} and {team_b} did not meet in any of the 4 replayed World Cup editions (2010, 2014, 2018, 2022).")
        return

    a_wins = sum(1 for m in meetings if (m["home_team"] == team_a and m["actual_result"] == "home_win") or (m["away_team"] == team_a and m["actual_result"] == "away_win"))
    b_wins = sum(1 for m in meetings if (m["home_team"] == team_b and m["actual_result"] == "home_win") or (m["away_team"] == team_b and m["actual_result"] == "away_win"))
    draws = sum(1 for m in meetings if m["actual_result"] == "draw")

    st.markdown(
        f'<div class="ha-h2h-summary">'
        f'<div class="ha-h2h-stat"><p class="n">{a_wins}</p><p class="l">{team_a} wins</p></div>'
        f'<div class="ha-h2h-stat"><p class="n">{draws}</p><p class="l">Draws</p></div>'
        f'<div class="ha-h2h-stat"><p class="n">{b_wins}</p><p class="l">{team_b} wins</p></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for m in sorted(meetings, key=lambda m: m["date"]):
        badge_cls = "correct" if m["correct"] else "wrong"
        badge_text = "✓ model correct" if m["correct"] else "✗ model missed"
        rows.append(
            f'<div class="ha-match-row">'
            f'<span class="m-year">{m["year"]}</span>'
            f'<span class="m-team">{flag_for(m["home_team"])} {m["home_team"]}</span>'
            f'<span class="m-score">{m["home_goals"]}–{m["away_goals"]}</span>'
            f'<span class="m-team away">{m["away_team"]} {flag_for(m["away_team"])}</span>'
            f'<span class="m-badge {badge_cls}">{badge_text}</span>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_model_reliability(metrics: Dict[str, Any]) -> None:
    section_title("Model Reliability")
    st.caption("Classification report and confusion matrix computed directly from the 256-match backtest.")

    conf = metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    report = metrics.get("classification_report", {})

    col1, col2 = st.columns([1, 1])
    with col1:
        cells = (
            '<div class="ha-conf-cell head"></div>'
            '<div class="ha-conf-cell head">Pred. Home</div>'
            '<div class="ha-conf-cell head">Pred. Draw</div>'
            '<div class="ha-conf-cell head">Pred. Away</div>'
        )
        row_labels = ["Actual Home", "Actual Draw", "Actual Away"]
        for i, label in enumerate(row_labels):
            cells += f'<div class="ha-conf-cell row-head">{label}</div>'
            for j in range(3):
                cls = "ha-conf-cell diag" if i == j else "ha-conf-cell"
                cells += f'<div class="{cls}">{conf[i][j]}</div>'
        st.markdown(f'<div class="ha-card"><h3>Confusion matrix</h3><div class="ha-conf-matrix">{cells}</div></div>', unsafe_allow_html=True)

    with col2:
        outcome_labels = {"home_win": "Home win", "draw": "Draw", "away_win": "Away win"}
        rows = ""
        for key, label in outcome_labels.items():
            recall = report.get(key, {}).get("recall", 0.0) * 100
            fill_cls = "ha-goal-fill low" if recall < 30 else "ha-goal-fill"
            rows += (
                f'<div class="ha-goal-row"><span class="g-label" style="width:80px;">{label}</span>'
                f'<div class="ha-goal-track"><div class="{fill_cls}" style="width:{recall:.0f}%;"></div></div>'
                f'<span class="g-val">{recall:.0f}% rec.</span></div>'
            )
        st.markdown(f'<div class="ha-card"><h3>Recall by outcome</h3>{rows}</div>', unsafe_allow_html=True)


def render_replay_explorer(predictions: List[dict]) -> None:
    section_title("Match Replay Explorer")
    st.caption("All replayed matches, searchable — not a static dump.")

    search = st.text_input("Search by team name", placeholder="e.g. Argentina", key="ha_replay_search")

    rows = predictions
    if search:
        needle = search.lower()
        rows = [p for p in rows if needle in p["home_team"].lower() or needle in p["away_team"].lower()]

    import pandas as pd

    if not rows:
        st.info("No matches found for that search.")
        return

    df = pd.DataFrame(rows)
    df["Result"] = df.apply(
        lambda r: f'✓ Correct ({outcome_label(r["actual_result"])})' if r["correct"] else f'✗ Actual: {outcome_label(r["actual_result"])}',
        axis=1,
    )
    df["Score"] = df.apply(lambda r: f'{r["home_goals"]}–{r["away_goals"]}', axis=1)
    df["Predicted"] = df["predicted_result"].map(outcome_label)
    display_df = df.rename(columns={
        "date": "Date", "year": "Edition", "home_team": "Home", "away_team": "Away",
    })[["Date", "Edition", "Home", "Score", "Away", "Predicted", "Result"]]

    st.dataframe(display_df.set_index("Date"), height=420, width="stretch")


def outcome_label(result: str) -> str:
    return {"home_win": "Home win", "draw": "Draw", "away_win": "Away win"}.get(result, result)


def render_insights(metrics: Dict[str, Any], predictions: List[dict]) -> None:
    section_title("Historical Insights")
    st.caption("Generated from the real classification report and per-edition accuracy.")

    report = metrics.get("classification_report", {})
    accuracy_by_year = metrics.get("accuracy_by_year", {})

    draw_recall = report.get("draw", {}).get("recall", 0.0) * 100
    draw_support = int(report.get("draw", {}).get("support", 0))
    draw_correct = round(draw_recall / 100 * draw_support) if draw_support else 0

    by_year: Dict[int, List[dict]] = defaultdict(list)
    for p in predictions:
        by_year[p["year"]].append(p)
    goals_by_year = {y: sum(m["home_goals"] + m["away_goals"] for m in ms) for y, ms in by_year.items()}
    years_sorted = sorted(goals_by_year)

    worst_year = min(accuracy_by_year, key=accuracy_by_year.get) if accuracy_by_year else "-"
    worst_acc = accuracy_by_year.get(worst_year, 0.0) * 100

    insights = [
        ("Model behavior", f"The model over-predicts away wins and under-calls draws — only {draw_correct} of {draw_support} real draws ({draw_recall:.0f}%) were correctly identified across all four editions."),
        ("Edition trend", f"{worst_year} was the model's toughest tournament to date at {worst_acc:.1f}% accuracy, against an overall backtest average of {metrics.get('overall_accuracy', 0.0)*100:.1f}%."),
    ]
    if len(years_sorted) >= 2:
        insights.append((
            "Goal trend",
            f"Total goals per edition moved from {goals_by_year[years_sorted[0]]} in {years_sorted[0]} "
            f"to {goals_by_year[years_sorted[-1]]} in {years_sorted[-1]}.",
        ))

    cards = "".join(
        f'<div class="ha-insight-card"><span class="tag">{tag}</span><p>{text}</p></div>'
        for tag, text in insights
    )
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;">{cards}</div>', unsafe_allow_html=True)


def show() -> None:
    """Render the Historical Analytics interface."""
    data = fetch_historical_data()
    metrics = data.get("metrics", FALLBACK_DATA["metrics"])
    predictions = data.get("predictions", FALLBACK_DATA["predictions"])

    render_analytics_hero(metrics, predictions)
    render_overview_kpis(metrics, predictions)

    st.markdown("<br>", unsafe_allow_html=True)
    render_edition_explorer(predictions)

    st.markdown("<br>", unsafe_allow_html=True)
    render_goal_analytics(predictions)

    st.markdown("<br>", unsafe_allow_html=True)
    render_head_to_head(predictions)

    st.markdown("<br>", unsafe_allow_html=True)
    render_model_reliability(metrics)

    st.markdown("<br>", unsafe_allow_html=True)
    render_replay_explorer(predictions)

    st.markdown("<br>", unsafe_allow_html=True)
    render_insights(metrics, predictions)


if __name__ == "__main__":
    show()
