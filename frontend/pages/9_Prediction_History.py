"""FIFA World Cup 2026 Prediction Portal - Prediction History.

Retrieves and lists prediction history for the authenticated user, or presents
sandbox historical session logs when running anonymously.
"""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from frontend.services.theme import render_hero, section_title, flag_for, EMERALD, BLUE_ACCENT, TEXT_MUTED


def fetch_prediction_history() -> List[Dict[str, Any]]:
    """Fetch user's prediction history from the backend, falling back to mock logs if unauthenticated."""
    if not st.session_state.get("authenticated", False):
        # High-fidelity mock history for unauthenticated/sandbox demo
        return [
            {
                "id": 101,
                "home_team": "Argentina",
                "away_team": "France",
                "tournament": "FIFA World Cup",
                "venue": "neutral",
                "match_date": "2026-06-11",
                "predicted_winner": "Argentina",
                "prob_home_win": 0.485,
                "prob_draw": 0.280,
                "prob_away_win": 0.235,
                "confidence_score": 0.485,
                "expected_goals": {"home_xg": 1.95, "away_xg": 1.25},
                "timestamp": "2026-07-03T18:32:00Z"
            },
            {
                "id": 102,
                "home_team": "Brazil",
                "away_team": "England",
                "tournament": "FIFA World Cup",
                "venue": "neutral",
                "match_date": "2026-06-12",
                "predicted_winner": "Brazil",
                "prob_home_win": 0.520,
                "prob_draw": 0.260,
                "prob_away_win": 0.220,
                "confidence_score": 0.520,
                "expected_goals": {"home_xg": 2.10, "away_xg": 1.10},
                "timestamp": "2026-07-03T18:35:00Z"
            },
            {
                "id": 103,
                "home_team": "USA",
                "away_team": "Mexico",
                "tournament": "CONCACAF Gold Cup",
                "venue": "home",
                "match_date": "2026-06-15",
                "predicted_winner": "USA",
                "prob_home_win": 0.450,
                "prob_draw": 0.310,
                "prob_away_win": 0.240,
                "confidence_score": 0.450,
                "expected_goals": {"home_xg": 1.65, "away_xg": 1.15},
                "timestamp": "2026-07-03T18:40:00Z"
            }
        ]
        
    try:
        api = st.session_state.api_client
        return api.get_prediction_history()
    except Exception as e:
        st.warning(f"Failed to fetch live history from FastAPI database ({e}). Showing session sandbox data.")
        return []


def show() -> None:
    """Render the Prediction History interface."""
    render_hero(
        kicker="Query Logs",
        title="📜 Prediction History Log",
        subtitle="Review, filter, and export the history of match predictions processed in your current profile session.",
        badges=[("⏳ Session & Database Logs", "gold", "")],
    )

    # Authentication Context Notice
    if not st.session_state.get("authenticated", False):
        st.info("💡 **Sandbox Mode**: You are running anonymously. Logging in via the sidebar will persist your predictions permanently in the SQLite/PostgreSQL database.")

    history = fetch_prediction_history()

    if not history:
        st.markdown("No prediction records found in your session or database history log yet.")
        return

    # Metrics Summary Card Grid
    total_queries = len(history)
    avg_conf = sum(h["confidence_score"] for h in history) / total_queries if total_queries > 0 else 0
    unique_matches = len(set(f"{h['home_team']}_vs_{h['away_team']}" for h in history))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">Total Queries</div>
                <div class="value">{total_queries}</div>
                <div class="delta" style="color: {EMERALD};">Predictions computed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">Average Confidence</div>
                <div class="value">{avg_conf*100:.1f}%</div>
                <div class="delta" style="color: {BLUE_ACCENT};">Mean model prediction scale</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="fifa-card">
                <div class="label">Distinct Matchups</div>
                <div class="value">{unique_matches}</div>
                <div class="delta" style="color: {TEXT_MUTED};">Unique team pair runs</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Prediction Audit Registry")
    
    # Process history data for table representation
    rows = []
    for pred in history:
        # Extract xG safely
        xg_data = pred.get("expected_goals", {})
        # xg_data can be a dictionary or a string of JSON in database
        if isinstance(xg_data, str):
            try:
                xg_data = json.loads(xg_data)
            except Exception:
                xg_data = {}
        
        home_xg = xg_data.get("home_xg", 0.0)
        away_xg = xg_data.get("away_xg", 0.0)
        
        rows.append({
            "Timestamp": pred["timestamp"][:19].replace("T", " "),
            "Home Team": f"{flag_for(pred['home_team'])} {pred['home_team']}",
            "Away Team": f"{flag_for(pred['away_team'])} {pred['away_team']}",
            "Tournament": pred["tournament"],
            "Venue": pred["venue"],
            "Predicted Winner": pred["predicted_winner"],
            "Home Win Prob": f"{pred['prob_home_win']*100:.1f}%",
            "Draw Prob": f"{pred['prob_draw']*100:.1f}%",
            "Away Win Prob": f"{pred['prob_away_win']*100:.1f}%",
            "Expected Score (xG)": f"{home_xg:.1f} - {away_xg:.1f}",
            "Confidence": f"{pred['confidence_score']*100:.1f}%"
        })
        
    df_history = pd.DataFrame(rows)
    
    # Search Box Filter
    search_query = st.text_input("🔍 Filter History by Team Name", placeholder="e.g. Brazil")
    if search_query:
        df_history = df_history[
            df_history["Home Team"].str.contains(search_query, case=False) |
            df_history["Away Team"].str.contains(search_query, case=False)
        ]
        
    st.dataframe(df_history.set_index("Timestamp"), height=350, width=1200)

    # Export history to CSV
    csv_data = df_history.to_csv(index=False)
    st.download_button(
        label="📥 Export Query History as CSV",
        data=csv_data,
        file_name="UserPredictionHistory.csv",
        mime="text/csv",
        width="stretch"
    )


if __name__ == "__main__":
    show()
