"""FIFA World Cup 2026 Prediction Portal - Settings.

Configures connection settings (FastAPI URL), local cache controls, default parameters,
and provides system metadata details.
"""

from __future__ import annotations

import streamlit as st
from typing import Dict, Any

from frontend.services.theme import render_hero


def check_api_status(url: str) -> bool:
    """Test connection status of a FastAPI base URL."""
    import requests
    try:
        response = requests.get(f"{url}/api/analytics/teams", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def show() -> None:
    """Render the Settings interface."""
    render_hero(
        kicker="System",
        title="🔧 Platform Configuration Settings",
        subtitle="Configure connection endpoints, reset session states, and review platform specifications.",
        badges=[("🔌 Backend Connectivity", "gold", "")],
    )

    tab_api, tab_defaults, tab_system = st.tabs([
        "🔌 Connection Settings", 
        "⚙️ Default Preferences", 
        "🖥️ System Metadata"
    ])

    with tab_api:
        st.markdown("#### FastAPI Server Endpoints")
        st.markdown("Specify the network URL of the FastAPI backend service.")
        
        api_url = st.text_input("FastAPI Server Base URL", value="http://localhost:8000")
        
        # Test connection button
        test_col, status_col = st.columns([1, 3])
        with test_col:
            test_btn = st.button("🔌 Test Connection", type="primary")
            
        with status_col:
            if test_btn:
                with st.spinner("Testing API connection..."):
                    online = check_api_status(api_url)
                    if online:
                        st.success("🟢 Connection Successful: FastAPI backend responded with HTTP 200.")
                        # Update client in session state
                        st.session_state.api_client.base_url = api_url
                    else:
                        st.error("🔴 Connection Failed: No response received from server endpoint.")
            else:
                st.write("")

    with tab_defaults:
        st.markdown("#### Simulation & Model UI Preferences")
        
        venue_pref = st.selectbox("Default Match Venue Configuration", ["neutral", "home", "away"])
        sim_pref = st.selectbox("Default Monte Carlo Simulation Count", [500, 1000, 2000, 5000], index=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Cache & Session State Controls")
        st.markdown("Clear session storage parameters (this will log out active sessions and clear simulated results).")
        
        clear_btn = st.button("🧹 Reset Local Client Cache")
        if clear_btn:
            # Clear critical session flags, but preserve api_client
            for key in ["authenticated", "auth_token", "user_profile", "sim_result", "sim_completed"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Session state cache cleared successfully.")

    with tab_system:
        st.markdown("#### Platform Specifications")
        st.markdown(
            """
            * **Application**: FIFA World Cup 2026 Match Prediction Portal
            * **Version**: `1.0.0-release`
            * **Technology Stack**:
              * **Frontend**: Python `Streamlit` (Aesthetics curated with Custom HSL & Google Fonts)
              * **Charts**: Interactive `Plotly` and `go.Figure` rendering vectors
              * **Backend**: `FastAPI` (endpoints for prediction service, SHAP explanations, analytics, authentication)
              * **Database**: `SQLAlchemy` (SQLite file database / PostgreSQL)
              * **ML Engine**: `LightGBM` (Gradient boosted decision trees trained on custom walk-forward metrics since 1998)
            """
        )


if __name__ == "__main__":
    show()
