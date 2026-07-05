"""FIFA World Cup 2026 Prediction Engine - Master Frontend Router.

Provides global page configuration, JWT authorization state, premium CSS styles,
and navigation routing using Streamlit 1.35.0's native st.navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path to guarantee clean relative module imports
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
from frontend.services.api_client import APIClient
from frontend.services.theme import inject_brand_css, render_top_header, EMERALD


def initialize_session_state() -> None:
    """Set up global session state variables for authentication and API communication."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"

    if "primary_color" not in st.session_state:
        st.session_state.primary_color = "#1f6feb"  # Default Royal Blue


def inject_global_css() -> None:
    """Inject the shared FIFA World Cup 2026 brand stylesheet plus sidebar-specific styling."""
    inject_brand_css()
    st.markdown(
        f"""
        <style>
            /* User profile card in sidebar */
            .user-profile {{
                background-color: #ffffff;
                border: 1px solid #e2e2df;
                border-left: 3px solid {EMERALD};
                border-radius: 8px;
                padding: 12px 14px;
                margin-bottom: 20px;
            }}
            .user-title {{
                font-size: 0.75rem;
                color: #5f6368 !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .user-name {{
                font-size: 1.05rem;
                font-weight: 700;
                color: {EMERALD} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_auth() -> None:
    """Render the account sign-in panel in the sidebar.

    Brand identity (logo + title) now lives solely in the top navigation bar,
    so the sidebar is reserved for account state only -- no duplicate branding.
    """
    st.sidebar.markdown(
        '<div style="font-family:\'Outfit\',sans-serif; font-weight:800; font-size:0.95rem; '
        'text-transform:uppercase; letter-spacing:0.08em; color:#5f6368; margin-bottom:14px;">Account</div>',
        unsafe_allow_html=True,
    )

    # Auth logic panel
    if not st.session_state.logged_in:
        auth_mode = st.sidebar.selectbox(
            "User Portal Login",
            ["Sign In", "Create Account"],
            label_visibility="collapsed"
        )
        
        with st.sidebar.form("auth_form"):
            username = st.text_input("Username", placeholder="e.g. analyst101")
            email = st.text_input("Email", placeholder="e.g. email@domain.com") if auth_mode == "Create Account" else None
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            submit_label = "Sign In" if auth_mode == "Sign In" else "Register"
            submit_btn = st.form_submit_button(submit_label, width="stretch")
            
            if submit_btn:
                if not username or not password:
                    st.sidebar.error("All credentials must be supplied.")
                else:
                    try:
                        if auth_mode == "Sign In":
                            res = st.session_state.api_client.login(username, password)
                            if "access_token" in res:
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.sidebar.success(f"Welcome back, {username}!")
                                st.rerun()
                        else:
                            if not email:
                                st.sidebar.error("Valid email is required.")
                            else:
                                st.session_state.api_client.register(username, email, password)
                                st.sidebar.success("Registration successful! Sign in now.")
                    except Exception as e:
                        st.sidebar.error(f"Auth Error: {e}")
    else:
        st.sidebar.markdown(
            f"""
            <div class="user-profile">
                <div class="user-title">Active Security Profile</div>
                <div class="user-name">👤 {st.session_state.username}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("🔐 Log Out Session", width="stretch"):
            st.session_state.api_client.clear_token()
            st.session_state.logged_in = False
            st.session_state.username = None
            st.sidebar.info("Logged out successfully.")
            st.rerun()

    st.sidebar.markdown("<br><hr style='border:1px solid #e2e2df'><br>", unsafe_allow_html=True)


def main() -> None:
    """Configure page configs, build navigation tree, and run Streamlit entry point."""
    st.set_page_config(
        page_title="FIFA World Cup 2026 | Prediction Portal",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()
    inject_global_css()
    render_sidebar_auth()

    # Define Navigation Structure using Streamlit 1.35+ Page objects.
    # url_path is set explicitly (rather than left to filename-derived defaults)
    # so the custom FIFA-style header can link to stable, predictable URLs.
    home_page = st.Page("pages/1_Home.py", title="Dashboard Home", icon="🏠", url_path="dashboard", default=True)
    match_pred_page = st.Page("pages/2_Match_Prediction.py", title="Match Winner Prediction", icon="🔮", url_path="match-predictor")
    team_comp_page = st.Page("pages/3_Team_Comparison.py", title="Team Head-to-Head", icon="⚔️", url_path="head-to-head")
    tournament_sim_page = st.Page("pages/4_Tournament_Simulator.py", title="Tournament Simulator", icon="🏆", url_path="tournament-simulator")
    team_anal_page = st.Page("pages/5_Team_Analytics.py", title="National Team Profiles", icon="📈", url_path="team-analytics")
    historical_page = st.Page("pages/6_Historical_Analysis.py", title="Historical Analytics", icon="📚", url_path="historical-analytics")
    model_perf_page = st.Page("pages/7_Model_Performance.py", title="Model Audit & Latency", icon="⚙️", url_path="model-performance")
    feature_imp_page = st.Page("pages/8_Feature_Importance.py", title="AI Insights & Explainability", icon="🧠", url_path="feature-importance")
    pred_history_page = st.Page("pages/9_Prediction_History.py", title="Query Logs & History", icon="⏳", url_path="prediction-history")
    settings_page = st.Page("pages/10_Settings.py", title="System Settings", icon="🔧", url_path="settings")

    # Group Pages logically
    navigation_sections = {
        "Overview": [home_page],
        "Prediction & Simulation": [match_pred_page, team_comp_page, tournament_sim_page],
        "Team & Historical Analytics": [team_anal_page, historical_page],
        "ML Auditing & Interpretability": [model_perf_page, feature_imp_page],
        "User & System Tools": [pred_history_page, settings_page],
    }

    # Native navigation stays wired for real routing/state, but visually hidden --
    # the FIFA-style header (rendered in render_top_header) provides the actual
    # on-screen nav using real <a> links to these same url_path values.
    pg = st.navigation(navigation_sections, position="hidden")
    render_top_header(current_url_path=pg.url_path)
    pg.run()


if __name__ == "__main__":
    main()
