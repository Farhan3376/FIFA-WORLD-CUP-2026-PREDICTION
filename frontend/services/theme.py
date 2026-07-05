"""Shared FIFA World Cup 2026 brand theme: colors, fonts, flags, and CSS injection.

Centralizes the visual identity used across all frontend pages so that the
portal reads as a single, consistent, human-designed product — a clean
light editorial look (white surfaces, black body text) rather than a
generic dark "AI dashboard".
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
PLAYERS_DIR = ASSETS_DIR / "players"

# --- Brand Palette -------------------------------------------------------
# Matched to FIFA.com's own on-screen system: bright FIFA blue for chrome/nav,
# pure black for promo banners, wine/maroon for "live" and hero surfaces,
# and a white canvas — gold reserved as the one non-FIFA accent color.
INK = "#111111"
INK_SOFT = "#3a3a3a"
PAPER = "#ffffff"
PAPER_ALT = "#F4F6FA"
PROMO_BLACK = "#0A0A0A"
FIFA_BLUE = "#0A3B8C"
FIFA_BLUE_LIGHT = "#1F7CE0"
WINE = "#7A1F3D"
WINE_DARK = "#5C1730"
PITCH_GREEN = "#0a3d2e"
PITCH_GREEN_LIGHT = "#127a52"
EMERALD = "#0a8f4f"
GOLD = "#C8A24A"
GOLD_LIGHT = "#8a6d00"
SURFACE = "#ffffff"
SURFACE_ALT = "#F4F6FA"
BORDER = "#e2e2df"
BORDER_STRONG = "#C7D0DE"
TEXT_MUTED = "#5f6368"
TEXT_BODY = "#111111"
RED_CARD = "#c62828"
BLUE_ACCENT = "#0A3B8C"

FONT_DISPLAY = "'Outfit', 'Segoe UI', sans-serif"
FONT_BODY = "'Inter', 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SF Mono', ui-monospace, monospace"

# --- Country flag emoji map (ISO-ish common names used across the app) --
FLAGS = {
    "Argentina": "🇦🇷", "Australia": "🇦🇺", "Belgium": "🇧🇪", "Brazil": "🇧🇷",
    "Cameroon": "🇨🇲", "Canada": "🇨🇦", "Croatia": "🇭🇷", "Denmark": "🇩🇰",
    "Ecuador": "🇪🇨", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪",
    "Ghana": "🇬🇭", "Iran": "🇮🇷", "Italy": "🇮🇹", "Japan": "🇯🇵",
    "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱", "Portugal": "🇵🇹",
    "Qatar": "🇶🇦", "Saudi Arabia": "🇸🇦", "Senegal": "🇸🇳", "South Korea": "🇰🇷",
    "Spain": "🇪🇸", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "USA": "🇺🇸",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Colombia": "🇨🇴",
    "Chile": "🇨🇱", "Poland": "🇵🇱", "Serbia": "🇷🇸", "Sweden": "🇸🇪",
    "Nigeria": "🇳🇬", "Egypt": "🇪🇬", "Algeria": "🇩🇿", "Peru": "🇵🇪",
    "Costa Rica": "🇨🇷", "Jamaica": "🇯🇲", "Panama": "🇵🇦", "New Zealand": "🇳🇿",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Turkey": "🇹🇷", "Austria": "🇦🇹", "Norway": "🇳🇴",
    "Ukraine": "🇺🇦", "Ivory Coast": "🇨🇮", "Cote d'Ivoire": "🇨🇮",
}

# --- Marquee player cards ------------------------------------------------
# Each entry points at a photo file in frontend/assets/players/ when available;
# falls back to a CSS jersey-color initials badge if the file is missing.
MARQUEE_PLAYERS = [
    {"name": "Lionel Messi", "initials": "LM", "team": "Argentina", "number": "10", "color": "#75AADB", "photo": "messi_argentina.jpeg"},
    {"name": "Cristiano Ronaldo", "initials": "CR", "team": "Portugal", "number": "7", "color": "#046A38", "photo": None},
    {"name": "Kylian Mbappé", "initials": "KM", "team": "France", "number": "10", "color": "#0055A4", "photo": None},
    {"name": "Neymar Jr.", "initials": "NJ", "team": "Brazil", "number": "10", "color": "#FFD700", "photo": None},
]


def flag_for(team: str) -> str:
    """Return a flag emoji for a team name, falling back to a generic globe."""
    return FLAGS.get(team, "🌍")


@st.cache_data(show_spinner=False)
def player_image_data_uri(filename: str) -> str | None:
    """Base64-encode a local asset image for inline CSS/HTML embedding, or None if missing."""
    path = PLAYERS_DIR / filename
    if not path.exists():
        return None
    ext = path.suffix.lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def inject_brand_css() -> None:
    """Inject the shared FIFA World Cup 2026 brand stylesheet (call once per page)."""
    trophy_uri = player_image_data_uri("trophy_wc26.jpeg") or ""

    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

            :root {{
                --fifa-header-logo: url('{trophy_uri}');
            }}

            .stApp {{
                background: {PAPER};
                color: {TEXT_BODY};
                font-family: {FONT_BODY};
            }}

            /* Primary buttons: bold, high-contrast white text on the FIFA-blue fill */
            div[data-testid="stButton"] > button[kind="primary"] {{
                font-weight: 800 !important;
                letter-spacing: 0.02em;
            }}
            div[data-testid="stButton"] > button[kind="primary"] p {{
                font-weight: 800 !important;
                color: #ffffff !important;
            }}

            /* Make Streamlit's default text/headers dark on the light background */
            [data-testid="stMarkdownContainer"], p, span, label, li {{
                color: {INK};
            }}

            section[data-testid="stSidebar"] {{
                background: {PAPER_ALT};
                border-right: 1px solid {BORDER};
            }}
            section[data-testid="stSidebar"] * {{
                color: {INK} !important;
            }}

            h1, h2, h3, h4 {{
                font-family: {FONT_DISPLAY} !important;
                color: {INK} !important;
            }}

            /* Native Streamlit header/toolbar is kept transparent and minimal --
               the custom FIFA-style header (render_top_header) is the real nav now. */
            [data-testid="stHeader"] {{
                background: transparent !important;
                height: 2.5rem !important;
            }}

            /* Promo strip (black banner, official FIFA.com furniture) */
            .promo-strip {{
                background: {PROMO_BLACK};
                border-radius: 10px;
                padding: 14px 22px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 22px;
                flex-wrap: wrap;
            }}
            .promo-left {{ display: flex; align-items: center; gap: 14px; }}
            .promo-badge {{
                width: 38px; height: 38px;
                border-radius: 7px;
                background: #0e0e0e;
                display: flex; align-items: center; justify-content: center;
                padding: 2px;
                flex-shrink: 0;
            }}
            .promo-badge img {{ width: 100%; height: 100%; object-fit: contain; }}
            .promo-title {{ color: #fff; font-weight: 700; font-size: 0.94rem; font-family: {FONT_BODY}; }}
            .promo-sub {{ color: rgba(255,255,255,0.55); font-size: 0.8rem; }}

            /* Match carousel strip (circular flag badges) */
            .match-carousel {{
                display: flex;
                gap: 22px;
                overflow-x: auto;
                padding: 4px 2px 14px 2px;
                margin-bottom: 22px;
            }}
            .match-chip {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 6px;
                flex-shrink: 0;
                width: 76px;
            }}
            .match-chip-badge {{
                width: 52px; height: 52px;
                border-radius: 50%;
                border: 2px solid {BORDER_STRONG};
                display: flex; align-items: center; justify-content: center;
                font-size: 1.35rem;
                position: relative;
                background: {SURFACE_ALT};
            }}
            .match-chip.is-live .match-chip-badge {{ border-color: {WINE}; }}
            .match-chip-live-tag {{
                position: absolute;
                bottom: -8px;
                left: 50%;
                transform: translateX(-50%);
                background: {WINE};
                color: #fff;
                font-family: {FONT_MONO};
                font-size: 0.55rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                padding: 1px 6px;
                border-radius: 999px;
                white-space: nowrap;
            }}
            .match-chip-score {{
                font-family: {FONT_MONO};
                font-size: 0.7rem;
                font-weight: 600;
                color: {INK};
                text-align: center;
                white-space: nowrap;
            }}

            /* Official-style hero banner (FIFA blue ground) */
            .fifa-hero {{
                background:
                    radial-gradient(120% 160% at 100% 0%, rgba(200,162,74,0.18) 0%, transparent 55%),
                    linear-gradient(120deg, {FIFA_BLUE} 0%, #062b6b 70%, #041c47 130%);
                border-radius: 10px;
                padding: 34px 36px;
                margin-bottom: 28px;
                position: relative;
                overflow: hidden;
            }}
            .fifa-hero .kicker {{
                display: inline-block;
                color: {GOLD};
                font-family: {FONT_MONO};
                font-weight: 700;
                font-size: 0.72rem;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                margin-bottom: 10px;
            }}
            .fifa-hero h1 {{
                color: #ffffff !important;
                font-size: 2.4rem;
                font-weight: 800;
                margin: 0 0 10px 0;
                letter-spacing: -0.01em;
            }}
            .fifa-hero p {{
                color: rgba(255,255,255,0.8);
                font-size: 1.05rem;
                max-width: 780px;
                line-height: 1.55;
                margin: 0;
            }}
            .fifa-badge-row {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 20px;
            }}
            .fifa-badge {{
                background-color: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.22);
                color: #ffffff;
                padding: 6px 14px;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
            }}
            .fifa-badge.live {{ color: #6be3a6; border-color: #6be3a6; }}
            .fifa-badge.gold {{ color: {GOLD}; border-color: {GOLD}; }}

            /* Stat / KPI cards — accent hairline + filled icon tile + sparkline/progress footer */
            .fifa-card {{
                position: relative;
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 14px;
                padding: 20px 22px 18px;
                margin-bottom: 18px;
                overflow: hidden;
                transition: transform 0.2s cubic-bezier(.2,.8,.3,1), box-shadow 0.2s ease, border-color 0.2s ease;
            }}
            .fifa-card::before {{
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 3px;
                background: linear-gradient(90deg, var(--kpi-accent, {FIFA_BLUE}), transparent 140%);
                opacity: 0.9;
            }}
            .fifa-card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 8px 24px -6px rgba(0,0,0,0.12);
                border-color: var(--kpi-accent, {FIFA_BLUE});
            }}
            .fifa-card .icon-row {{ display: flex; align-items: flex-start; justify-content: flex-end; margin-bottom: 10px; }}
            .fifa-card .icon-tile {{ display: none; }}
            .fifa-card .trend-pill {{
                font-family: {FONT_MONO};
                font-size: 0.64rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                color: var(--kpi-accent, {FIFA_BLUE});
                background: color-mix(in srgb, var(--kpi-accent, {FIFA_BLUE}) 12%, transparent);
                padding: 3px 9px;
                border-radius: 999px;
            }}
            .fifa-card .label {{
                color: {TEXT_MUTED};
                font-size: 0.92rem;
                text-transform: uppercase;
                font-weight: 700;
                letter-spacing: 0.06em;
            }}
            .fifa-card .value {{
                color: {INK};
                font-size: 2.7rem;
                font-weight: 800;
                font-family: {FONT_MONO};
                letter-spacing: -0.01em;
                margin-top: 6px;
                line-height: 1;
            }}
            .fifa-card .delta {{
                font-size: 0.95rem;
                font-weight: 600;
                margin-top: 8px;
                margin-bottom: 14px;
            }}
            .kpi-spark {{
                display: flex;
                align-items: flex-end;
                gap: 3px;
                height: 20px;
                margin-top: 12px;
            }}
            .kpi-spark i {{
                display: block;
                flex: 1;
                border-radius: 2px 2px 0 0;
                background: var(--kpi-accent, {FIFA_BLUE});
                opacity: 0.22;
            }}
            .kpi-spark i:last-child {{ opacity: 1; }}
            .kpi-progress-track {{
                height: 5px;
                border-radius: 3px;
                background: {SURFACE_ALT};
                border: 1px solid {BORDER};
                overflow: hidden;
                margin-top: 12px;
            }}
            .kpi-progress-fill {{
                height: 100%;
                border-radius: 3px;
                background: var(--kpi-accent, {FIFA_BLUE});
            }}

            /* Match card (fixture style) */
            .match-card {{
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 14px;
                padding: 26px 30px;
                margin: 10px 0 24px 0;
            }}
            .match-teams {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
            }}
            .match-team {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
                flex: 1;
            }}
            .match-team .flag {{
                font-size: 3rem;
                line-height: 1;
            }}
            .match-team .name {{
                font-family: {FONT_DISPLAY};
                font-weight: 700;
                font-size: 1.15rem;
                color: {INK};
                text-align: center;
            }}
            .match-vs {{
                font-family: {FONT_DISPLAY};
                font-weight: 800;
                color: {GOLD};
                font-size: 1.1rem;
                padding: 0 6px;
            }}

            .section-title {{
                font-family: {FONT_DISPLAY};
                font-weight: 700;
                font-size: 1.3rem;
                color: {INK};
                margin: 6px 0 14px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .section-title .accent-bar {{
                width: 5px;
                height: 22px;
                background: linear-gradient(180deg, {FIFA_BLUE}, {GOLD});
                border-radius: 3px;
                display: inline-block;
            }}

            .fifa-footer {{
                text-align: center;
                padding: 36px 0 18px 0;
                font-size: 0.82rem;
                color: {TEXT_MUTED};
                border-top: 1px solid {BORDER};
                margin-top: 50px;
            }}

            hr.fifa-divider {{
                border: none;
                border-top: 1px solid {BORDER};
                margin: 28px 0;
            }}

            /* Marquee player cards (illustrated jersey badges) */
            .player-row {{
                display: flex;
                gap: 18px;
                flex-wrap: wrap;
                margin: 6px 0 10px 0;
            }}
            .player-card {{
                flex: 1;
                min-width: 150px;
                background: {SURFACE};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 18px 14px;
                text-align: center;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .player-card:hover {{
                transform: translateY(-3px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.08);
            }}
            .player-jersey {{
                width: 84px;
                height: 84px;
                margin: 0 auto 10px auto;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: {FONT_DISPLAY};
                font-weight: 800;
                font-size: 1.3rem;
                color: #ffffff;
                border: 3px solid {PAPER};
                box-shadow: 0 0 0 2px {BORDER};
                background-size: cover;
                background-position: center top;
            }}
            .player-name {{
                font-family: {FONT_DISPLAY};
                font-weight: 700;
                font-size: 0.95rem;
                color: {INK};
            }}
            .player-meta {{
                font-size: 0.78rem;
                color: {TEXT_MUTED};
                margin-top: 2px;
            }}

            /* --- Quick Match Predictor (Home page component) --- */
            .qmp {{
                background:
                    radial-gradient(ellipse 700px 300px at 15% 0%, rgba(10,59,140,0.05) 0%, transparent 60%),
                    radial-gradient(ellipse 600px 300px at 85% 100%, rgba(200,162,74,0.06) 0%, transparent 60%),
                    #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 22px;
                padding: 36px 40px 32px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(11,31,58,0.10), 0 2px 6px rgba(11,31,58,0.06);
            }}
            .qmp-subtitle {{ color: {TEXT_MUTED}; font-size: 0.96rem; margin: -8px 0 22px 0; }}
            .qmp-teams {{ display: flex; align-items: stretch; gap: 20px; margin-bottom: 22px; }}
            .qmp-team-card {{
                flex: 1;
                background: {SURFACE_ALT};
                border: 1px solid {BORDER};
                border-radius: 18px;
                padding: 22px 20px;
                text-align: center;
                transition: transform 0.22s cubic-bezier(.2,.8,.3,1), border-color 0.22s ease, box-shadow 0.22s ease;
            }}
            .qmp-team-card:hover {{
                transform: translateY(-4px);
                border-color: var(--qc-accent, {FIFA_BLUE});
                box-shadow: 0 16px 40px -16px var(--qc-accent, {FIFA_BLUE});
            }}
            .qmp-team-card .flag {{ font-size: 3.2rem; line-height: 1; margin-bottom: 8px; display: block; }}
            .qmp-team-card .name {{
                font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 1.5rem; color: {INK};
                text-transform: uppercase; letter-spacing: 0.01em; margin-bottom: 12px;
            }}
            .qmp-stat-row {{
                display: flex; align-items: center; justify-content: center; gap: 8px;
                font-size: 0.82rem; color: {TEXT_MUTED}; margin-top: 6px;
            }}
            .qmp-stat-row b {{ color: {INK}; font-family: {FONT_MONO}; font-weight: 700; }}
            .qmp-stat-badge {{
                font-family: {FONT_MONO}; font-size: 0.7rem; font-weight: 700;
                padding: 2px 9px; border-radius: 999px; color: var(--qc-accent, {FIFA_BLUE});
                background: color-mix(in srgb, var(--qc-accent, {FIFA_BLUE}) 14%, transparent);
                display: inline-block;
            }}
            .qmp-vs-col {{ display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; flex-shrink: 0; }}
            .qmp-vs-label {{ font-family: {FONT_DISPLAY}; font-weight: 800; color: {GOLD}; font-size: 1rem; }}
            .qmp-predict-btn-wrap div[data-testid="stButton"] > button[kind="primary"] {{
                background: linear-gradient(135deg, {FIFA_BLUE} 0%, #003d82 100%) !important;
                border-radius: 14px !important;
                border: none !important;
                padding: 0.9rem 1rem !important;
                box-shadow: 0 12px 32px -10px rgba(10,59,140,0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }}
            .qmp-prob-grid {{ display: flex; gap: 18px; margin: 26px 0 8px 0; }}
            .qmp-prob-card {{
                flex: 1;
                background: {SURFACE_ALT}; border: 1px solid {BORDER};
                border-radius: 16px; padding: 20px 14px; text-align: center; position: relative;
            }}
            .qmp-prob-card.leader {{ border-color: var(--qc-color); box-shadow: 0 0 0 1px var(--qc-color), 0 12px 30px -16px var(--qc-color); }}
            .qmp-ring-wrap {{ position: relative; width: 100px; height: 100px; margin: 0 auto 12px; }}
            .qmp-ring-wrap svg {{ transform: rotate(-90deg); }}
            .qmp-ring-bg {{ fill: none; stroke: {BORDER}; stroke-width: 8; }}
            .qmp-ring-fill {{ fill: none; stroke-width: 8; stroke-linecap: round; stroke: var(--qc-color); }}
            .qmp-ring-value {{
                position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                font-family: {FONT_MONO}; font-weight: 800; font-size: 1.5rem; color: {INK};
            }}
            .qmp-prob-label {{
                font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 0.8rem;
                letter-spacing: 0.07em; text-transform: uppercase; color: {TEXT_MUTED};
            }}
            @media (max-width: 900px) {{
                .qmp-teams {{ flex-direction: column; }}
                .qmp-prob-grid {{ flex-direction: column; }}
            }}
            @media (prefers-reduced-motion: reduce) {{
                .qmp-team-card {{ transition: none !important; }}
            }}

            /* --- Match Prediction page: match hero + win probability + comparison --- */
            .mp-hero {{
                position: relative; border-radius: 22px; overflow: hidden; padding: 40px 44px;
                background:
                    radial-gradient(ellipse 800px 400px at 20% 0%, rgba(31,124,224,0.22) 0%, transparent 60%),
                    radial-gradient(ellipse 700px 400px at 80% 100%, rgba(200,162,74,0.12) 0%, transparent 60%),
                    linear-gradient(150deg, #0B1F3A 0%, #071427 100%);
                color: #fff; margin-bottom: 22px;
            }}
            .mp-hero::before {{
                content: ""; position: absolute; inset: 0;
                background-image: repeating-linear-gradient(115deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 44px);
                pointer-events: none;
            }}
            .mp-kicker {{ position: relative; font-family: {FONT_MONO}; font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: {GOLD}; text-align: center; margin-bottom: 22px; }}
            .mp-matchup {{ position: relative; display: flex; align-items: center; gap: 20px; }}
            .mp-team {{ flex: 1; text-align: center; }}
            .mp-team .flag {{ font-size: 4.2rem; line-height: 1; margin-bottom: 10px; display: block; }}
            .mp-team .name {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 1.8rem; text-transform: uppercase; letter-spacing: 0.01em; margin-bottom: 12px; }}
            .mp-team-stats {{ display: flex; flex-direction: column; gap: 6px; align-items: center; }}
            .mp-stat-badge {{ font-family: {FONT_MONO}; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; color: var(--qc-color); background: color-mix(in srgb, var(--qc-color) 18%, transparent); }}
            .mp-stat-line {{ font-size: 0.8rem; color: rgba(255,255,255,0.6); }}
            .mp-stat-line b {{ color: #fff; font-family: {FONT_MONO}; font-weight: 700; }}
            .mp-vs {{ font-family: {FONT_DISPLAY}; font-weight: 800; color: {GOLD}; font-size: 1.2rem; flex-shrink: 0; }}

            .mp-wp-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 20px; padding: 30px 34px; box-shadow: 0 8px 24px rgba(11,31,58,0.10); margin-bottom: 22px; }}
            .mp-wp-title {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 22px; text-align: center; color: {INK}; }}
            .mp-wp-grid {{ display: flex; gap: 22px; }}
            .mp-wp-ring-card {{ flex: 1; text-align: center; }}
            .mp-ring-wrap {{ position: relative; width: 140px; height: 140px; margin: 0 auto 12px; }}
            .mp-ring-wrap svg {{ transform: rotate(-90deg); }}
            .mp-ring-bg {{ fill: none; stroke: {BORDER}; stroke-width: 10; }}
            .mp-ring-fill {{ fill: none; stroke-width: 10; stroke-linecap: round; }}
            .mp-ring-value {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: {FONT_MONO}; font-weight: 800; font-size: 1.9rem; color: {INK}; }}
            .mp-ring-label {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.05em; color: {TEXT_MUTED}; }}
            .mp-conf-row {{ display: flex; align-items: center; justify-content: center; gap: 14px; margin-top: 24px; padding-top: 20px; border-top: 1px solid {BORDER}; flex-wrap: wrap; }}
            .mp-conf-label {{ font-family: {FONT_MONO}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: {TEXT_MUTED}; }}
            .mp-conf-value {{ font-family: {FONT_MONO}; font-weight: 800; font-size: 1.3rem; color: {EMERALD}; }}
            .mp-conf-track {{ width: 180px; height: 8px; border-radius: 4px; background: {SURFACE_ALT}; overflow: hidden; }}
            .mp-conf-fill {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, {EMERALD}, #5FD69A); }}

            .mp-cmp-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 18px; padding: 24px 26px; box-shadow: 0 1px 2px rgba(11,31,58,0.06); height: 100%; }}
            .mp-cmp-title {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 16px; color: {INK}; }}
            .mp-bar-row {{ display: grid; grid-template-columns: 56px 1fr 56px; align-items: center; gap: 10px; padding: 8px 0; }}
            .mp-bar-metric {{ grid-column: 1 / -1; font-size: 0.76rem; font-weight: 600; color: {TEXT_MUTED}; text-align: center; margin-bottom: 3px; }}
            .mp-bar-val {{ font-family: {FONT_MONO}; font-size: 0.76rem; font-weight: 700; }}
            .mp-bar-val.home {{ text-align: right; color: {EMERALD}; }}
            .mp-bar-val.away {{ text-align: left; color: {FIFA_BLUE_LIGHT}; }}
            .mp-track-pair {{ display: flex; gap: 3px; height: 8px; }}
            .mp-track-home, .mp-track-away {{ flex: 1; background: {SURFACE_ALT}; border-radius: 4px; overflow: hidden; display: flex; }}
            .mp-track-home {{ justify-content: flex-end; }}
            .mp-fill-home {{ height: 100%; background: {EMERALD}; border-radius: 4px 0 0 4px; }}
            .mp-fill-away {{ height: 100%; background: {FIFA_BLUE_LIGHT}; border-radius: 0 4px 4px 0; }}

            .mp-radar-legend {{ display: flex; gap: 20px; margin-top: 14px; font-size: 0.8rem; justify-content: center; }}
            .mp-radar-legend span {{ display: flex; align-items: center; gap: 6px; }}
            .mp-radar-legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}

            .mp-factor-row {{ display: grid; grid-template-columns: 170px 1fr 56px; align-items: center; gap: 14px; padding: 10px 0; }}
            .mp-factor-name {{ font-size: 0.88rem; font-weight: 600; color: {INK}; }}
            .mp-factor-track {{ height: 9px; background: {SURFACE_ALT}; border-radius: 5px; overflow: hidden; }}
            .mp-factor-fill {{ height: 100%; border-radius: 5px; }}
            .mp-factor-pct {{ font-family: {FONT_MONO}; font-size: 0.8rem; text-align: right; color: {TEXT_MUTED}; }}
            .mp-insight-card {{ background: {SURFACE_ALT}; border: 1px solid {BORDER}; border-left: 3px solid {FIFA_BLUE}; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; font-size: 0.88rem; color: {INK}; }}
            .mp-insight-card:last-child {{ margin-bottom: 0; }}

            .mp-predict-btn-wrap div[data-testid="stButton"] > button[kind="primary"] {{
                background: linear-gradient(135deg, {FIFA_BLUE} 0%, #003d82 100%) !important;
                border-radius: 14px !important;
                border: none !important;
                padding: 1rem 1rem !important;
                font-size: 1.05rem !important;
                box-shadow: 0 12px 32px -10px rgba(10,59,140,0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }}

            @media (max-width: 900px) {{
                .mp-matchup {{ flex-direction: column; }}
                .mp-wp-grid {{ flex-direction: column; }}
            }}
            @media (prefers-reduced-motion: reduce) {{
                .mp-hero, .mp-cmp-card {{ transition: none !important; }}
            }}

            /* --- Teams & Stats page: key-stat cards, form index, tournament strength, edge card --- */
            .ks-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
            .ks-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px; text-align: center; position: relative; overflow: hidden; }}
            .ks-card::before {{ content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: {BORDER_STRONG}; }}
            .ks-card.leader-home::before {{ background: {EMERALD}; }}
            .ks-card.leader-away::before {{ background: {FIFA_BLUE_LIGHT}; }}
            .ks-metric-label {{ font-family: {FONT_MONO}; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; color: {TEXT_MUTED}; margin-bottom: 10px; }}
            .ks-values {{ display: flex; align-items: center; justify-content: center; gap: 10px; }}
            .ks-val {{ font-family: {FONT_MONO}; font-weight: 800; font-size: 1.25rem; color: {INK}; }}
            .ks-val.leader {{ color: {EMERALD}; }}
            .ks-val.leader-blue {{ color: {FIFA_BLUE_LIGHT}; }}
            .ks-sep {{ color: {BORDER_STRONG}; font-size: 0.9rem; }}

            .form-row {{ display: flex; align-items: center; gap: 14px; padding: 14px 0; border-bottom: 1px solid {BORDER}; }}
            .form-row:last-child {{ border-bottom: none; }}
            .form-team-label {{ display: flex; align-items: center; gap: 8px; width: 140px; flex-shrink: 0; font-weight: 700; font-size: 0.9rem; color: {INK}; }}
            .form-track {{ flex: 1; height: 10px; border-radius: 5px; background: {SURFACE_ALT}; overflow: hidden; }}
            .form-fill {{ height: 100%; border-radius: 5px; }}
            .form-rating {{ font-family: {FONT_MONO}; font-weight: 700; font-size: 0.9rem; width: 60px; text-align: right; flex-shrink: 0; }}

            .ts-row {{ display: grid; grid-template-columns: 170px 1fr 1fr; gap: 16px; align-items: center; padding: 10px 0; }}
            .ts-label {{ font-size: 0.86rem; font-weight: 600; color: {INK}; }}
            .ts-bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
            .ts-bar-track {{ flex: 1; height: 9px; border-radius: 5px; background: {SURFACE_ALT}; overflow: hidden; }}
            .ts-bar-fill {{ height: 100%; border-radius: 5px; }}
            .ts-bar-pct {{ font-family: {FONT_MONO}; font-size: 0.78rem; font-weight: 700; width: 50px; text-align: right; }}

            .edge-card {{
                background: linear-gradient(160deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
                border: 1px solid {BORDER}; border-radius: 20px; padding: 30px 34px; text-align: center;
                box-shadow: 0 8px 24px rgba(11,31,58,0.10);
            }}
            .edge-label {{ font-family: {FONT_MONO}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: {TEXT_MUTED}; margin-bottom: 14px; }}
            .edge-team {{ display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 20px; }}
            .edge-team .flag {{ font-size: 2.6rem; }}
            .edge-team .name {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 1.7rem; text-transform: uppercase; color: {EMERALD}; }}
            .edge-factors {{ display: flex; flex-direction: column; gap: 8px; max-width: 360px; margin: 0 auto 22px; text-align: left; }}
            .edge-factor {{ display: flex; align-items: center; gap: 10px; font-size: 0.9rem; color: {INK}; }}
            .edge-factor .check {{ color: {EMERALD}; font-weight: 800; }}
            .edge-confidence {{ display: flex; align-items: center; justify-content: center; gap: 14px; padding-top: 20px; border-top: 1px solid {BORDER}; flex-wrap: wrap; }}
            .edge-conf-label {{ font-family: {FONT_MONO}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: {TEXT_MUTED}; }}
            .edge-conf-track {{ width: 180px; height: 8px; border-radius: 4px; background: {SURFACE_ALT}; overflow: hidden; }}
            .edge-conf-fill {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, {EMERALD}, #5FD69A); }}
            .edge-conf-value {{ font-family: {FONT_MONO}; font-weight: 800; font-size: 1.3rem; color: {EMERALD}; }}

            @media (max-width: 900px) {{
                .ks-grid {{ grid-template-columns: repeat(2, 1fr); }}
                .ts-row {{ grid-template-columns: 1fr; }}
            }}

            /* --- Historical Analytics page: edition explorer, goal analytics, H2H, reliability --- */
            .ha-hero {{
                position: relative; border-radius: 22px; overflow: hidden; padding: 40px 44px;
                background:
                    radial-gradient(ellipse 800px 400px at 20% 0%, rgba(31,124,224,0.20) 0%, transparent 60%),
                    radial-gradient(ellipse 700px 400px at 80% 100%, rgba(200,162,74,0.12) 0%, transparent 60%),
                    linear-gradient(150deg, #0B1F3A 0%, #071427 100%);
                color: #fff; margin-bottom: 26px;
            }}
            .ha-hero::before {{
                content: ""; position: absolute; inset: 0;
                background-image:
                    repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 64px),
                    repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 64px);
                pointer-events: none;
            }}
            .ha-hero-inner {{ position: relative; z-index: 1; max-width: 720px; }}
            .ha-hero-inner * {{ color: #ffffff; }}
            .ha-kicker {{ font-family: {FONT_MONO}; font-size: 0.85rem; letter-spacing: 0.18em; text-transform: uppercase; color: {GOLD} !important; margin: 0 0 12px; }}
            .ha-hero h1 {{ font-family: {FONT_DISPLAY} !important; font-size: clamp(1.7rem, 3vw, 2.3rem); font-weight: 800; margin: 0 0 12px; color: #ffffff !important; letter-spacing: -0.01em; }}
            .ha-hero p {{ font-size: 1.05rem; color: rgba(255,255,255,0.88) !important; margin: 0 0 22px; max-width: 560px; line-height: 1.55; }}
            .ha-hero-chips {{ display: flex; gap: 10px; flex-wrap: wrap; }}
            .ha-chip {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.18); border-radius: 999px; padding: 7px 16px; font-size: 0.92rem; font-family: {FONT_MONO}; color: #fff; }}
            .ha-chip b {{ color: {GOLD} !important; }}

            .ha-edition-tabs {{ display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
            .ha-edition-tab {{
                font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 1rem; padding: 9px 20px; border-radius: 8px;
                border: 1px solid {BORDER_STRONG}; background: {SURFACE}; color: {TEXT_MUTED};
            }}
            .ha-edition-tab.active {{ background: {FIFA_BLUE}; border-color: {FIFA_BLUE}; color: #fff; }}

            .ha-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 22px 24px; }}
            .ha-card h3 {{ font-family: {FONT_DISPLAY} !important; font-size: 1.12rem; margin: 0 0 16px; color: {INK} !important; }}

            .ha-goal-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }}
            .ha-goal-row:last-child {{ margin-bottom: 0; }}
            .ha-goal-row .g-label {{ width: 66px; flex-shrink: 0; font-family: {FONT_MONO}; font-size: 0.95rem; color: {TEXT_MUTED}; }}
            .ha-goal-track {{ flex: 1; height: 12px; background: {SURFACE_ALT}; border-radius: 6px; overflow: hidden; }}
            .ha-goal-fill {{ height: 100%; background: linear-gradient(90deg, {FIFA_BLUE}, {FIFA_BLUE_LIGHT}); border-radius: 6px; }}
            .ha-goal-fill.low {{ background: linear-gradient(90deg, #b5541f, #e0a45a); }}
            .ha-goal-row .g-val {{ width: 82px; flex-shrink: 0; text-align: right; font-family: {FONT_MONO}; font-weight: 700; font-size: 0.95rem; color: {INK}; }}

            .ha-donut-wrap {{ display: flex; align-items: center; gap: 24px; justify-content: center; padding: 6px 0; flex-wrap: wrap; }}
            .ha-donut-wrap svg {{ transform: rotate(-90deg); }}
            .ha-donut-legend {{ display: flex; flex-direction: column; gap: 10px; font-size: 0.98rem; color: {INK}; }}
            .ha-donut-legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; }}

            .ha-h2h-summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }}
            .ha-h2h-stat {{ background: {SURFACE_ALT}; border-radius: 10px; padding: 14px; text-align: center; }}
            .ha-h2h-stat .n {{ font-family: {FONT_DISPLAY}; font-size: 1.75rem; font-weight: 800; color: {INK}; }}
            .ha-h2h-stat .l {{ font-family: {FONT_MONO}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; color: {TEXT_MUTED}; }}

            .ha-match-row {{
                display: grid; grid-template-columns: 70px 1fr auto 1fr 150px; align-items: center; gap: 10px;
                padding: 12px 16px; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 9px; font-size: 1rem;
                margin-bottom: 8px;
            }}
            .ha-match-row:last-child {{ margin-bottom: 0; }}
            .ha-match-row .m-year {{ font-family: {FONT_MONO}; color: {TEXT_MUTED}; font-size: 0.92rem; }}
            .ha-match-row .m-team {{ text-align: right; font-weight: 700; color: {INK}; }}
            .ha-match-row .m-team.away {{ text-align: left; }}
            .ha-match-row .m-score {{ font-family: {FONT_MONO}; font-weight: 800; background: {SURFACE_ALT}; padding: 3px 12px; border-radius: 6px; color: {INK}; }}
            .ha-match-row .m-badge {{ font-family: {FONT_MONO}; font-size: 0.85rem; text-align: right; }}
            .ha-match-row .m-badge.correct {{ color: {EMERALD}; }}
            .ha-match-row .m-badge.wrong {{ color: {TEXT_MUTED}; }}

            .ha-conf-matrix {{ display: grid; grid-template-columns: auto repeat(3, 1fr); gap: 3px; font-family: {FONT_MONO}; font-size: 0.95rem; }}
            .ha-conf-cell {{ background: {SURFACE_ALT}; padding: 12px; text-align: center; border-radius: 5px; color: {INK}; }}
            .ha-conf-cell.head {{ background: transparent; color: {TEXT_MUTED}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.03em; }}
            .ha-conf-cell.diag {{ background: color-mix(in srgb, {EMERALD} 14%, transparent); font-weight: 800; color: {EMERALD}; }}
            .ha-conf-cell.row-head {{ text-align: right; color: {TEXT_MUTED}; font-size: 0.85rem; padding-right: 12px; background: transparent; }}

            .ha-insight-card {{ background: {SURFACE_ALT}; border: 1px solid {BORDER}; border-left: 3px solid {FIFA_BLUE}; border-radius: 0 10px 10px 0; padding: 16px 18px; margin-bottom: 10px; }}
            .ha-insight-card:last-child {{ margin-bottom: 0; }}
            .ha-insight-card .tag {{ font-family: {FONT_MONO}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: {FIFA_BLUE_LIGHT}; margin-bottom: 6px; display: block; }}
            .ha-insight-card p {{ margin: 0; font-size: 1.02rem; color: {INK}; }}

            .ha-note {{ font-size: 0.92rem; color: {TEXT_MUTED}; margin-top: 12px; }}

            @media (max-width: 900px) {{
                .ha-two-col {{ grid-template-columns: 1fr !important; }}
                .ha-h2h-summary {{ grid-template-columns: 1fr; }}
                .ha-match-row {{ grid-template-columns: 1fr; text-align: center; }}
                .ha-match-row .m-team.away, .ha-match-row .m-team {{ text-align: center; }}
                .ha-match-row .m-badge {{ text-align: center; }}
            }}

            /* --- AI Insights & Explainability: hero, feature importance, SHAP, confidence --- */
            .ai-hero {{
                position: relative; border-radius: 22px; overflow: hidden; padding: 48px 44px;
                background: linear-gradient(160deg, #0B1F3A 0%, #071427 100%);
                color: #fff; margin-bottom: 26px;
            }}
            .ai-hero-mesh {{
                position: absolute; inset: 0; pointer-events: none;
                background-image:
                    radial-gradient(1.5px 1.5px at 15% 20%, rgba(0,174,239,0.55) 50%, transparent 52%),
                    radial-gradient(1.5px 1.5px at 35% 65%, rgba(0,174,239,0.45) 50%, transparent 52%),
                    radial-gradient(1.5px 1.5px at 60% 30%, rgba(0,174,239,0.5) 50%, transparent 52%),
                    radial-gradient(1.5px 1.5px at 80% 70%, rgba(0,174,239,0.4) 50%, transparent 52%),
                    radial-gradient(1.5px 1.5px at 90% 15%, rgba(0,174,239,0.5) 50%, transparent 52%),
                    radial-gradient(1.5px 1.5px at 45% 85%, rgba(0,174,239,0.4) 50%, transparent 52%),
                    repeating-linear-gradient(115deg, rgba(0,174,239,0.05) 0px, rgba(0,174,239,0.05) 1px, transparent 1px, transparent 90px),
                    repeating-linear-gradient(25deg, rgba(0,174,239,0.04) 0px, rgba(0,174,239,0.04) 1px, transparent 1px, transparent 70px);
            }}
            .ai-hero-inner {{ position: relative; z-index: 1; max-width: 720px; }}
            .ai-hero-inner * {{ color: #ffffff; }}
            .ai-kicker {{ font-family: {FONT_MONO}; font-size: 0.85rem; letter-spacing: 0.14em; text-transform: uppercase; color: #00AEEF !important; margin: 0 0 12px; }}
            .ai-hero h1 {{ font-family: {FONT_DISPLAY} !important; font-size: clamp(1.7rem, 3vw, 2.4rem); font-weight: 800; margin: 0 0 12px; color: #ffffff !important; letter-spacing: -0.01em; }}
            .ai-hero p {{ font-size: 1.05rem; color: rgba(255,255,255,0.88) !important; margin: 0 0 22px; max-width: 580px; line-height: 1.55; }}
            .ai-hero-chips {{ display: flex; gap: 10px; flex-wrap: wrap; }}
            .ai-chip {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(0,174,239,0.35); border-radius: 999px; padding: 7px 16px; font-size: 0.92rem; font-family: {FONT_MONO}; color: #fff; }}
            .ai-chip b {{ color: #00AEEF !important; }}

            .ai-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 22px 24px; }}
            .ai-card h3 {{ font-family: {FONT_DISPLAY} !important; font-size: 1.12rem; margin: 0 0 4px; color: {INK} !important; }}
            .ai-card .card-sub {{ font-size: 0.96rem; color: {TEXT_MUTED}; margin: 0 0 16px; }}

            .fi-row {{ display: grid; grid-template-columns: 200px 1fr 70px; align-items: center; gap: 12px; padding: 9px 0; }}
            .fi-rank {{ font-family: {FONT_MONO}; color: #00AEEF; font-weight: 700; font-size: 0.9rem; width: 22px; display: inline-block; }}
            .fi-name {{ font-size: 0.98rem; font-weight: 600; color: {INK}; }}
            .fi-track {{ height: 14px; background: {SURFACE_ALT}; border-radius: 7px; overflow: hidden; }}
            .fi-fill {{ height: 100%; border-radius: 7px; background: linear-gradient(90deg, {FIFA_BLUE}, #00AEEF); }}
            .fi-pct {{ font-family: {FONT_MONO}; font-weight: 700; font-size: 0.92rem; text-align: right; color: {INK}; }}

            .shap-legend {{ display: flex; gap: 18px; margin-bottom: 14px; font-size: 0.95rem; color: {INK}; }}
            .shap-legend span {{ display: flex; align-items: center; gap: 6px; }}
            .shap-legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}

            .shap-bar-row {{ display: grid; grid-template-columns: 170px 1fr 72px; align-items: center; gap: 10px; padding: 7px 0; }}
            .shap-bar-name {{ font-size: 0.95rem; color: {INK}; }}
            .shap-bar-track {{ position: relative; height: 15px; background: {SURFACE_ALT}; border-radius: 4px; }}
            .shap-bar-center {{ position: absolute; left: 50%; top: -3px; bottom: -3px; width: 1px; background: {BORDER_STRONG}; }}
            .shap-bar-fill {{ position: absolute; top: 0; height: 100%; border-radius: 3px; }}
            .shap-bar-fill.pos {{ background: {EMERALD}; left: 50%; }}
            .shap-bar-fill.neg {{ background: {RED_CARD}; right: 50%; }}
            .shap-bar-val {{ font-family: {FONT_MONO}; font-size: 0.9rem; font-weight: 700; text-align: right; }}
            .shap-bar-val.pos {{ color: {EMERALD}; }}
            .shap-bar-val.neg {{ color: {RED_CARD}; }}

            .shap-impact-cards {{ display: flex; flex-direction: column; gap: 10px; }}
            .shap-impact-card {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 14px; border-radius: 10px; background: {SURFACE_ALT}; border-left: 3px solid {EMERALD}; }}
            .shap-impact-card.neg {{ border-left-color: {RED_CARD}; }}
            .shap-impact-card .name {{ font-size: 0.98rem; font-weight: 600; color: {INK}; }}
            .shap-impact-card .val {{ font-family: {FONT_MONO}; font-weight: 700; font-size: 0.96rem; }}
            .shap-impact-card.pos .val {{ color: {EMERALD}; }}
            .shap-impact-card.neg .val {{ color: {RED_CARD}; }}

            .explain-card {{ background: linear-gradient(160deg, #0B1F3A 0%, #071427 100%); border-radius: 20px; padding: 32px 36px; color: #ffffff; }}
            .explain-card * {{ color: #ffffff; }}
            .explain-teams {{ display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
            .explain-team {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 1.4rem; text-transform: uppercase; color: #ffffff; }}
            .explain-vs {{ color: {GOLD} !important; font-family: {FONT_MONO}; font-weight: 700; }}
            .explain-verdict {{ text-align: center; margin-bottom: 22px; }}
            .explain-verdict .pred {{ font-family: {FONT_DISPLAY}; font-size: 1.2rem; color: #00AEEF !important; font-weight: 700; }}
            .explain-verdict .conf {{ font-family: {FONT_MONO}; font-size: 2rem; font-weight: 800; margin-top: 4px; color: #ffffff; }}
            .explain-reasons {{ display: flex; flex-direction: column; gap: 10px; max-width: 640px; margin: 0 auto; }}
            .explain-reason {{ display: flex; gap: 10px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16); border-radius: 10px; padding: 12px 16px; font-size: 1rem; color: #ffffff; }}
            .explain-reason .ico {{ color: #00AEEF !important; flex-shrink: 0; }}

            .conf-ring-card {{ text-align: center; padding: 18px; }}
            .conf-ring-wrap {{ width: 130px; height: 130px; margin: 0 auto 10px; position: relative; }}
            .conf-ring-wrap svg {{ transform: rotate(-90deg); }}
            .conf-ring-value {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-family: {FONT_MONO}; font-weight: 800; font-size: 1.5rem; color: {INK}; }}
            .conf-ring-label {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 0.96rem; text-transform: uppercase; letter-spacing: 0.04em; color: {TEXT_MUTED}; }}

            .calib-bin-row {{ display: grid; grid-template-columns: 110px 1fr 1fr 76px; align-items: center; gap: 10px; padding: 7px 0; font-size: 0.92rem; }}
            .calib-bin-row .bin-range {{ font-family: {FONT_MONO}; color: {TEXT_MUTED}; font-size: 0.86rem; }}
            .calib-track {{ height: 10px; background: {SURFACE_ALT}; border-radius: 5px; overflow: hidden; }}
            .calib-fill {{ height: 100%; border-radius: 5px; }}
            .calib-fill.conf {{ background: {FIFA_BLUE_LIGHT}; }}
            .calib-fill.acc {{ background: {GOLD}; }}
            .calib-dev {{ font-family: {FONT_MONO}; font-weight: 700; text-align: right; font-size: 0.9rem; }}
            .calib-dev.over {{ color: {EMERALD}; }}
            .calib-dev.under {{ color: {RED_CARD}; }}
            .calib-legend {{ display: flex; gap: 18px; margin-bottom: 12px; font-size: 0.92rem; color: {INK}; }}
            .calib-legend span {{ display: flex; align-items: center; gap: 6px; }}
            .calib-legend i {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; }}

            .trust-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }}
            .trust-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; }}
            .trust-card .t-label {{ font-family: {FONT_MONO}; font-size: 0.78rem; text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 6px; }}
            .trust-card .t-value {{ font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 1.12rem; color: {INK}; }}

            @media (max-width: 900px) {{
                .fi-row {{ grid-template-columns: 130px 1fr 56px; }}
                .shap-bar-row {{ grid-template-columns: 110px 1fr 56px; }}
                .calib-bin-row {{ grid-template-columns: 70px 1fr 1fr 56px; }}
            }}

            /* --- Tournament Simulator page: hero, control center, ranking, bracket, final, matrix, upsets --- */
            .tsim-hero {{
                position: relative; border-radius: 22px; overflow: hidden; padding: 52px 44px;
                background: linear-gradient(160deg, #0B1F3A 0%, #071427 100%);
                color: #fff; margin-bottom: 26px;
            }}
            .tsim-hero-mesh {{
                position: absolute; inset: 0; pointer-events: none;
                background-image:
                    radial-gradient(1.6px 1.6px at 12% 25%, rgba(200,162,74,0.5) 50%, transparent 52%),
                    radial-gradient(1.6px 1.6px at 30% 70%, rgba(0,174,239,0.4) 50%, transparent 52%),
                    radial-gradient(1.6px 1.6px at 55% 20%, rgba(200,162,74,0.4) 50%, transparent 52%),
                    radial-gradient(1.6px 1.6px at 78% 65%, rgba(0,174,239,0.45) 50%, transparent 52%),
                    radial-gradient(1.6px 1.6px at 92% 30%, rgba(200,162,74,0.4) 50%, transparent 52%),
                    repeating-linear-gradient(60deg, rgba(255,255,255,0.03) 0, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 70px),
                    repeating-linear-gradient(-30deg, rgba(255,255,255,0.025) 0, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 90px);
            }}
            .tsim-hero-inner {{ position: relative; z-index: 1; max-width: 760px; }}
            .tsim-hero-inner * {{ color: #ffffff; }}
            .tsim-kicker {{ font-family: {FONT_MONO}; font-size: 0.85rem; letter-spacing: 0.16em; text-transform: uppercase; color: {GOLD} !important; margin: 0 0 12px; }}
            .tsim-hero h1 {{ font-family: {FONT_DISPLAY} !important; font-size: clamp(1.9rem, 3.4vw, 2.7rem); font-weight: 800; margin: 0 0 14px; color: #ffffff !important; letter-spacing: -0.01em; }}
            .tsim-hero p {{ font-size: 1.05rem; color: rgba(255,255,255,0.88) !important; margin: 0 0 26px; max-width: 600px; }}
            .tsim-hero-chips {{ display: flex; gap: 10px; flex-wrap: wrap; }}
            .tsim-chip {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(0,174,239,0.35); border-radius: 999px; padding: 7px 16px; font-size: 0.92rem; font-family: {FONT_MONO}; color: #fff; }}
            .tsim-chip b {{ color: #00AEEF !important; }}

            .tsim-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 22px 24px; }}
            .tsim-card h3 {{ font-family: {FONT_DISPLAY} !important; font-size: 1.08rem; margin: 0 0 4px; color: {INK} !important; }}
            .tsim-card .card-sub {{ font-size: 0.94rem; color: {TEXT_MUTED}; margin: 0 0 14px; }}

            .tsim-champ-row {{
                display: grid; grid-template-columns: 34px 1fr 1fr 74px; align-items: center; gap: 14px;
                padding: 12px 16px; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; margin-bottom: 10px;
            }}
            .tsim-champ-row:last-child {{ margin-bottom: 0; }}
            .tsim-champ-row.rank-1 {{ border-color: {GOLD}; }}
            .tsim-champ-rank {{ font-family: {FONT_MONO}; font-weight: 800; color: {TEXT_MUTED}; }}
            .tsim-champ-row.rank-1 .tsim-champ-rank {{ color: {GOLD_LIGHT}; }}
            .tsim-champ-team {{ display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1rem; color: {INK}; }}
            .tsim-champ-track {{ height: 12px; background: {SURFACE_ALT}; border-radius: 6px; overflow: hidden; }}
            .tsim-champ-fill {{ height: 100%; border-radius: 6px; background: linear-gradient(90deg, {GOLD}, #e0c068); }}
            .tsim-champ-pct {{ font-family: {FONT_MONO}; font-weight: 800; text-align: right; color: {INK}; }}

            .tsim-bracket-scroll {{ overflow-x: auto; padding: 8px 0 16px; }}
            .tsim-bracket {{ display: flex; gap: 22px; min-width: 900px; }}
            .tsim-bracket-round {{ display: flex; flex-direction: column; justify-content: space-around; gap: 12px; flex: 1; min-width: 160px; }}
            .tsim-bracket-round-label {{ font-family: {FONT_MONO}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; color: {TEXT_MUTED}; text-align: center; margin-bottom: 4px; }}
            .tsim-bracket-match {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 10px; font-size: 0.86rem; }}
            .tsim-bracket-team {{ display: flex; justify-content: space-between; padding: 3px 0; color: {INK}; }}
            .tsim-bracket-team.winner {{ font-weight: 700; color: {EMERALD}; }}
            .tsim-bracket-prob {{ font-family: {FONT_MONO}; font-size: 0.78rem; color: {TEXT_MUTED}; }}

            .tsim-final-card {{ background: linear-gradient(160deg, #0B1F3A 0%, #071427 100%); border-radius: 20px; padding: 34px 38px; color: #ffffff; text-align: center; }}
            .tsim-final-card * {{ color: #ffffff; }}
            .tsim-final-teams {{ display: flex; align-items: center; justify-content: center; gap: 26px; margin-bottom: 18px; flex-wrap: wrap; }}
            .tsim-final-team {{ font-family: {FONT_DISPLAY}; font-weight: 800; font-size: 1.6rem; text-transform: uppercase; }}
            .tsim-final-vs {{ color: {GOLD} !important; font-family: {FONT_MONO}; font-weight: 700; }}
            .tsim-final-pct {{ font-family: {FONT_MONO}; font-size: 2.2rem; font-weight: 800; }}
            .tsim-final-label {{ font-family: {FONT_MONO}; font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.05em; color: rgba(255,255,255,0.72) !important; margin-top: 4px; }}

            .tsim-matrix-wrap {{ overflow-x: auto; border: 1px solid {BORDER}; border-radius: 12px; }}

            .tsim-upset-card {{
                background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {GOLD};
                border-radius: 0 10px 10px 0; padding: 16px 18px;
            }}
            .tsim-upset-card .tag {{ font-family: {FONT_MONO}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: {GOLD_LIGHT}; display: block; margin-bottom: 8px; }}
            .tsim-upset-card h4 {{ font-family: {FONT_DISPLAY} !important; margin: 0 0 6px; font-size: 1.08rem; color: {INK} !important; }}
            .tsim-upset-card p {{ margin: 0; font-size: 0.94rem; color: {TEXT_MUTED}; }}
            .tsim-upset-card .delta {{ font-family: {FONT_MONO}; font-weight: 700; color: {EMERALD}; }}

            .tsim-insight-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {FIFA_BLUE}; border-radius: 0 10px 10px 0; padding: 16px 18px; }}
            .tsim-insight-card .tag {{ font-family: {FONT_MONO}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: {FIFA_BLUE_LIGHT}; display: block; margin-bottom: 8px; }}
            .tsim-insight-card p {{ margin: 0; font-size: 0.94rem; color: {INK}; }}

            .tsim-hist-wrap {{ display: flex; align-items: flex-end; gap: 3px; height: 140px; padding: 10px; background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
            .tsim-hist-bar {{ flex: 1; background: linear-gradient(180deg, #00AEEF, {FIFA_BLUE}); border-radius: 2px 2px 0 0; min-width: 2px; }}

            @media (max-width: 900px) {{
                .tsim-champ-row {{ grid-template-columns: 26px 1fr 60px; }}
                .tsim-champ-track {{ display: none; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, subtitle: str, badges: list[tuple[str, str, str]] | None = None) -> None:
    """Render a shared official-style hero banner.

    Args:
        kicker: Small uppercase eyebrow label above the title.
        title: Main heading (emoji allowed).
        subtitle: Supporting description text.
        badges: Optional list of (label, css_class, _unused) tuples rendered as pills.
    """
    badges_html = ""
    if badges:
        pills = "".join(
            f'<span class="fifa-badge {cls}">{label}</span>' for label, cls, _ in badges
        )
        badges_html = f'<div class="fifa-badge-row">{pills}</div>'

    st.markdown(
        f"""
        <div class="fifa-hero">
            <span class="kicker">{kicker}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {badges_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    """Render a consistent section header with an accent bar."""
    st.markdown(
        f'<div class="section-title"><span class="accent-bar"></span>{text}</div>',
        unsafe_allow_html=True,
    )


def render_hero_image_banner(filename: str, caption: str = "", max_height: str = "320px") -> bool:
    """Render a real photo (e.g. the stars collage) below the hero, if the asset file exists.

    Returns True if the image was rendered, False if the asset was missing
    (callers can skip gracefully rather than showing a broken image).
    """
    uri = player_image_data_uri(filename)
    if not uri:
        return False
    cap_html = f'<div style="text-align:center; color:{TEXT_MUTED}; font-size:0.8rem; margin-top:8px;">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div style="width:100%; height:{max_height}; border:1px solid {BORDER}; border-radius:12px; overflow:hidden; margin-bottom:20px; background:{SURFACE};">
            <img src="{uri}" style="width:100%; height:100%; object-fit:cover; object-position:center 20%; display:block;" />
        </div>
        {cap_html}
        """,
        unsafe_allow_html=True,
    )
    return True


def render_marquee_players() -> None:
    """Render a row of marquee player cards (Messi, Ronaldo, Mbappé, Neymar).

    Uses a real photo from frontend/assets/players/ when available, falling
    back to a CSS jersey-color initials badge otherwise.
    """
    card_html = []
    for p in MARQUEE_PLAYERS:
        photo_uri = player_image_data_uri(p["photo"]) if p.get("photo") else None
        if photo_uri:
            jersey_style = f"background-image:url('{photo_uri}');"
            jersey_content = ""
        else:
            jersey_style = f"background:{p['color']};"
            jersey_content = p["initials"]
        card_html.append(
            f'<div class="player-card">'
            f'<div class="player-jersey" style="{jersey_style}">{jersey_content}</div>'
            f'<div class="player-name">{p["name"]}</div>'
            f'<div class="player-meta">{flag_for(p["team"])} {p["team"]} · #{p["number"]}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="player-row">{"".join(card_html)}</div>', unsafe_allow_html=True)


def render_promo_strip(title: str, subtitle: str) -> None:
    """Render the black promo banner used on FIFA.com's own homepage, with the trophy badge."""
    trophy_uri = player_image_data_uri("trophy_wc26.jpeg")
    badge_html = (
        f'<div class="promo-badge"><img src="{trophy_uri}" alt="FIFA World Cup 2026 trophy" /></div>'
        if trophy_uri else '<div class="promo-badge" style="display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🏆</div>'
    )
    st.markdown(
        f"""
        <div class="promo-strip">
            <div class="promo-left">
                {badge_html}
                <div>
                    <div class="promo-title">{title}</div>
                    <div class="promo-sub">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_carousel(matches: list[dict]) -> None:
    """Render a horizontal scrollable strip of match chips (circular flag badges).

    Args:
        matches: list of dicts with keys: team (for flag lookup), score (display text),
                 and optional is_live (bool).
    """
    chips = []
    for m in matches:
        live_class = " is-live" if m.get("is_live") else ""
        live_tag = '<span class="match-chip-live-tag">LIVE</span>' if m.get("is_live") else ""
        chips.append(
            f'<div class="match-chip{live_class}">'
            f'<div class="match-chip-badge">{flag_for(m["team"])}{live_tag}</div>'
            f'<div class="match-chip-score">{m["score"]}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="match-carousel">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_kpi_card(
    icon_svg: str,
    trend_label: str,
    value: str,
    label: str,
    accent: str = FIFA_BLUE,
    unit: str = "",
    sparkline: list[int] | None = None,
    progress_pct: float | None = None,
) -> None:
    """Render an advanced KPI card: accent hairline, filled icon tile, trend pill,
    tabular-mono value, and a footer signal (sparkline for counts, progress bar for %).
    """
    unit_html = f'<span style="font-family:{FONT_MONO}; font-size:0.9rem; font-weight:600; color:{TEXT_MUTED};">{unit}</span>' if unit else ""

    footer_html = ""
    if sparkline:
        bars = "".join(f'<i style="height:{max(h, 8)}%;"></i>' for h in sparkline)
        footer_html = f'<div class="kpi-spark">{bars}</div>'
    elif progress_pct is not None:
        pct = max(0.0, min(100.0, progress_pct))
        footer_html = f'<div class="kpi-progress-track"><div class="kpi-progress-fill" style="width:{pct}%;"></div></div>'

    st.markdown(
        f"""
        <div class="fifa-card" style="--kpi-accent:{accent};">
            <div class="icon-row">
                <div class="icon-tile">{icon_svg}</div>
                <span class="trend-pill">{trend_label}</span>
            </div>
            <div class="value">{value}{unit_html}</div>
            <div class="delta" style="color:{TEXT_MUTED};">{label}</div>
            {footer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Top navigation header ------------------------------------------------
# Real page IA, mirroring the approved design proposal: 6 primary items plus
# a 7th "More" group for account-adjacent pages that don't fit the primary IA.
TOP_NAV_ITEMS = [
    {"label": "Dashboard", "url_path": "dashboard"},
    {"label": "Match Predictor", "url_path": "match-predictor"},
    {
        "label": "Teams & Stats",
        "children": [
            {"label": "Head-to-Head", "url_path": "head-to-head"},
            {"label": "Team Analytics", "url_path": "team-analytics"},
        ],
    },
    {"label": "Tournament Simulator", "url_path": "tournament-simulator"},
    {
        "label": "Analytics",
        "children": [
            {"label": "Historical Analytics", "url_path": "historical-analytics"},
        ],
    },
    {
        "label": "AI Insights",
        "children": [
            {"label": "AI Insights & Explainability", "url_path": "feature-importance"},
            {"label": "Model Performance", "url_path": "model-performance"},
        ],
    },
]

TOP_NAV_MORE_ITEMS = [
    {"label": "Prediction History", "url_path": "prediction-history"},
    {"label": "Settings", "url_path": "settings"},
]


def _nav_item_is_active(item: dict, current_url_path: str) -> bool:
    """Return True if this nav item (or one of its children) matches the current page."""
    if item.get("url_path") == current_url_path:
        return True
    return any(child.get("url_path") == current_url_path for child in item.get("children", []))


def render_top_header(current_url_path: str = "") -> None:
    """Render the FIFA-style top header: utility strip + black nav row with dropdowns.

    Real <a href="/{url_path}"> links drive navigation (Streamlit multipage apps
    serve each st.Page at /<url_path>), so this works without JS routing hacks --
    the tradeoff is a full page reload per navigation, same as any plain link.
    """
    trophy_uri = player_image_data_uri("trophy_wc26.jpeg")
    brand_mark_html = (
        f'<img src="{trophy_uri}" alt="FIFA World Cup 2026 trophy" />'
        if trophy_uri else "🏆"
    )

    caret_svg = '<svg class="caret" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 4l4 4 4-4"/></svg>'

    def render_item(item: dict) -> str:
        active = _nav_item_is_active(item, current_url_path)
        active_cls = " active" if active else ""
        children = item.get("children")
        if children:
            rows = "".join(
                f'<a href="/{c["url_path"]}"><span class="dot"></span>{c["label"]}</a>'
                for c in children
            )
            # The label itself links to the first child page (a real destination);
            # the caret is a separate toggle so opening the dropdown doesn't
            # require navigating away first. summary's native click-to-toggle
            # is suppressed on the link/caret via onclick handlers below.
            primary_url = children[0]["url_path"]
            return (
                f'<details class="nav-item{active_cls}">'
                f'<summary class="nav-item-btn" onclick="event.preventDefault();">'
                f'<a class="nav-item-label-link" href="/{primary_url}" onclick="event.stopPropagation();">{item["label"]}</a>'
                f'<span class="nav-caret-toggle" onclick="this.closest(\'details\').open = !this.closest(\'details\').open; event.stopPropagation();">{caret_svg}</span>'
                f'</summary>'
                f'<div class="nav-dropdown">{rows}</div>'
                f'</details>'
            )
        return f'<a class="nav-item nav-item-link{active_cls}" href="/{item["url_path"]}"><span class="nav-item-btn">{item["label"]}</span></a>'

    nav_items_html = "".join(render_item(item) for item in TOP_NAV_ITEMS)
    more_rows_html = "".join(
        f'<a href="/{item["url_path"]}"><span class="dot"></span>{item["label"]}</a>'
        for item in TOP_NAV_MORE_ITEMS
    )
    more_active = any(_nav_item_is_active(item, current_url_path) for item in TOP_NAV_MORE_ITEMS)
    more_dropdown_html = (
        f'<details class="nav-item{" active" if more_active else ""}">'
        f'<summary class="nav-item-btn">More {caret_svg}</summary>'
        f'<div class="nav-dropdown">{more_rows_html}</div>'
        f'</details>'
    )

    st.markdown(
        f"""
        <style>
            .fifa-topheader {{ margin: -1rem -1rem 1.5rem -1rem; }}
            .fifa-util-strip {{
                background: {FIFA_BLUE};
                display: flex; align-items: center; justify-content: flex-end; gap: 18px;
                padding: 7px 28px; font-size: 0.76rem; color: rgba(255,255,255,0.82);
                font-family: {FONT_BODY};
            }}
            .fifa-util-strip .sep {{ width: 1px; height: 12px; background: rgba(255,255,255,0.25); }}

            .fifa-main-nav {{
                background: {PROMO_BLACK};
                display: flex; align-items: center; gap: 8px;
                padding: 0 28px; height: 66px; position: relative;
            }}
            .fifa-nav-brand {{
                display: flex; align-items: center; gap: 12px; padding: 8px 20px 8px 8px; margin-right: 8px;
                border-right: 1px solid rgba(255,255,255,0.12); flex-shrink: 0;
            }}
            .fifa-nav-brand-mark {{
                width: 36px; height: 36px; border-radius: 8px; background: #ffffff;
                display: flex; align-items: center; justify-content: center;
                flex-shrink: 0; overflow: hidden; padding: 3px; font-size: 1.1rem;
            }}
            .fifa-nav-brand-mark img {{ width: 100%; height: 100%; object-fit: contain; }}
            .fifa-nav-brand-title {{
                font-family: {FONT_DISPLAY}; font-weight: 700; color: #fff; font-size: 1.02rem;
                letter-spacing: 0.02em; white-space: nowrap; text-transform: uppercase;
            }}
            .fifa-nav-brand-title sup {{ font-size: 0.5em; }}

            .fifa-nav-items {{
                display: flex; align-items: center; height: 100%; gap: 2px; flex: 1;
                flex-wrap: nowrap; min-width: 0; overflow-x: auto; overflow-y: hidden;
                scrollbar-width: none;
            }}
            .fifa-nav-items::-webkit-scrollbar {{ display: none; }}
            .nav-item {{ position: relative; height: 100%; display: flex; align-items: center; flex-shrink: 0; }}
            .nav-item[open] {{ }}
            details.nav-item {{ display: flex; }}
            details.nav-item summary::-webkit-details-marker {{ display: none; }}
            details.nav-item summary {{ list-style: none; }}
            .fifa-nav-items a.nav-item-link,
            .fifa-nav-items a.nav-item-link:hover,
            .fifa-nav-items a.nav-item-link:visited {{
                text-decoration: none !important;
            }}
            .nav-item-btn {{
                all: unset; box-sizing: border-box; cursor: pointer; display: flex; align-items: center; gap: 6px;
                height: 66px; padding: 0 16px; color: rgba(255,255,255,0.75);
                font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 0.92rem; letter-spacing: 0.01em;
                white-space: nowrap; flex-shrink: 0; position: relative;
                transition: color 0.15s ease, background 0.15s ease;
            }}
            .nav-item-btn:hover {{ color: #fff; background: rgba(255,255,255,0.06); }}
            .nav-item.active .nav-item-btn {{ color: #fff; }}
            .nav-item.active .nav-item-btn::after {{
                content: ""; position: absolute; left: 16px; right: 16px; bottom: 0; height: 3px;
                border-radius: 3px 3px 0 0; background: {GOLD};
            }}
            .nav-item-btn .caret {{ width: 9px; height: 9px; transition: transform 0.18s ease; opacity: 0.7; }}
            details.nav-item[open] .caret {{ transform: rotate(180deg); }}
            .nav-item-label-link {{
                color: inherit !important; text-decoration: none !important;
                cursor: pointer;
            }}
            .nav-caret-toggle {{
                display: flex; align-items: center; padding: 4px; margin: -4px -8px -4px 2px;
                cursor: pointer;
            }}

            .nav-dropdown {{
                position: absolute; top: 100%; left: 0; min-width: 230px;
                background: #fff; border: 1px solid {BORDER_STRONG}; border-radius: 10px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.18); padding: 8px; margin-top: 6px; z-index: 30;
            }}
            .nav-dropdown a {{
                display: flex; align-items: center; gap: 10px; padding: 9px 12px; border-radius: 7px;
                color: {INK} !important; text-decoration: none !important; font-size: 0.86rem; font-weight: 500;
                font-family: {FONT_BODY};
                transition: background 0.12s ease, padding-left 0.12s ease;
            }}
            .nav-dropdown a:hover {{ background: {SURFACE_ALT}; padding-left: 16px; color: {FIFA_BLUE}; }}
            .nav-dropdown a .dot {{ width: 5px; height: 5px; border-radius: 50%; background: {GOLD}; flex-shrink: 0; }}

            .fifa-nav-right {{ display: flex; align-items: center; gap: 4px; flex-shrink: 0; margin-left: auto; padding-left: 12px; }}

            @media (max-width: 900px) {{
                .fifa-util-strip {{ display: none; }}
                .fifa-nav-brand-title {{ display: none; }}
            }}
            @media (prefers-reduced-motion: reduce) {{
                .nav-item-btn, .caret {{ transition: none !important; }}
            }}
        </style>

        <div class="fifa-topheader">
            <div class="fifa-util-strip">
                <span>🌐 English ▾</span>
                <span class="sep"></span>
                <span>Help</span>
                <span class="sep"></span>
                <span>API Status</span>
            </div>
            <nav class="fifa-main-nav">
                <div class="fifa-nav-brand">
                    <div class="fifa-nav-brand-mark">{brand_mark_html}</div>
                    <div class="fifa-nav-brand-title">FIFA World Cup 2026<sup>&trade;</sup></div>
                </div>
                <div class="fifa-nav-items">{nav_items_html}{more_dropdown_html}</div>
            </nav>
        </div>
        """,
        unsafe_allow_html=True,
    )
