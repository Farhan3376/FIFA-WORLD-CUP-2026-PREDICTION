"""Phase 1 - Step 4: Feature engineering.

Loads ``data/interim/matches_merged.csv`` and produces two output files:

* ``data/interim/matches_features.csv``  — merged data + all engineered features.
* ``data/processed/matches_final.csv``   — feature-only ML-ready dataset.

Leakage-free design
-------------------
Every rolling or expanding statistic is computed with a ``shift(1)`` before
the aggregation window so the *current* match is never included in its own
features.  The core technique:

1. Build a "team perspective" table — each match appears twice (once per
   team), recording goals scored, goals conceded, and the team's result.
2. Sort by ``(team, date)`` and apply ``groupby + shift(1) + expanding /
   rolling`` to derive historical stats.
3. Merge home-team stats and away-team stats back onto the main match table
   using ``(date, team)`` join keys.

Features generated
------------------
**Rating features**
    ``elo_diff``, ``home_elo_before``, ``away_elo_before``,
    ``home_fifa_rank``, ``away_fifa_rank``, ``fifa_rank_diff``

**Historical performance (all past matches)**
    ``home_overall_win_pct``, ``home_draw_pct``, ``home_loss_pct``
    ``away_overall_win_pct``, ``away_draw_pct``, ``away_loss_pct``
    ``home_avg_goals_scored``, ``home_avg_goals_conceded``, ``home_avg_goal_diff``
    ``away_avg_goals_scored``, ``away_avg_goals_conceded``, ``away_avg_goal_diff``
    ``home_games_played``, ``away_games_played``

**Venue-split performance**
    ``home_home_win_pct``  (home team's record when playing at home)
    ``away_away_win_pct``  (away team's record when playing away)

**Recent form (last 5 matches)**
    ``home_form_5``, ``away_form_5``
    Points scale: Win=3, Draw=1, Loss=0, normalised to [0, 1].

**Temporal / contextual**
    ``home_rest_days``, ``away_rest_days``
    ``year``, ``month``, ``is_neutral``
    ``tournament_importance``, ``tournament_type``

Run directly::

    python -m src.feature_builder
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    COL_AWAY_GOALS,
    COL_AWAY_TEAM,
    COL_DATE,
    COL_HOME_GOALS,
    COL_HOME_TEAM,
    COL_NEUTRAL,
    COL_RESULT,
    COL_TOURNAMENT,
    INTERIM_DIR,
    PROCESSED_DIR,
    RECENT_FORM_WINDOW,
    REPORTS_DIR,
    Settings,
    ensure_directories,
    load_settings,
    set_global_seed,
    RESULT_HOME_WIN,
    RESULT_DRAW,
    RESULT_AWAY_WIN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tournament importance weights (used as a continuous feature, not for Elo).
# ---------------------------------------------------------------------------
_TOURNAMENT_IMPORTANCE: Dict[str, float] = {
    "fifa world cup": 1.00,
    "confederations cup": 0.85,
    "copa america": 0.85,
    "africa cup of nations": 0.85,
    "uefa euro": 0.85,
    "gold cup": 0.75,
    "afc asian cup": 0.75,
    "ofc nations cup": 0.75,
    "olympic games": 0.70,
    "fifa world cup qualification": 0.65,
    "copa america qualification": 0.55,
    "uefa euro qualification": 0.55,
    "african cup of nations qualification": 0.55,
    "friendly": 0.20,
}
_DEFAULT_IMPORTANCE = 0.45

# Broad tournament category (used for one-hot / ordinal encoding later).
_TOURNAMENT_TYPE: Dict[str, str] = {
    "fifa world cup": "world_cup",
    "confederations cup": "continental_cup",
    "copa america": "continental_cup",
    "africa cup of nations": "continental_cup",
    "uefa euro": "continental_cup",
    "gold cup": "continental_cup",
    "afc asian cup": "continental_cup",
    "ofc nations cup": "continental_cup",
    "olympic games": "other_major",
    "fifa world cup qualification": "world_cup_qual",
    "copa america qualification": "continental_qual",
    "uefa euro qualification": "continental_qual",
    "african cup of nations qualification": "continental_qual",
    "friendly": "friendly",
}
_DEFAULT_TOURNAMENT_TYPE = "other"

# Form score mapping: team result → points.
_FORM_POINTS: Dict[str, float] = {"W": 3.0, "D": 1.0, "L": 0.0}
_FORM_MAX = 3.0  # maximum points per game (used for normalisation)


# ===========================================================================
# Team-perspective table
# ===========================================================================

def _build_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape match data into one row per team per match.

    Each match becomes two rows — one for the home team and one for the
    away team — making it straightforward to compute per-team running
    statistics with standard ``groupby`` operations.

    Args:
        df: Match DataFrame (sorted chronologically).

    Returns:
        Long-format DataFrame with columns:
        ``date, team, opponent, goals_scored, goals_conceded,
          team_result (W/D/L), is_home, match_idx``
    """
    # Home-team rows.
    home = pd.DataFrame({
        "date": df[COL_DATE].values,
        "team": df[COL_HOME_TEAM].values,
        "opponent": df[COL_AWAY_TEAM].values,
        "goals_scored": df[COL_HOME_GOALS].values,
        "goals_conceded": df[COL_AWAY_GOALS].values,
        "result_code": df[COL_RESULT].values,   # 0=H win,1=draw,2=A win
        "is_home": True,
        "match_idx": df.index.values,
    })
    home["team_result"] = home["result_code"].map(
        {RESULT_HOME_WIN: "W", RESULT_DRAW: "D", RESULT_AWAY_WIN: "L"}
    )

    # Away-team rows.
    away = pd.DataFrame({
        "date": df[COL_DATE].values,
        "team": df[COL_AWAY_TEAM].values,
        "opponent": df[COL_HOME_TEAM].values,
        "goals_scored": df[COL_AWAY_GOALS].values,
        "goals_conceded": df[COL_HOME_GOALS].values,
        "result_code": df[COL_RESULT].values,
        "is_home": False,
        "match_idx": df.index.values,
    })
    away["team_result"] = away["result_code"].map(
        {RESULT_HOME_WIN: "L", RESULT_DRAW: "D", RESULT_AWAY_WIN: "W"}
    )

    combined = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "date", "match_idx"])
        .reset_index(drop=True)
    )

    # Numeric win/draw/loss flags for vectorised aggregation.
    combined["is_win"] = (combined["team_result"] == "W").astype(float)
    combined["is_draw"] = (combined["team_result"] == "D").astype(float)
    combined["is_loss"] = (combined["team_result"] == "L").astype(float)
    combined["form_pts"] = combined["team_result"].map(_FORM_POINTS)
    combined["goal_diff"] = combined["goals_scored"] - combined["goals_conceded"]

    return combined


# ===========================================================================
# Rolling statistics (shift-before-aggregate = zero leakage)
# ===========================================================================

def _shifted_expanding_mean(series: pd.Series) -> pd.Series:
    """Expanding mean over all *past* values (current excluded via shift)."""
    return series.shift(1).expanding().mean()


def _shifted_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the last ``window`` *past* values."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shifted_expanding_sum(series: pd.Series) -> pd.Series:
    """Expanding count of past matches (shift excludes current)."""
    return series.shift(1).expanding().sum()


def compute_team_stats(
    perspective: pd.DataFrame,
    form_window: int = RECENT_FORM_WINDOW,
) -> pd.DataFrame:
    """Compute all per-team rolling statistics on the perspective table.

    Stats derived (all shifted so current match is excluded):

    * Overall: win %, draw %, loss %, avg goals scored/conceded/diff,
      games played, form score (last N matches, normalised 0-1).
    * Home-only and away-only win % (venue-split).
    * Rest days since last match.

    Args:
        perspective: Output of :func:`_build_team_perspective`.
        form_window: Number of most recent matches for the form feature.

    Returns:
        ``perspective`` DataFrame with new statistic columns appended.
    """
    logger.info("Computing rolling team statistics ...")

    grp = perspective.groupby("team", sort=False)

    # --- Overall stats (all venues) ----------------------------------------
    perspective["overall_win_pct"] = grp["is_win"].transform(_shifted_expanding_mean)
    perspective["draw_pct"] = grp["is_draw"].transform(_shifted_expanding_mean)
    perspective["loss_pct"] = grp["is_loss"].transform(_shifted_expanding_mean)
    perspective["avg_goals_scored"] = grp["goals_scored"].transform(_shifted_expanding_mean)
    perspective["avg_goals_conceded"] = grp["goals_conceded"].transform(_shifted_expanding_mean)
    perspective["avg_goal_diff"] = grp["goal_diff"].transform(_shifted_expanding_mean)
    perspective["games_played"] = grp["is_win"].transform(_shifted_expanding_sum)

    # --- Recent form (last N matches, normalised) ---------------------------
    perspective["form"] = (
        grp["form_pts"]
        .transform(lambda s: _shifted_rolling_mean(s, form_window))
        / _FORM_MAX
    )

    # --- Venue-split stats --------------------------------------------------
    # Home win % — only computed from rows where is_home=True.
    home_mask = perspective["is_home"]
    perspective["venue_win_pct"] = np.nan

    home_view = perspective[home_mask].copy()
    home_view["venue_win_pct"] = (
        home_view.groupby("team")["is_win"].transform(_shifted_expanding_mean)
    )
    away_view = perspective[~home_mask].copy()
    away_view["venue_win_pct"] = (
        away_view.groupby("team")["is_win"].transform(_shifted_expanding_mean)
    )
    perspective.loc[home_mask, "venue_win_pct"] = home_view["venue_win_pct"].values
    perspective.loc[~home_mask, "venue_win_pct"] = away_view["venue_win_pct"].values

    # --- Rest days ----------------------------------------------------------
    perspective["rest_days"] = (
        perspective.groupby("team")["date"]
        .transform(lambda s: s.diff().dt.days)
    )
    # First match of each team has no prior rest — cap at 365 (a full year).
    perspective["rest_days"] = perspective["rest_days"].clip(upper=365).fillna(365)

    logger.info("Rolling statistics computed.")
    return perspective


# ===========================================================================
# Merge stats back onto the match table
# ===========================================================================

def _pivot_to_match_features(
    perspective: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Re-join per-team stats back to the original match-level DataFrame.

    Args:
        perspective: Team-perspective table with computed stats.
        df: Original match DataFrame.

    Returns:
        Match DataFrame with home_* and away_* feature columns added.
    """
    stat_cols = [
        "overall_win_pct", "draw_pct", "loss_pct",
        "avg_goals_scored", "avg_goals_conceded", "avg_goal_diff",
        "games_played", "form", "venue_win_pct", "rest_days",
    ]

    # Home team stats (is_home == True rows).
    home_stats = (
        perspective[perspective["is_home"]][["match_idx"] + stat_cols]
        .set_index("match_idx")
        .rename(columns={c: f"home_{c}" for c in stat_cols})
    )

    # Away team stats (is_home == False rows).
    away_stats = (
        perspective[~perspective["is_home"]][["match_idx"] + stat_cols]
        .set_index("match_idx")
        .rename(columns={c: f"away_{c}" for c in stat_cols})
    )

    df = df.join(home_stats, how="left")
    df = df.join(away_stats, how="left")

    # Rename venue-split columns to clearer names.
    df.rename(columns={
        "home_venue_win_pct": "home_home_win_pct",
        "away_venue_win_pct": "away_away_win_pct",
    }, inplace=True)

    return df


# ===========================================================================
# Tournament and temporal features
# ===========================================================================

def add_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``tournament_importance`` (float) and ``tournament_type`` (str).

    Args:
        df: Match DataFrame with a ``tournament`` column.

    Returns:
        DataFrame with two new columns.
    """
    key = df[COL_TOURNAMENT].str.lower().str.strip()
    df["tournament_importance"] = key.map(_TOURNAMENT_IMPORTANCE).fillna(_DEFAULT_IMPORTANCE)
    df["tournament_type"] = key.map(_TOURNAMENT_TYPE).fillna(_DEFAULT_TOURNAMENT_TYPE)
    logger.info("Tournament features added.")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``year``, ``month``, and ``is_neutral`` columns.

    Args:
        df: Match DataFrame with a parsed ``date`` column.

    Returns:
        DataFrame with temporal features added.
    """
    df["year"] = df[COL_DATE].dt.year
    df["month"] = df[COL_DATE].dt.month
    df["is_neutral"] = df[COL_NEUTRAL].astype(int)   # bool -> 0/1 for ML
    logger.info("Temporal features added.")
    return df


# ===========================================================================
# Rating difference features
# ===========================================================================

def add_rating_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived rating difference columns.

    Args:
        df: Match DataFrame with Elo and optional FIFA rank columns.

    Returns:
        DataFrame with ``elo_diff`` and ``fifa_rank_diff`` added.
    """
    df["elo_diff"] = (df["home_elo_before"] - df["away_elo_before"]).round(2)

    # FIFA rank: lower number = better rank, so positive diff = home team
    # ranked worse than away team (away advantage in ranking terms).
    if "home_fifa_rank" in df.columns and "away_fifa_rank" in df.columns:
        df["fifa_rank_diff"] = df["away_fifa_rank"] - df["home_fifa_rank"]
    else:
        df["fifa_rank_diff"] = np.nan

    logger.info("Rating difference features added.")
    return df


# ===========================================================================
# Build final ML-ready feature set
# ===========================================================================

# The exact columns that go into the ML feature matrix.
ML_FEATURE_COLUMNS: List[str] = [
    # Rating features
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    # Historical performance
    "home_overall_win_pct",
    "home_draw_pct",
    "home_loss_pct",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_avg_goal_diff",
    "home_games_played",
    "away_overall_win_pct",
    "away_draw_pct",
    "away_loss_pct",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_avg_goal_diff",
    "away_games_played",
    # Venue-split performance
    "home_home_win_pct",
    "away_away_win_pct",
    # Recent form
    "home_form",
    "away_form",
    # Contextual
    "home_rest_days",
    "away_rest_days",
    "is_neutral",
    "tournament_importance",
    "year",
    "month",
    # Target
    COL_RESULT,
]


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the ML-ready feature matrix from the enriched DataFrame.

    Rows where any feature is still NaN (typically the very first few
    matches per team that have no historical stats yet) are kept but flagged;
    the model training stage can decide how to handle them.

    Args:
        df: Fully enriched match DataFrame.

    Returns:
        Feature-only DataFrame with columns defined by :data:`ML_FEATURE_COLUMNS`.
    """
    available = [c for c in ML_FEATURE_COLUMNS if c in df.columns]
    missing_cols = [c for c in ML_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning("Missing ML columns (will be absent from final dataset): %s", missing_cols)

    ml_df = df[available].copy()

    nan_rows = ml_df.drop(columns=[COL_RESULT], errors="ignore").isna().any(axis=1).sum()
    logger.info(
        "ML dataset: %d rows, %d features. Rows with any NaN feature: %d (%.1f%%).",
        len(ml_df), ml_df.shape[1] - 1, nan_rows, nan_rows / len(ml_df) * 100,
    )
    return ml_df


# ===========================================================================
# Summary report
# ===========================================================================

def write_feature_summary(df: pd.DataFrame, ml_df: pd.DataFrame) -> Path:
    """Write feature engineering summary to ``reports/feature_summary.txt``.

    Args:
        df: Fully enriched DataFrame.
        ml_df: ML-ready feature matrix.

    Returns:
        Path to the written report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "feature_summary.txt"

    elo_stats = df["elo_diff"].describe()
    form_stats = df["home_form"].describe()

    lines = textwrap.dedent(f"""
    ============================================================
    FIFA World Cup 2026 - Feature Engineering Summary
    ============================================================

    Enriched dataset shape   : {df.shape[0]:,} rows  x  {df.shape[1]} columns
    ML feature matrix shape  : {ml_df.shape[0]:,} rows  x  {ml_df.shape[1]} columns

    Features generated       : {ml_df.shape[1] - 1}  (excl. target)
    Target column            : result  (0=Home Win, 1=Draw, 2=Away Win)

    Elo difference (home - away)
    ----------------------------
      Mean   : {elo_stats['mean']:>8.1f}
      Std    : {elo_stats['std']:>8.1f}
      Min    : {elo_stats['min']:>8.1f}
      Max    : {elo_stats['max']:>8.1f}

    Home form score (last {RECENT_FORM_WINDOW}, normalised 0-1)
    ----------------------------
      Mean   : {form_stats['mean']:>8.3f}
      Std    : {form_stats['std']:>8.3f}

    Tournament type distribution
    ----------------------------
    {df['tournament_type'].value_counts().to_string()}

    NaN counts in ML feature matrix
    --------------------------------
    {ml_df.isna().sum().to_string()}

    ============================================================
    """).strip()

    out_path.write_text(lines, encoding="utf-8")
    logger.info("Feature summary written -> %s", out_path)
    return out_path


# ===========================================================================
# Orchestration
# ===========================================================================

def run(
    interim_dir: Path = INTERIM_DIR,
    processed_dir: Path = PROCESSED_DIR,
    settings: Optional[Settings] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full feature engineering pipeline.

    Args:
        interim_dir: Contains ``matches_merged.csv``; also receives
            ``matches_features.csv``.
        processed_dir: Destination for the ML-ready ``matches_final.csv``.
        settings: Project settings; loaded from default config if omitted.

    Returns:
        A tuple of ``(enriched_df, ml_df)``.

    Raises:
        FileNotFoundError: If ``matches_merged.csv`` is absent.
    """
    settings = settings or load_settings()
    ensure_directories()
    set_global_seed(settings.random_seed)

    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline.")
    logger.info("=" * 60)

    # -- Load ----------------------------------------------------------------
    merged_path = interim_dir / "matches_merged.csv"
    if not merged_path.is_file():
        raise FileNotFoundError(
            f"matches_merged.csv not found in {interim_dir}. "
            "Run merge_data.py first."
        )
    df = pd.read_csv(merged_path, low_memory=False)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values([COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM]).reset_index(drop=True)
    logger.info("Loaded %d merged matches (%d columns).", len(df), df.shape[1])

    # -- Rating diff features ------------------------------------------------
    df = add_rating_diff_features(df)

    # -- Tournament & temporal features --------------------------------------
    df = add_tournament_features(df)
    df = add_temporal_features(df)

    # -- Rolling per-team stats ----------------------------------------------
    perspective = _build_team_perspective(df)
    perspective = compute_team_stats(perspective, form_window=RECENT_FORM_WINDOW)
    df = _pivot_to_match_features(perspective, df)

    # -- ML feature matrix ---------------------------------------------------
    ml_df = build_ml_dataset(df)

    # -- Save ----------------------------------------------------------------
    # Drop post-match Elo columns before saving the features file.
    # They were needed in matches_merged.csv to feed the Elo chain, but
    # including them here would constitute data leakage into features.
    drop_cols = [c for c in df.columns if c.endswith("_elo_after")]
    features_path = interim_dir / "matches_features.csv"
    df.drop(columns=drop_cols).to_csv(features_path, index=False)
    logger.info("Features dataset written -> %s  (%d rows, %d cols)",
                features_path, len(df), df.shape[1])

    processed_dir.mkdir(parents=True, exist_ok=True)
    final_path = processed_dir / "matches_final.csv"
    ml_df.to_csv(final_path, index=False)
    logger.info("ML-ready dataset written -> %s  (%d rows, %d cols)",
                final_path, len(ml_df), ml_df.shape[1])

    # -- Summary report ------------------------------------------------------
    write_feature_summary(df, ml_df)

    logger.info("Feature engineering complete.")
    return df, ml_df


def main() -> None:
    """CLI entry point."""
    try:
        df, ml_df = run()
    except Exception as exc:
        logger.exception("Feature engineering failed: %s", exc)
        raise

    print("\nFeature engineering complete")
    print("-" * 40)
    print(f"  Enriched rows  : {len(df):,}")
    print(f"  Enriched cols  : {df.shape[1]}")
    print(f"  ML feature rows: {len(ml_df):,}")
    print(f"  ML features    : {ml_df.shape[1] - 1}  (+1 target)")
    print(f"\n  Features file  : {INTERIM_DIR / 'matches_features.csv'}")
    print(f"  Final ML file  : {PROCESSED_DIR / 'matches_final.csv'}")
    print(f"  Report         : {REPORTS_DIR / 'feature_summary.txt'}")
    print("\nML feature columns:")
    for col in ml_df.columns:
        null_pct = ml_df[col].isna().mean() * 100
        print(f"  {col:<35} NaN: {null_pct:.1f}%")


if __name__ == "__main__":
    main()
