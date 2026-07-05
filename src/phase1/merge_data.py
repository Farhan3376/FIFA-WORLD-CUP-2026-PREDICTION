"""Phase 1 - Step 3: Data merging and rating attachment.

Loads ``data/interim/matches_clean.csv`` and enriches each match with
pre-match team ratings, then saves ``data/interim/matches_merged.csv``.

Two rating systems are supported:

**Elo ratings (always computed)**
    Derived directly from the match history using the World Football Elo
    formula.  Ratings are computed in strict chronological order; the
    values recorded for each match are the ratings *before* that match
    is processed — so there is zero information leakage into features.

    Key parameters (tuned to international football):
    * Starting rating: 1 500 (FIFA standard baseline)
    * K-factor: tournament-dependent (World Cup = 60, Friendly = 20)
    * Home advantage: +100 Elo points added to the home team's expected
      score during the expectation calculation (not added to the stored
      rating itself).
    * Draw adjustment: Elo uses a draw weight of 0.5 (standard).

**FIFA rankings (optional)**
    If ``data/raw/fifa_ranking.csv`` exists and has the expected schema,
    it is loaded and merged as-of each match date using a backward
    asof-join (the most recent ranking published *on or before* the
    match date is used).  When the file is absent the fifa_rank columns
    are filled with ``NaN`` and feature_builder.py falls back to Elo.

Run directly::

    python -m src.merge_data
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import (
    COL_AWAY_TEAM,
    COL_DATE,
    COL_HOME_TEAM,
    COL_NEUTRAL,
    COL_RESULT,
    COL_TOURNAMENT,
    INTERIM_DIR,
    RAW_DIR,
    REPORTS_DIR,
    Settings,
    ensure_directories,
    load_settings,
    set_global_seed,
    RESULT_HOME_WIN,
    RESULT_DRAW,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Elo configuration
# ---------------------------------------------------------------------------
ELO_START = 1500.0          # initial rating for every team
ELO_HOME_ADVANTAGE = 100.0  # added to home team's rating for expectation calc
ELO_DRAW_RATE = 0.5         # score assigned to each team for a draw

# Tournament K-factors: higher K = larger rating swings for big matches.
_KFACTOR_MAP: Dict[str, float] = {
    "fifa world cup": 60.0,
    "confederations cup": 50.0,
    "copa america": 50.0,
    "africa cup of nations": 50.0,
    "uefa euro": 50.0,
    "gold cup": 45.0,
    "afc asian cup": 45.0,
    "ofc nations cup": 45.0,
    "olympic games": 40.0,
    "fifa world cup qualification": 40.0,
    "copa america qualification": 35.0,
    "uefa euro qualification": 35.0,
    "african cup of nations qualification": 35.0,
    "friendly": 20.0,
}
_DEFAULT_KFACTOR = 30.0     # for unrecognised tournament names


# ===========================================================================
# Elo computation
# ===========================================================================

def _kfactor(tournament: str) -> float:
    """Return the Elo K-factor for a given tournament name.

    Args:
        tournament: Tournament name (any casing).

    Returns:
        K-factor as a float.
    """
    return _KFACTOR_MAP.get(str(tournament).lower().strip(), _DEFAULT_KFACTOR)


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Compute expected score for team A against team B.

    Uses the standard Elo logistic formula with a 400-point scale factor.

    Args:
        rating_a: Team A's current Elo rating (home advantage already added).
        rating_b: Team B's current Elo rating.

    Returns:
        Expected score for team A in [0, 1].
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-match Elo ratings for every row in ``df`` in-place.

    The DataFrame **must** be sorted ascending by date before calling this
    function (guaranteed by :func:`run`).

    For each match the algorithm:
    1. Records the *current* ratings (before this match) as features.
    2. Calculates expected scores with the home advantage offset.
    3. Updates both teams' ratings based on the actual result.

    This strict ordering ensures features are always derived from past data.

    Args:
        df: Match DataFrame sorted by date ascending, containing
            ``home_team``, ``away_team``, ``result``, ``neutral``, and
            ``tournament`` columns.

    Returns:
        The same DataFrame with four new columns:
        ``home_elo_before``, ``away_elo_before``,
        ``home_elo_after``, ``away_elo_after``.
    """
    logger.info("Computing Elo ratings for %d matches ...", len(df))

    ratings: Dict[str, float] = {}  # team -> current Elo

    home_elo_before: list[float] = []
    away_elo_before: list[float] = []
    home_elo_after: list[float] = []
    away_elo_after: list[float] = []

    for row in df.itertuples(index=False):
        home: str = getattr(row, COL_HOME_TEAM)
        away: str = getattr(row, COL_AWAY_TEAM)
        result: int = getattr(row, COL_RESULT)
        neutral: bool = getattr(row, COL_NEUTRAL)
        tournament: str = getattr(row, COL_TOURNAMENT)

        r_home = ratings.get(home, ELO_START)
        r_away = ratings.get(away, ELO_START)

        home_elo_before.append(r_home)
        away_elo_before.append(r_away)

        # Apply home-ground advantage unless the match is on neutral soil.
        r_home_adj = r_home if neutral else r_home + ELO_HOME_ADVANTAGE

        exp_home = _elo_expected(r_home_adj, r_away)
        exp_away = 1.0 - exp_home

        # Actual scores: 1=win, 0.5=draw, 0=loss.
        if result == RESULT_HOME_WIN:
            s_home, s_away = 1.0, 0.0
        elif result == RESULT_DRAW:
            s_home = s_away = ELO_DRAW_RATE
        else:  # RESULT_AWAY_WIN
            s_home, s_away = 0.0, 1.0

        k = _kfactor(tournament)
        r_home_new = r_home + k * (s_home - exp_home)
        r_away_new = r_away + k * (s_away - exp_away)

        ratings[home] = r_home_new
        ratings[away] = r_away_new

        home_elo_after.append(r_home_new)
        away_elo_after.append(r_away_new)

    df = df.copy()
    df["home_elo_before"] = home_elo_before
    df["away_elo_before"] = away_elo_before
    df["home_elo_after"] = home_elo_after
    df["away_elo_after"] = away_elo_after

    # Round to 2 decimal places for readability.
    for col in ("home_elo_before", "away_elo_before", "home_elo_after", "away_elo_after"):
        df[col] = df[col].round(2)

    logger.info(
        "Elo ratings computed. Rating range: [%.0f, %.0f]",
        df["home_elo_before"].min(),
        df["home_elo_before"].max(),
    )
    return df


# ===========================================================================
# FIFA ranking merge (optional)
# ===========================================================================

def _load_fifa_ranking(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Load the optional ``fifa_ranking.csv`` file.

    Expected schema (flexible column detection):
    * date column   → ``rank_date``
    * team column   → ``country_full`` or ``team``
    * rank column   → ``rank``

    Args:
        raw_dir: Directory containing the raw CSV files.

    Returns:
        Normalised DataFrame or ``None`` if the file is absent / invalid.
    """
    path = raw_dir / "fifa_ranking.csv"
    if not path.is_file():
        logger.info("fifa_ranking.csv not found - skipping FIFA ranking merge.")
        return None

    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read fifa_ranking.csv: %s", exc)
        return None

    # Flexible column detection.
    col_map: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in raw.columns}

    for candidate in ("rank_date", "date"):
        if candidate in lower_cols:
            col_map["date"] = lower_cols[candidate]
            break
    for candidate in ("country_full", "team", "country"):
        if candidate in lower_cols:
            col_map["team"] = lower_cols[candidate]
            break
    for candidate in ("rank", "total_points", "fifa_rank"):
        if candidate in lower_cols:
            col_map["rank"] = lower_cols[candidate]
            break

    if len(col_map) < 3:
        logger.warning(
            "fifa_ranking.csv does not have expected columns (need date, team, rank). "
            "Found: %s. Skipping.", list(raw.columns)
        )
        return None

    df = raw[[col_map["date"], col_map["team"], col_map["rank"]]].copy()
    df.columns = ["rank_date", "team", "fifa_rank"]
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    df = df.dropna(subset=["rank_date", "team", "fifa_rank"])
    df["team"] = df["team"].astype(str).str.strip().str.title()
    df["fifa_rank"] = pd.to_numeric(df["fifa_rank"], errors="coerce")
    df = df.sort_values("rank_date").reset_index(drop=True)

    logger.info("Loaded %d FIFA ranking records (%d unique teams).",
                len(df), df["team"].nunique())
    return df


def _asof_join_ranking(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    team_col: str,
    rank_col_out: str,
) -> pd.Series:
    """Return a Series of FIFA ranks aligned to each match via an asof-join.

    For each match row, finds the most recent ranking entry for the given
    team published *on or before* the match date.

    Args:
        matches: Match DataFrame (must have ``date`` and ``team_col`` columns).
        rankings: Ranking DataFrame with ``rank_date``, ``team``, ``fifa_rank``.
        team_col: Column in ``matches`` holding the team name.
        rank_col_out: Name for the output Series.

    Returns:
        A Series with the same index as ``matches``.
    """
    result_values = np.full(len(matches), np.nan)

    # Build a per-team lookup series for fast asof access.
    for team, group in rankings.groupby("team"):
        group = group.sort_values("rank_date")
        mask = matches[team_col] == team
        if not mask.any():
            continue
        match_dates = matches.loc[mask, COL_DATE]
        # pd.merge_asof requires sorted left key.
        temp = pd.DataFrame({"date": match_dates, "idx": match_dates.index})
        temp = temp.sort_values("date")
        merged = pd.merge_asof(
            temp,
            group[["rank_date", "fifa_rank"]].rename(columns={"rank_date": "date"}),
            on="date",
            direction="backward",
        )
        result_values[merged["idx"].values] = merged["fifa_rank"].values

    return pd.Series(result_values, index=matches.index, name=rank_col_out)


def merge_fifa_rankings(
    df: pd.DataFrame, raw_dir: Path = RAW_DIR
) -> pd.DataFrame:
    """Attach pre-match FIFA rankings to every row (if data is available).

    Adds columns ``home_fifa_rank`` and ``away_fifa_rank``. Values are
    ``NaN`` for matches before the FIFA ranking era or when the file is absent.

    Args:
        df: Match DataFrame (sorted by date).
        raw_dir: Directory containing ``fifa_ranking.csv``.

    Returns:
        DataFrame with ranking columns added.
    """
    rankings = _load_fifa_ranking(raw_dir)

    if rankings is None:
        df["home_fifa_rank"] = np.nan
        df["away_fifa_rank"] = np.nan
        return df

    df["home_fifa_rank"] = _asof_join_ranking(df, rankings, COL_HOME_TEAM, "home_fifa_rank")
    df["away_fifa_rank"] = _asof_join_ranking(df, rankings, COL_AWAY_TEAM, "away_fifa_rank")

    filled_home = df["home_fifa_rank"].notna().sum()
    filled_away = df["away_fifa_rank"].notna().sum()
    logger.info(
        "FIFA ranking attached: home=%d rows filled, away=%d rows filled.",
        filled_home, filled_away,
    )
    return df


# ===========================================================================
# Summary report
# ===========================================================================

def write_merge_summary(df: pd.DataFrame) -> Path:
    """Write a merge summary to ``reports/merge_summary.txt``.

    Args:
        df: Merged DataFrame.

    Returns:
        Path to the written report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "merge_summary.txt"

    elo_coverage = df["home_elo_before"].notna().mean() * 100
    fifa_coverage = df["home_fifa_rank"].notna().mean() * 100

    # Top 10 teams by final Elo.
    last_elo = (
        df.sort_values(COL_DATE)
        .groupby(COL_HOME_TEAM)["home_elo_after"]
        .last()
        .sort_values(ascending=False)
        .head(10)
    )
    top10_lines = "\n".join(
        f"  {rank:>2}. {team:<30} {elo:.0f}"
        for rank, (team, elo) in enumerate(last_elo.items(), start=1)
    )

    lines = textwrap.dedent(f"""
    ============================================================
    FIFA World Cup 2026 - Merge Summary
    ============================================================

    Total matches            : {len(df):>10,}
    Elo coverage             : {elo_coverage:>9.1f}%
    FIFA ranking coverage    : {fifa_coverage:>9.1f}%

    New columns added
    -----------------
      home_elo_before   (pre-match Elo for home team)
      away_elo_before   (pre-match Elo for away team)
      home_elo_after    (post-match Elo for home team)
      away_elo_after    (post-match Elo for away team)
      home_fifa_rank    (FIFA rank as-of match date, if available)
      away_fifa_rank    (FIFA rank as-of match date, if available)

    Elo rating statistics
    ---------------------
      Min home Elo  : {df['home_elo_before'].min():.0f}
      Max home Elo  : {df['home_elo_before'].max():.0f}
      Mean home Elo : {df['home_elo_before'].mean():.0f}
      Std home Elo  : {df['home_elo_before'].std():.0f}

    Top 10 teams by final Elo rating
    ---------------------------------
{top10_lines}

    ============================================================
    """).strip()

    out_path.write_text(lines, encoding="utf-8")
    logger.info("Merge summary written -> %s", out_path)
    return out_path


# ===========================================================================
# Orchestration
# ===========================================================================

def run(
    interim_dir: Path = INTERIM_DIR,
    raw_dir: Path = RAW_DIR,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """Execute the full merge pipeline end-to-end.

    Args:
        interim_dir: Directory containing ``matches_clean.csv`` and the
            destination for ``matches_merged.csv``.
        raw_dir: Directory containing optional ranking CSV files.
        settings: Project settings; loaded from default config if omitted.

    Returns:
        The enriched :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If ``matches_clean.csv`` is absent.
    """
    settings = settings or load_settings()
    ensure_directories()
    set_global_seed(settings.random_seed)

    logger.info("=" * 60)
    logger.info("Starting merge pipeline.")
    logger.info("=" * 60)

    # -- Load clean data -----------------------------------------------------
    clean_path = interim_dir / "matches_clean.csv"
    if not clean_path.is_file():
        raise FileNotFoundError(
            f"matches_clean.csv not found in {interim_dir}. "
            "Run preprocessing.py first."
        )
    df = pd.read_csv(clean_path, low_memory=False)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    logger.info("Loaded %d clean matches.", len(df))

    # Sort chronologically — required for correct Elo computation.
    df = df.sort_values([COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM]).reset_index(drop=True)

    # -- Elo ratings (always) ------------------------------------------------
    df = compute_elo_ratings(df)

    # -- FIFA rankings (optional) --------------------------------------------
    df = merge_fifa_rankings(df, raw_dir)

    # -- Save ----------------------------------------------------------------
    out_path = interim_dir / "matches_merged.csv"
    df.to_csv(out_path, index=False)
    logger.info("Merged dataset written -> %s  (%d rows, %d cols)",
                out_path, len(df), df.shape[1])

    # -- Summary -------------------------------------------------------------
    write_merge_summary(df)

    logger.info("Merge pipeline complete.")
    return df


def main() -> None:
    """CLI entry point: run the merge pipeline with the default configuration."""
    try:
        df = run()
    except Exception as exc:
        logger.exception("Merge pipeline failed: %s", exc)
        raise

    print("\nMerge complete")
    print("-" * 40)
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"  Output       : {INTERIM_DIR / 'matches_merged.csv'}")
    print(f"  Report       : {REPORTS_DIR / 'merge_summary.txt'}")
    print(f"\n  Elo coverage : {df['home_elo_before'].notna().mean()*100:.1f}%")
    print(f"  FIFA rank    : {df['home_fifa_rank'].notna().mean()*100:.1f}%")


if __name__ == "__main__":
    main()
