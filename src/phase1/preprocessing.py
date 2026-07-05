"""Phase 1 - Step 2: Data preprocessing.

Takes raw ``data/raw/results.csv`` (and optionally ``shootouts.csv``) and
produces ``data/interim/matches_clean.csv``.

Pipeline steps (executed in order by :func:`run`):
1.  Load raw CSV files.
2.  Rename columns to the canonical schema defined in ``config.py``.
3.  Parse and validate date formats.
4.  Remove duplicate rows.
5.  Remove matches with impossible or invalid scores.
6.  Standardize team names (whitespace, casing, alias resolution).
7.  Standardize text fields (tournament, city, country).
8.  Handle missing values via fill / imputation / drop.
9.  Enforce correct ``dtype`` for every column.
10. Merge penalty-shootout winner information.
11. Derive the ``winner`` column (home team name, away team name or ``"Draw"``).
12. Derive the ``result`` target variable (0 / 1 / 2).
13. Detect and flag outliers (extreme scorelines).
14. Filter to a sensible date range.
15. Write ``data/interim/matches_clean.csv``.
16. Write a human-readable preprocessing summary to ``reports/``.

Run directly with::

    python -m src.preprocessing
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.config import (
    COL_AWAY_GOALS,
    COL_AWAY_TEAM,
    COL_DATE,
    COL_HOME_GOALS,
    COL_HOME_TEAM,
    COL_NEUTRAL,
    COL_RESULT,
    COL_TOURNAMENT,
    COL_CITY,
    COL_COUNTRY,
    COL_WINNER,
    INTERIM_DIR,
    MAX_GOALS_PER_TEAM,
    MIN_MATCH_YEAR,
    RAW_DIR,
    REPORTS_DIR,
    Settings,
    ensure_directories,
    load_settings,
    set_global_seed,
    RESULT_AWAY_WIN,
    RESULT_DRAW,
    RESULT_HOME_WIN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Raw -> canonical column renaming
# ---------------------------------------------------------------------------
# The martj42 dataset uses *_score; our canonical schema uses *_goals.
_RAW_RENAME: Dict[str, str] = {
    "home_score": COL_HOME_GOALS,
    "away_score": COL_AWAY_GOALS,
}

# ---------------------------------------------------------------------------
# Team-name alias table (common spelling variations → canonical name)
# Extend this table as new sources are integrated.
# ---------------------------------------------------------------------------
_TEAM_ALIASES: Dict[str, str] = {
    # USA variants
    "united states": "USA",
    "united states of america": "USA",
    "us": "USA",
    # UK nations
    "northern ireland": "Northern Ireland",
    "republic of ireland": "Republic of Ireland",
    "eire": "Republic of Ireland",
    # Korea variants
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "dpr korea": "North Korea",
    # China
    "china pr": "China",
    "peoples republic of china": "China",
    # Congo variants
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "congo republic": "Republic of Congo",
    # Other common mismatches
    "cape verde islands": "Cape Verde",
    "cape verde is.": "Cape Verde",
    "trinidad & tobago": "Trinidad and Tobago",
    "trinidad & tobago": "Trinidad and Tobago",
    "antigua & barbuda": "Antigua and Barbuda",
    "st. kitts and nevis": "Saint Kitts and Nevis",
    "st. vincent and the grenadines": "Saint Vincent and the Grenadines",
    "st. lucia": "Saint Lucia",
    "st kitts & nevis": "Saint Kitts and Nevis",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "chinese taipei": "Taiwan",
    "iran": "Iran",
    "ir iran": "Iran",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "republic of north macedonia": "North Macedonia",
    "fyr macedonia": "North Macedonia",
    "uae": "United Arab Emirates",
}

# ---------------------------------------------------------------------------
# Outlier threshold: flag (but keep) matches with combined goals above this.
# ---------------------------------------------------------------------------
_OUTLIER_COMBINED_GOALS = 20


# ===========================================================================
# Step 1 & 2 - Load and rename
# ===========================================================================

def load_raw_results(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load ``results.csv`` and rename columns to the canonical schema.

    Args:
        raw_dir: Directory that contains the raw CSV files.

    Returns:
        DataFrame with canonical column names.

    Raises:
        FileNotFoundError: If ``results.csv`` is absent.
    """
    path = raw_dir / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"results.csv not found in {raw_dir}. Run data_collection.py first."
        )
    logger.info("Loading raw results from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns.", len(df), df.shape[1])
    df.rename(columns=_RAW_RENAME, inplace=True)
    return df


def load_shootouts(raw_dir: Path = RAW_DIR) -> Optional[pd.DataFrame]:
    """Load ``shootouts.csv`` if available.

    Args:
        raw_dir: Directory that contains the raw CSV files.

    Returns:
        DataFrame with columns ``[date, home_team, away_team, winner]`` or
        ``None`` if the file is absent.
    """
    path = raw_dir / "shootouts.csv"
    if not path.is_file():
        logger.warning("shootouts.csv not found - shootout data will be skipped.")
        return None
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d shootout records.", len(df))
    return df


# ===========================================================================
# Step 3 - Date parsing
# ===========================================================================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``date`` column to :class:`pandas.Timestamp`.

    Rows with unparseable dates are dropped with a warning.

    Args:
        df: DataFrame containing a ``date`` column.

    Returns:
        DataFrame with ``date`` as ``datetime64[ns]``.
    """
    original = len(df)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    invalid = df[COL_DATE].isna().sum()
    if invalid:
        logger.warning("Dropping %d rows with unparseable dates.", invalid)
        df = df.dropna(subset=[COL_DATE])
    logger.info("Parsed dates: %d valid rows (dropped %d).", len(df), original - len(df))
    return df


# ===========================================================================
# Step 4 - Deduplicate
# ===========================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows.

    Deduplication key: ``[date, home_team, away_team]`` — a pair can only play
    once on a given date.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame without duplicates.
    """
    before = len(df)
    df = df.drop_duplicates(
        subset=[COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM], keep="first"
    )
    dropped = before - len(df)
    if dropped:
        logger.warning("Removed %d duplicate rows.", dropped)
    else:
        logger.info("No duplicate rows found.")
    return df


# ===========================================================================
# Step 5 - Invalid / impossible score removal
# ===========================================================================

def remove_invalid_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with negative or impossibly high goal counts.

    Args:
        df: DataFrame with ``home_goals`` and ``away_goals`` columns.

    Returns:
        Filtered DataFrame.
    """
    before = len(df)

    goals_cols = [COL_HOME_GOALS, COL_AWAY_GOALS]
    df[goals_cols] = df[goals_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows where goals are NaN (could not be coerced).
    df = df.dropna(subset=goals_cols)

    negative_mask = (df[COL_HOME_GOALS] < 0) | (df[COL_AWAY_GOALS] < 0)
    too_high_mask = (
        (df[COL_HOME_GOALS] > MAX_GOALS_PER_TEAM)
        | (df[COL_AWAY_GOALS] > MAX_GOALS_PER_TEAM)
    )

    invalid = negative_mask | too_high_mask
    if invalid.any():
        logger.warning(
            "Removing %d rows with invalid scores (negative or > %d).",
            invalid.sum(),
            MAX_GOALS_PER_TEAM,
        )
        df = df[~invalid]

    logger.info(
        "Score validation: %d rows retained (removed %d).",
        len(df),
        before - len(df),
    )
    return df


# ===========================================================================
# Step 6 - Team name standardization
# ===========================================================================

def _normalize_team_name(name: str) -> str:
    """Apply alias lookup and light normalization to a single team name.

    Processing order:
    1. Strip whitespace.
    2. Title-case the name.
    3. Look up the lowercase version in the alias table.

    Args:
        name: Raw team name string.

    Returns:
        Canonical team name.
    """
    if not isinstance(name, str):
        return str(name)
    cleaned = " ".join(name.strip().split())  # collapse internal whitespace
    lowered = cleaned.lower()
    if lowered in _TEAM_ALIASES:
        return _TEAM_ALIASES[lowered]
    return cleaned.title()


def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical team name normalization to home and away team columns.

    Args:
        df: DataFrame with ``home_team`` and ``away_team`` columns.

    Returns:
        DataFrame with standardized team names.
    """
    for col in (COL_HOME_TEAM, COL_AWAY_TEAM):
        df[col] = df[col].astype(str).map(_normalize_team_name)
    logger.info("Team names standardized.")
    return df


# ===========================================================================
# Step 7 - Standardize text fields
# ===========================================================================

def standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and apply consistent Title Casing to text columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with cleaned text columns.
    """
    text_cols = [COL_TOURNAMENT, COL_CITY, COL_COUNTRY]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace("Nan", pd.NA)
                .replace("None", pd.NA)
            )
    logger.info("Text fields standardized.")
    return df


# ===========================================================================
# Step 8 - Missing values
# ===========================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute or drop missing values with a documented strategy.

    Strategy:
    * ``tournament`` → fill with ``"Unknown"`` (match still usable).
    * ``city`` / ``country`` → fill with ``"Unknown"`` (informational only).
    * ``neutral`` → fill with ``False`` (conservative assumption).
    * ``home_goals`` / ``away_goals`` → already handled in score validation.
    * Rows still missing ``home_team``, ``away_team`` or ``date`` are dropped.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    fill_map = {
        COL_TOURNAMENT: "Unknown",
        COL_CITY: "Unknown",
        COL_COUNTRY: "Unknown",
        COL_NEUTRAL: False,
    }
    for col, fill_value in fill_map.items():
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].fillna(fill_value)
            if before:
                logger.info("Filled %d missing values in '%s' with %r.", before, col, fill_value)

    # Critical columns - cannot proceed without them.
    critical = [COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM]
    before = len(df)
    df = df.dropna(subset=critical)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows missing critical fields.", dropped)

    return df


# ===========================================================================
# Step 9 - Enforce dtypes
# ===========================================================================

def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every column has the correct, downstream-ready dtype.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with enforced dtypes.
    """
    df[COL_HOME_GOALS] = df[COL_HOME_GOALS].astype(int)
    df[COL_AWAY_GOALS] = df[COL_AWAY_GOALS].astype(int)
    df[COL_NEUTRAL] = df[COL_NEUTRAL].astype(bool)
    df[COL_HOME_TEAM] = df[COL_HOME_TEAM].astype(str)
    df[COL_AWAY_TEAM] = df[COL_AWAY_TEAM].astype(str)
    df[COL_TOURNAMENT] = df[COL_TOURNAMENT].astype(str)
    logger.info("Data types enforced.")
    return df


# ===========================================================================
# Step 10 - Merge shootout data
# ===========================================================================

def merge_shootouts(
    df: pd.DataFrame, shootouts: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Attach penalty-shootout winner information to the match table.

    For knockout matches that end in a draw and go to penalties, the actual
    winner is in ``shootouts.csv``. This information is stored in a new column
    ``shootout_winner`` so downstream logic can handle it without losing the
    regular-time score.

    Args:
        df: Clean match DataFrame.
        shootouts: Shootout DataFrame (or ``None`` if unavailable).

    Returns:
        DataFrame with an optional ``shootout_winner`` column added.
    """
    if shootouts is None:
        df["shootout_winner"] = pd.NA
        logger.info("No shootout data - 'shootout_winner' set to NA for all rows.")
        return df

    shootouts = shootouts.copy()
    shootouts[COL_DATE] = pd.to_datetime(shootouts[COL_DATE], errors="coerce")
    shootouts = shootouts.rename(columns={"winner": "shootout_winner"})
    shootouts = shootouts[[COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM, "shootout_winner"]]

    # Standardize team names in shootouts to match the main table.
    for col in (COL_HOME_TEAM, COL_AWAY_TEAM):
        shootouts[col] = shootouts[col].astype(str).map(_normalize_team_name)

    merged = df.merge(
        shootouts,
        on=[COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM],
        how="left",
    )
    n_matched = merged["shootout_winner"].notna().sum()
    logger.info("Merged %d shootout records.", n_matched)
    return merged


# ===========================================================================
# Step 11 & 12 - Winner and result target variable
# ===========================================================================

def derive_winner(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``winner`` column: home team name, away team name, or ``"Draw"``.

    When a match went to penalties, the *shootout* winner takes precedence
    for the ``winner`` column; the ``result`` target still reflects the
    regular-time scoreline (0/1/2) so models can learn from draws that had
    knockout implications.

    Args:
        df: DataFrame with goals and optional ``shootout_winner``.

    Returns:
        DataFrame with ``winner`` column added.
    """
    conditions = [
        df[COL_HOME_GOALS] > df[COL_AWAY_GOALS],
        df[COL_HOME_GOALS] == df[COL_AWAY_GOALS],
        df[COL_HOME_GOALS] < df[COL_AWAY_GOALS],
    ]
    choices = [df[COL_HOME_TEAM], "Draw", df[COL_AWAY_TEAM]]
    df[COL_WINNER] = np.select(conditions, choices, default="Unknown")

    # Override with shootout winner where applicable (column may not exist yet).
    if "shootout_winner" in df.columns:
        mask = df["shootout_winner"].notna()
        df.loc[mask, COL_WINNER] = df.loc[mask, "shootout_winner"]

    logger.info("'winner' column derived.")
    return df


def derive_result(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``result`` target variable (0=Home Win, 1=Draw, 2=Away Win).

    This always reflects the **regular-time** scoreline and is the label used
    for model training.

    Args:
        df: DataFrame with ``home_goals`` and ``away_goals`` columns.

    Returns:
        DataFrame with ``result`` column added (int8).
    """
    conditions = [
        df[COL_HOME_GOALS] > df[COL_AWAY_GOALS],
        df[COL_HOME_GOALS] == df[COL_AWAY_GOALS],
        df[COL_HOME_GOALS] < df[COL_AWAY_GOALS],
    ]
    choices = [RESULT_HOME_WIN, RESULT_DRAW, RESULT_AWAY_WIN]
    df[COL_RESULT] = np.select(conditions, choices, default=-1).astype("int8")
    logger.info("'result' target variable derived (0=Home, 1=Draw, 2=Away).")
    return df


# ===========================================================================
# Step 13 - Outlier flagging
# ===========================================================================

def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean ``is_outlier`` column for unusually high-scoring matches.

    Outliers are **kept** in the dataset. The flag lets downstream stages
    (validation reports, model evaluation) identify and handle them explicitly.

    Criteria:
    * Combined goals in a single match exceed ``_OUTLIER_COMBINED_GOALS``.
    * Either team scores more than ``MAX_GOALS_PER_TEAM - 1`` goals
      (retained for historical completeness; validation removes truly
      impossible values earlier).

    Args:
        df: DataFrame with goals columns.

    Returns:
        DataFrame with ``is_outlier`` (bool) column.
    """
    combined = df[COL_HOME_GOALS] + df[COL_AWAY_GOALS]
    high_scoring = combined > _OUTLIER_COMBINED_GOALS
    df["is_outlier"] = high_scoring
    n = high_scoring.sum()
    if n:
        logger.info("Flagged %d outlier matches (combined goals > %d).", n, _OUTLIER_COMBINED_GOALS)
    return df


# ===========================================================================
# Step 14 - Date-range filter
# ===========================================================================

def filter_date_range(df: pd.DataFrame, min_year: int = MIN_MATCH_YEAR) -> pd.DataFrame:
    """Remove matches before the earliest credible international fixture.

    Args:
        df: DataFrame with a parsed ``date`` column.
        min_year: Earliest year to retain (default: 1872, first recorded match).

    Returns:
        Filtered DataFrame.
    """
    before = len(df)
    df = df[df[COL_DATE].dt.year >= min_year]
    removed = before - len(df)
    if removed:
        logger.warning("Removed %d rows outside valid year range (< %d).", removed, min_year)
    return df


# ===========================================================================
# Step 16 - Preprocessing summary
# ===========================================================================

def _count_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Compute summary statistics over the clean DataFrame."""
    return {
        "total_matches": len(df),
        "date_min": str(df[COL_DATE].min().date()),
        "date_max": str(df[COL_DATE].max().date()),
        "unique_teams": len(
            pd.unique(df[[COL_HOME_TEAM, COL_AWAY_TEAM]].values.ravel())
        ),
        "unique_tournaments": df[COL_TOURNAMENT].nunique(),
        "home_wins": int((df[COL_RESULT] == RESULT_HOME_WIN).sum()),
        "draws": int((df[COL_RESULT] == RESULT_DRAW).sum()),
        "away_wins": int((df[COL_RESULT] == RESULT_AWAY_WIN).sum()),
        "missing_values": int(df.isna().sum().sum()),
        "outliers_flagged": int(df["is_outlier"].sum()),
    }


def write_summary(stats: Dict[str, object], raw_count: int) -> Path:
    """Write the preprocessing summary to ``reports/preprocessing_summary.txt``.

    Args:
        stats: Dictionary of summary statistics.
        raw_count: Number of rows in the original raw file.

    Returns:
        Path to the written report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "preprocessing_summary.txt"

    total = stats["total_matches"]
    removed = raw_count - int(total)

    lines = textwrap.dedent(f"""
    ============================================================
    FIFA World Cup 2026 - Preprocessing Summary
    ============================================================

    Raw rows loaded         : {raw_count:>10,}
    Rows removed (all steps): {removed:>10,}
    Clean rows retained     : {total:>10,}

    Date range              : {stats['date_min']}  ->  {stats['date_max']}
    Unique teams            : {stats['unique_teams']:>10,}
    Unique tournaments      : {stats['unique_tournaments']:>10,}

    Result distribution
    -------------------
      Home wins             : {stats['home_wins']:>10,}  ({stats['home_wins']/total*100:.1f}%)
      Draws                 : {stats['draws']:>10,}  ({stats['draws']/total*100:.1f}%)
      Away wins             : {stats['away_wins']:>10,}  ({stats['away_wins']/total*100:.1f}%)

    Remaining missing values: {stats['missing_values']:>10,}
    Outlier matches flagged : {stats['outliers_flagged']:>10,}
    ============================================================
    """).strip()

    out_path.write_text(lines, encoding="utf-8")
    logger.info("Preprocessing summary written -> %s", out_path)
    return out_path


# ===========================================================================
# Orchestration
# ===========================================================================

def run(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline end-to-end.

    Args:
        raw_dir: Directory containing raw CSV files.
        interim_dir: Directory to write ``matches_clean.csv``.
        settings: Project settings; loaded from default config if omitted.

    Returns:
        The clean :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If required raw data is absent.
    """
    settings = settings or load_settings()
    ensure_directories()
    set_global_seed(settings.random_seed)

    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline.")
    logger.info("=" * 60)

    # -- Load ----------------------------------------------------------------
    df = load_raw_results(raw_dir)
    shootouts = load_shootouts(raw_dir)
    raw_count = len(df)

    # -- Clean ---------------------------------------------------------------
    df = parse_dates(df)
    df = remove_duplicates(df)
    df = remove_invalid_scores(df)
    df = standardize_team_names(df)
    df = standardize_text_fields(df)
    df = handle_missing_values(df)
    df = enforce_dtypes(df)

    # -- Enrich --------------------------------------------------------------
    df = merge_shootouts(df, shootouts)
    df = derive_winner(df)
    df = derive_result(df)
    df = flag_outliers(df)
    df = filter_date_range(df)

    # -- Sort for reproducibility -------------------------------------------
    df = df.sort_values([COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM]).reset_index(drop=True)

    # -- Save ----------------------------------------------------------------
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "matches_clean.csv"
    df.to_csv(out_path, index=False)
    logger.info("Clean dataset written -> %s  (%d rows)", out_path, len(df))

    # -- Summary report ------------------------------------------------------
    stats = _count_stats(df)
    write_summary(stats, raw_count)

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


def main() -> None:
    """CLI entry point: run preprocessing with the default configuration."""
    try:
        df = run()
    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        raise

    print("\nPreprocessing complete")
    print("-" * 40)
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Output : {INTERIM_DIR / 'matches_clean.csv'}")
    print(f"  Report : {REPORTS_DIR / 'preprocessing_summary.txt'}")
    print("\nColumn dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<25} {dtype}")


if __name__ == "__main__":
    main()
