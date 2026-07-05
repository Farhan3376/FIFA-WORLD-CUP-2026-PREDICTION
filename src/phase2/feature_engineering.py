"""Phase 2 - Step 4: Feature Engineering, Scaling, and Encoding.

This module implements the feature engineering, imputation, and standardization steps:
1. Calculates advanced football features (Elo win probabilities, rating logs, goal/form differentials).
2. Imputes missing rolling features using median values.
3. Standardizes continuous numerical variables using a StandardScaler.
4. Supports categorical variable encoding (via OneHotEncoder) if categorical columns are present.
5. Saves fitted processors (imputer, scaler, encoder) to the `models/` directory for inference.
6. Saves the final engineered dataset to `data/processed/matches_engineered.csv`.

Run directly::

    python -m src.feature_engineering
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    PROCESSED_DIR,
    MODELS_DIR,
    COL_RESULT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="phase2.log")


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build interaction and differential football features.

    Args:
        df: Input DataFrame containing base features.

    Returns:
        pd.DataFrame: DataFrame with new advanced features added.
    """
    df = df.copy()
    logger.info("Engineering advanced football features...")

    # 1. Elo win probability for the Home Team
    # We use a standard home advantage of 100 Elo points (when match is not neutral)
    home_adv = 100 * (1 - df["is_neutral"])
    df["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** (- (df["elo_diff"] + home_adv) / 400.0))

    # 2. Sign-preserving logarithmic Elo difference to compress extreme rating differentials
    df["log_elo_diff"] = np.sign(df["elo_diff"]) * np.log1p(np.abs(df["elo_diff"]))

    # 3. Rolling goal scoring difference (Home Avg - Away Avg)
    df["goal_avg_diff"] = df["home_avg_goals_scored"] - df["away_avg_goals_scored"]

    # 4. Rolling goal conceded difference (Home Avg - Away Avg)
    df["goal_conceded_avg_diff"] = df["home_avg_goals_conceded"] - df["away_avg_goals_conceded"]

    # 5. Rolling form difference (Home Form - Away Form)
    df["form_diff"] = df["home_form"] - df["away_form"]

    # 6. Rest days difference (Home Rest - Away Rest)
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

    # 7. Games played difference (experience gap)
    df["games_played_diff"] = df["home_games_played"] - df["away_games_played"]

    # 8. Historical win percentage gap
    df["win_pct_diff"] = df["home_overall_win_pct"] - df["away_overall_win_pct"]

    # 9. Home-field win percentage vs Away-field win percentage gap
    df["home_vs_away_field_pct"] = df["home_home_win_pct"] - df["away_away_win_pct"]

    logger.info("Feature engineering complete. Added 9 new interaction features.")
    return df


def process_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, SimpleImputer, StandardScaler, OneHotEncoder | None]:
    """Perform imputation, scaling, and categorical encoding on the feature set.

    Args:
        df: Input DataFrame with advanced features.

    Returns:
        Tuple: (Processed Feature DataFrame, fitted Imputer, fitted Scaler, fitted Encoder or None)
    """
    df = df.copy()

    # Drop entirely empty columns (e.g. FIFA ranks)
    empty_cols = ["home_fifa_rank", "away_fifa_rank", "fifa_rank_diff"]
    cols_to_drop = [c for c in empty_cols if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping completely empty columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    # Separate target
    target = df[COL_RESULT].copy()
    features = df.drop(columns=[COL_RESULT])

    # 1. Identify column types
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cols = ["is_neutral"]

    # Continuous numeric columns exclude target, binary, and categorical columns
    continuous_cols = [
        c for c in features.select_dtypes(include=[np.number]).columns
        if c not in binary_cols and c not in categorical_cols
    ]

    logger.info("Feature columns classified:")
    logger.info("  - Continuous numerical features: %d", len(continuous_cols))
    logger.info("  - Binary features: %d", len(binary_cols))
    logger.info("  - Categorical features: %d", len(categorical_cols))

    # 2. Imputation
    logger.info("Fitting and applying SimpleImputer (strategy='median')...")
    imputer = SimpleImputer(strategy="median")
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features.drop(columns=categorical_cols)),
        columns=features.drop(columns=categorical_cols).columns,
        index=features.index
    )

    # 3. Scaling continuous numerical variables
    logger.info("Fitting and applying StandardScaler to continuous variables...")
    scaler = StandardScaler()
    features_imputed[continuous_cols] = scaler.fit_transform(features_imputed[continuous_cols])

    # 4. Encoding categorical features (if any are present)
    encoder = None
    if categorical_cols:
        logger.info("Fitting and applying OneHotEncoder to categorical features...")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_arr = encoder.fit_transform(features[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=features.index)
        processed_features = pd.concat([features_imputed, encoded_df], axis=1)
    else:
        logger.info("No categorical columns detected. Skipping encoding.")
        processed_features = features_imputed

    # Re-attach target variable
    processed_df = pd.concat([processed_features, target], axis=1)
    return processed_df, imputer, scaler, encoder


def run() -> Path:
    """Run the feature engineering, imputation, and scaling pipeline."""
    input_path = PROCESSED_DIR / "matches_final.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found at: {input_path}")

    logger.info("=" * 60)
    logger.info("Starting Phase 2 Feature Engineering, Scaling, and Encoding")
    logger.info("=" * 60)

    # Load dataset
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows for feature processing.", len(df))

    # Add advanced interaction features
    df_advanced = add_advanced_features(df)

    # Run processing
    df_processed, imputer, scaler, encoder = process_features(df_advanced)

    # Save processed dataset
    output_path = PROCESSED_DIR / "matches_engineered.csv"
    df_processed.to_csv(output_path, index=False)
    logger.info("Saved engineered and scaled dataset to %s", output_path)

    # Save fitted processors
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    if encoder is not None:
        with open(MODELS_DIR / "encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

    logger.info("Processor objects saved successfully under %s", MODELS_DIR)

    # Quick integrity checks
    null_count = df_processed.isna().sum().sum()
    if null_count > 0:
        logger.warning("Processed dataset contains %d missing values! Review imputation.", null_count)
    else:
        logger.info("Integrity check passed: 0 missing values remain.")

    # Log scaling diagnostics
    continuous_cols = [c for c in df_processed.columns if c not in ["is_neutral", COL_RESULT]]
    means = df_processed[continuous_cols].mean()
    stds = df_processed[continuous_cols].std()
    logger.info(
        "Scaling diagnostics (Continuous features): Mean max absolute dev = %.3e, Std dev max absolute dev = %.3e",
        np.abs(means).max(),
        np.abs(stds - 1.0).max()
    )

    logger.info("Feature engineering and processing pipeline complete.")
    return output_path


def main() -> None:
    """CLI entry point."""
    try:
        path = run()
        print(f"\nFeature Engineering Complete. Dataset saved to: {path}")
    except Exception as exc:
        logger.exception("Feature engineering failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
