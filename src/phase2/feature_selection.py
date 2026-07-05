"""Phase 2 - Step 5: Feature Selection and Outlier Capping.

This module implements the feature selection and outlier handling pipeline:
1. Applies Winsorization (capping outliers at the 1st and 99th percentiles) on continuous variables.
2. Filters features using Variance Thresholding (threshold=0.01) to remove near-constant columns.
3. Filters highly collinear features (|r| > 0.85) using Mutual Information to resolve ties.
4. Runs Recursive Feature Elimination (RFE) using a baseline Random Forest estimator to select the top 15 features.
5. Computes Permutation Importance on a validation split to prune features with zero or negative contribution.
6. Saves the final selected features list to `models/feature_selector.pkl`.
7. Exports the final dataset to `data/processed/matches_selected.csv`.
8. Generates a detailed text report `reports/eda/feature_selection_report.txt`.

Run directly::

    python -m src.feature_selection
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, VarianceThreshold, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import (
    COL_RESULT,
    EDA_REPORTS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="phase2.log")


def winsorize_series(
    series: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> Tuple[pd.Series, int]:
    """Winsorize (cap) a numeric series at specified quantiles.

    Args:
        series: Input pandas Series.
        lower_quantile: Lower quantile threshold (default 0.01).
        upper_quantile: Upper quantile threshold (default 0.99).

    Returns:
        Tuple[pd.Series, int]: The winsorized series and the count of capped values.
    """
    q_low = series.quantile(lower_quantile)
    q_high = series.quantile(upper_quantile)

    # Count how many values are capped
    capped_lower = (series < q_low).sum()
    capped_upper = (series > q_high).sum()
    total_capped = int(capped_lower + capped_upper)

    capped_series = series.clip(lower=q_low, upper=q_high)
    return capped_series, total_capped


def apply_variance_threshold(
    df: pd.DataFrame,
    features_to_check: List[str],
    threshold: float = 0.01
) -> Tuple[List[str], List[str]]:
    """Filter out features with variance below threshold.

    Args:
        df: Input DataFrame.
        features_to_check: List of feature names.
        threshold: Variance threshold value.

    Returns:
        Tuple[List[str], List[str]]: Selected feature names and dropped feature names.
    """
    selector = VarianceThreshold(threshold=threshold)
    # Fit selector
    selector.fit(df[features_to_check])

    support = selector.get_support()
    selected_features = [f for f, s in zip(features_to_check, support) if s]
    dropped_features = [f for f, s in zip(features_to_check, support) if not s]

    return selected_features, dropped_features


def apply_correlation_filtering(
    df: pd.DataFrame,
    features_to_check: List[str],
    target_col: str,
    threshold: float = 0.85
) -> Tuple[List[str], List[str], List[str]]:
    """Remove highly correlated features by dropping the one with lower Mutual Information.

    Args:
        df: Input DataFrame.
        features_to_check: List of feature names.
        target_col: Target column name.
        threshold: Absolute correlation threshold.

    Returns:
        Tuple[List[str], List[str], List[str]]: Selected feature names, dropped feature names, and details of drops.
    """
    selected_features = list(features_to_check)
    dropped_features = []
    drop_details = []

    # Compute correlation matrix
    corr_matrix = df[selected_features].corr(method="pearson").abs()

    # Find highly correlated pairs
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]
            val = corr_matrix.iloc[i, j]
            if val > threshold:
                pairs.append((col_a, col_b, val))

    # Sort pairs by correlation value descending
    pairs.sort(key=lambda x: x[2], reverse=True)

    if not pairs:
        return selected_features, dropped_features, drop_details

    # We will compute Mutual Information for features involved in collinearity
    collinear_features = set()
    for col_a, col_b, _ in pairs:
        collinear_features.add(col_a)
        collinear_features.add(col_b)

    collinear_list = list(collinear_features)

    # Compute Mutual Info (using a sample to speed it up if necessary)
    sample_size = min(20000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    mi_scores = mutual_info_classif(
        df_sample[collinear_list].fillna(0),
        df_sample[target_col],
        random_state=42
    )
    mi_dict = dict(zip(collinear_list, mi_scores))

    # Process pairs and drop features
    already_dropped = set()
    for col_a, col_b, corr_val in pairs:
        if col_a in already_dropped or col_b in already_dropped:
            continue

        mi_a = mi_dict[col_a]
        mi_b = mi_dict[col_b]

        if mi_a >= mi_b:
            drop_col = col_b
            keep_col = col_a
            lower_mi, higher_mi = mi_b, mi_a
        else:
            drop_col = col_a
            keep_col = col_b
            lower_mi, higher_mi = mi_a, mi_b

        already_dropped.add(drop_col)
        dropped_features.append(drop_col)
        selected_features.remove(drop_col)

        detail_msg = (
            f"Dropped '{drop_col}' due to correlation with '{keep_col}' (|r| = {corr_val:.4f}). "
            f"MI('{keep_col}'): {higher_mi:.5f} >= MI('{drop_col}'): {lower_mi:.5f}"
        )
        drop_details.append(detail_msg)
        logger.info(detail_msg)

    return selected_features, dropped_features, drop_details


def apply_rfe(
    df: pd.DataFrame,
    features_to_check: List[str],
    target_col: str,
    n_features_to_select: int = 15
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Run Recursive Feature Elimination to select the top N features.

    Args:
        df: Input DataFrame.
        features_to_check: List of feature names.
        target_col: Target column name.
        n_features_to_select: Number of features to select.

    Returns:
        Tuple[List[str], List[str], Dict[str, int]]: Selected feature names, dropped feature names, and ranking dictionary.
    """
    sample_size = min(20000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    X = df_sample[features_to_check]
    y = df_sample[target_col]

    estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    support = rfe.support_
    ranking = rfe.ranking_

    selected_features = [f for f, s in zip(features_to_check, support) if s]
    dropped_features = [f for f, s in zip(features_to_check, support) if not s]

    ranking_dict = dict(zip(features_to_check, ranking))

    return selected_features, dropped_features, ranking_dict


def apply_permutation_importance(
    df: pd.DataFrame,
    features_to_check: List[str],
    target_col: str
) -> Tuple[List[str], List[str], Dict[str, float]]:
    """Prune features with zero or negative permutation importance.

    Args:
        df: Input DataFrame.
        features_to_check: List of feature names.
        target_col: Target column name.

    Returns:
        Tuple[List[str], List[str], Dict[str, float]]: Selected feature names, dropped feature names, and importance scores dictionary.
    """
    X = df[features_to_check]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    result = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1
    )

    importances_mean = result.importances_mean
    importance_dict = dict(zip(features_to_check, importances_mean))

    selected_features = []
    dropped_features = []

    for f in features_to_check:
        score = importance_dict[f]
        if score > 0:
            selected_features.append(f)
        else:
            dropped_features.append(f)
            logger.info(
                "Dropped '%s' due to zero or negative permutation importance (%f)",
                f, score
            )

    return selected_features, dropped_features, importance_dict


def run() -> Path:
    """Run the feature selection pipeline."""
    input_path = PROCESSED_DIR / "matches_engineered.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Engineered dataset not found at: {input_path}")

    logger.info("=" * 60)
    logger.info("Starting Phase 2 Feature Selection Pipeline")
    logger.info("=" * 60)

    # Load dataset
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    logger.info("Loaded engineered dataset: %s", initial_shape)

    # 1. Outlier Capping (Winsorization) on continuous variables
    logger.info("Applying Winsorization (1% - 99% clipping) to continuous features...")
    df_capped = df.copy()

    # Identify continuous columns to winsorize (exclude result, is_neutral, year, month)
    exclude_cols = [COL_RESULT, "is_neutral", "year", "month"]
    continuous_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    winsorize_stats = {}
    for col in continuous_cols:
        df_capped[col], total_capped = winsorize_series(df[col], 0.01, 0.99)
        winsorize_stats[col] = total_capped

    logger.info("Winsorization complete. Capped outliers in %d columns.", len(continuous_cols))

    # Initialize features list (all columns except the target result)
    initial_features = [c for c in df_capped.columns if c != COL_RESULT]

    # 2. Variance Thresholding
    logger.info("Applying Variance Thresholding...")
    var_selected, var_dropped = apply_variance_threshold(
        df_capped, initial_features, threshold=0.01
    )
    logger.info("Variance Thresholding: %d features selected, %d features dropped.", len(var_selected), len(var_dropped))

    # 3. Correlation-based filtering with Mutual Information
    logger.info("Applying Correlation-based filtering (|r| > 0.85)...")
    corr_selected, corr_dropped, corr_details = apply_correlation_filtering(
        df_capped, var_selected, COL_RESULT, threshold=0.85
    )
    logger.info("Correlation filtering: %d features selected, %d features dropped.", len(corr_selected), len(corr_dropped))

    # 4. Recursive Feature Elimination (RFE)
    logger.info("Applying RFE (selecting top 15 features)...")
    rfe_selected, rfe_dropped, rfe_rankings = apply_rfe(
        df_capped, corr_selected, COL_RESULT, n_features_to_select=15
    )
    logger.info("RFE selection complete. Top 15 features selected.")

    # 5. Permutation Importance
    logger.info("Applying Permutation Importance pruning...")
    final_selected, perm_dropped, perm_importances = apply_permutation_importance(
        df_capped, rfe_selected, COL_RESULT
    )
    logger.info("Permutation importance pruning: %d features selected, %d features dropped.", len(final_selected), len(perm_dropped))

    # Save final selected dataset
    output_df = df_capped[final_selected + [COL_RESULT]]
    output_path = PROCESSED_DIR / "matches_selected.csv"
    output_df.to_csv(output_path, index=False)
    logger.info("Saved selected feature dataset to %s (shape: %s)", output_path, output_df.shape)

    # Save selector list
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selector_path = MODELS_DIR / "feature_selector.pkl"
    with open(selector_path, "wb") as f:
        pickle.dump(final_selected, f)
    logger.info("Saved feature selector object to %s", selector_path)

    # Generate Report
    report_lines = [
        "=" * 70,
        "         FIFA WORLD CUP 2026 PREDICTION - FEATURE SELECTION REPORT",
        "=" * 70,
        f"Initial Dataset Shape : {initial_shape}",
        f"Final Dataset Shape   : {output_df.shape}",
        f"Total Initial Features: {len(initial_features)}",
        f"Total Selected Features: {len(final_selected)}",
        f"Target Column         : {COL_RESULT}",
        "\n" + "-" * 70,
        "1. OUTLIER WINSORIZATION DETAILS (1% and 99% quantiles)",
        "-" * 70,
        f"{'Continuous Feature':<30} | {'Total Capped Values':<20} | {'% of Dataset':<10}",
        "-" * 70,
    ]

    for col, cnt in sorted(winsorize_stats.items()):
        pct = (cnt / len(df)) * 100
        report_lines.append(f"{col:<30} | {cnt:<20,} | {pct:<10.2f}%")

    report_lines.extend([
        "\n" + "-" * 70,
        "2. VARIANCE THRESHOLDING (threshold=0.01)",
        "-" * 70,
        f"Dropped Features ({len(var_dropped)}): {var_dropped or 'None'}",
    ])

    report_lines.extend([
        "\n" + "-" * 70,
        "3. CORRELATION FILTERING (|r| > 0.85 with Mutual Info tie-breaking)",
        "-" * 70,
        f"Dropped Features ({len(corr_dropped)}):",
    ])
    for detail in corr_details:
        report_lines.append(f"  - {detail}")
    if not corr_dropped:
        report_lines.append("  - None")

    report_lines.extend([
        "\n" + "-" * 70,
        "4. RECURSIVE FEATURE ELIMINATION (RFE rankings, Top 15 Selected)",
        "-" * 70,
        f"{'Feature Name':<30} | {'RFE Rank':<10} | {'Selected':<10}",
        "-" * 70,
    ])
    sorted_rfe = sorted(rfe_rankings.items(), key=lambda x: x[1])
    for col, rank in sorted_rfe:
        sel_str = "Yes" if rank == 1 else "No"
        report_lines.append(f"{col:<30} | {rank:<10} | {sel_str:<10}")

    report_lines.extend([
        "\n" + "-" * 70,
        "5. PERMUTATION IMPORTANCE PRUNING (Evaluated on validation split)",
        "-" * 70,
        f"{'Feature Name':<30} | {'Mean Validation Importance':<30} | {'Status':<10}",
        "-" * 70,
    ])
    for col in rfe_selected:
        imp = perm_importances[col]
        status = "Kept" if col in final_selected else "Dropped"
        report_lines.append(f"{col:<30} | {imp:<30.6f} | {status:<10}")

    report_lines.extend([
        "\n" + "=" * 70,
        "FINAL SELECTED FEATURE SET",
        "=" * 70,
    ])
    for i, col in enumerate(final_selected, 1):
        report_lines.append(f" {i:2d}. {col}")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EDA_REPORTS_DIR / "feature_selection_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved feature selection report to %s", report_path)

    logger.info("Feature selection pipeline completed successfully.")
    return report_path


def main() -> None:
    """CLI Entry Point."""
    try:
        path = run()
        print(f"\nFeature Selection Complete. Report saved to: {path}")
    except Exception as exc:
        logger.exception("Feature selection pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
