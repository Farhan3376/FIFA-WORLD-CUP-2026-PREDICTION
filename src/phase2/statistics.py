"""Phase 2 - Step 3: Statistical Testing and Analysis.

Performs rigorous statistical testing on the processed international football match dataset:
- 95% Confidence Intervals for key feature means.
- Correlation hypothesis testing (Pearson & Spearman) to identify significant linear/rank relations.
- One-Way ANOVA tests to assess if numeric feature means vary across match outcomes (Win, Draw, Loss).
- Chi-Square Contingency tests to determine if categorical variables are independent of match outcomes.

Results are formatted and written to `reports/eda/statistical_testing.txt`.
Logs are written to `logs/phase2.log`.

Run directly::

    python -m src.statistics
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    PROCESSED_DIR,
    EDA_REPORTS_DIR,
    COL_RESULT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="phase2.log")


def calculate_confidence_intervals(df: pd.DataFrame, confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """Calculate 95% confidence intervals for the mean of key numeric features.

    Args:
        df: Processed matches DataFrame.
        confidence: Confidence level (e.g. 0.95).

    Returns:
        Dict: Feature name mapping to lower and upper bounds.
    """
    features = [
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "home_form",
        "away_form",
        "home_rest_days",
        "away_rest_days",
    ]

    ci_results = {}
    for col in features:
        if col not in df.columns:
            continue
        data = df[col].dropna().to_numpy()
        n = len(data)
        if n < 2:
            continue

        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean

        # Using t-distribution (or normal distribution as n is large)
        h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)
        ci_results[col] = (mean - h, mean + h)

    logger.info("Calculated confidence intervals for %d features.", len(ci_results))
    return ci_results


def run_correlation_tests(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Test correlation significance between key numeric features and the result target.

    Args:
        df: Processed matches DataFrame.

    Returns:
        Dict: Feature name mapping to correlation metrics.
    """
    features = [
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "home_avg_goal_diff",
        "away_avg_goal_diff",
        "home_form",
        "away_form",
        "home_rest_days",
        "away_rest_days",
    ]

    results = {}

    for col in features:
        if col not in df.columns:
            continue
        # Drop row pairs where feature is missing
        mask = df[col].notna()
        x = df.loc[mask, col].to_numpy()
        y = df.loc[mask, COL_RESULT].to_numpy()

        if len(x) < 5:
            continue

        try:
            pearson_r, pearson_p = stats.pearsonr(x, y)
        except Exception as exc:
            logger.warning("Pearson correlation failed for %s: %s", col, exc)
            pearson_r, pearson_p = np.nan, np.nan

        try:
            spearman_r, spearman_p = stats.spearmanr(x, y)
        except Exception as exc:
            logger.warning("Spearman correlation failed for %s: %s", col, exc)
            spearman_r, spearman_p = np.nan, np.nan

        results[col] = {
            "pearson_coef": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_coef": spearman_r,
            "spearman_p_value": spearman_p,
        }

    logger.info("Run correlation tests completed for %d features.", len(results))
    return results


def run_anova_tests(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Perform One-Way ANOVA tests for numerical features across target outcomes.

    Assesses if feature means are statistically distinct between Home Wins, Draws, and Away Wins.

    Args:
        df: Processed matches DataFrame.

    Returns:
        Dict: Feature name mapping to (F-statistic, p-value).
    """
    features = [
        "home_elo_before",
        "away_elo_before",
        "elo_diff",
        "home_avg_goals_scored",
        "away_avg_goals_scored",
        "home_avg_goal_diff",
        "away_avg_goal_diff",
        "home_form",
        "away_form",
        "home_rest_days",
        "away_rest_days",
    ]

    results = {}
    for col in features:
        if col not in df.columns:
            continue

        # Split data by outcome classes
        g0 = df.loc[df[COL_RESULT] == 0, col].dropna().to_numpy()
        g1 = df.loc[df[COL_RESULT] == 1, col].dropna().to_numpy()
        g2 = df.loc[df[COL_RESULT] == 2, col].dropna().to_numpy()

        if len(g0) < 5 or len(g1) < 5 or len(g2) < 5:
            continue

        try:
            f_stat, p_val = stats.f_oneway(g0, g1, g2)
            results[col] = (f_stat, p_val)
        except Exception as exc:
            logger.error("ANOVA test failed for %s: %s", col, exc)

    logger.info("ANOVA tests completed for %d features.", len(results))
    return results


def run_chisquare_tests(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Perform Chi-Square Contingency tests of independence for categorical variables vs target outcome.

    Args:
        df: Processed matches DataFrame.

    Returns:
        Dict: Feature name mapping to chi2, p-value, dof, and expected matrix.
    """
    features = ["is_neutral", "tournament_importance", "month"]
    results = {}

    for col in features:
        if col not in df.columns:
            continue

        # Construct contingency table
        contingency_table = pd.crosstab(df[col], df[COL_RESULT])

        try:
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
            results[col] = {
                "chi2": chi2,
                "p_value": p_val,
                "dof": dof,
                "contingency_table": contingency_table,
            }
        except Exception as exc:
            logger.error("Chi-Square test failed for %s: %s", col, exc)

    logger.info("Chi-Square tests completed for %d features.", len(results))
    return results


def format_report(
    ci: Dict[str, Tuple[float, float]],
    corr: Dict[str, Dict[str, Any]],
    anova: Dict[str, Tuple[float, float]],
    chi2: Dict[str, Dict[str, Any]],
    total_records: int
) -> str:
    """Format all statistical testing outputs into a professional text report.

    Args:
        ci: Confidence intervals dict.
        corr: Correlation test results dict.
        anova: ANOVA results dict.
        chi2: Chi-square test results dict.
        total_records: Count of matches evaluated.

    Returns:
        str: Formatted report text.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("         FIFA WORLD CUP 2026 PREDICTION - STATISTICAL TESTING REPORT")
    lines.append("=" * 70)
    lines.append(f"Total Matches Analyzed: {total_records:,}")
    lines.append("\n" + "-" * 70)
    lines.append("1. 95% CONFIDENCE INTERVALS FOR FEATURE MEANS")
    lines.append("-" * 70)
    lines.append(f"{'Feature Name':<30} | {'Lower Bound':<15} | {'Upper Bound':<15}")
    lines.append("-" * 70)
    for col, bounds in sorted(ci.items()):
        lines.append(f"{col:<30} | {bounds[0]:<15.5f} | {bounds[1]:<15.5f}")

    lines.append("\n" + "-" * 70)
    lines.append("2. CORRELATION TESTING WITH MATCH OUTCOME TARGET")
    lines.append("-" * 70)
    lines.append(
        f"{'Feature Name':<20} | "
        f"{'Pearson R':<10} | "
        f"{'Pearson p':<10} | "
        f"{'Spearman R':<10} | "
        f"{'Spearman p':<10}"
    )
    lines.append("-" * 70)
    for col, metrics in sorted(corr.items()):
        pearson_p = metrics["pearson_p_value"]
        spearman_p = metrics["spearman_p_value"]
        p_str = f"{pearson_p:.3e}" if pearson_p > 0 and not math.isnan(pearson_p) else "0.000e+00"
        s_p_str = f"{spearman_p:.3e}" if spearman_p > 0 and not math.isnan(spearman_p) else "0.000e+00"
        lines.append(
            f"{col:<20} | "
            f"{metrics['pearson_coef']:<10.5f} | "
            f"{p_str:<10} | "
            f"{metrics['spearman_coef']:<10.5f} | "
            f"{s_p_str:<10}"
        )

    lines.append("\n" + "-" * 70)
    lines.append("3. ONE-WAY ANOVA TESTS ACROSS RESULT STAGES (HOME, DRAW, AWAY)")
    lines.append("-" * 70)
    lines.append(f"{'Feature Name':<25} | {'F-Statistic':<15} | {'p-value':<15} | {'Significant (alpha=0.05)':<12}")
    lines.append("-" * 70)
    for col, (f_stat, p_val) in sorted(anova.items()):
        sig_str = "Yes" if p_val < 0.05 else "No"
        p_str = f"{p_val:.3e}" if p_val > 0 else "0.000e+00"
        lines.append(f"{col:<25} | {f_stat:<15.5f} | {p_str:<15} | {sig_str:<12}")

    lines.append("\n" + "-" * 70)
    lines.append("4. CHI-SQUARE CONTINGENCY TESTS FOR INDEPENDENCE")
    lines.append("-" * 70)
    for col, data in sorted(chi2.items()):
        p_val = data["p_value"]
        sig_str = "Dependent (p < 0.05)" if p_val < 0.05 else "Independent (p >= 0.05)"
        p_str = f"{p_val:.3e}" if p_val > 0 else "0.000e+00"
        lines.append(f"Categorical Feature: {col}")
        lines.append(f"  - Chi2 Statistic : {data['chi2']:.5f}")
        lines.append(f"  - p-value        : {p_str}")
        lines.append(f"  - DoF            : {data['dof']}")
        lines.append(f"  - Conclusion     : {sig_str}")
        lines.append("\n  - Contingency Table (Counts):")

        # Format contingency table
        ct = data["contingency_table"]
        # Columns represent result classes: 0, 1, 2
        header = f"    {'Value':<12} | {'Home Win (0)':<12} | {'Draw (1)':<12} | {'Away Win (2)':<12}"
        lines.append(header)
        lines.append("    " + "-" * len(header))
        for val, row in ct.iterrows():
            lines.append(f"    {str(val):<12} | {row[0]:<12,} | {row[1]:<12,} | {row[2]:<12,}")
        lines.append("\n" + "-" * 70)

    return "\n".join(lines)


def run() -> Path:
    """Run the statistics pipeline."""
    dataset_path = PROCESSED_DIR / "matches_final.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found at: {dataset_path}")

    logger.info("=" * 60)
    logger.info("Starting Phase 2 Statistical Analysis Pipeline")
    logger.info("=" * 60)

    df = pd.read_csv(dataset_path)

    # Perform statistical tests
    ci = calculate_confidence_intervals(df)
    corr = run_correlation_tests(df)
    anova = run_anova_tests(df)
    chi2 = run_chisquare_tests(df)

    # Generate text report
    report_text = format_report(ci, corr, anova, chi2, len(df))

    # Save report
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EDA_REPORTS_DIR / "statistical_testing.txt"
    report_path.write_text(report_text, encoding="utf-8")

    logger.info("Saved statistical testing report -> %s", report_path)
    logger.info("Statistical analysis pipeline completed successfully.")

    return report_path


def main() -> None:
    """CLI entry point."""
    try:
        path = run()
        print(f"\nStatistical Testing Complete. Report saved to: {path}")
    except Exception as exc:
        logger.exception("Statistical analysis failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
