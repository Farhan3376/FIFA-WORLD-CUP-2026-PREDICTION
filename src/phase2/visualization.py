"""Phase 2 - Step 2: Exploratory Data Visualization.

Generates 20 publication-quality, 300 DPI figures and saves them in `reports/figures/`:
- Missing value distributions
- Target class balance
- Chronological match counts and decadal result trends
- Categorical impact (neutral venue, tournament importance)
- Feature correlation matrices (Pearson & Spearman)
- Feature distributions (histograms + KDEs) for key numerical variables
- Bivariate relations (feature difference vs match outcome boxplots)

Run directly::

    python -m src.visualization
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from src.config import (
    PROCESSED_DIR,
    FIGURES_DIR,
    COL_RESULT,
    RESULT_LABELS,
)
from src.utils.logger import get_logger

matplotlib.use("Agg")  # Non-interactive backend
logger = get_logger(__name__, log_filename="phase2.log")

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
_PALETTE = ["#00d4aa", "#ffd93d", "#ff6b6b", "#6bcbef", "#c77dff"]
_BG_COLOR = "#0f1117"
_GRID_COLOR = "#2a2d3e"
_TEXT_COLOR = "#e0e0e0"
_DPI = 300


def _apply_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply unified dark portfolio styling to an Axes object.

    Args:
        ax: Matplotlib axes.
        title: Title of the subplot.
        xlabel: Label for X-axis.
        ylabel: Label for Y-axis.
    """
    ax.set_facecolor(_BG_COLOR)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)

    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure at 300 DPI to the figures directory.

    Args:
        fig: Matplotlib Figure instance.
        name: Name of the output image file.

    Returns:
        Path: Path to the saved figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(
        path,
        dpi=_DPI,
        bbox_inches="tight",
        facecolor=_BG_COLOR,
        edgecolor="none"
    )
    plt.close(fig)
    logger.info("Saved figure -> %s", path)
    return path


def plot_missing_distribution(df: pd.DataFrame) -> Path:
    """1. Bar plot showing missing values percentages per feature."""
    missing_pcts = df.isna().mean() * 100
    missing_pcts = missing_pcts[missing_pcts > 0].sort_values()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG_COLOR)
    if missing_pcts.empty:
        ax.text(0.5, 0.5, "No missing values detected in any feature.",
                ha="center", va="center", color=_TEXT_COLOR, fontsize=12)
        ax.axis("off")
    else:
        bars = ax.barh(
            missing_pcts.index.to_numpy(),
            missing_pcts.to_numpy(),
            color="#ff6b6b",
            height=0.6,
            alpha=0.9
        )
        for bar, val in zip(bars, missing_pcts.values):
            ax.text(
                val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=_TEXT_COLOR, fontsize=8
            )
        ax.set_xlim(0, 110)
        _apply_style(ax, "Missing Values Percentage by Feature", "Percentage (%)", "")
        ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.5)
        ax.grid(axis="y", visible=False)

    return _save(fig, "eda_01_missing_values.png")


def plot_target_distribution(df: pd.DataFrame) -> Path:
    """2. Target variable distribution class count & percentage."""
    counts = df[COL_RESULT].value_counts().reindex([0, 1, 2])
    labels = [RESULT_LABELS[i] for i in [0, 1, 2]]
    pcts = (counts / len(df)) * 100

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG_COLOR)
    bars = ax.bar(labels, counts.to_numpy(), color=["#00d4aa", "#ffd93d", "#ff6b6b"], width=0.6, alpha=0.85)

    for bar, val, pct in zip(bars, counts.values, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + len(df) * 0.015,
            f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", color=_TEXT_COLOR, fontsize=9
        )

    ax.set_ylim(0, counts.max() * 1.15)
    _apply_style(ax, "Match Outcome Class Balance (Target Variable)", "Result Outcome", "Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return _save(fig, "eda_02_target_distribution.png")


def plot_match_trends(df: pd.DataFrame) -> Path:
    """3. Chronological match count trends per year."""
    counts = df.groupby("year").size()
    x = counts.index.to_numpy()
    y = counts.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=_BG_COLOR)
    ax.plot(x, y, color="#6bcbef", linewidth=2, alpha=0.9)
    ax.fill_between(x, y, color="#6bcbef", alpha=0.15)

    _apply_style(ax, "Historical International Football Matches Over Time", "Year", "Matches Played")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return _save(fig, "eda_03_match_trends.png")


def plot_neutral_impact(df: pd.DataFrame) -> Path:
    """4. Stacked bar chart showing match outcomes for neutral vs home venue."""
    # Group results by neutral venue indicator
    grouped = df.groupby(["is_neutral", COL_RESULT]).size().unstack(fill_value=0)
    pcts = grouped.div(grouped.sum(axis=1), axis=0) * 100

    labels = ["Home / Away Stadium", "Neutral Venue"]
    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG_COLOR)
    ax.bar(x, pcts[0].to_numpy(), width, label="Home Win", color="#00d4aa", alpha=0.85)
    ax.bar(x, pcts[1].to_numpy(), width, bottom=pcts[0].to_numpy(), label="Draw", color="#ffd93d", alpha=0.85)
    ax.bar(
        x,
        pcts[2].to_numpy(),
        width,
        bottom=(pcts[0] + pcts[1]).to_numpy(),
        label="Away Win",
        color="#ff6b6b",
        alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", facecolor=_BG_COLOR, labelcolor=_TEXT_COLOR, framealpha=0.5)
    _apply_style(ax, "Impact of Stadium Neutrality on Match Outcomes", "Venue Status", "Percentage (%)")
    return _save(fig, "eda_04_neutral_impact.png")


def plot_tournament_types(df: pd.DataFrame) -> Path:
    """5. Match count distribution by tournament importance level."""
    counts = df["tournament_importance"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG_COLOR)
    bars = ax.bar(counts.index.astype(str).to_numpy(), counts.to_numpy(), color="#c77dff", width=0.5, alpha=0.85)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + len(df) * 0.01,
            f"{val:,}", ha="center", va="bottom", color=_TEXT_COLOR, fontsize=8
        )

    ax.set_ylim(0, counts.max() * 1.12)
    _apply_style(ax, "Match Counts by Tournament Weight (Importance Factor)", "Importance Score", "Matches Played")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return _save(fig, "eda_05_tournament_types.png")


def _plot_correlation_matrix(df: pd.DataFrame, method: str, filename: str) -> Path:
    """Helper to generate correlation matrices."""
    # Filter numerical variables, dropping columns that are completely empty
    numerical = df.select_dtypes(include=[np.number]).dropna(how='all', axis=1)

    # Exclude result targets and temporal variables to keep heatmap tidy
    to_exclude = [COL_RESULT, "year", "month"]
    cols = [c for c in numerical.columns if c not in to_exclude]

    # Fill remaining NaNs with median for corr calculation
    corr_data = df[cols].fillna(df[cols].median())
    corr_matrix = corr_data.corr(method=method)

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=_BG_COLOR)

    # Darker mask/map
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6.5, "color": "#000000" if _TEXT_COLOR == "#e0e0e0" else _TEXT_COLOR},
        linewidths=0.5,
        linecolor=_BG_COLOR,
        cbar_kws={"shrink": 0.8}
    )

    # Style heatmap elements
    ax.tick_params(axis="both", colors=_TEXT_COLOR, labelsize=7.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COLOR, labelcolor=_TEXT_COLOR)

    title_str = f"Feature Correlation Matrix Heatmap ({method.capitalize()})"
    _apply_style(ax, title_str, "", "")
    ax.grid(visible=False)

    return _save(fig, filename)


def plot_pearson_correlation(df: pd.DataFrame) -> Path:
    """6. Pearson correlation heatmap."""
    return _plot_correlation_matrix(df, "pearson", "eda_06_pearson_correlation.png")


def plot_spearman_correlation(df: pd.DataFrame) -> Path:
    """7. Spearman correlation heatmap."""
    return _plot_correlation_matrix(df, "spearman", "eda_07_spearman_correlation.png")


def _plot_density_histogram(df: pd.DataFrame, col: str, title: str, filename: str, color: str) -> Path:
    """Helper to generate feature distribution plot using manual histogram/KDE to avoid pandas/sns conflicts."""
    data = df[col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=_BG_COLOR)

    if len(data) > 0:
        data_arr = data.to_numpy()
        # Matplotlib manual histogram
        ax.hist(
            data_arr,
            bins=30,
            density=True,
            color=color,
            alpha=0.45,
            edgecolor="none"
        )

        # Scipy manual KDE plot
        try:
            kde = gaussian_kde(data_arr)
            xs = np.linspace(float(data_arr.min()), float(data_arr.max()), 300)
            ax.plot(xs, kde(xs), color=color, linewidth=1.8)
        except Exception as exc:
            logger.warning("Failed to estimate KDE for %s: %s", col, exc)

        # Add visual reference markers
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color="#ffd93d", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="#ff6b6b", linestyle=":", linewidth=1.2, label=f"Median: {median_val:.2f}")
        ax.legend(fontsize=8, facecolor=_BG_COLOR, labelcolor=_TEXT_COLOR, framealpha=0.5)

    _apply_style(ax, title, col, "Density")
    return _save(fig, filename)


def plot_feature_distributions(df: pd.DataFrame) -> List[Path]:
    """8 to 16. Plot distributions for Elo metrics, average goals, and form."""
    dists = [
        ("home_elo_before", "Home Team pre-match Elo rating", "eda_08_home_elo_dist.png", "#00d4aa"),
        ("away_elo_before", "Away Team pre-match Elo rating", "eda_09_away_elo_dist.png", "#ff6b6b"),
        ("elo_diff", "Elo Rating Difference (Home - Away)", "eda_10_elo_diff_dist.png", "#ffd93d"),
        ("home_avg_goals_scored", "Home Team Rolling Avg Goals Scored",
         "eda_11_home_goals_dist.png", "#6bcbef"),
        ("away_avg_goals_scored", "Away Team Rolling Avg Goals Scored",
         "eda_12_away_goals_dist.png", "#c77dff"),
        ("home_avg_goals_conceded", "Home Team Rolling Avg Goals Conceded",
         "eda_13_home_goals_conceded_dist.png", "#ff6b6b"),
        ("away_avg_goals_conceded", "Away Team Rolling Avg Goals Conceded",
         "eda_14_away_goals_conceded_dist.png", "#00d4aa"),
        ("home_form", "Home Team Form Rating Index (0-1)", "eda_15_home_form_dist.png", "#00d4aa"),
        ("away_form", "Away Team Form Rating Index (0-1)", "eda_16_away_form_dist.png", "#ff6b6b"),
    ]

    saved_paths = []
    for col, title, fname, color in dists:
        try:
            if col in df.columns:
                saved_paths.append(_plot_density_histogram(df, col, title, fname, color))
        except Exception as exc:
            logger.error("Failed to generate distribution plot for %s: %s", col, exc)

    return saved_paths


def plot_elo_diff_vs_result(df: pd.DataFrame) -> Path:
    """17. Boxplot of Elo difference grouped by match results."""
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=_BG_COLOR)
    sns.boxplot(
        data=df,
        x=COL_RESULT,
        y="elo_diff",
        ax=ax,
        hue=COL_RESULT,
        legend=False,
        palette=["#00d4aa", "#ffd93d", "#ff6b6b"],
        width=0.45,
        flierprops={"marker": "x", "markeredgecolor": "#ff6b6b", "markersize": 3, "alpha": 0.4}
    )

    # Use FixedLocator to set ticks explicitly, satisfying matplotlib warnings
    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 2]))
    ax.set_xticklabels([RESULT_LABELS[i] for i in [0, 1, 2]])
    _apply_style(ax, "Elo Rating Difference by Match Outcome", "Match Result Target", "Elo Difference (Home - Away)")
    return _save(fig, "eda_17_elo_diff_vs_result.png")


def plot_goal_diff_vs_result(df: pd.DataFrame) -> Path:
    """18. Boxplot of rolling goals difference difference vs match outcome."""
    df = df.copy()
    # Compute relative difference of rolling averages
    df["rolling_goal_diff_diff"] = df["home_avg_goal_diff"] - df["away_avg_goal_diff"]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=_BG_COLOR)
    sns.boxplot(
        data=df,
        x=COL_RESULT,
        y="rolling_goal_diff_diff",
        ax=ax,
        hue=COL_RESULT,
        legend=False,
        palette=["#00d4aa", "#ffd93d", "#ff6b6b"],
        width=0.45,
        flierprops={"marker": "x", "markeredgecolor": "#6bcbef", "markersize": 3, "alpha": 0.4}
    )

    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 2]))
    ax.set_xticklabels([RESULT_LABELS[i] for i in [0, 1, 2]])
    _apply_style(
        ax,
        "Rolling Goal Differential Difference by Match Outcome",
        "Match Result Target",
        "Rolling Goal Diff Difference (Home - Away)"
    )
    return _save(fig, "eda_18_goal_diff_vs_result.png")


def plot_form_vs_result(df: pd.DataFrame) -> Path:
    """19. Boxplot of recent form difference vs match outcome."""
    df = df.copy()
    df["form_diff"] = df["home_form"] - df["away_form"]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=_BG_COLOR)
    sns.boxplot(
        data=df,
        x=COL_RESULT,
        y="form_diff",
        ax=ax,
        hue=COL_RESULT,
        legend=False,
        palette=["#00d4aa", "#ffd93d", "#ff6b6b"],
        width=0.45,
        flierprops={"marker": "x", "markeredgecolor": "#c77dff", "markersize": 3, "alpha": 0.4}
    )

    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 2]))
    ax.set_xticklabels([RESULT_LABELS[i] for i in [0, 1, 2]])
    _apply_style(ax, "Recent Form Index Difference by Match Outcome", "Match Result Target", "Form Diff (Home - Away)")
    return _save(fig, "eda_19_form_vs_result.png")


def plot_decadal_trends(df: pd.DataFrame) -> Path:
    """20. Stacked area chart showing outcome trends over decades."""
    df = df.copy()
    df["decade"] = (df["year"] // 10) * 10
    grouped = df.groupby(["decade", COL_RESULT]).size().unstack(fill_value=0)
    pcts = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # Ensure all target options are mapped
    pcts = pcts.reindex(columns=[0, 1, 2], fill_value=0.0)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=_BG_COLOR)
    ax.stackplot(
        pcts.index.to_numpy(),
        pcts[0].to_numpy(), pcts[1].to_numpy(), pcts[2].to_numpy(),
        labels=["Home Win", "Draw", "Away Win"],
        colors=["#00d4aa", "#ffd93d", "#ff6b6b"],
        alpha=0.8
    )

    ax.set_xlim(pcts.index.min(), pcts.index.max())
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", facecolor=_BG_COLOR, labelcolor=_TEXT_COLOR, framealpha=0.5)
    _apply_style(ax, "Decadal Trends in Match Outcome Percentages", "Decade", "Outcome Share (%)")
    return _save(fig, "eda_20_decadal_trends.png")


def run() -> List[Path]:
    """Load the final dataset and generate all 20 publication-quality plots."""
    dataset_path = PROCESSED_DIR / "matches_final.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Processed training dataset not found at: {dataset_path}")

    logger.info("=" * 60)
    logger.info("Starting Phase 2 Exploratory Visualization Pipeline (300 DPI)")
    logger.info("=" * 60)

    df = pd.read_csv(dataset_path)
    logger.info("Loaded %d rows for visualization.", len(df))

    saved_paths: List[Path] = []

    generators = [
        ("Missing distribution", lambda: plot_missing_distribution(df)),
        ("Target distribution", lambda: plot_target_distribution(df)),
        ("Match trends", lambda: plot_match_trends(df)),
        ("Neutral impact", lambda: plot_neutral_impact(df)),
        ("Tournament types", lambda: plot_tournament_types(df)),
        ("Pearson Correlation", lambda: plot_pearson_correlation(df)),
        ("Spearman Correlation", lambda: plot_spearman_correlation(df)),
        ("Feature distributions", lambda: plot_feature_distributions(df)),
        ("Elo diff vs result", lambda: plot_elo_diff_vs_result(df)),
        ("Goal diff vs result", lambda: plot_goal_diff_vs_result(df)),
        ("Form vs result", lambda: plot_form_vs_result(df)),
        ("Decadal trends", lambda: plot_decadal_trends(df)),
    ]

    for name, fn in generators:
        try:
            res = fn()
            if isinstance(res, list):
                saved_paths.extend(res)
                logger.info("Generated: %s (%d plots)", name, len(res))
            else:
                saved_paths.append(res)
                logger.info("Generated: %s", name)
        except Exception as exc:
            logger.error("Failed to generate '%s': %s", name, exc)

    logger.info("Visualization pipeline complete. Generated %d plots.", len(saved_paths))
    return saved_paths


def main() -> None:
    """CLI entry point."""
    try:
        paths = run()
        print(f"\nVisualization complete. Generated {len(paths)} plots at reports/figures/")
    except Exception as exc:
        logger.exception("Visualization failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
