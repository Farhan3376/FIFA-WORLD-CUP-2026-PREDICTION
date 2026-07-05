"""Phase 1 - Step 6: Exploratory visualizations.

Generates and saves five diagnostic figures to ``reports/figures/``:

1. ``missing_values_heatmap.png``   — NaN pattern across the feature dataset.
2. ``matches_by_year.png``          — match count per year (1872 → present).
3. ``top20_teams.png``              — teams with most international matches.
4. ``goals_distribution.png``       — home/away goals histogram + KDE.
5. ``tournament_distribution.png``  — top 15 tournaments by match count.

Design notes
------------
* All figures use a consistent dark-background style suitable for a portfolio.
* Each function is self-contained and can be called independently.
* Figures are saved at 150 dpi (print-quality without excessive file size).
* ``matplotlib`` is configured non-interactively so this module runs safely
  in headless / server environments with no display.

Run directly::

    python -m src.visualizations
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # headless / no-display environments

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (  # noqa: E402
    COL_AWAY_GOALS,
    COL_AWAY_TEAM,
    COL_DATE,
    COL_HOME_GOALS,
    COL_HOME_TEAM,
    COL_RESULT,
    COL_TOURNAMENT,
    FIGURES_DIR,
    INTERIM_DIR,
    Settings,
    ensure_directories,
    load_settings,
    set_global_seed,
    RESULT_AWAY_WIN,
    RESULT_DRAW,
    RESULT_HOME_WIN,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
_PALETTE = ["#00d4aa", "#ff6b6b", "#ffd93d", "#6bcbef", "#c77dff"]
_BG_COLOR = "#0f1117"
_GRID_COLOR = "#2a2d3e"
_TEXT_COLOR = "#e0e0e0"
_DPI = 150
_FIGSIZE_WIDE = (14, 6)
_FIGSIZE_SQUARE = (10, 8)
_FIGSIZE_TALL = (12, 8)


def _apply_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply the shared dark style to an Axes object."""
    ax.set_facecolor(_BG_COLOR)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=10)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to ``reports/figures/`` and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=_DPI, bbox_inches="tight",
                facecolor=_BG_COLOR, edgecolor="none")
    plt.close(fig)
    logger.info("Figure saved -> %s", path)
    return path


# ===========================================================================
# Figure 1 — Missing value heatmap
# ===========================================================================

def plot_missing_heatmap(df: pd.DataFrame) -> Path:
    """Heatmap of missing values across the feature dataset.

    Rows are sampled to keep the figure legible (max 2 000 rows shown).

    Args:
        df: The feature-enriched DataFrame.

    Returns:
        Path to the saved figure.
    """
    # Keep only columns that have at least one NaN.
    null_cols = df.columns[df.isna().any()].tolist()
    if not null_cols:
        logger.info("No missing values — skipping heatmap.")
        # Save a simple text figure so downstream report doesn't break.
        fig, ax = plt.subplots(figsize=(6, 3), facecolor=_BG_COLOR)
        ax.text(0.5, 0.5, "No missing values detected",
                ha="center", va="center", color=_TEXT_COLOR, fontsize=14)
        ax.axis("off")
        return _save(fig, "missing_values_heatmap.png")

    sample = df[null_cols].sample(min(2000, len(df)), random_state=42)
    null_matrix = sample.isna().astype(int)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, facecolor=_BG_COLOR)
    sns.heatmap(
        null_matrix.T,
        ax=ax,
        cmap=["#1a1d2e", "#ff6b6b"],
        cbar=False,
        yticklabels=True,
        xticklabels=False,
        linewidths=0,
    )
    _apply_style(ax, "Missing Value Pattern (red = NaN)", xlabel="Sampled Rows")
    ax.set_ylabel("Feature Column", fontsize=10)
    ax.tick_params(axis="y", labelsize=8, colors=_TEXT_COLOR)

    # Add NaN % annotations.
    for i, col in enumerate(null_cols):
        pct = df[col].isna().mean() * 100
        ax.text(
            len(sample) + len(sample) * 0.01, i + 0.5,
            f" {pct:.1f}%", va="center", color="#ff6b6b", fontsize=7.5,
        )

    fig.tight_layout()
    return _save(fig, "missing_values_heatmap.png")


# ===========================================================================
# Figure 2 — Match distribution by year
# ===========================================================================

def plot_matches_by_year(df: pd.DataFrame) -> Path:
    """Bar chart of international matches played per year.

    Args:
        df: DataFrame with a parsed ``date`` column.

    Returns:
        Path to the saved figure.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df[COL_DATE]).dt.year
    counts = df.groupby("year").size().reset_index(name="matches")

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, facecolor=_BG_COLOR)

    # Colour bars by era for visual interest.
    colors = [
        "#6bcbef" if y < 1930
        else "#00d4aa" if y < 1970
        else "#ffd93d" if y < 2000
        else "#ff6b6b"
        for y in counts["year"]
    ]
    ax.bar(counts["year"], counts["matches"], color=colors, width=0.85, alpha=0.9)

    # Era labels.
    for era, year, label in [
        ("#6bcbef", 1900, "Early era"),
        ("#00d4aa", 1950, "Post-WW2"),
        ("#ffd93d", 1985, "Modern era"),
        ("#ff6b6b", 2010, "Recent"),
    ]:
        ax.text(year, counts["matches"].max() * 0.92, label,
                color=era, fontsize=9, alpha=0.8)

    _apply_style(ax, "International Football Matches by Year", "Year", "Matches Played")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return _save(fig, "matches_by_year.png")


# ===========================================================================
# Figure 3 — Top 20 teams by matches played
# ===========================================================================

def plot_top20_teams(df: pd.DataFrame) -> Path:
    """Horizontal bar chart of the 20 most active international teams.

    Args:
        df: DataFrame with ``home_team`` and ``away_team`` columns.

    Returns:
        Path to the saved figure.
    """
    all_teams = pd.concat([
        df[COL_HOME_TEAM].rename("team"),
        df[COL_AWAY_TEAM].rename("team"),
    ])
    top20 = all_teams.value_counts().head(20).sort_values()

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=_BG_COLOR)

    cmap = plt.cm.get_cmap("cool", len(top20))
    colors = [cmap(i) for i in range(len(top20))]

    bars = ax.barh(top20.index, top20.values, color=colors, height=0.75, alpha=0.92)

    # Value labels on bars.
    for bar, val in zip(bars, top20.values):
        ax.text(
            val + top20.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", color=_TEXT_COLOR, fontsize=9,
        )

    _apply_style(ax, "Top 20 Teams by Total International Matches Played",
                 "Matches Played", "")
    ax.tick_params(axis="y", labelsize=10, colors=_TEXT_COLOR)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="x", color=_GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "top20_teams.png")


# ===========================================================================
# Figure 4 — Goals distribution
# ===========================================================================

def plot_goals_distribution(df: pd.DataFrame) -> Path:
    """Histogram + KDE of home and away goals per match.

    Args:
        df: DataFrame with ``home_goals`` and ``away_goals`` columns.

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE, facecolor=_BG_COLOR)

    for ax, col, color, label in [
        (axes[0], COL_HOME_GOALS, "#00d4aa", "Home Goals"),
        (axes[1], COL_AWAY_GOALS, "#ff6b6b", "Away Goals"),
    ]:
        data = df[col].dropna().astype(int)
        max_goals = min(int(data.max()), 15)
        bins = range(0, max_goals + 2)

        ax.hist(data, bins=list(bins), color=color, alpha=0.6,
                density=True, edgecolor=_BG_COLOR, linewidth=0.5)

        # Overlay KDE.
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method=0.3)
        xs = np.linspace(0, max_goals, 300)
        ax.plot(xs, kde(xs), color=color, linewidth=2.5)

        mean_val = data.mean()
        ax.axvline(mean_val, color="#ffd93d", linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_val:.2f}")
        ax.legend(fontsize=9, facecolor=_BG_COLOR, labelcolor=_TEXT_COLOR)

        _apply_style(ax, f"{label} Distribution per Match", label, "Density")
        ax.set_facecolor(_BG_COLOR)

    fig.suptitle("Goals Distribution — International Football (1872–present)",
                 color=_TEXT_COLOR, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "goals_distribution.png")


# ===========================================================================
# Figure 5 — Tournament distribution
# ===========================================================================

def plot_tournament_distribution(df: pd.DataFrame) -> Path:
    """Pie + bar dual panel showing top 15 tournaments by match count.

    Args:
        df: DataFrame with a ``tournament`` column.

    Returns:
        Path to the saved figure.
    """
    counts = df[COL_TOURNAMENT].value_counts().head(15)

    fig, (ax_bar, ax_pie) = plt.subplots(
        1, 2, figsize=(16, 7), facecolor=_BG_COLOR,
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # --- Bar chart ---
    cmap = plt.cm.get_cmap("plasma", len(counts))
    bar_colors = [cmap(i / len(counts)) for i in range(len(counts))]

    bars = ax_bar.barh(counts.index[::-1], counts.values[::-1],
                       color=bar_colors[::-1], height=0.75, alpha=0.92)
    for bar, val in zip(bars, counts.values[::-1]):
        ax_bar.text(
            val + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", color=_TEXT_COLOR, fontsize=8.5,
        )
    _apply_style(ax_bar, "Top 15 Tournaments by Match Count", "Matches", "")
    ax_bar.tick_params(axis="y", labelsize=8.5, colors=_TEXT_COLOR)
    ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_bar.grid(axis="x", color=_GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    ax_bar.grid(axis="y", visible=False)

    # --- Pie chart (top 8 + Others) ---
    top8 = counts.head(8)
    other_count = counts.iloc[8:].sum()
    pie_data = pd.concat([top8, pd.Series({"Others": other_count})])
    pie_colors = [cmap(i / 8) for i in range(8)] + ["#555577"]

    wedges, texts, autotexts = ax_pie.pie(
        pie_data.values,
        labels=None,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"linewidth": 0.5, "edgecolor": _BG_COLOR},
    )
    for at in autotexts:
        at.set_color(_TEXT_COLOR)
        at.set_fontsize(7.5)

    ax_pie.legend(
        wedges, pie_data.index.tolist(),
        loc="lower center", bbox_to_anchor=(0.5, -0.18),
        ncol=2, fontsize=7.5, facecolor=_BG_COLOR, labelcolor=_TEXT_COLOR,
        framealpha=0.4,
    )
    ax_pie.set_facecolor(_BG_COLOR)
    ax_pie.set_title("Tournament Share (top 8 + Others)",
                     color=_TEXT_COLOR, fontsize=12, fontweight="bold")

    fig.tight_layout()
    return _save(fig, "tournament_distribution.png")


# ===========================================================================
# Figure 6 (bonus) — Result distribution over time (decade)
# ===========================================================================

def plot_result_trend(df: pd.DataFrame) -> Path:
    """Stacked area chart of result % (home win / draw / away win) by decade.

    Args:
        df: DataFrame with ``date`` and ``result`` columns.

    Returns:
        Path to the saved figure.
    """
    df = df.copy()
    df["decade"] = (pd.to_datetime(df[COL_DATE]).dt.year // 10) * 10
    grp = df.groupby(["decade", COL_RESULT]).size().unstack(fill_value=0)

    # Normalise to percentage.
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    pct = pct.reindex(columns=[RESULT_HOME_WIN, RESULT_DRAW, RESULT_AWAY_WIN], fill_value=0)

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE, facecolor=_BG_COLOR)
    ax.stackplot(
        pct.index, pct[RESULT_HOME_WIN], pct[RESULT_DRAW], pct[RESULT_AWAY_WIN],
        labels=["Home Win", "Draw", "Away Win"],
        colors=["#00d4aa", "#ffd93d", "#ff6b6b"],
        alpha=0.82,
    )
    ax.legend(loc="upper right", fontsize=10,
              facecolor=_BG_COLOR, labelcolor=_TEXT_COLOR, framealpha=0.5)
    _apply_style(ax, "Match Result Trend by Decade (%)", "Decade", "Percentage (%)")
    ax.set_xlim(pct.index.min(), pct.index.max())
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return _save(fig, "result_trend_by_decade.png")


# ===========================================================================
# Orchestration
# ===========================================================================

def run(
    interim_dir: Path = INTERIM_DIR,
    settings: Optional[Settings] = None,
) -> list[Path]:
    """Generate all diagnostic figures and save them to ``reports/figures/``.

    Args:
        interim_dir: Directory containing ``matches_features.csv``.
        settings: Project settings; loaded from default config if omitted.

    Returns:
        List of paths to the saved figure files.

    Raises:
        FileNotFoundError: If ``matches_features.csv`` is absent.
    """
    settings = settings or load_settings()
    ensure_directories()
    set_global_seed(settings.random_seed)

    logger.info("=" * 60)
    logger.info("Starting visualization pipeline.")
    logger.info("=" * 60)

    features_path = interim_dir / "matches_features.csv"
    if not features_path.is_file():
        raise FileNotFoundError(
            f"matches_features.csv not found in {interim_dir}. "
            "Run feature_builder.py first."
        )

    df = pd.read_csv(features_path, low_memory=False)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    logger.info("Loaded %d rows for visualization.", len(df))

    saved: list[Path] = []

    generators = [
        ("Missing value heatmap", lambda: plot_missing_heatmap(df)),
        ("Matches by year", lambda: plot_matches_by_year(df)),
        ("Top 20 teams", lambda: plot_top20_teams(df)),
        ("Goals distribution", lambda: plot_goals_distribution(df)),
        ("Tournament distribution", lambda: plot_tournament_distribution(df)),
        ("Result trend by decade", lambda: plot_result_trend(df)),
    ]

    for name, fn in generators:
        try:
            path = fn()
            saved.append(path)
            logger.info("Generated: %s", name)
        except Exception as exc:
            logger.error("Failed to generate '%s': %s", name, exc)

    logger.info("Visualization pipeline complete. %d figure(s) saved.", len(saved))
    return saved


def main() -> None:
    """CLI entry point."""
    try:
        paths = run()
    except Exception as exc:
        logger.exception("Visualization failed: %s", exc)
        raise

    print("\nVisualization complete")
    print("-" * 40)
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
