"""Phase 1 — End-to-end pipeline runner.

Executes the full data-engineering pipeline in order:

    Step 1  data_collection  → data/raw/
    Step 2  preprocessing    → data/interim/matches_clean.csv
    Step 3  merge_data       → data/interim/matches_merged.csv
    Step 4  feature_builder  → data/interim/matches_features.csv
                               data/processed/matches_final.csv
    Step 5  validation       → reports/validation_report.txt
    Step 6  visualizations   → reports/figures/*.png

Usage::

    python run_pipeline.py                  # normal run (uses cached raw data)
    python run_pipeline.py --force-download # re-download all raw files
    python run_pipeline.py --skip-viz       # skip figure generation
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

from src.config import (
    FIGURES_DIR,
    INTERIM_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    ensure_directories,
    load_settings,
    set_global_seed,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="pipeline.log")


def _banner(text: str) -> None:
    width = 64
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def _step(
    label: str,
    fn: Callable,
    *args,
    **kwargs,
) -> object:
    """Run a pipeline step with timing and error handling."""
    print(f"\n  ▶  {label} ...", flush=True)
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"     ✓  Done  ({elapsed:.1f}s)")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"     ✗  FAILED ({elapsed:.1f}s): {exc}")
        logger.exception("Step '%s' failed.", label)
        raise


def main(force_download: bool = False, skip_viz: bool = False) -> int:
    """Run the complete Phase 1 pipeline.

    Args:
        force_download: If ``True``, re-download raw files even if cached.
        skip_viz: If ``True``, skip the visualization step.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    _banner("FIFA World Cup 2026 — Phase 1 Data Engineering Pipeline")

    settings = load_settings()
    ensure_directories()
    set_global_seed(settings.random_seed)

    wall_start = time.perf_counter()

    # Import lazily so import errors surface as clear step failures.
    from src.phase1 import data_collection, preprocessing, merge_data, feature_builder, validation

    steps = [
        ("Step 1 — Data collection",
         lambda: data_collection.collect_all(settings=settings, force=force_download)),

        ("Step 2 — Preprocessing",
         lambda: preprocessing.run(settings=settings)),

        ("Step 3 — Merge & Elo ratings",
         lambda: merge_data.run(settings=settings)),

        ("Step 4 — Feature engineering",
         lambda: feature_builder.run(settings=settings)),

        ("Step 5 — Validation",
         lambda: validation.run(settings=settings)),
    ]

    if not skip_viz:
        from src.phase1 import visualizations
        steps.append(
            ("Step 6 — Visualizations",
             lambda: visualizations.run(settings=settings))
        )

    any_failed = False
    for label, fn in steps:
        try:
            _step(label, fn)
        except Exception:
            any_failed = True
            print(f"\n  Pipeline aborted at: {label}")
            print("  Check logs/pipeline.log for details.")
            return 1

    wall_elapsed = time.perf_counter() - wall_start

    _banner(f"Phase 1 Complete  ({wall_elapsed:.1f}s total)")

    print(f"""
  Output files
  ─────────────────────────────────────────────────────────
  Raw data          {RAW_DIR}
  Clean dataset     {INTERIM_DIR / 'matches_clean.csv'}
  Merged dataset    {INTERIM_DIR / 'matches_merged.csv'}
  Feature dataset   {INTERIM_DIR / 'matches_features.csv'}
  ML-ready dataset  {PROCESSED_DIR / 'matches_final.csv'}

  Reports
  ─────────────────────────────────────────────────────────
  Validation        {REPORTS_DIR / 'validation_report.txt'}
  Preprocessing     {REPORTS_DIR / 'preprocessing_summary.txt'}
  Merge             {REPORTS_DIR / 'merge_summary.txt'}
  Features          {REPORTS_DIR / 'feature_summary.txt'}
  Figures           {FIGURES_DIR}

  Logs              {LOGS_DIR / 'pipeline.log'}
""")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 — Phase 1 pipeline runner."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download all raw data files even if already cached.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip the visualization step (faster for re-runs).",
    )
    args = parser.parse_args()
    sys.exit(main(force_download=args.force_download, skip_viz=args.skip_viz))
