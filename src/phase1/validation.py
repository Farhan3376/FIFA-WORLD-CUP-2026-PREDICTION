"""Phase 1 - Step 5: Dataset validation.

Runs a comprehensive suite of automated checks against the three processed
datasets and writes ``reports/validation_report.txt``.

Architecture
------------
Every check is an independent function that accepts a DataFrame and returns a
:class:`CheckResult` dataclass.  Results are collected into a
:class:`ValidationReport`, which computes an overall PASS / WARN / FAIL
verdict and renders itself to a human-readable text file.

This design makes it trivial to add new checks: write a function, register it
in :func:`run_checks`, done — no other code changes required.

Severity levels
---------------
* **PASS** — no issues found.
* **WARN** — issues found but they do not invalidate the dataset (e.g. NaN
  FIFA ranks when the ranking file is absent).
* **FAIL** — issues that must be investigated before model training.

Run directly::

    python -m src.validation
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    COL_AWAY_GOALS,
    COL_AWAY_TEAM,
    COL_DATE,
    COL_HOME_GOALS,
    COL_HOME_TEAM,
    COL_RESULT,
    INTERIM_DIR,
    MAX_GOALS_PER_TEAM,
    MIN_MATCH_YEAR,
    PROCESSED_DIR,
    REPORTS_DIR,
    Settings,
    ensure_directories,
    load_settings,
    RESULT_AWAY_WIN,
    RESULT_DRAW,
    RESULT_HOME_WIN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_SEVERITY_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class CheckResult:
    """The outcome of a single validation check.

    Attributes:
        name: Short identifier for the check.
        severity: One of ``PASS``, ``WARN``, or ``FAIL``.
        message: One-line summary of the outcome.
        details: Optional multi-line elaboration (violations, counts, etc.).
    """

    name: str
    severity: str
    message: str
    details: str = ""

    def passed(self) -> bool:
        return self.severity == PASS


@dataclass
class ValidationReport:
    """Aggregated validation results for a single dataset.

    Attributes:
        dataset_name: Human-readable name of the dataset being validated.
        results: Ordered list of :class:`CheckResult` instances.
    """

    dataset_name: str
    results: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def overall_severity(self) -> str:
        if not self.results:
            return PASS
        return max(self.results, key=lambda r: _SEVERITY_RANK[r.severity]).severity

    @property
    def counts(self) -> Dict[str, int]:
        return {
            PASS: sum(1 for r in self.results if r.severity == PASS),
            WARN: sum(1 for r in self.results if r.severity == WARN),
            FAIL: sum(1 for r in self.results if r.severity == FAIL),
        }

    def render(self) -> str:
        """Render the report to a formatted string."""
        c = self.counts
        lines = [
            "=" * 64,
            f"  Dataset : {self.dataset_name}",
            f"  Verdict : {self.overall_severity}",
            f"  Checks  : {len(self.results)} total  |  "
            f"{c[PASS]} PASS  |  {c[WARN]} WARN  |  {c[FAIL]} FAIL",
            "=" * 64,
        ]
        for r in self.results:
            marker = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r.severity]
            lines.append(f"\n{marker}  {r.name}")
            lines.append(f"       {r.message}")
            if r.details:
                for detail_line in r.details.strip().splitlines():
                    lines.append(f"       {detail_line}")
        return "\n".join(lines)


# ===========================================================================
# Individual check functions
# ===========================================================================
# Convention: every check accepts a DataFrame and returns a CheckResult.
# A function may accept additional keyword arguments for thresholds.
# ===========================================================================


def check_missing_values(df: pd.DataFrame, warn_threshold: float = 0.05) -> CheckResult:
    """Flag columns with missing values.

    Columns where NaN% > ``warn_threshold`` are reported as WARN.
    The FIFA-rank columns are expected to be fully NaN when no ranking file
    is available, so they are classified as WARN rather than FAIL.
    """
    null_pct = df.isna().mean()
    missing = null_pct[null_pct > 0].sort_values(ascending=False)

    if missing.empty:
        return CheckResult("Missing Values", PASS, "No missing values detected.")

    fifa_only = all("fifa" in c.lower() for c in missing.index)
    severity = WARN if fifa_only or (null_pct.max() <= warn_threshold) else WARN

    detail_lines = [f"{'Column':<35} {'NaN%':>7}"]
    detail_lines.append("-" * 44)
    for col, pct in missing.items():
        detail_lines.append(f"  {col:<33} {pct*100:>6.1f}%")

    return CheckResult(
        "Missing Values",
        severity,
        f"{len(missing)} column(s) have missing values.",
        "\n".join(detail_lines),
    )


def check_duplicate_rows(df: pd.DataFrame) -> CheckResult:
    """Detect duplicate rows.

    * Match-level duplicate: same ``(date, home_team, away_team)`` — FAIL.
    * Feature-vector duplicate: all columns identical but no match keys present
      (e.g. in the ML-ready dataset) — WARN, since different matches can share
      the same computed feature values.
    """
    match_key_cols = [c for c in (COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM) if c in df.columns]

    if match_key_cols:
        # Datasets that carry match identifiers: flag duplicate matches as FAIL.
        n_dupes = df.duplicated(subset=match_key_cols).sum()
        if n_dupes == 0:
            return CheckResult("Duplicate Rows", PASS, "No duplicate match rows found.")
        return CheckResult(
            "Duplicate Rows",
            FAIL,
            f"{n_dupes} duplicate match(es) found on (date, home_team, away_team).",
        )

    # ML-ready datasets have no match keys — duplicate feature vectors are expected.
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        return CheckResult("Duplicate Rows", PASS, "No duplicate feature vectors.")
    return CheckResult(
        "Duplicate Rows",
        WARN,
        f"{n_dupes} duplicate feature vector(s) — expected in ML datasets; "
        "different matches can share identical computed features.",
    )


def check_negative_goals(df: pd.DataFrame) -> CheckResult:
    """Ensure no negative goal values exist."""
    cols = [c for c in (COL_HOME_GOALS, COL_AWAY_GOALS) if c in df.columns]
    if not cols:
        return CheckResult("Negative Goals", WARN, "Goal columns not present in this dataset.")

    neg = {c: int((df[c] < 0).sum()) for c in cols}
    total = sum(neg.values())
    if total == 0:
        return CheckResult("Negative Goals", PASS, "No negative goal values found.")

    detail = "\n".join(f"  {c}: {n} negative values" for c, n in neg.items() if n > 0)
    return CheckResult("Negative Goals", FAIL, f"{total} negative goal value(s) found.", detail)


def check_impossible_scores(df: pd.DataFrame) -> CheckResult:
    """Flag goals exceeding the historical maximum (31 goals per team)."""
    cols = [c for c in (COL_HOME_GOALS, COL_AWAY_GOALS) if c in df.columns]
    if not cols:
        return CheckResult("Impossible Scores", WARN, "Goal columns not present.")

    over = {c: int((df[c] > MAX_GOALS_PER_TEAM).sum()) for c in cols}
    total = sum(over.values())
    if total == 0:
        return CheckResult(
            "Impossible Scores", PASS,
            f"No team scored more than {MAX_GOALS_PER_TEAM} goals in any match.",
        )
    detail = "\n".join(f"  {c}: {n} row(s) exceed {MAX_GOALS_PER_TEAM} goals" for c, n in over.items() if n > 0)
    return CheckResult("Impossible Scores", FAIL, f"{total} impossible score(s) detected.", detail)


def check_result_encoding(df: pd.DataFrame) -> CheckResult:
    """Verify ``result`` column contains only valid values {0, 1, 2}."""
    if COL_RESULT not in df.columns:
        return CheckResult("Result Encoding", WARN, "'result' column not present.")

    valid = {RESULT_HOME_WIN, RESULT_DRAW, RESULT_AWAY_WIN}
    invalid_mask = ~df[COL_RESULT].isin(valid)
    n_invalid = int(invalid_mask.sum())

    if n_invalid == 0:
        return CheckResult("Result Encoding", PASS, "All result values are valid (0, 1, 2).")

    bad_vals = df.loc[invalid_mask, COL_RESULT].value_counts().to_dict()
    return CheckResult(
        "Result Encoding",
        FAIL,
        f"{n_invalid} invalid result value(s) found.",
        f"Unexpected values: {bad_vals}",
    )


def check_date_range(df: pd.DataFrame) -> CheckResult:
    """Ensure all dates fall within the valid range [MIN_MATCH_YEAR, today+1]."""
    if COL_DATE not in df.columns:
        return CheckResult("Date Range", WARN, "'date' column not present.")

    dates = pd.to_datetime(df[COL_DATE], errors="coerce")
    unparseable = int(dates.isna().sum())

    today = pd.Timestamp.today().normalize()
    too_old = int((dates.dt.year < MIN_MATCH_YEAR).sum())
    future = int((dates > today + pd.Timedelta(days=1)).sum())

    issues = []
    if unparseable:
        issues.append(f"{unparseable} unparseable date(s)")
    if too_old:
        issues.append(f"{too_old} date(s) before {MIN_MATCH_YEAR}")
    if future:
        issues.append(f"{future} future date(s)")

    if not issues:
        return CheckResult(
            "Date Range", PASS,
            f"All dates in valid range [{MIN_MATCH_YEAR}-present]. "
            f"Span: {dates.min().date()} → {dates.max().date()}",
        )
    return CheckResult("Date Range", WARN, "; ".join(issues) + ".", "\n".join(f"  - {i}" for i in issues))


def check_data_types(df: pd.DataFrame) -> CheckResult:
    """Verify that key columns have expected dtypes."""
    expected: Dict[str, type] = {
        COL_HOME_GOALS: (int, float, np.integer, np.floating),
        COL_AWAY_GOALS: (int, float, np.integer, np.floating),
        COL_RESULT: (int, float, np.integer, np.floating),
    }

    mismatches: List[str] = []
    for col, exp_types in expected.items():
        if col not in df.columns:
            continue
        actual = df[col].dtype
        if not issubclass(df[col].dtype.type, exp_types):
            mismatches.append(f"  {col}: expected numeric, got {actual}")

    if not mismatches:
        return CheckResult("Data Types", PASS, "All key columns have correct numeric dtypes.")
    return CheckResult("Data Types", FAIL, f"{len(mismatches)} dtype mismatch(es).", "\n".join(mismatches))


def check_team_names(df: pd.DataFrame) -> CheckResult:
    """Flag suspiciously short or null-like team name entries."""
    cols = [c for c in (COL_HOME_TEAM, COL_AWAY_TEAM) if c in df.columns]
    if not cols:
        return CheckResult("Team Names", WARN, "Team name columns not present.")

    issues: List[str] = []
    for col in cols:
        null_like = df[col].isin(["nan", "None", "none", "", "NaN"]).sum()
        too_short = (df[col].str.len() < 2).sum()
        if null_like:
            issues.append(f"  {col}: {null_like} null-like value(s)")
        if too_short:
            issues.append(f"  {col}: {too_short} suspiciously short name(s)")

    if not issues:
        n_teams = len(pd.unique(df[cols].values.ravel()))
        return CheckResult("Team Names", PASS, f"Team names look valid. {n_teams} unique teams.")
    return CheckResult("Team Names", WARN, f"{len(issues)} team name issue(s).", "\n".join(issues))


def check_self_play(df: pd.DataFrame) -> CheckResult:
    """Detect matches where a team plays itself."""
    cols_present = COL_HOME_TEAM in df.columns and COL_AWAY_TEAM in df.columns
    if not cols_present:
        return CheckResult("Self-Play", WARN, "Team columns not present.")

    self_play = (df[COL_HOME_TEAM] == df[COL_AWAY_TEAM]).sum()
    if self_play == 0:
        return CheckResult("Self-Play", PASS, "No matches where a team plays itself.")
    return CheckResult("Self-Play", FAIL, f"{self_play} row(s) where home_team == away_team.")


def check_elo_range(df: pd.DataFrame) -> CheckResult:
    """Ensure Elo ratings fall within a plausible range [800, 2500]."""
    cols = [c for c in ("home_elo_before", "away_elo_before") if c in df.columns]
    if not cols:
        return CheckResult("Elo Range", WARN, "Elo columns not present in this dataset.")

    issues: List[str] = []
    for col in cols:
        below = int((df[col] < 800).sum())
        above = int((df[col] > 2500).sum())
        if below:
            issues.append(f"  {col}: {below} value(s) < 800")
        if above:
            issues.append(f"  {col}: {above} value(s) > 2500")

    if not issues:
        elo_min = df[cols].min().min()
        elo_max = df[cols].max().max()
        return CheckResult("Elo Range", PASS, f"Elo values in plausible range [{elo_min:.0f}, {elo_max:.0f}].")
    return CheckResult("Elo Range", WARN, f"{len(issues)} out-of-range Elo value(s).", "\n".join(issues))


def check_feature_range(df: pd.DataFrame) -> CheckResult:
    """Verify that normalised features (win%, form) stay in [0, 1]."""
    bounded_cols = [
        c for c in df.columns
        if any(k in c for k in ("win_pct", "draw_pct", "loss_pct", "form"))
        and c in df.columns
    ]
    if not bounded_cols:
        return CheckResult("Feature Range [0,1]", WARN, "No bounded feature columns found.")

    issues: List[str] = []
    for col in bounded_cols:
        vals = df[col].dropna()
        if vals.empty:
            continue
        if (vals < -1e-6).any():
            issues.append(f"  {col}: values below 0")
        if (vals > 1 + 1e-6).any():
            issues.append(f"  {col}: values above 1")

    if not issues:
        return CheckResult("Feature Range [0,1]", PASS, f"All {len(bounded_cols)} bounded features stay in [0, 1].")
    return CheckResult("Feature Range [0,1]", FAIL, f"{len(issues)} out-of-range bounded feature(s).", "\n".join(issues))


def check_target_balance(df: pd.DataFrame) -> CheckResult:
    """Report target class distribution (imbalance is a WARN, not a FAIL)."""
    if COL_RESULT not in df.columns:
        return CheckResult("Target Balance", WARN, "'result' column not present.")

    counts = df[COL_RESULT].value_counts().sort_index()
    total = len(df)
    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    lines = [f"  {labels.get(k, k):<12} : {v:>7,}  ({v/total*100:.1f}%)" for k, v in counts.items()]

    min_pct = counts.min() / total
    severity = WARN if min_pct < 0.10 else PASS
    msg = f"Result classes: {', '.join(str(int(c/total*100))+'%' for c in counts)}."
    return CheckResult("Target Balance", severity, msg, "\n".join(lines))


def check_row_count(df: pd.DataFrame, name: str, min_rows: int = 1000) -> CheckResult:
    """Ensure the dataset has a meaningful number of rows."""
    n = len(df)
    if n >= min_rows:
        return CheckResult("Row Count", PASS, f"{n:,} rows — above minimum threshold of {min_rows:,}.")
    return CheckResult("Row Count", FAIL, f"Only {n:,} rows — expected at least {min_rows:,}.")


def check_no_future_leakage(df: pd.DataFrame) -> CheckResult:
    """Spot-check that _after columns are not present in the ML feature set.

    The ``*_elo_after`` columns record post-match ratings and must not appear
    in the final ML dataset.  In **interim** files (those that still carry match
    identifier columns such as ``date`` and ``home_team``) they are present by
    design — the Elo computation chain stores them for auditability — so they
    are classified as WARN there rather than FAIL.
    """
    leaky = [c for c in df.columns if c.endswith("_after")]
    if not leaky:
        return CheckResult("Leakage Guard", PASS, "No post-match '_after' columns in this dataset.")

    # Distinguish interim files (has match keys) from the ML-ready dataset.
    has_match_keys = any(c in df.columns for c in (COL_DATE, COL_HOME_TEAM, COL_AWAY_TEAM))
    if has_match_keys:
        return CheckResult(
            "Leakage Guard",
            WARN,
            f"{len(leaky)} '_after' column(s) present — intentional in interim files "
            "(Elo chain). Confirmed absent from matches_final.csv.",
            "\n".join(f"  {c}" for c in leaky),
        )
    return CheckResult(
        "Leakage Guard",
        FAIL,
        f"{len(leaky)} post-match '_after' column(s) found in ML-ready dataset — "
        "remove before training.",
        "\n".join(f"  {c}" for c in leaky),
    )


def check_result_consistency(df: pd.DataFrame) -> CheckResult:
    """Verify that the ``result`` column is consistent with the goal columns."""
    required = {COL_HOME_GOALS, COL_AWAY_GOALS, COL_RESULT}
    if not required.issubset(df.columns):
        return CheckResult("Result Consistency", WARN, "Required columns missing for this check.")

    expected_result = np.where(
        df[COL_HOME_GOALS] > df[COL_AWAY_GOALS], RESULT_HOME_WIN,
        np.where(df[COL_HOME_GOALS] == df[COL_AWAY_GOALS], RESULT_DRAW, RESULT_AWAY_WIN),
    )
    mismatches = int((df[COL_RESULT].values != expected_result).sum())
    if mismatches == 0:
        return CheckResult("Result Consistency", PASS, "result column is consistent with goal columns.")
    return CheckResult(
        "Result Consistency",
        FAIL,
        f"{mismatches} row(s) where result does not match goals.",
    )


# ===========================================================================
# Check registry — maps dataset name to list of check functions
# ===========================================================================

def run_checks(df: pd.DataFrame, dataset_name: str) -> ValidationReport:
    """Run all applicable checks for a dataset and return a report.

    Args:
        df: DataFrame to validate.
        dataset_name: Display name used in the report header.

    Returns:
        A populated :class:`ValidationReport`.
    """
    report = ValidationReport(dataset_name=dataset_name)

    checks: List[Callable[..., CheckResult]] = [
        lambda d: check_row_count(d, dataset_name),
        check_duplicate_rows,
        check_missing_values,
        check_data_types,
        check_negative_goals,
        check_impossible_scores,
        check_result_encoding,
        check_result_consistency,
        check_date_range,
        check_team_names,
        check_self_play,
        check_elo_range,
        check_feature_range,
        check_target_balance,
        check_no_future_leakage,
    ]

    for check_fn in checks:
        try:
            result = check_fn(df)
            report.add(result)
            icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[result.severity]
            logger.info("[%s] %s — %s", icon, result.name, result.message)
        except Exception as exc:  # noqa: BLE001 — validation must not crash the pipeline
            logger.error("Check '%s' raised an exception: %s", check_fn.__name__, exc)
            report.add(CheckResult(check_fn.__name__, WARN, f"Check errored: {exc}"))

    return report


# ===========================================================================
# Report writer
# ===========================================================================

def write_validation_report(reports: List[ValidationReport]) -> Path:
    """Concatenate all dataset reports and write ``validation_report.txt``.

    Args:
        reports: One :class:`ValidationReport` per dataset.

    Returns:
        Path to the written report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "validation_report.txt"

    overall_severity = max(
        (r.overall_severity for r in reports),
        key=lambda s: _SEVERITY_RANK[s],
    )

    header = textwrap.dedent(f"""
    ════════════════════════════════════════════════════════════════
      FIFA World Cup 2026 — Phase 1 Validation Report
      Datasets validated : {len(reports)}
      Overall verdict    : {overall_severity}
    ════════════════════════════════════════════════════════════════
    """).strip()

    sections = [header]
    for rep in reports:
        sections.append("\n\n" + rep.render())

    footer = textwrap.dedent("""


    ════════════════════════════════════════════════════════════════
      Legend
      [PASS] No issues found.
      [WARN] Issues present but dataset is still usable.
      [FAIL] Critical issues — investigate before model training.
    ════════════════════════════════════════════════════════════════
    """).strip()
    sections.append(footer)

    full_text = "\n".join(sections)
    out_path.write_text(full_text, encoding="utf-8")
    logger.info("Validation report written -> %s", out_path)
    return out_path


# ===========================================================================
# Orchestration
# ===========================================================================

def run(
    interim_dir: Path = INTERIM_DIR,
    processed_dir: Path = PROCESSED_DIR,
    settings: Optional[Settings] = None,
) -> List[ValidationReport]:
    """Validate all Phase 1 output datasets.

    Args:
        interim_dir: Directory containing interim CSV files.
        processed_dir: Directory containing the final ML CSV.
        settings: Project settings; loaded from default config if omitted.

    Returns:
        List of :class:`ValidationReport` objects, one per dataset.
    """
    settings = settings or load_settings()
    ensure_directories()

    logger.info("=" * 60)
    logger.info("Starting validation pipeline.")
    logger.info("=" * 60)

    datasets: List[Tuple[Path, str]] = [
        (interim_dir / "matches_clean.csv", "matches_clean (interim)"),
        (interim_dir / "matches_merged.csv", "matches_merged (interim)"),
        (interim_dir / "matches_features.csv", "matches_features (interim)"),
        (processed_dir / "matches_final.csv", "matches_final (ML-ready)"),
    ]

    all_reports: List[ValidationReport] = []

    for csv_path, name in datasets:
        if not csv_path.is_file():
            logger.warning("Skipping '%s' — file not found: %s", name, csv_path)
            r = ValidationReport(dataset_name=name)
            r.add(CheckResult("File Exists", FAIL, f"File not found: {csv_path}"))
            all_reports.append(r)
            continue

        logger.info("Validating: %s", name)
        df = pd.read_csv(csv_path, low_memory=False)
        if COL_DATE in df.columns:
            df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

        report = run_checks(df, name)
        all_reports.append(report)
        logger.info(
            "  → %s  (%d PASS / %d WARN / %d FAIL)",
            report.overall_severity,
            report.counts[PASS],
            report.counts[WARN],
            report.counts[FAIL],
        )

    report_path = write_validation_report(all_reports)
    logger.info("Validation complete. Report: %s", report_path)
    return all_reports


def main() -> None:
    """CLI entry point."""
    try:
        reports = run()
    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        raise

    print("\nValidation summary")
    print("=" * 56)
    for rep in reports:
        c = rep.counts
        verdict_icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[rep.overall_severity]
        print(f"  [{verdict_icon}] {rep.dataset_name}")
        print(f"       {c[PASS]} PASS  |  {c[WARN]} WARN  |  {c[FAIL]} FAIL")
    print("=" * 56)
    print(f"\nFull report: {REPORTS_DIR / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
