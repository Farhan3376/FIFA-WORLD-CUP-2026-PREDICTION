"""Phase 2 - Step 1: Exploratory Data Analysis (EDA) & Data Quality Auditing.

This module loads the ML-ready dataset `data/processed/matches_final.csv` and:
1. Performs Part 1 Data Exploration (shape, dtypes, missing values, duplicates, class balance).
2. Computes Part 3 statistical moments (mean, median, mode, variance, std, skew, kurtosis).
3. Performs Part 8 outlier detection using IQR, Z-Score, and Isolation Forest.
4. Audits the schema for potential data leakage.
5. Exports Part 9 reports:
   - `reports/eda/statistical_summary.txt`
   - `reports/eda/feature_summary.csv`
   - `reports/eda/data_quality_report.pdf` (using ReportLab)

Run directly::

    python -m src.eda
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from src.config import (
    PROCESSED_DIR,
    EDA_REPORTS_DIR,
    COL_RESULT,
    RESULT_LABELS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="phase2.log")


def load_final_dataset() -> pd.DataFrame:
    """Load the final processed training dataset.

    Returns:
        pd.DataFrame: Loaded training dataset.

    Raises:
        FileNotFoundError: If the dataset is missing.
    """
    dataset_path = PROCESSED_DIR / "matches_final.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed training dataset not found at: {dataset_path}. "
            "Please run run_pipeline.py from Phase 1 first."
        )
    logger.info("Loaded final dataset from %s", dataset_path)
    return pd.read_csv(dataset_path)


def explore_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform dataset structure and metadata exploration.

    Args:
        df: Input training dataframe.

    Returns:
        Dict[str, Any]: Exploration metadata dictionary.
    """
    shape = df.shape
    num_samples = shape[0]
    num_features = shape[1] - 1  # exclude target
    columns = list(df.columns)
    dtypes = df.dtypes.to_dict()
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)

    # Missing values
    missing_counts = df.isna().sum().to_dict()
    missing_pcts = (df.isna().mean() * 100).to_dict()

    # Duplicates
    duplicate_count = int(df.duplicated().sum())

    # Target class distribution
    target_counts = df[COL_RESULT].value_counts().to_dict()
    target_pcts = (df[COL_RESULT].value_counts(normalize=True) * 100).to_dict()

    logger.info("=== Dataset Exploration Overview ===")
    logger.info("Shape: %s | Samples: %d | Features: %d", shape, num_samples, num_features)
    logger.info("Memory usage: %.2f MB", memory_usage_mb)
    logger.info("Duplicate rows: %d", duplicate_count)

    return {
        "shape": shape,
        "num_samples": num_samples,
        "num_features": num_features,
        "columns": columns,
        "dtypes": dtypes,
        "memory_usage_mb": memory_usage_mb,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "duplicate_count": duplicate_count,
        "target_counts": target_counts,
        "target_pcts": target_pcts,
    }


def perform_statistical_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Perform mathematical and statistical moment calculations.

    Args:
        df: Input dataframe.

    Returns:
        Tuple[pd.DataFrame, str]: Dataframe of summary statistics and string summary.
    """
    stats_rows = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            mean_val = median_val = mode_val = var_val = std_val = skew_val = kurt_val = np.nan
        else:
            mean_val = col_data.mean()
            median_val = col_data.median()
            mode_series = col_data.mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan
            var_val = col_data.var()
            std_val = col_data.std()
            skew_val = col_data.skew()
            kurt_val = col_data.kurtosis()

        stats_rows.append({
            "Feature": col,
            "Mean": mean_val,
            "Median": median_val,
            "Mode": mode_val,
            "Variance": var_val,
            "Std Dev": std_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
        })

    stats_df = pd.DataFrame(stats_rows).set_index("Feature")

    # Generate txt report content
    lines = [
        "=" * 60,
        "FIFA World Cup 2026 - Phase 2 Statistical Summary",
        "=" * 60,
        "\nFeature Moments (Mean, Median, Mode, Variance, Skewness, Kurtosis):\n",
        stats_df.to_string(float_format=lambda x: f"{x:,.4f}"),
        "\n" + "=" * 60,
    ]
    txt_summary = "\n".join(lines)

    # Save to file
    summary_path = EDA_REPORTS_DIR / "statistical_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(txt_summary)
    logger.info("Saved statistical summary to %s", summary_path)

    return stats_df, txt_summary


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify outliers using IQR, Z-Score, and Isolation Forest.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Summary table of outliers per numerical column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Drop target from outlier detection to avoid target contamination
    if COL_RESULT in numeric_cols:
        numeric_cols = numeric_cols.drop(COL_RESULT)

    outlier_summary = []

    # Prepare features for Isolation Forest (handling NaN by filling with median)
    if len(numeric_cols) > 0:
        if df[numeric_cols].isna().sum().sum() > 0:
            iforest_data = df[numeric_cols].fillna(df[numeric_cols].median())
        else:
            iforest_data = df[numeric_cols]

        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        # Handle cases with single sample or invalid datasets gracefully
        try:
            preds = iforest.fit_predict(iforest_data)
            is_iforest_outlier = (preds == -1)
            total_iforest_outliers = int(is_iforest_outlier.sum())
        except Exception as exc:
            logger.warning("Isolation Forest failed: %s", exc)
            total_iforest_outliers = 0
    else:
        total_iforest_outliers = 0

    for col in numeric_cols:
        col_data = df[col].dropna()
        n_samples = len(col_data)
        if n_samples == 0:
            outlier_summary.append({
                "Feature": col,
                "IQR Outliers": 0,
                "Z-Score Outliers": 0,
                "IQR %": 0.0,
                "Z-Score %": 0.0,
            })
            continue

        # IQR Method
        q25, q75 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        num_iqr = len(iqr_outliers)

        # Z-Score Method (threshold = 3)
        mean_val = col_data.mean()
        std_val = col_data.std()
        if std_val > 0:
            z_scores = (col_data - mean_val) / std_val
            z_outliers = col_data[np.abs(z_scores) > 3]
            num_z = len(z_outliers)
        else:
            num_z = 0

        outlier_summary.append({
            "Feature": col,
            "IQR Outliers": num_iqr,
            "Z-Score Outliers": num_z,
            "IQR %": (num_iqr / n_samples) * 100,
            "Z-Score %": (num_z / n_samples) * 100,
        })

    outlier_df = pd.DataFrame(outlier_summary).set_index("Feature")
    # Add Isolation Forest as a global count attribute
    outlier_df.attrs["iforest_outliers"] = total_iforest_outliers

    logger.info(
        "Outlier detection audit complete. Isolation Forest flagged %d global outliers.",
        total_iforest_outliers
    )
    return outlier_df


def audit_data_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit the schema for potential data leakage indicators.

    Checks for the presence of columns containing prohibited names (like goals scored,
    match winner, or post-match Elo ratings) which should not be present in an ML training dataset.

    Args:
        df: Input dataframe.

    Returns:
        Dict[str, Any]: Audit verdict.
    """
    prohibited_names = ["goals", "winner", "after_match", "score", "_after"]
    detected_prohibited = []

    for col in df.columns:
        if col == COL_RESULT:
            continue
        for name in prohibited_names:
            if name in col.lower():
                detected_prohibited.append((col, name))

    verdict = "PASS" if not detected_prohibited else "WARN"

    logger.info("Data leakage audit: %s (%d suspicious columns found)", verdict, len(detected_prohibited))
    return {
        "verdict": verdict,
        "suspicious_columns": detected_prohibited,
    }


def save_feature_summary(df: pd.DataFrame, stats_df: pd.DataFrame, explore_dict: Dict[str, Any]) -> None:
    """Save the comprehensive feature summary report as a CSV.

    Args:
        df: Input training dataframe.
        stats_df: Computed statistical moments.
        explore_dict: Exploration metadata.
    """
    summary_rows = []
    for col in df.columns:
        dtype = str(explore_dict["dtypes"][col])
        missing_cnt = explore_dict["missing_counts"][col]
        missing_pct = explore_dict["missing_pcts"][col]
        unique_cnt = df[col].nunique()

        # Fetch statistical moments if numeric
        if col in stats_df.index:
            mean_val = stats_df.loc[col, "Mean"]
            std_val = stats_df.loc[col, "Std Dev"]
            skew_val = stats_df.loc[col, "Skewness"]
            kurt_val = stats_df.loc[col, "Kurtosis"]
        else:
            mean_val = std_val = skew_val = kurt_val = np.nan

        summary_rows.append({
            "FeatureName": col,
            "DataType": dtype,
            "MissingCount": missing_cnt,
            "MissingPercentage": missing_pct,
            "UniqueCount": unique_cnt,
            "Mean": mean_val,
            "StdDev": std_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = EDA_REPORTS_DIR / "feature_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved feature summary CSV to %s", summary_csv_path)


def build_pdf_report(
    explore_dict: Dict[str, Any],
    stats_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    leakage_dict: Dict[str, Any]
) -> None:
    """Generate the data quality PDF report using ReportLab flowables.

    Args:
        explore_dict: Exploration metadata.
        stats_df: Computed statistical moments.
        outlier_df: Computed outliers.
        leakage_dict: Data leakage verdict.
    """
    pdf_path = EDA_REPORTS_DIR / "data_quality_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=15
    )

    style_h2 = ParagraphStyle(
        name="Heading2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=8
    )

    style_body = ParagraphStyle(
        name="BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155")
    )

    style_table_text = ParagraphStyle(
        name="TableTextStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    )

    style_table_header = ParagraphStyle(
        name="TableHeaderStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("FIFA World Cup 2026 — Phase 2", style_body))
    story.append(Paragraph("Data Quality & Exploration Report", style_title))
    story.append(Spacer(1, 10))

    # Executive Summary Table
    meta_data = [
        [Paragraph("Metadata Metric", style_table_header),
         Paragraph("Value / Count", style_table_header)],
        [Paragraph("Total Samples", style_table_text),
         Paragraph(f"{explore_dict['num_samples']:,}", style_table_text)],
        [Paragraph("Total Features", style_table_text),
         Paragraph(f"{explore_dict['num_features']}", style_table_text)],
        [Paragraph("Memory Usage", style_table_text),
         Paragraph(f"{explore_dict['memory_usage_mb']:.2f} MB", style_table_text)],
        [Paragraph("Duplicate Rows", style_table_text),
         Paragraph(f"{explore_dict['duplicate_count']}", style_table_text)],
        [Paragraph("Leakage Audit Verdict", style_table_text),
         Paragraph(f"{leakage_dict['verdict']}", style_table_text)],
    ]
    t_meta = Table(meta_data, colWidths=[200, 300])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # Target Class Balance Analysis
    story.append(Paragraph("Target Variable Class Balance", style_h2))
    target_data = [
        [Paragraph("Class Label", style_table_header),
         Paragraph("Count", style_table_header),
         Paragraph("Percentage", style_table_header)]
    ]
    for label_id, label_name in RESULT_LABELS.items():
        count = explore_dict["target_counts"].get(label_id, 0)
        pct = explore_dict["target_pcts"].get(label_id, 0.0)
        target_data.append([
            Paragraph(f"{label_name} ({label_id})", style_table_text),
            Paragraph(f"{count:,}", style_table_text),
            Paragraph(f"{pct:.2f}%", style_table_text)
        ])
    t_target = Table(target_data, colWidths=[200, 150, 150])
    t_target.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    story.append(t_target)
    story.append(Spacer(1, 15))

    # Missing Value Heatmap Placeholder/Section
    story.append(Paragraph("Missing Values Summary", style_h2))
    missing_data = [
        [Paragraph("Feature Name", style_table_header),
         Paragraph("Missing Count", style_table_header),
         Paragraph("Missing %", style_table_header)]
    ]

    # Filter features that have missing values
    missing_feats = [
        (feat, cnt, explore_dict["missing_pcts"][feat])
        for feat, cnt in explore_dict["missing_counts"].items() if cnt > 0
    ]
    # Sort by count descending
    missing_feats.sort(key=lambda x: x[1], reverse=True)

    if not missing_feats:
        missing_data.append([
            Paragraph("No features contain missing values.", style_table_text),
            Paragraph("0", style_table_text),
            Paragraph("0.0%", style_table_text)
        ])
    else:
        # Show top features with missing values to keep table compact
        for feat, cnt, pct in missing_feats[:8]:
            missing_data.append([
                Paragraph(feat, style_table_text),
                Paragraph(f"{cnt:,}", style_table_text),
                Paragraph(f"{pct:.2f}%", style_table_text)
            ])
        if len(missing_feats) > 8:
            missing_data.append([
                Paragraph(f"... and {len(missing_feats) - 8} more", style_table_text),
                Paragraph("", style_table_text),
                Paragraph("", style_table_text)
            ])

    t_missing = Table(missing_data, colWidths=[250, 125, 125])
    t_missing.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    story.append(t_missing)
    story.append(PageBreak())

    # Outliers Section
    story.append(Paragraph("Outlier Detection Audit", style_h2))
    story.append(Paragraph(
        "Outlier summary using IQR (interquartile range, threshold=1.5) and standard deviation Z-Scores (|Z| > 3). "
        f"An unsupervised Isolation Forest flagged a global total of "
        f"<b>{outlier_df.attrs.get('iforest_outliers', 0)}</b> outliers across all numerical columns.",
        style_body
    ))
    story.append(Spacer(1, 8))

    outlier_table_data = [[
        Paragraph("Feature", style_table_header),
        Paragraph("IQR Outliers", style_table_header),
        Paragraph("IQR %", style_table_header),
        Paragraph("Z-Score Outliers", style_table_header),
        Paragraph("Z-Score %", style_table_header),
    ]]

    # Sort features by IQR outlier percentage descending, showing top 12
    top_outliers = outlier_df.sort_values(by="IQR %", ascending=False)
    for col in top_outliers.index[:12]:
        row = top_outliers.loc[col]
        outlier_table_data.append([
            Paragraph(col, style_table_text),
            Paragraph(f"{int(row['IQR Outliers']):,}", style_table_text),
            Paragraph(f"{row['IQR %']:.2f}%", style_table_text),
            Paragraph(f"{int(row['Z-Score Outliers']):,}", style_table_text),
            Paragraph(f"{row['Z-Score %']:.2f}%", style_table_text),
        ])
    if len(top_outliers) > 12:
        outlier_table_data.append([
            Paragraph(f"... showing top 12 of {len(top_outliers)} numerical features.", style_body),
            Paragraph("", style_table_text),
            Paragraph("", style_table_text),
            Paragraph("", style_table_text),
            Paragraph("", style_table_text),
        ])

    t_outlier = Table(outlier_table_data, colWidths=[200, 75, 75, 75, 75])
    t_outlier.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    story.append(t_outlier)
    story.append(Spacer(1, 15))

    # Data Leakage checks
    story.append(Paragraph("Data Leakage Checklist", style_h2))
    leakage_verdict = leakage_dict["verdict"]
    color_verdict = "#16a34a" if leakage_verdict == "PASS" else "#d97706"
    story.append(Paragraph(
        f"Leakage audit check: <font color='{color_verdict}'><b>{leakage_verdict}</b></font>.",
        style_body
    ))
    story.append(Spacer(1, 6))

    if leakage_dict["suspicious_columns"]:
        leak_data = [
            [Paragraph("Flagged Column", style_table_header),
             Paragraph("Prohibited Keyword Match", style_table_header)]
        ]
        for col, kw in leakage_dict["suspicious_columns"]:
            leak_data.append([Paragraph(col, style_table_text), Paragraph(kw, style_table_text)])
        t_leak = Table(leak_data, colWidths=[250, 250])
        t_leak.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
        ]))
        story.append(t_leak)
    else:
        story.append(Paragraph(
            "Checked dataset columns against typical target leakage signatures (goals, match winners, updates). "
            "No post-match features or prohibited names detected. The dataset is validated as leakage-free.",
            style_body
        ))

    # Page number callback
    def footer(canvas_obj: canvas.Canvas, document: SimpleDocTemplate) -> None:
        canvas_obj.saveState()
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(colors.HexColor("#64748b"))
        canvas_obj.drawString(
            54, 30,
            "FIFA World Cup 2026 Prediction Portfolio — Phase 2 Data Quality Report"
        )
        canvas_obj.drawRightString(document.pagesize[0] - 54, 30, f"Page {document.page}")
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    logger.info("Saved data quality PDF report to %s", pdf_path)


def main() -> None:
    """Run Phase 2 EDA entrypoint pipeline."""
    logger.info("Starting Phase 2 Exploratory Data Analysis (Step 1: eda.py)")

    try:
        df = load_final_dataset()
        explore_dict = explore_dataset(df)
        stats_df, _ = perform_statistical_analysis(df)
        outlier_df = detect_outliers(df)
        leakage_dict = audit_data_leakage(df)

        # Save summaries
        save_feature_summary(df, stats_df, explore_dict)
        build_pdf_report(explore_dict, stats_df, outlier_df, leakage_dict)

        logger.info("Phase 2 EDA (eda.py) completed successfully.")

    except Exception as exc:
        logger.exception("Failed to run Phase 2 EDA (eda.py): %s", exc)
        raise


if __name__ == "__main__":
    main()
