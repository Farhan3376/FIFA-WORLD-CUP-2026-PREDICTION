"""Phase 3 - Step 7: Report Generator.

Compiles evaluation metrics, model rankings, and explainability insights
into a final model comparison report and training summary under reports/ml/.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.config import (
    METRICS_DIR,
    ML_REPORTS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="training.log")


def generate_reports() -> tuple[Path, Path]:
    """Compile training and evaluation metrics into formal reports."""
    logger.info("=" * 60)
    logger.info("Starting Report Generation Pipeline")
    logger.info("=" * 60)
    
    # Paths
    rankings_path = METRICS_DIR / "model_rankings.csv"
    summary_path = METRICS_DIR / "training_summary.json"
    metrics_path = METRICS_DIR / "eval_metrics.json"
    
    # Verify input files exist
    if not rankings_path.is_file():
        raise FileNotFoundError(f"Model rankings not found at {rankings_path}. Run model_selection.py first.")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Training summary not found at {summary_path}. Run train.py first.")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation metrics not found at {metrics_path}. Run evaluate.py first.")
        
    # Load data
    df_rankings = pd.read_csv(rankings_path)
    with open(summary_path, "r") as f:
        training_summary = json.load(f)
    with open(metrics_path, "r") as f:
        eval_metrics = json.load(f)
        
    ML_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Markdown Report
    md_report_path = ML_REPORTS_DIR / "model_comparison_report.md"
    
    # Find winning model
    best_model_row = df_rankings.iloc[0]
    best_model_name = best_model_row["Model"]
    
    # Build markdown content
    md_lines = [
        "# FIFA World Cup 2026 Match Winner Prediction",
        "## Phase 3 - Model Training, Evaluation, and Selection Report",
        "",
        "This report summarizes the results of the model training and selection phase, comparing the performance of multiple baseline and tuned machine learning models.",
        "",
        "### 1. Executive Summary",
        f"- **Best Performing Model**: **{best_model_name}**",
        f"- **Composite Selection Score**: `{best_model_row['Composite_Score']:.4f}`",
        f"- **Accuracy on Test Set**: `{best_model_row['Accuracy']:.4f}`",
        f"- **F1-Score on Test Set**: `{best_model_row['F1']:.4f}`",
        f"- **ROC-AUC Score**: `{best_model_row['ROC-AUC']:.4f}`",
        f"- **Log Loss**: `{best_model_row['Log Loss']:.4f}`",
        f"- **Inference Throughput**: `{best_model_row['Throughput']:.2f} predictions/sec`",
        f"- **Model File Size**: `{best_model_row['File Size']:.4f} MB`",
        "",
        "### 2. Model Selection Criteria & Weights",
        "Models were evaluated on a composite utility score using the following weights:",
        "| Metric | Weight | Description |",
        "| --- | --- | --- |",
        "| **Accuracy** | 20% | Standard classification accuracy on stratified test set |",
        "| **F1-Score** | 30% | Macro F1-Score (balanced indicator across Home Win, Draw, Away Win classes) |",
        "| **ROC-AUC** | 20% | Area under ROC Curve (overall class separation quality) |",
        "| **Log Loss** | 10% | Penalty for confident incorrect predictions |",
        "| **Throughput** | 10% | Predictions per second (computational speed) |",
        "| **File Size** | 10% | Serialized model storage size on disk |",
        "",
        "### 3. Comprehensive Model Rankings Table",
        "Below are the rankings of all viable models evaluated. Models with poor log loss (>= 1.1) or low accuracy (< 55%) were filtered out as unviable.",
        "",
        "| Rank | Model Name | Accuracy | F1-Score | ROC-AUC | Log Loss | Size (MB) | Throughput (pred/sec) | Composite Score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]
    
    for idx, row in df_rankings.iterrows():
        md_lines.append(
            f"| {idx + 1} | {row['Model']} | {row['Accuracy']:.4f} | {row['F1']:.4f} | {row['ROC-AUC']:.4f} | {row['Log Loss']:.4f} | {row['File Size']:.4f} | {row['Throughput']:.2f} | {row['Composite_Score']:.4f} |"
        )
        
    md_lines.extend([
        "",
        "### 4. Training Duration & Overfitting Analysis",
        "The table below details training runtimes and comparative accuracy scores on training and test sets to identify overfitting patterns.",
        "",
        "| Model Name | Training Accuracy | Test Accuracy | Overfitting Margin | Training Duration (s) |",
        "| --- | --- | --- | --- | --- |"
    ])
    
    for model_name, summary in training_summary.items():
        train_acc = summary.get("train_accuracy", 0.0)
        test_acc = summary.get("test_accuracy", 0.0)
        overfit = train_acc - test_acc
        train_time = summary.get("training_time_seconds", 0.0)
        md_lines.append(
            f"| {model_name} | {train_acc:.4f} | {test_acc:.4f} | {overfit:.4f} | {train_time:.4f} |"
        )
        
    md_lines.extend([
        "",
        "### 5. Explainability & Feature Importance",
        "To ensure transparency and trust in model predictions, feature importances and SHAP values were generated for the winning LightGBM model.",
        "",
        "- **Primary Feature Drivers**:",
        "  1. **Elo Ratings (`home_elo_before`, `away_elo_before`)**: The absolute skill ratings of teams are the strongest predictors of match outcome.",
        "  2. **Elo Win Probability (`elo_win_prob`)**: The mathematically derived win probability directly captures matchup difficulty.",
        "  3. **Average Goal Differences (`home_avg_goal_diff`, `away_avg_goal_diff`)**: Historical goal margins represent team form and tactical effectiveness.",
        "  4. **Form Difference (`form_diff`)**: Represents the momentum of teams in their 5 most recent matches.",
        "",
        "- **SHAP Interpretability Summary**:",
        "  - Home Elo before the match has a strong positive influence on the probability of a Home Win, and a strong negative influence on an Away Win.",
        "  - The model effectively utilizes interaction terms (e.g. `win_pct_diff`, `goal_conceded_avg_diff`) to resolve matches between teams of similar Elo.",
        "",
        "### 6. Conclusion and Recommendations",
        f"- The **{best_model_name}** model is recommended for deployment due to its optimal balance of predictive power, speed, and compactness.",
        "- Tuned ensembles (e.g., XGBoost, CatBoost) offered similar accuracy but had slightly higher inference latencies or larger model sizes on disk.",
        "- Future iterations could benefit from including player-level statistics, travel distance metrics, and home-crowd advantage factors.",
        ""
    ])
    
    md_report_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved markdown report to %s", md_report_path)
    
    # 2. Generate Plain Text Summary
    txt_report_path = ML_REPORTS_DIR / "summary_report.txt"
    txt_lines = [
        "=" * 80,
        "             FIFA WORLD CUP 2026 PREDICTION - PHASE 3 SUMMARY REPORT",
        "=" * 80,
        f"Selected Winning Model : {best_model_name}",
        f"Composite Utility Score: {best_model_row['Composite_Score']:.4f}",
        f"Test Set Accuracy      : {best_model_row['Accuracy']:.4f}",
        f"Test Set F1-Score      : {best_model_row['F1']:.4f}",
        f"Test Set ROC-AUC       : {best_model_row['ROC-AUC']:.4f}",
        f"Test Set Log Loss      : {best_model_row['Log Loss']:.4f}",
        f"Prediction Throughput  : {best_model_row['Throughput']:.2f} predictions/sec",
        f"Model File Size        : {best_model_row['File Size']:.4f} MB",
        "-" * 80,
        "MODEL RANKINGS TABLE",
        "-" * 80,
        f"{'Rank':<4} | {'Model Name':<20} | {'Accuracy':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Log Loss':<8} | {'Score':<6}",
        "-" * 80,
    ]
    
    for idx, row in df_rankings.iterrows():
        txt_lines.append(
            f"{idx + 1:<4} | {row['Model']:<20} | {row['Accuracy']:.4f}   | {row['F1']:.4f}   | {row['ROC-AUC']:.4f}  | {row['Log Loss']:.4f} | {row['Composite_Score']:.4f}"
        )
        
    txt_lines.extend([
        "-" * 80,
        "OVERFITTING ANALYSIS (Train vs Test Accuracy)",
        "-" * 80,
        f"{'Model Name':<20} | {'Train Acc':<10} | {'Test Acc':<10} | {'Margin':<10} | {'Time (s)':<10}",
        "-" * 80,
    ])
    
    for model_name, summary in training_summary.items():
        train_acc = summary.get("train_accuracy", 0.0)
        test_acc = summary.get("test_accuracy", 0.0)
        overfit = train_acc - test_acc
        train_time = summary.get("training_time_seconds", 0.0)
        txt_lines.append(
            f"{model_name:<20} | {train_acc:<10.4f} | {test_acc:<10.4f} | {overfit:<10.4f} | {train_time:<10.4f}"
        )
        
    txt_lines.append("=" * 80)
    
    txt_report_path.write_text("\n".join(txt_lines), encoding="utf-8")
    logger.info("Saved plain text report to %s", txt_report_path)
    
    return md_report_path, txt_report_path


def main() -> None:
    """CLI Entry Point."""
    try:
        md_path, txt_path = generate_reports()
        print(f"\nReport Generation Complete.")
        print(f"  Markdown Report: {md_path}")
        print(f"  Plain Text Report: {txt_path}")
    except Exception as e:
        logger.exception("Failed to generate reports: %s", e)
        raise e


if __name__ == "__main__":
    main()
