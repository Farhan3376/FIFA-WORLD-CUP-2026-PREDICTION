"""Phase 3 - Step 4: Model Selection and Ranking.

Automatically ranks all trained baseline and tuned models based on a weighted
utility score that incorporates Accuracy, F1, ROC-AUC, Log Loss, Prediction
Throughput, and Model Size on disk. Serializes the best overall model to
models/trained/best_model.pkl.

Execution::

    python -m src.model_selection
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.config import (
    METRICS_DIR,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="training.log")

# Configurable Weights for Model Selection (must sum to 1.0)
WEIGHTS = {
    "Accuracy": 0.20,
    "F1": 0.30,
    "ROC-AUC": 0.20,
    "Log Loss": 0.10,
    "Throughput": 0.10,    # Predictions per second (1 / Prediction Time)
    "File Size": 0.10,     # Size on disk (smaller is better)
}


def load_metrics() -> Dict[str, Dict[str, float]]:
    """Load evaluation metrics JSON file."""
    metrics_path = METRICS_DIR / "eval_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation metrics not found at {metrics_path}. Run evaluate.py first.")
    
    with open(metrics_path, "r") as f:
        return json.load(f)


def calculate_file_size(model_name: str) -> float:
    """Calculate the size of the saved model pickle file in MB."""
    file_path = TRAINED_MODELS_DIR / f"{model_name.lower()}.pkl"
    if not file_path.is_file():
        logger.warning("Pickle file not found for %s at %s", model_name, file_path)
        return 999.0  # Assign a large dummy value if file is missing
    
    # Return size in MB
    return file_path.stat().st_size / (1024 * 1024)


def perform_selection() -> None:
    """Rank models and select/export the best model."""
    logger.info("=" * 60)
    logger.info("Starting Model Selection & Ranking Pipeline")
    logger.info("=" * 60)

    # 1. Load metrics
    raw_metrics = load_metrics()
    
    # 2. Build DataFrame and add File Size & Throughput
    rows = []
    for model_name, metrics in raw_metrics.items():
        # Avoid ranking 'best_model' itself if it somehow exists
        if model_name.lower() == "best_model":
            continue
            
        file_size_mb = calculate_file_size(model_name)
        pred_time = metrics.get("Prediction Time (s)", 1.0)
        throughput = 1.0 / pred_time if pred_time > 0 else 0.0

        rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("Accuracy", 0.0),
            "F1": metrics.get("F1", 0.0),
            "ROC-AUC": metrics.get("ROC-AUC", 0.5),
            "Log Loss": metrics.get("Log Loss", 9.9),
            "Throughput": throughput,
            "File Size": file_size_mb,
            "Raw Prediction Time": pred_time
        })
        
    df = pd.DataFrame(rows)
    
    if df.empty:
        logger.error("No models found for ranking.")
        return

    # Filter out models that are completely unviable (e.g. worse log loss than random guessing)
    viable_mask = (df["Log Loss"] < 1.1) & (df["Accuracy"] >= 0.55)
    filtered_out = df[~viable_mask]["Model"].tolist()
    if filtered_out:
        logger.info("Filtered out unviable models (Log Loss >= 1.1 or Accuracy < 0.55): %s", filtered_out)
        print(f"Filtered out unviable models: {filtered_out}")
    df = df[viable_mask].reset_index(drop=True)

    # 3. Normalize metrics (Min-Max Scaling to 0-1 range)
    # Higher is better: Accuracy, F1, ROC-AUC, Throughput
    # Lower is better: Log Loss, File Size
    norm_cols = {}
    for col in ["Accuracy", "F1", "ROC-AUC", "Throughput"]:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            norm_cols[f"Norm_{col}"] = (df[col] - col_min) / (col_max - col_min)
        else:
            norm_cols[f"Norm_{col}"] = 1.0  # Equal values get maximum score

    for col in ["Log Loss", "File Size"]:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            norm_cols[f"Norm_{col}"] = (col_max - df[col]) / (col_max - col_min)
        else:
            norm_cols[f"Norm_{col}"] = 1.0

    df_norm = pd.DataFrame(norm_cols)
    df = pd.concat([df, df_norm], axis=1)

    # 4. Compute Weighted score
    df["Composite_Score"] = (
        df["Norm_Accuracy"] * WEIGHTS["Accuracy"] +
        df["Norm_F1"] * WEIGHTS["F1"] +
        df["Norm_ROC-AUC"] * WEIGHTS["ROC-AUC"] +
        df["Norm_Log Loss"] * WEIGHTS["Log Loss"] +
        df["Norm_Throughput"] * WEIGHTS["Throughput"] +
        df["Norm_File Size"] * WEIGHTS["File Size"]
    )

    # Sort models by composite score descending
    df_sorted = df.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)

    # 5. Log and Display rankings
    logger.info("\n--- Model Comparison & Rankings ---")
    print("\n--- Model Comparison & Rankings ---")
    rank_headers = f"{'Rank':<5} | {'Model':<22} | {'Accuracy':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Size (MB)':<9} | {'Score':<6}"
    logger.info(rank_headers)
    print(rank_headers)
    logger.info("-" * 80)
    print("-" * 80)
    
    for idx, row in df_sorted.iterrows():
        rank_str = f"{idx + 1:<5} | {row['Model']:<22} | {row['Accuracy']:0.4f}   | {row['F1']:0.4f}   | {row['ROC-AUC']:0.4f}  | {row['File Size']:0.2f}      | {row['Composite_Score']:0.4f}"
        logger.info(rank_str)
        print(rank_str)

    # 6. Save rankings to CSV and JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rankings_csv_path = METRICS_DIR / "model_rankings.csv"
    df_sorted.to_csv(rankings_csv_path, index=False)
    logger.info("Saved model ranking table to %s", rankings_csv_path)

    # 7. Select overall winner and export it
    best_model_row = df_sorted.iloc[0]
    best_model_name = best_model_row["Model"]
    
    logger.info("\n🏆 Selected Best Overall Model: %s (Composite Score: %.4f)", best_model_name, best_model_row["Composite_Score"])
    print(f"\n🏆 Selected Best Overall Model: {best_model_name} (Composite Score: {best_model_row['Composite_Score']:0.4f})")

    src_path = TRAINED_MODELS_DIR / f"{best_model_name.lower()}.pkl"
    dest_path = TRAINED_MODELS_DIR / "best_model.pkl"
    
    shutil.copy2(src_path, dest_path)
    logger.info("Copied best model (%s) -> %s", src_path.name, dest_path)
    print(f"Serialized best model successfully to: {dest_path}")


def main() -> None:
    """Run model selection ranking process."""
    try:
        perform_selection()
    except Exception as e:
        logger.exception("Error during model selection ranking:")
        raise e


if __name__ == "__main__":
    main()
