"""Phase 3 - Step 5: Model Explainability and Interpretation.

Computes and saves publication-quality global Permutation Importance and SHAP-based
interpretability plots (summary, bar, waterfall, and decision plots) using the 
overall best model.

Execution::

    python -m src.explainability
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.inspection import permutation_importance

from src.config import (
    PROCESSED_DIR,
    TRAINED_MODELS_DIR,
    ML_REPORTS_DIR,
)
from src.utils.logger import get_logger

# Use non-interactive backend for matplotlib
plt.switch_backend("Agg")

logger = get_logger(__name__, log_filename="training.log")


def load_best_model_and_data():
    """Load best model, test features, and labels."""
    model_path = TRAINED_MODELS_DIR / "best_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Best model not found at {model_path}. Run model_selection.py first.")
    
    logger.info("Loading best model from: %s", model_path)
    model = joblib.load(model_path)
    
    X_test_path = PROCESSED_DIR / "X_test.csv"
    y_test_path = PROCESSED_DIR / "y_test.csv"
    
    if not X_test_path.is_file() or not y_test_path.is_file():
        raise FileNotFoundError("Processed X_test.csv or y_test.csv not found.")
        
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    return model, X_test, y_test


def compute_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Compute and plot Permutation Feature Importance."""
    logger.info("Computing Permutation Feature Importance...")
    
    # Run permutation importance with 5 repeats
    result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )
    
    sorted_importances_idx = result.importances_mean.argsort()
    
    # Create pandas DataFrame of importances
    importance_df = pd.DataFrame({
        "Feature": X_test.columns[sorted_importances_idx],
        "Importance_Mean": result.importances_mean[sorted_importances_idx],
        "Importance_Std": result.importances_std[sorted_importances_idx],
    })
    
    logger.info("\n--- Permutation Importance Rankings (Descending) ---")
    for _, row in importance_df.iloc[::-1].iterrows():
        logger.info("%-25s: %.5f (+/- %.5f)", row["Feature"], row["Importance_Mean"], row["Importance_Std"])

    # Save to CSV
    ML_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.iloc[::-1].to_csv(ML_REPORTS_DIR / "permutation_importance.csv", index=False)
    
    # Plot horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance_Mean"],
        xerr=importance_df["Importance_Std"],
        color="#2c3e50",
        edgecolor="none",
        height=0.6
    )
    plt.title("Permutation Feature Importance on Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Decrease in Accuracy Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    
    plot_path = ML_REPORTS_DIR / "permutation_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved permutation importance plot to: %s", plot_path)


def compute_shap_plots(model, X_test: pd.DataFrame) -> None:
    """Compute SHAP values and save summary, bar, waterfall, and decision plots."""
    logger.info("Starting SHAP Explainability computation...")
    
    # Downsample validation set for SHAP computation (500 stratified samples for speed/robustness)
    # We sample the first 500 rows for simplicity or use randomized state
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    
    # 1. Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle SHAP output shape differences
    # For LightGBM multiclass, shap_values shape is typically (n_samples, n_features, n_classes)
    # Convert to list of matrices if 3D array
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_values_list = shap_values

    # Determine base values for the first class (Home Win)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        ev_class_0 = expected_value[0]
    else:
        ev_class_0 = expected_value

    class_names = ["Home Win", "Draw", "Away Win"]
    
    # 2. Save Multiclass SHAP Summary Bar Plot
    # Shows mean absolute impact across all three classes
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_list,
        X_sample,
        class_names=class_names,
        show=False
    )
    plt.title("SHAP Global Feature Impact (Multiclass)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = ML_REPORTS_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot: %s", summary_path)

    # 3. Create Explanation Object for Class 0 (Home Win) to generate waterfall/bar plots
    # Get values for class 0 (Home Win)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_0_values = shap_values[:, :, 0]
    elif isinstance(shap_values, list):
        class_0_values = shap_values[0]
    else:
        class_0_values = shap_values
        
    exp = shap.Explanation(
        values=class_0_values,
        base_values=ev_class_0,
        data=X_sample.values,
        feature_names=X_sample.columns.tolist()
    )

    # 4. Save SHAP Global Bar Plot for Home Win
    plt.figure(figsize=(10, 6))
    shap.plots.bar(exp, show=False)
    plt.title("SHAP Feature Importance (Home Win class)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    bar_path = ML_REPORTS_DIR / "shap_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP global bar plot: %s", bar_path)

    # 5. Save SHAP Waterfall Plot for the First Sample in the batch
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp[0], show=False)
    plt.title("SHAP Local Prediction Explanation (Home Win class)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    waterfall_path = ML_REPORTS_DIR / "shap_waterfall.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP waterfall plot: %s", waterfall_path)

    # 6. Save SHAP Decision Plot for Class 0 (first 50 samples)
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        ev_class_0,
        class_0_values[:50],
        features=X_sample.iloc[:50],
        show=False
    )
    plt.title("SHAP Decision Plot (Home Win class - First 50 samples)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    decision_path = ML_REPORTS_DIR / "shap_decision.png"
    plt.savefig(decision_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP decision plot: %s", decision_path)


def main() -> None:
    """Run explainability pipeline."""
    try:
        model, X_test, y_test = load_best_model_and_data()
        compute_permutation_importance(model, X_test, y_test)
        compute_shap_plots(model, X_test)
        logger.info("Explainability Pipeline execution completed successfully!")
    except Exception as e:
        logger.exception("Error in Explainability Pipeline:")
        raise e


if __name__ == "__main__":
    main()
