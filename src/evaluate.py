"""Phase 3 - Step 2: Model Evaluation Pipeline.

Computes comprehensive performance metrics for all trained models and generates
publication-quality figures (Confusion Matrices, ROC, PR, Learning, Validation curves).

Execution::

    python -m src.evaluate
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, validation_curve

from src.config import (
    ML_REPORTS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger
from src.utils.models import FastSVC, PyTorchClassifier, SimpleNN

logger = get_logger(__name__, log_filename="training.log")


# Set premium plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 150,
})

# Color palette definition (Home Win, Draw, Away Win colors)
CLASS_COLORS = ["#2b5c8f", "#7f7f7f", "#d95f02"]  # Blue, Grey, Orange
MODEL_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


# =====================================================================
# 1. Helper Functions
# =====================================================================

def load_data_and_models() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Load train/test splits and all trained models.

    Returns:
        Tuple of X_train, X_test, y_train, y_test, and models dict.
    """
    logger.info("Loading train/test data splits...")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    models = {}
    model_files = list(TRAINED_MODELS_DIR.glob("*.pkl"))
    logger.info("Found %d model file(s) in %s", len(model_files), TRAINED_MODELS_DIR)

    for path in model_files:
        model_name = path.stem
        # Capitalize name logically for plotting
        display_name = {
            "logisticregression": "LogisticRegression",
            "decisiontree": "DecisionTree",
            "randomforest": "RandomForest",
            "svm": "SVM",
            "knn": "KNN",
            "naivebayes": "NaiveBayes",
            "extratrees": "ExtraTrees",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
            "histgradientboosting": "HistGradientBoosting",
            "neuralnetwork": "NeuralNetwork",
        }.get(model_name, model_name)

        models[display_name] = joblib.load(path)
        logger.info("Loaded model: %s", display_name)

    return X_train, X_test, y_train, y_test, models


def compute_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute all evaluation metrics, predictions and probabilities.

    Args:
        model: Trained model estimator.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        Tuple of metrics dictionary, predictions, and probabilities.
    """
    t_start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    prediction_time = time.perf_counter() - t_start

    # Flatten predictions if they are 2D
    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    # One-vs-Rest AUC Score calculation
    # result target values are 0 (Home Win), 1 (Draw), 2 (Away Win)
    auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "ROC-AUC": float(auc_ovr),
        "Log Loss": float(log_loss(y_test, y_prob)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "Cohen's Kappa": float(cohen_kappa_score(y_test, y_pred)),
        "Prediction Time (s)": float(prediction_time),
    }

    return metrics, y_pred, y_prob


# =====================================================================
# 2. Plotting Functions
# =====================================================================

def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generate and save confusion matrices for top models."""
    top_models = ["RandomForest", "LightGBM", "XGBoost", "LogisticRegression"]
    
    for name in top_models:
        if name not in models:
            continue
        model = models[name]
        y_pred = model.predict(X_test)
        if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
            y_pred = y_pred.ravel()
            
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Home Win", "Draw", "Away Win"],
            yticklabels=["Home Win", "Draw", "Away Win"],
            cbar=True,
            square=True,
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Outcome")
        plt.ylabel("Actual Outcome")
        plt.tight_layout()
        
        fig_path = output_dir / f"confusion_matrix_{name.lower()}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved confusion matrix: %s", fig_path)


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generate combined ROC Curves for top models (One-vs-Rest, average macro)."""
    plt.figure(figsize=(8, 6))
    
    # Target classes: 0 (Home Win), 1 (Draw), 2 (Away Win)
    # Convert y_test to one-hot encoding for multi-class ROC plotting
    y_onehot = pd.get_dummies(y_test).values
    n_classes = y_onehot.shape[1]

    top_models = ["RandomForest", "LightGBM", "XGBoost", "LogisticRegression", "NeuralNetwork"]
    
    for idx, name in enumerate(top_models):
        if name not in models:
            continue
        model = models[name]
        y_prob = model.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class, then average
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_prob[:, i])
            roc_auc[i] = roc_auc_score(y_onehot[:, i], y_prob[:, i])
            
        # Micro-average ROC curve
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        
        plt.plot(
            fpr_grid,
            mean_tpr,
            label=f"{name} (AUC = {macro_auc:.3f})",
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            linewidth=2,
        )
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.500)", alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curves (Macro Average)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    fig_path = output_dir / "roc_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curves: %s", fig_path)


def plot_pr_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generate combined Precision-Recall curves for top models (Macro Average)."""
    plt.figure(figsize=(8, 6))
    y_onehot = pd.get_dummies(y_test).values
    n_classes = y_onehot.shape[1]

    top_models = ["RandomForest", "LightGBM", "XGBoost", "LogisticRegression", "NeuralNetwork"]
    
    for idx, name in enumerate(top_models):
        if name not in models:
            continue
        model = models[name]
        y_prob = model.predict_proba(X_test)
        
        # Calculate macro PR curve
        recall_grid = np.linspace(0.0, 1.0, 1000)
        mean_precision = np.zeros_like(recall_grid)
        
        for i in range(n_classes):
            p, r, _ = precision_recall_curve(y_onehot[:, i], y_prob[:, i])
            # Interpolate precision to match recall grid
            mean_precision += np.interp(recall_grid, r[::-1], p[::-1])
            
        mean_precision /= n_classes
        
        plt.plot(
            recall_grid,
            mean_precision,
            label=f"{name}",
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            linewidth=2,
        )
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Macro Average)")
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    
    fig_path = output_dir / "pr_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved PR curves: %s", fig_path)


def plot_learning_curves(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path
) -> None:
    """Generate learning curves for top model to diagnose bias/variance."""
    # We choose LightGBM as the representative model for learning curves
    name = "LightGBM"
    if name not in models:
        return
        
    model = models[name]
    logger.info("Computing learning curves for %s (this may take a moment)...", name)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=3,
        scoring="accuracy",
        n_jobs=1,
        random_state=42,
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#e41a1c", label="Training Score", linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#e41a1c")
    
    plt.plot(train_sizes, val_mean, "s-", color="#377eb8", label="Cross-Validation Score", linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#377eb8")
    
    plt.title(f"Learning Curves ({name})")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    
    fig_path = output_dir / f"learning_curves_{name.lower()}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved learning curves: %s", fig_path)


def plot_validation_curves(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path
) -> None:
    """Generate validation curves to show hyperparameter behavior."""
    # We choose LightGBM max_depth as the representative hyperparameter
    name = "LightGBM"
    if name not in models:
        return
        
    model = models[name]
    param_name = "max_depth"
    param_range = [3, 5, 7, 10, 15]
    
    logger.info("Computing validation curves for %s on %s...", name, param_name)
    
    train_scores, val_scores = validation_curve(
        model,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        scoring="accuracy",
        n_jobs=1,
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(7, 5))
    plt.plot(param_range, train_mean, "o-", color="#e41a1c", label="Training Score", linewidth=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#e41a1c")
    
    plt.plot(param_range, val_mean, "s-", color="#377eb8", label="Cross-Validation Score", linewidth=2)
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#377eb8")
    
    plt.title(f"Validation Curves ({name}) - max_depth")
    plt.xlabel("Max Depth Value")
    plt.ylabel("Accuracy")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    
    fig_path = output_dir / f"validation_curves_{name.lower()}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved validation curves: %s", fig_path)


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate model comparison bar charts."""
    metrics_to_plot = ["Accuracy", "F1", "Prediction Time (s)"]
    
    for metric in metrics_to_plot:
        # Sort values logically for presentation
        df_sorted = metrics_df.sort_values(by=metric, ascending=(metric != "Prediction Time (s)"))
        
        plt.figure(figsize=(10, 5))
        bars = plt.barh(
            df_sorted.index,
            df_sorted[metric],
            color="#2b5c8f" if metric != "Prediction Time (s)" else "#d95f02",
            height=0.6,
        )
        
        # Add labels on the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + (width * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}" if metric != "Prediction Time (s)" else f"{width:.4f}s",
                ha="left",
                va="center",
                fontsize=9,
            )
            
        plt.title(f"Model Comparison - {metric}")
        plt.xlabel(metric)
        plt.ylabel("Model Name")
        plt.tight_layout()
        
        metric_file = metric.lower().replace(" (s)", "").replace(" ", "_")
        fig_path = output_dir / f"model_comparison_{metric_file}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved comparison plot: %s", fig_path)


# =====================================================================
# 3. Main Orchestration
# =====================================================================

def main() -> None:
    """Run model evaluation pipeline."""
    # Ensure output directories exist
    ML_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = MODELS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load splits and models
    X_train, X_test, y_train, y_test, models = load_data_and_models()

    if not models:
        logger.error("No trained models found! Please run train.py first.")
        raise FileNotFoundError("Trained models directory is empty.")

    eval_results = {}
    logger.info("=" * 60)
    logger.info("Evaluating all trained models")
    logger.info("=" * 60)

    for name, model in models.items():
        logger.info("Evaluating: %s ...", name)
        
        # Compute metrics
        metrics, y_pred, y_prob = compute_metrics(model, X_test, y_test)
        
        # Print metrics to stdout
        logger.info("%s Test Accuracy = %.4f | F1 = %.4f | Log Loss = %.4f", name, metrics["Accuracy"], metrics["F1"], metrics["Log Loss"])
        
        # Store results
        eval_results[name] = metrics

        # Log detailed classification report
        report = classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"])
        logger.info("Classification Report for %s:\n%s", name, report)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(eval_results).T
    metrics_df.index.name = "Model"

    # Save metrics table to JSON & CSV
    metrics_json_path = metrics_dir / "eval_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(eval_results, f, indent=4)
        
    metrics_csv_path = metrics_dir / "eval_metrics.csv"
    metrics_df.to_csv(metrics_csv_path)
    logger.info("Saved evaluation metrics table to %s and %s", metrics_json_path, metrics_csv_path)

    # Generate figures
    logger.info("Generating publication-quality visualization plots...")
    plot_confusion_matrices(models, X_test, y_test, ML_REPORTS_DIR)
    plot_roc_curves(models, X_test, y_test, ML_REPORTS_DIR)
    plot_pr_curves(models, X_test, y_test, ML_REPORTS_DIR)
    plot_learning_curves(models, X_train, y_train, ML_REPORTS_DIR)
    plot_validation_curves(models, X_train, y_train, ML_REPORTS_DIR)
    plot_model_comparison(metrics_df, ML_REPORTS_DIR)

    print("\nModel Evaluation Complete! Evaluation reports and plots saved to reports/ml/.")
    print("Metrics table saved to models/metrics/eval_metrics.json.")


if __name__ == "__main__":
    main()
