"""Phase 3 - Step 1: Model Training Pipeline.

Trains baseline and advanced ML models (including a PyTorch Feedforward NN),
performs cross-validation, and serializes the trained models to disk.

Execution::

    python -m src.train
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Advanced Gradient Boosting Classifiers
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import (
    COL_RESULT,
    LOGS_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    TRAINED_MODELS_DIR,
    ensure_directories,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="training.log")


# =====================================================================
# 1. Custom Shared Models and Wrappers
# =====================================================================

from src.utils.models import FastSVC, PyTorchClassifier, SimpleNN


# =====================================================================
# 3. Main Training Pipeline Functions
# =====================================================================

def load_and_split_data(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the dataset and perform an 80/20 stratified split.

    Args:
        dataset_path: Path to the processed training CSV.

    Returns:
        Tuple of training and testing features and targets.
    """
    logger.info("Loading training dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path)

    # Separate target
    if COL_RESULT not in df.columns:
        raise ValueError(f"Target column '{COL_RESULT}' not found in the dataset.")

    X = df.drop(columns=[COL_RESULT])
    y = df[COL_RESULT]

    # Stratified Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info("Dataset split complete. Train size: %d, Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def get_models(input_dim: int) -> Dict[str, Any]:
    """Initialize all baseline, advanced, and deep learning models.

    Args:
        input_dim: Number of input features.

    Returns:
        Dict mapping model name to model estimator.
    """
    models = {
        # Baseline Models
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "SVM": FastSVC(max_samples=5000, random_state=42),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "NaiveBayes": GaussianNB(),
        "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),

        # Advanced Models
        "XGBoost": XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0, thread_count=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),

        # Deep Learning Model
        "NeuralNetwork": PyTorchClassifier(input_dim=input_dim, random_state=42),
    }
    return models


def evaluate_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> None:
    """Implement 5-Fold, 10-Fold, and Repeated Stratified K-Fold CV strategies.

    Outputs findings for key representative models to compare folds and stability.
    """
    logger.info("Initializing Cross-Validation evaluation...")
    
    cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Let's run CV on a subset of models to demonstrate stability
    demo_models = ["RandomForest", "LightGBM", "LogisticRegression"]

    for name in demo_models:
        if name not in models:
            continue
        model = models[name]
        logger.info("Running cross-validation evaluations for model: %s", name)

        # 5-Fold
        scores_5 = cross_val_score(model, X_train, y_train, cv=cv_5, scoring="accuracy")
        logger.info("%s: 5-Fold CV Accuracy: %.4f +/- %.4f", name, scores_5.mean(), scores_5.std())

        # 10-Fold
        scores_10 = cross_val_score(model, X_train, y_train, cv=cv_10, scoring="accuracy")
        logger.info("%s: 10-Fold CV Accuracy: %.4f +/- %.4f", name, scores_10.mean(), scores_10.std())

        # Repeated Stratified K-Fold
        scores_rep = cross_val_score(model, X_train, y_train, cv=cv_repeated, scoring="accuracy")
        logger.info("%s: Repeated Stratified K-Fold Accuracy (5 splits, 3 reps): %.4f +/- %.4f", name, scores_rep.mean(), scores_rep.std())


def train_and_save_all(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """Train all models, log their test accuracy, and serialize them to disk."""
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting model training pipeline")
    logger.info("=" * 60)

    summary_metrics = {}

    for name, model in models.items():
        logger.info("Training model: %s ...", name)
        t_start = time.perf_counter()
        
        # Fit model
        model.fit(X_train, y_train)
        
        train_time = time.perf_counter() - t_start
        logger.info("%s trained successfully in %.3f seconds.", name, train_time)

        # Simple accuracy check
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)

        # Flatten predictions to prevent broadcasting/dimension issues (e.g. CatBoost)
        if hasattr(preds_train, "ndim") and preds_train.ndim > 1:
            preds_train = preds_train.ravel()
        if hasattr(preds_test, "ndim") and preds_test.ndim > 1:
            preds_test = preds_test.ravel()

        train_acc = np.mean(preds_train == y_train)
        test_acc = np.mean(preds_test == y_test)
        logger.info("%s scores: Train Acc = %.4f | Test Acc = %.4f", name, train_acc, test_acc)

        summary_metrics[name] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "training_time_seconds": float(train_time),
        }

        # Save model
        model_path = TRAINED_MODELS_DIR / f"{name.lower()}.pkl"
        joblib.dump(model, model_path)
        logger.info("Saved %s -> %s", name, model_path)

    # Save summary metrics to models/metrics/
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
    logger.info("Saved training summary report to %s", summary_path)


def main() -> None:
    """Orchestrate training process."""
    ensure_directories()
    
    dataset_path = PROCESSED_DIR / "processed_training_dataset.csv"
    if not dataset_path.is_file():
        logger.error("Processed dataset not found at %s", dataset_path)
        raise FileNotFoundError(f"Dataset missing: {dataset_path}")

    # Load and Split Data
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_path)

    # Save splits to prevent leakage and ensure evaluation runs on identical partitions
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    logger.info("Splits saved to %s for cross-module consistency.", PROCESSED_DIR)

    # Initialize Models
    models = get_models(input_dim=X_train.shape[1])

    # Run Cross-Validation strategies demo
    evaluate_cross_validation(models, X_train, y_train)

    # Train and Save Models
    train_and_save_all(models, X_train, y_train, X_test, y_test)

    print("\nModel Training Complete! All models saved to models/trained/.")
    print("Logs saved to logs/training.log.")


if __name__ == "__main__":
    main()
