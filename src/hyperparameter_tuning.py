"""Phase 3 - Step 3: Hyperparameter Tuning.

Optimizes the hyperparameters of top-performing models (Random Forest, XGBoost,
LightGBM, CatBoost) using RandomizedSearchCV with 5-Fold Cross-Validation.

Execution::

    python -m src.hyperparameter_tuning
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.config import (
    METRICS_DIR,
    PROCESSED_DIR,
    TRAINED_MODELS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="training.log")


def load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load train features and targets."""
    logger.info("Loading training dataset for hyperparameter tuning...")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

    # Stratified downsample for faster tuning execution (standard practice)
    max_tuning_samples = 10000
    if len(X_train) > max_tuning_samples:
        from sklearn.model_selection import train_test_split
        _, X_sample, _, y_sample = train_test_split(
            X_train, y_train,
            test_size=max_tuning_samples / len(X_train),
            random_state=42,
            stratify=y_train
        )
        logger.info("Downsampled tuning dataset to %d stratified samples for speed.", max_tuning_samples)
        return X_sample, y_sample

    return X_train, y_train


def get_search_spaces() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Define models and their respective hyperparameter search spaces."""
    search_spaces = {}

    # 1. Random Forest
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search_spaces["RandomForest"] = (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_grid
    )

    # 2. XGBoost
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2],
    }
    search_spaces["XGBoost"] = (
        XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42
        ),
        xgb_grid
    )

    # 3. LightGBM
    lgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    search_spaces["LightGBM"] = (
        LGBMClassifier(random_state=42, verbose=-1),
        lgb_grid
    )

    # 4. CatBoost
    cat_grid = {
        "iterations": [100, 200, 300],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
    }
    search_spaces["CatBoost"] = (
        CatBoostClassifier(random_seed=42, verbose=0),
        cat_grid
    )

    return search_spaces


def main() -> None:
    """Run hyperparameter tuning process."""
    X_train, y_train = load_training_data()
    search_spaces = get_search_spaces()

    tuned_params_summary = {}

    logger.info("=" * 60)
    logger.info("Starting Hyperparameter Tuning Pipeline")
    logger.info("=" * 60)

    for name, (model, grid) in search_spaces.items():
        logger.info("Optimizing %s ...", name)
        t_start = time.perf_counter()

        # We set n_iter=5 to balance optimization capability with execution time
        # cv=5 provides robust stratified splits
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=grid,
            n_iter=5,
            cv=5,
            scoring="accuracy",
            n_jobs=1,  # Set to 1 to prevent joblib nested deadlock with multithreaded estimators
            random_state=42,
        )

        search.fit(X_train, y_train)
        duration = time.perf_counter() - t_start

        best_params = search.best_params_
        best_score = search.best_score_
        logger.info("%s optimization finished in %.2f seconds.", name, duration)
        logger.info("%s Best CV Accuracy: %.4f", name, best_score)
        logger.info("%s Best Params: %s", name, best_params)

        tuned_params_summary[name] = {
            "best_params": best_params,
            "best_cv_score": float(best_score),
            "tuning_time_seconds": float(duration),
        }

        # Save the tuned model to TRAINED_MODELS_DIR
        tuned_model_path = TRAINED_MODELS_DIR / f"{name.lower()}_tuned.pkl"
        joblib.dump(search.best_estimator_, tuned_model_path)
        logger.info("Saved tuned model: %s", tuned_model_path)

    # Save best parameter JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / "tuned_parameters.json"
    with open(summary_path, "w") as f:
        json.dump(tuned_params_summary, f, indent=4)
    logger.info("Saved tuned parameters summary to %s", summary_path)

    print("\nHyperparameter Tuning Complete! Tuned models saved to models/trained/*_tuned.pkl.")
    print("Parameter logs saved to models/metrics/tuned_parameters.json.")


if __name__ == "__main__":
    main()
