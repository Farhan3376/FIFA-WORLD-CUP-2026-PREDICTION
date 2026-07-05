"""Shared custom models and wrappers for Phase 3.

Houses SimpleNN, PyTorchClassifier, and FastSVC to ensure they are serialized
under a stable module path ('src.utils.models') rather than '__main__'.
"""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.utils.logger import get_logger

logger = get_logger(__name__, log_filename="training.log")


class SimpleNN(nn.Module):
    """Feedforward PyTorch Neural Network for multiclass prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchClassifier:
    """Scikit-learn compatible wrapper for the PyTorch Neural Network."""

    def __init__(
        self,
        input_dim: int,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 0.001,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.model = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> PyTorchClassifier:
        """Fit the PyTorch model."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = SimpleNN(self.input_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_np = X.values if hasattr(X, "values") else np.array(X)
        y_np = y.values if hasattr(y, "values") else np.array(y)

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute class probabilities."""
        X_np = X.values if hasattr(X, "values") else np.array(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


class FastSVC:
    """Wrapper for Support Vector Classifier to ensure fast training on large data."""

    def __init__(self, max_samples: int = 5000, random_state: int = 42):
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = SVC(probability=True, C=1.0, kernel="rbf", random_state=random_state)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> FastSVC:
        """Fit SVM on a stratified subset if data size exceeds limit."""
        if len(X) > self.max_samples:
            # Draw stratified sample
            X_df = pd.DataFrame(X)
            y_ser = pd.Series(y)
            _, X_sample, _, y_sample = train_test_split(
                X_df, y_ser,
                test_size=self.max_samples / len(X_df),
                random_state=self.random_state,
                stratify=y_ser
            )
            self.model.fit(X_sample, y_sample)
            logger.info("FastSVC: trained on a stratified sample of %d rows (downsampled from %d)", self.max_samples, len(X))
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
