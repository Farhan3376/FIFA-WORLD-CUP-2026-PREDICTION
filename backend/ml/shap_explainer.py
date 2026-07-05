"""SHAP Explainer service for match predictions.

Computes local feature contribution values for the predicted outcome class
using the thread-safe model loader.
"""

from __future__ import annotations

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

from backend.ml.model_loader import ml_loader

logger = logging.getLogger("backend")


class SHAPExplainerService:
    """Service to compute and format SHAP values for match predictions."""

    @staticmethod
    def get_match_explanation(
        X_model_input: pd.DataFrame,
        predicted_class_idx: int
    ) -> Dict[str, Any]:
        """Compute SHAP values for the specified model input and predicted class.

        Args:
            X_model_input: DataFrame containing the 15 scaled and selected features.
            predicted_class_idx: Index of the predicted class (0: home_win, 1: draw, 2: away_win).

        Returns:
            Dictionary containing base value, feature contributions, and raw SHAP arrays.
        """
        try:
            explainer = ml_loader.shap_explainer
            feature_names = X_model_input.columns.tolist()

            # Run SHAP on the input row
            # X_model_input shape is (1, 15)
            shap_values = explainer.shap_values(X_model_input)
            
            # Retrieve expected (base) values
            # explainer.expected_value can be a list or float depending on SHAP version
            expected_values = explainer.expected_value
            if isinstance(expected_values, (list, np.ndarray)):
                base_value = float(expected_values[predicted_class_idx])
            else:
                base_value = float(expected_values)

            # Resolve multi-class list of arrays vs 3D array output from shap_values
            # shape: (n_classes, n_samples, n_features) or list of [n_samples, n_features]
            if isinstance(shap_values, list):
                # Standard for multi-class TreeExplainer in older SHAP versions
                class_shap = shap_values[predicted_class_idx][0]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # Shape is (n_samples, n_features, n_classes)
                class_shap = shap_values[0, :, predicted_class_idx]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                # Binary or regression case, or multi-class compressed
                class_shap = shap_values[0]
            else:
                class_shap = np.zeros(len(feature_names))

            # Pair features with their SHAP impact values
            contributions = {}
            for name, val in zip(feature_names, class_shap):
                contributions[name] = float(val)

            # Map predicted class index to name
            class_names = {0: "home_win", 1: "draw", 2: "away_win"}
            
            return {
                "predicted_outcome": class_names.get(predicted_class_idx, "unknown"),
                "base_value": base_value,
                "contributions": contributions,
                "total_impact": float(np.sum(np.abs(class_shap)))
            }

        except Exception as e:
            logger.error("Error generating SHAP explanation: %s", e, exc_info=True)
            # Safe fallback if SHAP fails or throws version/NumPy errors
            return {
                "predicted_outcome": "error",
                "base_value": 0.33,
                "contributions": {col: 0.0 for col in X_model_input.columns},
                "total_impact": 0.0,
                "warning": f"SHAP engine fallback: {str(e)}"
            }
