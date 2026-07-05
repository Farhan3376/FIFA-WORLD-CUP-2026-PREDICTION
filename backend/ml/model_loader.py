"""Machine Learning Model Loader.

Implements a thread-safe lazy-loading wrapper for pre-trained models and assets,
reusing the MatchPredictor and ProbabilityEngine from previous phases.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import shap
from backend.config import settings
from simulation.match_predictor import MatchPredictor
from simulation.probability_engine import ProbabilityEngine

logger = logging.getLogger("backend")


class ModelLoader:
    """Thread-safe lazy loader for prediction models and preprocessing pipelines."""

    _instance: Optional[ModelLoader] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton instantiation."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """Initialize the model loader instance (runs once)."""
        if self._initialized:
            return

        self._predictor: Optional[MatchPredictor] = None
        self._prob_engine: Optional[ProbabilityEngine] = None
        self._shap_explainer: Optional[shap.TreeExplainer] = None
        self._initialized = True
        self._load_lock = threading.Lock()

    def load_assets(self) -> None:
        """Force load all models, databases, and pipelines into memory."""
        with self._load_lock:
            if self._predictor is not None:
                return

            logger.info("Loading pre-trained ML models and preprocessing components...")
            try:
                # Re-use MatchPredictor from simulation/match_predictor.py
                self._predictor = MatchPredictor(
                    model_path=settings.MODEL_PATH,
                    scaler_path=settings.SCALER_PATH,
                    imputer_path=settings.IMPUTER_PATH,
                    selector_path=settings.SELECTOR_PATH,
                    team_db_path=settings.TEAM_DB_PATH,
                )
                
                # Re-use ProbabilityEngine from simulation/probability_engine.py
                self._prob_engine = ProbabilityEngine(self._predictor)

                # Initialize SHAP explainer on the LightGBM model
                logger.info("Initializing SHAP TreeExplainer...")
                self._shap_explainer = shap.TreeExplainer(self._predictor.model)
                logger.info("ML assets loaded successfully.")

            except Exception as e:
                logger.error("Failed to load machine learning assets: %s", e, exc_info=True)
                raise RuntimeError(f"Error loading ML pipelines: {e}")

    @property
    def predictor(self) -> MatchPredictor:
        """Get the MatchPredictor instance (loads assets if not already loaded)."""
        if self._predictor is None:
            self.load_assets()
        assert self._predictor is not None
        return self._predictor

    @property
    def prob_engine(self) -> ProbabilityEngine:
        """Get the ProbabilityEngine instance (loads assets if not already loaded)."""
        if self._prob_engine is None:
            self.load_assets()
        assert self._prob_engine is not None
        return self._prob_engine

    @property
    def shap_explainer(self) -> shap.TreeExplainer:
        """Get the SHAP Explainer instance (loads assets if not already loaded)."""
        if self._shap_explainer is None:
            self.load_assets()
        assert self._shap_explainer is not None
        return self._shap_explainer


# Instantiate model loader singleton
ml_loader = ModelLoader()
