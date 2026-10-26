"""
F1 Sentinel Isolation Forest Anomaly Detector
Fits an ensemble of isolation trees on windowed telemetry features.
Converts decision function depths to continuous [0, 1] anomaly scores.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """
    Isolation Forest for high-dimensional windowed telemetry anomaly detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.name = "Isolation_Forest"
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self._score_min = -0.5
        self._score_max = 0.5

    def fit(self, X: np.ndarray):
        """Fit scaler and Isolation Forest on training feature matrix."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Calibrate empirical score distribution for smooth [0, 1] normalization
        raw_scores = self.model.decision_function(X_scaled)
        self._score_min = float(np.percentile(raw_scores, 1))
        self._score_max = float(np.percentile(raw_scores, 99))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous [0, 1] anomaly scores.
        Higher score = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        # raw_scores: lower is more anomalous. Invert and normalize to [0, 1]
        denom = max(self._score_max - self._score_min, 1e-5)
        norm = (self._score_max - raw_scores) / denom
        return np.clip(norm, 0.0, 1.0)

    def score_single(self, feature_vector: np.ndarray) -> float:
        """Score a single feature vector in streaming mode."""
        feat_2d = feature_vector.reshape(1, -1)
        scores = self.score(feat_2d)
        return float(scores[0])
