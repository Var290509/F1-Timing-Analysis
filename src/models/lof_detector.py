"""
F1 Sentinel Local Outlier Factor (LOF) Detector
Density-based local outlier detection operating in novelty detection mode.
Measures local reachability density relative to k-nearest neighbors in feature space.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class LOFDetector:
    """
    Novelty-mode Local Outlier Factor for real-time telemetry anomaly detection.
    """

    def __init__(
        self,
        n_neighbors: int = 25,
        contamination: float = 0.05
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.name = "Local_Outlier_Factor"
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1
        )
        self.is_fitted = False
        self._score_min = -1.5
        self._score_max = 0.5

    def fit(self, X: np.ndarray):
        """Fit scaler and LOF model in novelty detection mode."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

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
            raise RuntimeError("LOF model is not fitted yet.")

        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        # raw_scores: lower is more anomalous
        denom = max(self._score_max - self._score_min, 1e-5)
        norm = (self._score_max - raw_scores) / denom
        return np.clip(norm, 0.0, 1.0)

    def score_single(self, feature_vector: np.ndarray) -> float:
        """Score a single feature vector in streaming mode."""
        feat_2d = feature_vector.reshape(1, -1)
        scores = self.score(feat_2d)
        return float(scores[0])
