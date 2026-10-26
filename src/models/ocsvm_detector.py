"""
F1 Sentinel One-Class Support Vector Machine (OCSVM) Detector
Constructs a maximal-margin non-linear RBF decision envelope in reproducing kernel Hilbert space.
Evaluates distance to the boundary to yield continuous [0, 1] anomaly scores.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class OCSVMDetector:
    """
    One-Class SVM with RBF kernel for telemetry novelty detection.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale"
    ):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.name = "OneClass_SVM"
        self.scaler = StandardScaler()
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )
        self.is_fitted = False
        self._score_min = -2.0
        self._score_max = 2.0

    def fit(self, X: np.ndarray):
        """Fit scaler and One-Class SVM on training feature matrix."""
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
            raise RuntimeError("One-Class SVM is not fitted yet.")

        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        denom = max(self._score_max - self._score_min, 1e-5)
        norm = (self._score_max - raw_scores) / denom
        return np.clip(norm, 0.0, 1.0)

    def score_single(self, feature_vector: np.ndarray) -> float:
        """Score a single feature vector in streaming mode."""
        feat_2d = feature_vector.reshape(1, -1)
        scores = self.score(feat_2d)
        return float(scores[0])
