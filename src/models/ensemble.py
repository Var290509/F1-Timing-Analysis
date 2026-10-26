"""
F1 Sentinel Hybrid Ensemble Anomaly Detector
Fuses statistical, tree-based, density-based, kernel, and deep learning detectors
via weighted score averaging and consensus voting to optimize Event-F1.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.models.zscore_detector import ZScoreAnomalyDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof_detector import LOFDetector
from src.models.ocsvm_detector import OCSVMDetector
from src.models.autoencoder import AutoencoderDetector


DEFAULT_WEIGHTS = {
    "ZScore_Detector": 0.20,
    "Isolation_Forest": 0.25,
    "Local_Outlier_Factor": 0.15,
    "OneClass_SVM": 0.15,
    "PyTorch_Autoencoder": 0.25
}


class HybridEnsembleDetector:
    """
    Multi-model hybrid ensemble combining 5 diverse anomaly detection paradigms:
    1. Statistical: Rolling Z-Score
    2. Tree-based: Isolation Forest
    3. Density-based: Local Outlier Factor
    4. Kernel boundary: One-Class SVM
    5. Neural reconstruction: PyTorch Autoencoder
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        default_meta_threshold: float = 0.55
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}
        self.default_meta_threshold = default_meta_threshold
        self.name = "Hybrid_Ensemble"

        # Instantiate sub-detectors
        self.zscore_model = ZScoreAnomalyDetector()
        self.iforest_model = IsolationForestDetector()
        self.lof_model = LOFDetector()
        self.ocsvm_model = OCSVMDetector()
        self.ae_model = AutoencoderDetector()

        self.models_dict = {
            "ZScore_Detector": self.zscore_model,
            "Isolation_Forest": self.iforest_model,
            "Local_Outlier_Factor": self.lof_model,
            "OneClass_SVM": self.ocsvm_model,
            "PyTorch_Autoencoder": self.ae_model
        }
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, series_train: Optional[pd.Series] = None):
        """
        Fit all trainable sub-detectors on normal/baseline training data.
        """
        self.zscore_model.fit()
        self.iforest_model.fit(X_train)
        self.lof_model.fit(X_train)
        self.ocsvm_model.fit(X_train)
        self.ae_model.fit(X_train)
        self.is_fitted = True
        return self

    def score_all_models(
        self,
        X: np.ndarray,
        series: pd.Series
    ) -> Dict[str, np.ndarray]:
        """
        Compute continuous [0, 1] anomaly scores for every individual detector and ensemble.
        
        Returns:
            Dictionary mapping model names to 1D score arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble models are not fitted yet.")

        z_scores = self.zscore_model.score_series(series)
        if_scores = self.iforest_model.score(X)
        lof_scores = self.lof_model.score(X)
        ocsvm_scores = self.ocsvm_model.score(X)
        ae_scores = self.ae_model.score(X)

        all_scores = {
            "ZScore_Detector": z_scores,
            "Isolation_Forest": if_scores,
            "Local_Outlier_Factor": lof_scores,
            "OneClass_SVM": ocsvm_scores,
            "PyTorch_Autoencoder": ae_scores
        }

        # Weighted blend
        ensemble_scores = np.zeros(len(X), dtype=float)
        for model_name, scores in all_scores.items():
            w = self.weights.get(model_name, 0.2)
            ensemble_scores += w * scores

        all_scores["Hybrid_Ensemble"] = np.clip(ensemble_scores, 0.0, 1.0)
        return all_scores

    def score_single(
        self,
        current_value: float,
        trailing_values: np.ndarray,
        feature_vector: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a single streaming telemetry observation across all models.
        
        Returns:
            ensemble_score: float in [0, 1]
            breakdown: dictionary of individual model scores
        """
        z_score, _, _ = self.zscore_model.score_single(current_value, trailing_values)
        if_score = self.iforest_model.score_single(feature_vector)
        lof_score = self.lof_model.score_single(feature_vector)
        ocsvm_score = self.ocsvm_model.score_single(feature_vector)
        ae_score, _ = self.ae_model.score_single(feature_vector)

        breakdown = {
            "ZScore_Detector": z_score,
            "Isolation_Forest": if_score,
            "Local_Outlier_Factor": lof_score,
            "OneClass_SVM": ocsvm_score,
            "PyTorch_Autoencoder": ae_score
        }

        ensemble_score = sum(self.weights[k] * breakdown[k] for k in breakdown)
        breakdown["Hybrid_Ensemble"] = float(np.clip(ensemble_score, 0.0, 1.0))
        return float(ensemble_score), breakdown
