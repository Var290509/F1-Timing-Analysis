"""
F1 Sentinel Statistical Baseline: Rolling Z-Score Anomaly Detector
Computes rolling dynamic mean and variance over a calibration window.
Normalizes anomaly score to continuous [0, 1] with an adaptive threshold.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class ZScoreAnomalyDetector:
    """
    Statistical baseline detector using rolling window Z-Score with adaptive deviation thresholds.
    """

    def __init__(self, window_size: int = 30, base_threshold: float = 3.0):
        self.window_size = window_size
        self.base_threshold = base_threshold
        self.name = "ZScore_Detector"

    def fit(self, X: np.ndarray = None):
        """Statistical detector is online/non-parametric, fit is a no-op."""
        return self

    def score_series(self, values: pd.Series) -> np.ndarray:
        """
        Compute continuous [0, 1] anomaly scores over a full time-series.
        Scores are computed strictly using trailing rolling statistics.
        """
        vals = values.astype(float)
        rolling = vals.rolling(window=self.window_size, min_periods=2)
        r_mean = rolling.mean().bfill()
        r_std = rolling.std().fillna(1e-5).replace(0, 1e-5)

        z = np.abs((vals - r_mean) / r_std)
        # Sigmoid-like smooth normalization: 0 at z=0, ~0.5 at base_threshold, ~1.0 at high z
        scores = 2.0 / (1.0 + np.exp(-z / (self.base_threshold / 2.0))) - 1.0
        return np.clip(scores, 0.0, 1.0).values

    def score_single(
        self,
        current_value: float,
        trailing_values: np.ndarray,
        adaptive_threshold: Optional[float] = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Score a single real-time telemetry observation against a trailing window.
        
        Returns:
            normalized_score: float in [0, 1]
            z_value: raw absolute z-score
            details: diagnostic dictionary
        """
        if len(trailing_values) < 2:
            return 0.0, 0.0, {"z": 0.0, "mean": current_value, "std": 1e-5}

        buf = trailing_values[-self.window_size:]
        mean = float(np.mean(buf))
        std = max(float(np.std(buf)), 1e-5)
        raw_z = (current_value - mean) / std
        abs_z = abs(raw_z)

        thresh = adaptive_threshold if adaptive_threshold is not None else self.base_threshold
        score = 2.0 / (1.0 + np.exp(-abs_z / (thresh / 2.0))) - 1.0
        norm_score = float(np.clip(score, 0.0, 1.0))

        details = {
            "raw_z": raw_z,
            "abs_z": abs_z,
            "rolling_mean": mean,
            "rolling_std": std,
            "threshold": thresh
        }
        return norm_score, abs_z, details
