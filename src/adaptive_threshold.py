"""
F1 Sentinel Adaptive Thresholding & Racing Regime Detection
Dynamically recalibrates anomaly decision thresholds over trailing score windows.
Includes an F1 Tactical Regime Shift detector (weather shifts, safety car pace deltas,
tyre compound swaps) to prevent false-alarm cascades.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class AdaptiveThresholdManager:
    """
    Dynamic Anomaly Thresholding with Tactical F1 Regime Shift Adaptation.
    
    Recalibrates threshold at time t using trailing score distribution:
        Threshold_t = min(max_threshold, max(min_threshold, mean(scores_window) + k * std(scores_window)))
        
    Regime shift layer:
        Detects significant shifts in baseline telemetry distribution (e.g. entering Safety Car
        or changing track conditions) and temporarily widens tolerance bounds.
    """

    def __init__(
        self,
        trailing_window: int = 150,
        base_k: float = 3.0,
        min_threshold: float = 0.68,
        max_threshold: float = 0.88,
        regime_detection: bool = True,
        regime_short_w: int = 20,
        regime_long_w: int = 100,
        regime_sensitivity: float = 2.5
    ):
        self.trailing_window = trailing_window
        self.base_k = base_k
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.regime_detection = regime_detection
        self.regime_short_w = regime_short_w
        self.regime_long_w = regime_long_w
        self.regime_sensitivity = regime_sensitivity

        # Online state
        self.scores_buffer: List[float] = []
        self.raw_values_buffer: List[float] = []
        self.regime_cooldown: int = 0
        self.current_regime: str = "NORMAL_RACE"

    def reset(self):
        self.scores_buffer.clear()
        self.raw_values_buffer.clear()
        self.regime_cooldown = 0
        self.current_regime = "NORMAL_RACE"

    def detect_regime_shift(self, new_val: float) -> Tuple[bool, str]:
        """
        Detect distribution regime change using short-window vs long-window discrepancy.
        Simulates F1 scenarios: Safety Car pace drop, weather wet/dry shift, tyre compound change.
        """
        self.raw_values_buffer.append(new_val)
        if len(self.raw_values_buffer) > self.regime_long_w * 2:
            self.raw_values_buffer.pop(0)

        n = len(self.raw_values_buffer)
        if n < self.regime_long_w:
            return False, "WARMUP"

        buf = np.array(self.raw_values_buffer)
        short_buf = buf[-self.regime_short_w:]
        long_buf = buf[-self.regime_long_w:]

        long_mean = float(np.mean(long_buf))
        long_std = max(float(np.std(long_buf)), 1e-5)
        short_mean = float(np.mean(short_buf))

        z_diff = abs(short_mean - long_mean) / long_std

        if z_diff > self.regime_sensitivity:
            if short_mean < long_mean:
                regime_tag = "REGIME_SHIFT:PACE_REDUCTION_OR_WET"
            else:
                regime_tag = "REGIME_SHIFT:HIGH_STRESS_OR_OVERHEAT"
            return True, regime_tag

        return False, "STABLE_RACE_PACE"

    def step(
        self,
        current_score: float,
        current_raw_value: Optional[float] = None
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Process incoming continuous anomaly score and determine dynamic threshold & label.
        
        Returns:
            threshold: current calibrated threshold
            is_anomaly: boolean decision
            metadata: diagnostic information
        """
        self.scores_buffer.append(float(current_score))
        if len(self.scores_buffer) > self.trailing_window:
            self.scores_buffer.pop(0)

        regime_active = False
        regime_desc = "NORMAL_RACE"

        if self.regime_detection and current_raw_value is not None:
            shift_detected, regime_desc = self.detect_regime_shift(current_raw_value)
            if shift_detected:
                self.regime_cooldown = 15  # 15 steps cooldown
                self.current_regime = regime_desc
            elif self.regime_cooldown > 0:
                self.regime_cooldown -= 1
                regime_active = True
            else:
                self.current_regime = "NORMAL_RACE"

        # Calculate base dynamic threshold from trailing scores
        if len(self.scores_buffer) < 20:
            threshold = self.max_threshold
            trailing_mean = float(np.mean(self.scores_buffer)) if self.scores_buffer else 0.35
            trailing_std = 0.05
            is_anomaly = False
        else:
            trailing_mean = float(np.mean(self.scores_buffer[:-1]))  # exclude current point
            trailing_std = float(np.std(self.scores_buffer[:-1]))
            # Dynamic k factor: if regime shift active, temporarily widen k to prevent false alarms
            effective_k = self.base_k * 1.35 if (self.regime_cooldown > 0 or regime_active) else self.base_k
            threshold = trailing_mean + effective_k * max(trailing_std, 0.04)
            threshold = float(np.clip(threshold, self.min_threshold, self.max_threshold))
            is_anomaly = bool(current_score >= threshold)

        # Outlier-resistant: if point was anomalous, remove from buffer so it does not inflate baseline
        if is_anomaly and len(self.scores_buffer) > 1:
            self.scores_buffer.pop()

        meta = {
            "threshold": threshold,
            "trailing_mean": trailing_mean,
            "trailing_std": trailing_std,
            "regime": self.current_regime,
            "regime_shift_active": bool(self.regime_cooldown > 0 or regime_active)
        }
        return threshold, is_anomaly, meta

    def compute_series(
        self,
        scores: np.ndarray,
        raw_values: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Batch compute dynamic thresholds and labels for an entire time-series.
        Strictly sequential (no lookahead).
        """
        self.reset()
        n = len(scores)
        thresholds = np.zeros(n, dtype=float)
        labels = np.zeros(n, dtype=bool)
        metas = []

        for i in range(n):
            score = scores[i]
            val = raw_values[i] if raw_values is not None else None
            t, flag, m = self.step(score, val)
            thresholds[i] = t
            labels[i] = flag
            metas.append(m)

        return thresholds, labels, metas
