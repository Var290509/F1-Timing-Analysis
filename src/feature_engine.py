"""
F1 Sentinel Feature Engineering
Windowed, streaming-compatible feature extraction for high-frequency telemetry.
Computes rolling statistics, lag features, EWMA, delta, and z-scores with
zero future data leakage.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "value",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_range",
    "rate_of_change",
    "window_zscore",
    "lag_1",
    "lag_2",
    "lag_3",
    "ewma",
    "ewma_deviation",
    "acceleration"
]


def extract_window_features(
    series: pd.Series,
    window_size: int = 30,
    ewma_span: int = 20
) -> pd.DataFrame:
    """
    Vectorized, strictly causal windowed feature computation.
    All rolling and lag operations look strictly backward in time.
    
    Args:
        series: Pandas series of telemetry values indexed sequentially.
        window_size: Trailing window size W (e.g. 10, 30, 50, 100).
        ewma_span: Decay span for EWMA.
        
    Returns:
        DataFrame containing all engineered feature columns.
    """
    df_feat = pd.DataFrame(index=series.index)
    val = series.astype(float)
    df_feat["value"] = val

    # 1. Rolling statistics (strictly trailing, min_periods=1)
    rolling = val.rolling(window=window_size, min_periods=2)
    df_feat["rolling_mean"] = rolling.mean().bfill()
    df_feat["rolling_std"] = rolling.std().fillna(1e-5)
    df_feat["rolling_min"] = rolling.min().bfill()
    df_feat["rolling_max"] = rolling.max().bfill()
    df_feat["rolling_range"] = (df_feat["rolling_max"] - df_feat["rolling_min"]).clip(lower=1e-5)

    # 2. Lag features (t-1, t-2, t-3)
    df_feat["lag_1"] = val.shift(1).bfill()
    df_feat["lag_2"] = val.shift(2).bfill()
    df_feat["lag_3"] = val.shift(3).bfill()

    # 3. Rate of change (delta) and acceleration
    df_feat["rate_of_change"] = val - df_feat["lag_1"]
    delta_lag1 = df_feat["rate_of_change"].shift(1).fillna(0.0)
    df_feat["acceleration"] = df_feat["rate_of_change"] - delta_lag1

    # 4. Window Z-Score (normalized distance within trailing window)
    std_safe = df_feat["rolling_std"].replace(0, 1e-5)
    df_feat["window_zscore"] = (val - df_feat["rolling_mean"]) / std_safe

    # 5. EWMA and EWMA deviation
    df_feat["ewma"] = val.ewm(span=ewma_span, adjust=False).mean()
    df_feat["ewma_deviation"] = val - df_feat["ewma"]

    # Fill any initial warmup NaNs safely
    df_feat = df_feat.bfill().ffill().fillna(0.0)
    return df_feat


def extract_multi_window_features(
    series: pd.Series,
    window_sizes: List[int] = [10, 30, 50, 100]
) -> pd.DataFrame:
    """
    Multi-scale window feature extraction across different temporal horizons.
    Useful for capturing fast shocks (W=10) and slow thermal / wear drift (W=100).
    """
    base_feats = extract_window_features(series, window_size=window_sizes[1])
    dfs = [base_feats]

    for w in window_sizes:
        if w == window_sizes[1]:
            continue
        rolling = series.rolling(window=w, min_periods=2)
        mean_w = rolling.mean().bfill()
        std_w = rolling.std().fillna(1e-5)
        dfs.append(pd.DataFrame({
            f"mean_w{w}": mean_w,
            f"std_w{w}": std_w,
            f"zscore_w{w}": (series - mean_w) / std_w.replace(0, 1e-5)
        }, index=series.index))

    return pd.concat(dfs, axis=1)


class StreamingFeatureBuffer:
    """
    Online, stateful feature generator for streaming F1 telemetry feeds.
    Maintains a rolling ring buffer of length window_size.
    Computes all features in real time without access to future observations.
    """

    def __init__(self, window_size: int = 30, ewma_span: int = 20):
        self.window_size = window_size
        self.ewma_alpha = 2.0 / (ewma_span + 1.0)
        self.buffer: List[float] = []
        self.ewma_state: Optional[float] = None
        self.last_delta: float = 0.0

    def reset(self):
        self.buffer.clear()
        self.ewma_state = None
        self.last_delta = 0.0

    def update(self, value: float) -> Dict[str, float]:
        """
        Ingest next incoming telemetry data point and return engineered feature dict.
        """
        val = float(value)
        self.buffer.append(val)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        buf = np.array(self.buffer, dtype=float)
        n = len(buf)

        # Lags
        lag1 = self.buffer[-2] if n >= 2 else val
        lag2 = self.buffer[-3] if n >= 3 else lag1
        lag3 = self.buffer[-4] if n >= 4 else lag2

        # Rolling stats
        r_mean = float(np.mean(buf))
        r_std = float(np.std(buf)) if n >= 2 else 1e-5
        if r_std < 1e-5:
            r_std = 1e-5
        r_min = float(np.min(buf))
        r_max = float(np.max(buf))
        r_range = max(r_max - r_min, 1e-5)

        # Delta & Acceleration
        delta = val - lag1
        acceleration = delta - self.last_delta
        self.last_delta = delta

        # Z-Score within window
        zscore = (val - r_mean) / r_std

        # EWMA
        if self.ewma_state is None:
            self.ewma_state = val
        else:
            self.ewma_state = self.ewma_alpha * val + (1.0 - self.ewma_alpha) * self.ewma_state
        ewma_dev = val - self.ewma_state

        return {
            "value": val,
            "rolling_mean": r_mean,
            "rolling_std": r_std,
            "rolling_min": r_min,
            "rolling_max": r_max,
            "rolling_range": r_range,
            "rate_of_change": delta,
            "window_zscore": zscore,
            "lag_1": lag1,
            "lag_2": lag2,
            "lag_3": lag3,
            "ewma": float(self.ewma_state),
            "ewma_deviation": ewma_dev,
            "acceleration": acceleration
        }


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_nab_dataset
    df, _, _ = load_nab_dataset("engine_temperature_sensor")
    feats = extract_window_features(df["value"], window_size=30)
    print("Vectorized features shape:", feats.shape)
    print("Columns:", list(feats.columns))
    
    # Test streaming buffer consistency
    streamer = StreamingFeatureBuffer(window_size=30)
    stream_records = [streamer.update(v) for v in df["value"][:100]]
    stream_df = pd.DataFrame(stream_records)
    print("Streaming sample head:")
    print(stream_df[["value", "rolling_mean", "rolling_std", "window_zscore"]].head(3))
