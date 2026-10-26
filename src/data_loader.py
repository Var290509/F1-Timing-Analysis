"""
F1 Sentinel Data Pipeline
Loads Numenta Anomaly Benchmark (NAB) time-series streams, maps them to
Formula 1 telemetry sensor channels, handles cleaning/resampling, and
provides a streaming simulation generator with zero future data leakage.
"""

import os
import json
from typing import Dict, List, Tuple, Generator, Optional, Any
import pandas as pd
import numpy as np

# Built-in Formula 1 Domain Telemetry Channel Profiles
F1_TELEMETRY_PROFILES = {
    "engine_temperature_sensor": {
        "dataset_file": "machine_temperature_system_failure.csv",
        "nab_key": "realKnownCause/machine_temperature_system_failure.csv",
        "telemetry_code": "TEL-ENG-01",
        "channel_name": "ICE Cylinder Head & Coolant Temp",
        "unit": "°C",
        "description": "Monitors internal combustion engine thermal load. Rapid spikes indicate cooling radiator blockage or thermal runaway.",
        "normal_operating_range": (70.0, 95.0),
        "sampling_rate": "5m"
    },
    "ers_mgu_k_load": {
        "dataset_file": "ec2_cpu_utilization_5f5533.csv",
        "nab_key": "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
        "telemetry_code": "TEL-ERS-02",
        "channel_name": "MGU-K Electrical Deployment Duty Cycle",
        "unit": "%",
        "description": "Monitors hybrid power unit harvesting and boost cycle. Dropouts signal inverter desync or battery cell degradation.",
        "normal_operating_range": (20.0, 60.0),
        "sampling_rate": "5m"
    },
    "lap_time_stream_delta": {
        "dataset_file": "art_daily_flatmiddle.csv",
        "nab_key": "artificialWithAnomaly/art_daily_flatmiddle.csv",
        "telemetry_code": "TEL-TIM-03",
        "channel_name": "Sector Micro-Sector Pace Delta",
        "unit": "s",
        "description": "Lap timing consistency stream against target delta. Shifts signify safety car regimes or severe tyre degradation cliff.",
        "normal_operating_range": (-0.5, 0.5),
        "sampling_rate": "5m"
    },
    "tyre_carcass_pressure_fl": {
        "dataset_file": "ambient_temperature_system_failure.csv",
        "nab_key": "realKnownCause/ambient_temperature_system_failure.csv",
        "telemetry_code": "TEL-TYR-04",
        "channel_name": "Front-Left Tyre Carcass Bulk Temperature",
        "unit": "°C",
        "description": "Monitors tyre thermal degradation curve. Outliers indicate blistering, asymmetric wear, or imminent puncture failure.",
        "normal_operating_range": (85.0, 110.0),
        "sampling_rate": "1h"
    }
}


def get_available_channels() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all available F1 telemetry channel profiles."""
    return F1_TELEMETRY_PROFILES


def load_nab_dataset(
    channel_key: str = "engine_temperature_sensor",
    data_dir: str = "data/nab",
    resample_freq: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Load a NAB CSV file mapped to an F1 telemetry channel.
    
    Returns:
        df: Cleaned DataFrame with ['timestamp', 'value']
        meta: Channel metadata dictionary
        windows: List of (start_timestamp, end_timestamp) ground-truth anomaly windows
    """
    if channel_key in F1_TELEMETRY_PROFILES:
        meta = F1_TELEMETRY_PROFILES[channel_key]
        filename = meta["dataset_file"]
        nab_key = meta["nab_key"]
    else:
        # Fallback if raw filename was passed
        filename = channel_key if channel_key.endswith(".csv") else f"{channel_key}.csv"
        nab_key = filename
        meta = {
            "telemetry_code": "TEL-GEN-00",
            "channel_name": channel_key,
            "unit": "units",
            "description": f"Custom F1 telemetry stream for {channel_key}",
            "normal_operating_range": (0.0, 100.0)
        }

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}. Run scripts/download_nab.py first.")

    # Load and clean CSV
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Handle missing / null values cleanly
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().any():
        df["value"] = df["value"].interpolate(method="linear").bfill().ffill()

    # Optional resampling
    if resample_freq:
        df = df.set_index("timestamp").resample(resample_freq).mean().interpolate().reset_index()

    # Load ground-truth anomaly windows
    windows = load_ground_truth_windows(nab_key, data_dir=data_dir)

    # Add ground-truth binary column for reference
    df["is_ground_truth"] = 0
    for start_t, end_t in windows:
        mask = (df["timestamp"] >= start_t) & (df["timestamp"] <= end_t)
        df.loc[mask, "is_ground_truth"] = 1

    return df, meta, windows


def load_ground_truth_windows(
    nab_key: str,
    data_dir: str = "data/nab"
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Load ground-truth anomaly window intervals from combined_windows.json.
    """
    json_path = os.path.join(data_dir, "combined_windows.json")
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r") as f:
            all_windows = json.load(f)
    except Exception:
        return []

    # Match nab_key or try suffix match
    windows_raw = all_windows.get(nab_key, [])
    if not windows_raw:
        for k, v in all_windows.items():
            if k.endswith(os.path.basename(nab_key)):
                windows_raw = v
                break

    parsed_windows = []
    for interval in windows_raw:
        if len(interval) == 2:
            t0 = pd.to_datetime(interval[0])
            t1 = pd.to_datetime(interval[1])
            parsed_windows.append((t0, t1))

    return parsed_windows


class TelemetryStreamSimulator:
    """
    Simulates real-time F1 telemetry streaming.
    Yields data row-by-row or in sliding windows with strict causal indexing (no future data leakage).
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 30):
        self.df = df.copy()
        self.window_size = window_size
        self.total_points = len(df)
        self.current_idx = 0

    def __len__(self) -> int:
        return self.total_points

    def stream_points(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields point-by-point telemetry dictionaries sequentially.
        """
        for idx, row in self.df.iterrows():
            yield {
                "index": int(idx),
                "timestamp": row["timestamp"],
                "value": float(row["value"]),
                "is_ground_truth": int(row.get("is_ground_truth", 0))
            }

    def stream_windows(self) -> Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Yields (historical_window_df, current_point_dict) for each step after initial warm-up.
        Ensures strict causality: window contains only timestamps <= current point.
        """
        for i in range(len(self.df)):
            current_row = self.df.iloc[i]
            point_info = {
                "index": i,
                "timestamp": current_row["timestamp"],
                "value": float(current_row["value"]),
                "is_ground_truth": int(current_row.get("is_ground_truth", 0))
            }
            # Start of window (clipped to 0)
            start_i = max(0, i - self.window_size + 1)
            window_df = self.df.iloc[start_i: i + 1].copy()
            yield window_df, point_info


if __name__ == "__main__":
    df, meta, windows = load_nab_dataset("engine_temperature_sensor")
    print(f"Loaded {meta['channel_name']} ({meta['telemetry_code']}): {len(df)} points")
    print(f"Ground-truth anomaly windows ({len(windows)}): {windows}")
    simulator = TelemetryStreamSimulator(df, window_size=30)
    for win, pt in simulator.stream_windows():
        if pt["index"] == 35:
            print(f"Sample window at idx 35: shape {win.shape}, current value = {pt['value']}")
            break
