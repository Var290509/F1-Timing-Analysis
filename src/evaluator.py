"""
F1 Sentinel Evaluation Engine
Implements authentic Event-Based F1 score (NAB standard), detection delay metrics,
pointwise precision/recall, and inference latency profiling for F1 telemetry anomaly detection.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def compute_event_based_metrics(
    timestamps: pd.Series,
    binary_predictions: np.ndarray,
    ground_truth_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
) -> Dict[str, Any]:
    """
    Compute Event-Based Precision, Recall, F1, and Detection Delay according to NAB benchmark standards.
    
    Rules:
    - True Positive (TP): At least one flag raised within ground-truth window [start, end].
      Counts at most 1 TP per ground-truth window.
    - False Negative (FN): Zero flags raised within ground-truth window.
    - False Positive (FP): Contiguous detection segments outside any ground-truth window.
    - Detection Delay: Timestamps (or time duration) between window start and first detection flag.
    """
    ts_array = pd.to_datetime(timestamps).values
    preds = np.asarray(binary_predictions, dtype=bool)
    n = len(ts_array)

    if len(ground_truth_windows) == 0:
        # No ground-truth anomalies in series
        fp_count = count_contiguous_events(preds)
        return {
            "TP": 0,
            "FP": fp_count,
            "FN": 0,
            "precision": 0.0 if fp_count > 0 else 1.0,
            "recall": 1.0,
            "event_f1": 0.0,
            "detection_delay_points": 0.0,
            "detection_delay_minutes": 0.0,
            "total_windows": 0,
            "detected_windows": 0
        }

    # Mark which timestamps fall inside any ground-truth window
    inside_gt_mask = np.zeros(n, dtype=bool)
    gt_window_masks = []

    for w_start, w_end in ground_truth_windows:
        t_start = pd.to_datetime(w_start).to_numpy()
        t_end = pd.to_datetime(w_end).to_numpy()
        w_mask = (ts_array >= t_start) & (ts_array <= t_end)
        gt_window_masks.append(w_mask)
        inside_gt_mask |= w_mask

    # 1. Evaluate True Positives & False Negatives per window
    tp = 0
    fn = 0
    delays_points = []
    delays_minutes = []

    for w_idx, (w_start, w_end) in enumerate(ground_truth_windows):
        w_mask = gt_window_masks[w_idx]
        window_indices = np.where(w_mask)[0]

        if len(window_indices) == 0:
            fn += 1
            continue

        flagged_in_window = np.where(w_mask & preds)[0]
        if len(flagged_in_window) > 0:
            tp += 1
            first_flag_idx = flagged_in_window[0]
            start_window_idx = window_indices[0]
            delay_pts = max(0, first_flag_idx - start_window_idx)
            delays_points.append(delay_pts)

            first_flag_time = pd.to_datetime(ts_array[first_flag_idx])
            start_time = pd.to_datetime(w_start)
            delay_min = max(0.0, (first_flag_time - start_time).total_seconds() / 60.0)
            delays_minutes.append(delay_min)
        else:
            fn += 1

    # 2. Evaluate False Positives: contiguous prediction runs OUTSIDE ground truth
    outside_flags = preds & (~inside_gt_mask)
    fp = count_contiguous_events(outside_flags)

    # 3. Compute Precision, Recall, and Event-F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        event_f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        event_f1 = 0.0

    avg_delay_pts = float(np.mean(delays_points)) if delays_points else 0.0
    avg_delay_min = float(np.mean(delays_minutes)) if delays_minutes else 0.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "event_f1": float(round(event_f1, 4)),
        "detection_delay_points": float(round(avg_delay_pts, 2)),
        "detection_delay_minutes": float(round(avg_delay_min, 2)),
        "total_windows": len(ground_truth_windows),
        "detected_windows": tp
    }


def count_contiguous_events(binary_mask: np.ndarray) -> int:
    """
    Count number of distinct contiguous runs of True in a boolean array.
    Used to group successive flagged points into single alert events.
    """
    if len(binary_mask) == 0 or not np.any(binary_mask):
        return 0
    diff = np.diff(binary_mask.astype(int))
    # A new event starts when binary_mask[0] is True or diff == 1
    events = int(np.sum(diff == 1))
    if binary_mask[0]:
        events += 1
    return events


def profile_inference_latency(
    model,
    X_sample: np.ndarray,
    n_iterations: int = 500
) -> float:
    """
    Profile per-window inference latency in milliseconds (ms).
    """
    sample_vec = X_sample[0:1]
    # Warmup
    for _ in range(20):
        if hasattr(model, "score"):
            _ = model.score(sample_vec)
        elif hasattr(model, "score_series"):
            _ = model.score_series(pd.Series(sample_vec.flatten()))

    start_t = time.perf_counter()
    for _ in range(n_iterations):
        if hasattr(model, "score"):
            _ = model.score(sample_vec)
        elif hasattr(model, "score_series"):
            _ = model.score_series(pd.Series(sample_vec.flatten()))
    elapsed_s = time.perf_counter() - start_t
    latency_ms = (elapsed_s / n_iterations) * 1000.0
    return float(round(latency_ms, 3))


def evaluate_all_models(
    timestamps: pd.Series,
    raw_values: pd.Series,
    model_scores_dict: Dict[str, np.ndarray],
    thresholds_dict: Dict[str, np.ndarray],
    ground_truth_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    models_dict: Optional[Dict[str, Any]] = None,
    X_features: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Evaluate all individual models and hybrid ensemble against ground-truth windows.
    Returns comparison DataFrame with Event-F1, Precision, Recall, Delays, and Latency.
    """
    results = []
    for model_name, scores in model_scores_dict.items():
        thresh = thresholds_dict.get(model_name)
        if thresh is None:
            thresh = 0.50
        
        if isinstance(thresh, np.ndarray):
            binary_preds = scores >= thresh
        else:
            binary_preds = scores >= float(thresh)

        metrics = compute_event_based_metrics(timestamps, binary_preds, ground_truth_windows)
        
        # Profile latency if model object and features available
        latency_ms = 0.0
        if models_dict and model_name in models_dict and X_features is not None:
            latency_ms = profile_inference_latency(models_dict[model_name], X_features)

        results.append({
            "Model": model_name,
            "Event-F1": metrics["event_f1"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Detection Delay (pts)": metrics["detection_delay_points"],
            "Detection Delay (min)": metrics["detection_delay_minutes"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Inference Latency (ms)": latency_ms
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Event-F1", ascending=False).reset_index(drop=True)
    return results_df
