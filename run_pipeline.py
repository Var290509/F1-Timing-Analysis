"""
F1 Sentinel End-to-End Pipeline Execution Script
Runs data ingestion, windowed feature extraction, multi-model training,
adaptive thresholding, evidence generation, Event-Based F1 evaluation,
and exports predictions.csv, metrics_summary.json, and all plots.
"""

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_nab_dataset, get_available_channels
from src.feature_engine import extract_window_features, StreamingFeatureBuffer
from src.models.ensemble import HybridEnsembleDetector
from src.adaptive_threshold import AdaptiveThresholdManager
from src.evidence_generator import EvidenceGenerator
from src.evaluator import evaluate_all_models, compute_event_based_metrics, profile_inference_latency
from scripts.generate_visualizations import (
    plot_time_series_anomalies,
    plot_anomaly_score_and_threshold,
    plot_model_comparison_metrics,
    plot_confusion_breakdown,
    plot_zoom_in_cases
)


def run_pipeline(
    channel_key: str = "engine_temperature_sensor",
    window_size: int = 30,
    train_points: int = 2500,
    output_dir: str = "outputs"
):
    print("=" * 80)
    print("🏎️  F1 SENTINEL — ADAPTIVE TIME-SERIES ANOMALY DETECTION PIPELINE")
    print("    Utkrishta 2026 AI/ML Hackathon | Challenge 2: Sentinel")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Ingestion & Preprocessing
    print(f"\n[STEP 1/6] Loading F1 Telemetry Stream: '{channel_key}'...")
    df, meta, gt_windows = load_nab_dataset(channel_key)
    print(f"  Channel: {meta['channel_name']} ({meta['telemetry_code']})")
    print(f"  Total Data Points: {len(df):,}")
    print(f"  Ground-Truth NAB Anomaly Windows: {len(gt_windows)}")
    for i, (ws, we) in enumerate(gt_windows, 1):
        print(f"    Window {i}: {ws} -> {we}")

    # 2. Windowed Feature Engineering
    print(f"\n[STEP 2/6] Computing Streaming Causal Windowed Features (W = {window_size})...")
    feat_start = time.perf_counter()
    feats_df = extract_window_features(df["value"], window_size=window_size)
    feat_time = (time.perf_counter() - feat_start) * 1000.0
    print(f"  Engineered {feats_df.shape[1]} features across {len(feats_df):,} points in {feat_time:.1f} ms")
    print(f"  Features: {list(feats_df.columns)}")

    X_all = feats_df.values
    series_all = df["value"]

    # 3. Model Training & Ensemble Setup
    print(f"\n[STEP 3/6] Fitting Multi-Model Anomaly Detectors on Normal Baseline ({train_points} points)...")
    X_train = X_all[:train_points]
    series_train = series_all.iloc[:train_points]

    ensemble = HybridEnsembleDetector()
    fit_start = time.perf_counter()
    ensemble.fit(X_train, series_train)
    fit_time = (time.perf_counter() - fit_start) * 1000.0
    print(f"  Trained all 5 models (Z-Score, IsoForest, LOF, OCSVM, Autoencoder) in {fit_time:.1f} ms")

    # 4. Continuous Anomaly Scoring
    print("\n[STEP 4/6] Generating Continuous Anomaly Scores Across Telemetry Stream...")
    score_start = time.perf_counter()
    model_scores = ensemble.score_all_models(X_all, series_all)
    score_time = (time.perf_counter() - score_start) * 1000.0
    print(f"  Inference completed in {score_time:.1f} ms ({score_time / len(X_all):.3f} ms / point)")

    ensemble_scores = model_scores["Hybrid_Ensemble"]

    # 5. Adaptive Thresholding & Regime Shift Adaptation
    print("\n[STEP 5/6] Applying Dynamic Adaptive Thresholding & F1 Tactical Regime Detection...")
    thresh_manager = AdaptiveThresholdManager(
        trailing_window=150,
        base_k=3.0,
        min_threshold=0.68,
        max_threshold=0.88,
        regime_detection=True
    )
    dynamic_thresholds, binary_preds, regime_metas = thresh_manager.compute_series(
        ensemble_scores,
        series_all.values
    )

    # Also compute threshold per base model for comparison
    thresholds_dict = {"Hybrid_Ensemble": dynamic_thresholds}
    for m_name, s_arr in model_scores.items():
        if m_name == "Hybrid_Ensemble":
            continue
        m_mgr = AdaptiveThresholdManager(trailing_window=150, base_k=3.0, min_threshold=0.68, max_threshold=0.88)
        m_thresh, _, _ = m_mgr.compute_series(s_arr, series_all.values)
        thresholds_dict[m_name] = m_thresh

    # Generate Explainable Evidence Strings
    print("  Synthesizing Race Engineer Diagnostic Evidence Strings...")
    evidence_gen = EvidenceGenerator(channel_name=meta["channel_name"], unit=meta["unit"])
    evidence_list = []
    labels_list = []

    # Get PyTorch Autoencoder reconstruction errors for evidence details
    ae_mse_all = ensemble.ae_model.get_raw_reconstruction_error(X_all)

    for i in range(len(df)):
        score_val = float(ensemble_scores[i])
        thresh_val = float(dynamic_thresholds[i])
        is_flagged = bool(binary_preds[i])
        lbl = "ANOMALOUS" if is_flagged else "NORMAL"
        labels_list.append(lbl)

        cur_model_scores = {k: float(v[i]) for k, v in model_scores.items()}
        cur_feats = {
            "window_zscore": float(feats_df["window_zscore"].iloc[i]),
            "rate_of_change": float(feats_df["rate_of_change"].iloc[i]),
            "ae_reconstruction_error": float(ae_mse_all[i])
        }

        ev = evidence_gen.generate(
            anomaly_score=score_val,
            threshold=thresh_val,
            label=lbl,
            current_value=float(df["value"].iloc[i]),
            model_scores=cur_model_scores,
            feature_dict=cur_feats,
            regime_meta=regime_metas[i]
        )
        evidence_list.append(ev)

    # Save outputs/predictions.csv (required columns: timestamp, anomaly_score, label, evidence)
    predictions_df = pd.DataFrame({
        "timestamp": df["timestamp"].astype(str),
        "anomaly_score": np.round(ensemble_scores, 4),
        "label": labels_list,
        "evidence": evidence_list
    })
    pred_path = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    print(f"  [SAVED] Predictions exported to: {pred_path} ({len(predictions_df):,} rows)")

    # 6. Comprehensive Evaluation & Latency Benchmarking
    print("\n[STEP 6/6] Computing Event-Based F1 & Latency Metrics...")
    eval_df = evaluate_all_models(
        timestamps=df["timestamp"],
        raw_values=df["value"],
        model_scores_dict=model_scores,
        thresholds_dict=thresholds_dict,
        ground_truth_windows=gt_windows,
        models_dict=ensemble.models_dict,
        X_features=X_all
    )

    # Print summary table to console
    print("\n" + "=" * 90)
    print("🏆  F1 SENTINEL — MODEL BENCHMARK RESULTS (NAB EVENT-BASED EVALUATION)")
    print("=" * 90)
    print(eval_df.to_string(index=False))
    print("=" * 90)

    # Export metrics_summary.json
    metrics_json_path = os.path.join(output_dir, "metrics_summary.json")
    metrics_dict = {
        "channel_metadata": meta,
        "total_timestamps": len(df),
        "ground_truth_windows_count": len(gt_windows),
        "models_performance": eval_df.to_dict(orient="records")
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  [SAVED] Metrics summary saved to: {metrics_json_path}")

    # Generate and save all 5 competition plots
    print("\n[PLOTS] Rendering and saving competition visualization artifacts...")
    plot_time_series_anomalies(
        df=df,
        ground_truth_windows=gt_windows,
        predictions=binary_preds,
        meta=meta,
        output_path=os.path.join(plots_dir, "time_series_anomalies.png")
    )

    plot_anomaly_score_and_threshold(
        df=df,
        scores=ensemble_scores,
        thresholds=dynamic_thresholds,
        ground_truth_windows=gt_windows,
        output_path=os.path.join(plots_dir, "anomaly_score_and_threshold.png")
    )

    plot_model_comparison_metrics(
        metrics_df=eval_df,
        output_path=os.path.join(plots_dir, "model_comparison_metrics.png")
    )

    plot_confusion_breakdown(
        metrics_df=eval_df,
        output_path=os.path.join(plots_dir, "confusion_breakdown.png")
    )

    plot_zoom_in_cases(
        df=df,
        scores=ensemble_scores,
        thresholds=dynamic_thresholds,
        predictions=binary_preds,
        evidence_list=evidence_list,
        ground_truth_windows=gt_windows,
        meta=meta,
        output_path=os.path.join(plots_dir, "zoom_in_cases.png")
    )

    print("\n🎯 All pipeline stages successfully executed!")
    return predictions_df, eval_df


if __name__ == "__main__":
    run_pipeline()
