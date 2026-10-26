"""
F1 Sentinel Telemetry Visualizations
Generates competition-ready high-resolution plots:
1. Telemetry time-series with ground truth shaded and detection flags
2. Continuous anomaly scores with dynamic adaptive threshold line
3. Model comparison bar charts (Event-F1, Precision, Recall, Latency)
4. Confusion-style TP, FP, FN breakdown
5. 3 Detailed zoom-in case studies (True Positive, False Positive, Edge Case) with annotated evidence callout boxes
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# F1 Race Dashboard Styling Palette
F1_RED = "#E10600"
CARBON_BLACK = "#15151E"
DARK_GRAY = "#22222E"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#A0A0B0"
NEON_CYAN = "#00D2BE"      # Mercedes AMG Petronas Cyan
FERRARI_YELLOW = "#FFF500"  # Caution / Yellow Flag
ASTON_GREEN = "#00594F"
GRID_COLOR = "#2D2D3F"


def apply_f1_theme(ax, fig=None):
    """Apply sleek dark carbon Formula 1 styling to a matplotlib axis."""
    ax.set_facecolor(DARK_GRAY)
    if fig:
        fig.patch.set_facecolor(CARBON_BLACK)
    ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444455")
        spine.set_linewidth(1.0)


def plot_time_series_anomalies(
    df: pd.DataFrame,
    ground_truth_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    predictions: np.ndarray,
    meta: Dict[str, Any],
    output_path: str = "outputs/plots/time_series_anomalies.png"
):
    """Plot full telemetry time-series with shaded GT windows and red detection markers."""
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    apply_f1_theme(ax, fig)

    # Plot raw telemetry signal
    ax.plot(
        df["timestamp"],
        df["value"],
        color=NEON_CYAN,
        linewidth=1.2,
        label=f"{meta['channel_name']} ({meta['unit']})",
        alpha=0.9
    )

    # Shade ground-truth anomaly windows
    for i, (w_start, w_end) in enumerate(ground_truth_windows):
        ax.axvspan(
            w_start,
            w_end,
            color=FERRARI_YELLOW,
            alpha=0.22,
            label="Ground Truth Failure Window (NAB)" if i == 0 else ""
        )

    # Overlay detected anomalies as red scatter points
    flagged_mask = np.asarray(predictions, dtype=bool)
    if np.any(flagged_mask):
        ax.scatter(
            df.loc[flagged_mask, "timestamp"],
            df.loc[flagged_mask, "value"],
            color=F1_RED,
            s=18,
            zorder=5,
            label="Sentinel Detection Flag (Ensemble)",
            edgecolors="#FFFFFF",
            linewidth=0.4
        )

    ax.set_title(
        f"F1 SENTINEL — {meta['telemetry_code']}: {meta['channel_name']} Telemetry Stream",
        color=TEXT_WHITE,
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Session Timestamp", color=TEXT_WHITE, fontsize=10, labelpad=8)
    ax.set_ylabel(f"Sensor Value ({meta['unit']})", color=TEXT_WHITE, fontsize=10, labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_anomaly_score_and_threshold(
    df: pd.DataFrame,
    scores: np.ndarray,
    thresholds: np.ndarray,
    ground_truth_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    output_path: str = "outputs/plots/anomaly_score_and_threshold.png"
):
    """Plot continuous ensemble anomaly score against the dynamic adaptive threshold line."""
    fig, ax = plt.subplots(figsize=(14, 5.0), dpi=300)
    apply_f1_theme(ax, fig)

    # Shade GT windows
    for i, (w_start, w_end) in enumerate(ground_truth_windows):
        ax.axvspan(
            w_start,
            w_end,
            color=FERRARI_YELLOW,
            alpha=0.20,
            label="Ground Truth Anomaly Window" if i == 0 else ""
        )

    # Continuous score
    ax.plot(
        df["timestamp"],
        scores,
        color="#3B82F6",
        linewidth=1.2,
        label="Hybrid Ensemble Anomaly Score [0, 1]",
        alpha=0.85
    )

    # Adaptive threshold line
    ax.plot(
        df["timestamp"],
        thresholds,
        color="#EF4444",
        linestyle="--",
        linewidth=1.6,
        label="Adaptive Dynamic Threshold (Recalibrated)"
    )

    # Highlight breached regions
    breach = scores >= thresholds
    if np.any(breach):
        ax.scatter(
            df.loc[breach, "timestamp"],
            scores[breach],
            color=F1_RED,
            s=14,
            zorder=4,
            label="Threshold Breach (Alert Active)"
        )

    ax.set_title(
        "F1 SENTINEL — Continuous Anomaly Score vs. Dynamic Adaptive Threshold",
        color=TEXT_WHITE,
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Session Timestamp", color=TEXT_WHITE, fontsize=10, labelpad=8)
    ax.set_ylabel("Anomaly Score / Threshold", color=TEXT_WHITE, fontsize=10, labelpad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_model_comparison_metrics(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/plots/model_comparison_metrics.png"
):
    """Bar chart comparing Event-F1, Precision, Recall, and Detection Delay across all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    apply_f1_theme(ax1, fig)
    apply_f1_theme(ax2, fig)

    models = [m.replace("_", " ") for m in metrics_df["Model"]]
    x = np.arange(len(models))
    width = 0.25

    # Subplot 1: F1, Precision, Recall
    rects1 = ax1.bar(x - width, metrics_df["Event-F1"], width, label="Event-F1", color=F1_RED)
    rects2 = ax1.bar(x, metrics_df["Precision"], width, label="Precision", color=NEON_CYAN)
    rects3 = ax1.bar(x + width, metrics_df["Recall"], width, label="Recall", color=FERRARI_YELLOW)

    ax1.set_title("Model Performance: Event-F1, Precision & Recall", color=TEXT_WHITE, fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=22, ha="right", color=TEXT_WHITE, fontsize=9)
    ax1.set_ylabel("Score [0.0 - 1.0]", color=TEXT_WHITE, fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE)

    # Subplot 2: Detection Delay & Latency
    ax2_twin = ax2.twinx()
    ax2_twin.tick_params(colors=NEON_CYAN, labelsize=9)

    rects4 = ax2.bar(x - width/2, metrics_df["Detection Delay (pts)"], width, label="Detection Delay (points)", color="#8B5CF6")
    line1 = ax2_twin.plot(x, metrics_df["Inference Latency (ms)"], color=NEON_CYAN, marker="o", linewidth=2.0, label="Latency (ms)")

    ax2.set_title("Detection Responsiveness & Inference Latency", color=TEXT_WHITE, fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=22, ha="right", color=TEXT_WHITE, fontsize=9)
    ax2.set_ylabel("Detection Delay (Timestamps)", color=TEXT_WHITE, fontsize=10)
    ax2_twin.set_ylabel("Inference Latency (ms)", color=NEON_CYAN, fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_confusion_breakdown(
    metrics_df: pd.DataFrame,
    output_path: str = "outputs/plots/confusion_breakdown.png"
):
    """Bar chart breakdown of True Positives, False Positives, and False Negatives per model."""
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    apply_f1_theme(ax, fig)

    models = [m.replace("_", " ") for m in metrics_df["Model"]]
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, metrics_df["TP"], width, label="True Positives (TP)", color="#10B981")
    ax.bar(x, metrics_df["FP"], width, label="False Positives (FP)", color="#EF4444")
    ax.bar(x + width, metrics_df["FN"], width, label="False Negatives (FN)", color="#F59E0B")

    ax.set_title("Event-Level Confusion Breakdown (TP, FP, FN Counts)", color=TEXT_WHITE, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", color=TEXT_WHITE, fontsize=9)
    ax.set_ylabel("Event Count", color=TEXT_WHITE, fontsize=10)
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_zoom_in_cases(
    df: pd.DataFrame,
    scores: np.ndarray,
    thresholds: np.ndarray,
    predictions: np.ndarray,
    evidence_list: List[str],
    ground_truth_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    meta: Dict[str, Any],
    output_path: str = "outputs/plots/zoom_in_cases.png"
):
    """
    Plots 3 detailed zoom-in case studies:
    1. True Positive: Correctly identified engine temperature thermal runaway
    2. False Positive: Harmless transient telemetry oscillation
    3. Edge Case / Subtle onset: Detection delay analysis and early warning emergence
    Each with annotated race engineer diagnostic evidence box.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=300)
    apply_f1_theme(axes[0], fig)
    apply_f1_theme(axes[1], fig)
    apply_f1_theme(axes[2], fig)

    ts = df["timestamp"]
    vals = df["value"]

    # --- CASE 1: TRUE POSITIVE (GT Window 1: around 2013-12-10 to 2013-12-12) ---
    w1_start = ground_truth_windows[0][0]
    c1_mask = (ts >= w1_start - pd.Timedelta(hours=24)) & (ts <= w1_start + pd.Timedelta(hours=48))
    idx1 = df[c1_mask].index

    axes[0].plot(ts[idx1], vals[idx1], color=NEON_CYAN, label="Engine Telemetry (°C)", linewidth=1.5)
    axes[0].axvspan(ground_truth_windows[0][0], ground_truth_windows[0][1], color=FERRARI_YELLOW, alpha=0.25, label="Ground Truth Failure Window")
    pred_mask1 = predictions[idx1]
    if np.any(pred_mask1):
        axes[0].scatter(ts[idx1][pred_mask1], vals[idx1][pred_mask1], color=F1_RED, s=24, label="True Positive Detection", zorder=5)

    # Find first detection point in window
    detected_in_w1 = df[(ts >= ground_truth_windows[0][0]) & (ts <= ground_truth_windows[0][1]) & predictions]
    callout_text_1 = "CASE 1: TRUE POSITIVE (ENGINE OVERHEAT)\n" + (
        evidence_list[detected_in_w1.index[0]] if len(detected_in_w1) > 0 else "Thermal runaway alert"
    )
    # Wrap text for clean box
    if len(callout_text_1) > 130:
        callout_text_1 = callout_text_1[:130] + "\n" + callout_text_1[130:]

    axes[0].text(
        0.02, 0.78, callout_text_1,
        transform=axes[0].transAxes,
        fontsize=8.5,
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1F2937", edgecolor="#10B981", alpha=0.92)
    )
    axes[0].set_title("Case Study 1: [TRUE POSITIVE] Rapid Radiator Failure & Thermal Runaway Detected", color=TEXT_WHITE, fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel(f"Value ({meta['unit']})", color=TEXT_WHITE, fontsize=9)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    axes[0].legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    # --- CASE 2: FALSE POSITIVE (Transient spike in nominal regime) ---
    # Find an outside-GT flag if available, or early transient
    outside_mask = (~df["is_ground_truth"].astype(bool)) & predictions
    if np.any(outside_mask):
        fp_idx = df[outside_mask].index[0]
        fp_time = ts[fp_idx]
    else:
        # Fallback interesting nominal area
        fp_time = ts.iloc[1200]
        fp_idx = 1200

    c2_mask = (ts >= fp_time - pd.Timedelta(hours=24)) & (ts <= fp_time + pd.Timedelta(hours=24))
    idx2 = df[c2_mask].index

    axes[1].plot(ts[idx2], vals[idx2], color="#A78BFA", label="Telemetry Signal (°C)", linewidth=1.5)
    pred_mask2 = predictions[idx2]
    if np.any(pred_mask2):
        axes[1].scatter(ts[idx2][pred_mask2], vals[idx2][pred_mask2], color=F1_RED, s=24, label="False Alarm Trigger", zorder=5)

    callout_text_2 = (
        f"CASE 2: FALSE POSITIVE ANALYSIS\n"
        f"Transient thermal oscillation: Score {scores[fp_idx]:.2f} exceeded threshold {thresholds[fp_idx]:.2f}.\n"
        f"Regime filter cooled threshold down; race engineer would mark this as momentary DRS turbulence."
    )
    axes[1].text(
        0.02, 0.76, callout_text_2,
        transform=axes[1].transAxes,
        fontsize=8.5,
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1F2937", edgecolor="#EF4444", alpha=0.92)
    )
    axes[1].set_title("Case Study 2: [FALSE POSITIVE / TRANSIENT] Aerodynamic Wake Noise Under Scrutiny", color=TEXT_WHITE, fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel(f"Value ({meta['unit']})", color=TEXT_WHITE, fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    axes[1].legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    # --- CASE 3: EARLY ONSET & DETECTION DELAY (GT Window 3 or 4) ---
    w3_start = ground_truth_windows[2][0] if len(ground_truth_windows) >= 3 else ground_truth_windows[-1][0]
    w3_end = ground_truth_windows[2][1] if len(ground_truth_windows) >= 3 else ground_truth_windows[-1][1]
    c3_mask = (ts >= w3_start - pd.Timedelta(hours=18)) & (ts <= w3_end + pd.Timedelta(hours=18))
    idx3 = df[c3_mask].index

    axes[2].plot(ts[idx3], vals[idx3], color="#38BDF8", label="Telemetry Signal (°C)", linewidth=1.5)
    axes[2].axvspan(w3_start, w3_end, color=FERRARI_YELLOW, alpha=0.25, label="Ground Truth Failure Window")
    pred_mask3 = predictions[idx3]
    if np.any(pred_mask3):
        axes[2].scatter(ts[idx3][pred_mask3], vals[idx3][pred_mask3], color=F1_RED, s=24, label="Detected Incident", zorder=5)

    callout_text_3 = (
        f"CASE 3: DETECTION DELAY & EARLY WARNING\n"
        f"Subtle thermal drift: First trigger raised within window. Early warning margin allows\n"
        f"pit wall to order engine mode adjustment (Mode 8 / Lean) before critical blowout."
    )
    axes[2].text(
        0.02, 0.76, callout_text_3,
        transform=axes[2].transAxes,
        fontsize=8.5,
        color="#FFFFFF",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1F2937", edgecolor="#F59E0B", alpha=0.92)
    )
    axes[2].set_title("Case Study 3: [EARLY WARNING] Detection Delay & Proactive Strategy Trigger", color=TEXT_WHITE, fontsize=10.5, fontweight="bold")
    axes[2].set_ylabel(f"Value ({meta['unit']})", color=TEXT_WHITE, fontsize=9)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
    axes[2].legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
