"""
F1 Sentinel Driver Race Timing Visualizations
Generates high-resolution multi-driver timing and anomaly detection plots:
1. Multi-driver lap-by-lap pace trajectories and tyre degradation curves
2. Full-grid lap times scatter plot with detected timing anomalies overlaid
3. Teammate head-to-head pace delta analysis
"""

import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.driver_timing_engine import DriverTimingEngine

F1_RED = "#E10600"
CARBON_BLACK = "#15151E"
DARK_GRAY = "#22222E"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#A0A0B0"
NEON_CYAN = "#00D2BE"
FERRARI_YELLOW = "#FFF500"
MCLAREN_PAPAYA = "#FF8000"
FERRARI_RED = "#DC0000"
MERCEDES_TEAL = "#00A19B"
RED_BULL_NAVY = "#3671C6"
GRID_COLOR = "#2D2D3F"


def apply_f1_theme(ax, fig=None):
    ax.set_facecolor(DARK_GRAY)
    if fig:
        fig.patch.set_facecolor(CARBON_BLACK)
    ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444455")
        spine.set_linewidth(1.0)


def plot_driver_lap_pace_comparison(output_path="outputs/plots/driver_lap_pace_comparison.png"):
    """Plot multi-driver lap-by-lap pace trajectory and tyre degradation curves."""
    engine = DriverTimingEngine()
    latest_id = engine.get_latest_race_id()
    grid_df = engine.get_race_telemetry_grid(latest_id)
    race_info = engine.get_race_info(latest_id)

    if grid_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=300)
    apply_f1_theme(ax, fig)

    top_drivers = ["ANT", "LEC", "VER", "HAM", "NOR", "PIA"]
    driver_colors = {
        "ANT": MERCEDES_TEAL,
        "LEC": FERRARI_RED,
        "VER": RED_BULL_NAVY,
        "HAM": "#E11D48",
        "NOR": MCLAREN_PAPAYA,
        "PIA": "#F59E0B"
    }

    for code in top_drivers:
        sub = grid_df[grid_df["code"] == code].sort_values("lap")
        if not sub.empty:
            ax.plot(
                sub["lap"],
                sub["lap_sec"],
                label=f"{code} ({sub['team_name'].iloc[0]})",
                color=driver_colors.get(code, NEON_CYAN),
                linewidth=1.8,
                alpha=0.9
            )

    ax.set_title(
        f"DRIVER LAP PACE COMPARISON — {race_info['year']} {race_info['name'].upper()}",
        color=TEXT_WHITE,
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Lap Number", color=TEXT_WHITE, fontsize=10)
    ax.set_ylabel("Lap Time (seconds)", color=TEXT_WHITE, fontsize=10)
    ax.set_ylim(95, 125)
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_driver_timing_anomalies_scatter(output_path="outputs/plots/driver_timing_anomalies_scatter.png"):
    """Scatter plot of all driver laps with flagged timing anomalies highlighted."""
    engine = DriverTimingEngine()
    latest_id = engine.get_latest_race_id()
    grid_df = engine.get_race_telemetry_grid(latest_id)
    anoms = engine.detect_driver_timing_anomalies(latest_id)
    race_info = engine.get_race_info(latest_id)

    if grid_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6.0), dpi=300)
    apply_f1_theme(ax, fig)

    # Plot all laps as soft background points
    ax.scatter(
        grid_df["lap"],
        grid_df["lap_sec"],
        color="#4B5563",
        alpha=0.35,
        s=12,
        label="Clean Nominal Race Laps (All Drivers)"
    )

    # Overlay detected timing anomalies
    if not anoms.empty:
        ax.scatter(
            anoms["lap"],
            anoms["lap_sec"],
            color=F1_RED,
            s=45,
            edgecolors="#FFFFFF",
            linewidth=0.6,
            zorder=5,
            label="Sentinel Driver Timing Anomalies (Spikes/Cliffs/Pit Delays)"
        )

        # Annotate top 4 anomalies
        top_anoms = anoms.sort_values("anomaly_score", ascending=False).head(4)
        for _, row in top_anoms.iterrows():
            ax.annotate(
                f"{row['driver_code']} L{row['lap']}: {row['anomaly_type'][:12]}",
                xy=(row["lap"], row["lap_sec"]),
                xytext=(row["lap"] + 1.2, row["lap_sec"] + 3.0),
                color=TEXT_WHITE,
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1F2937", edgecolor=F1_RED, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=F1_RED, lw=0.8)
            )

    ax.set_title(
        f"MULTI-DRIVER TIMING ANOMALY DISTRIBUTION — {race_info['year']} {race_info['name'].upper()}",
        color=TEXT_WHITE,
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Lap Number", color=TEXT_WHITE, fontsize=10)
    ax.set_ylabel("Lap Time (seconds)", color=TEXT_WHITE, fontsize=10)
    ax.set_ylim(95, 145)
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_teammate_pace_deltas(output_path="outputs/plots/teammate_pace_deltas.png"):
    """Plot intra-team lap-by-lap pace deltas for top constructor pairings."""
    engine = DriverTimingEngine()
    latest_id = engine.get_latest_race_id()
    teammates_df = engine.compare_teammates(latest_id)
    race_info = engine.get_race_info(latest_id)

    if teammates_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)
    apply_f1_theme(ax, fig)

    teams = ["Ferrari", "McLaren", "Mercedes"]
    team_colors = {"Ferrari": FERRARI_RED, "McLaren": MCLAREN_PAPAYA, "Mercedes": MERCEDES_TEAL}

    for t in teams:
        sub = teammates_df[teammates_df["team"].str.contains(t, case=False, na=False)].sort_values("lap")
        if not sub.empty:
            d1, d2 = sub["driver_1"].iloc[0], sub["driver_2"].iloc[0]
            ax.plot(
                sub["lap"],
                sub["delta_d1_minus_d2"],
                label=f"{t}: {d1} vs {d2} (Delta s)",
                color=team_colors.get(t, NEON_CYAN),
                linewidth=1.8,
                alpha=0.9
            )

    ax.axhline(0, color="#6B7280", linestyle="--", linewidth=1.0)
    ax.axhspan(-1.2, 1.2, color="#10B981", alpha=0.10, label="Nominal Teammate Delta Envelope (±1.2s)")

    ax.set_title(
        f"INTRA-TEAM HEAD-TO-HEAD PACE DELTA & ANOMALIES — {race_info['year']} {race_info['name'].upper()}",
        color=TEXT_WHITE,
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Lap Number", color=TEXT_WHITE, fontsize=10)
    ax.set_ylabel("Lap Delta: Driver 1 - Driver 2 (seconds)", color=TEXT_WHITE, fontsize=10)
    ax.set_ylim(-6.0, 6.0)
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_driver_lap_pace_comparison()
    plot_driver_timing_anomalies_scatter()
    plot_teammate_pace_deltas()
