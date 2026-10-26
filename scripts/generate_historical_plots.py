"""
F1 Sentinel Historical Visualizations (1950–2026)
Renders high-resolution competition plots comparing 76 seasons of historical F1
with the current running game (2022–2026):
1. Reliability & Attrition Evolution (1950–2026)
2. Pit Stop Duration & Precision Evolution
3. 6 Pillars of Current Game Transformation (Radar/Bar Analysis)
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
from src.historical_engine import F1HistoricalEngine, F1_ERAS

F1_RED = "#E10600"
CARBON_BLACK = "#15151E"
DARK_GRAY = "#22222E"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#A0A0B0"
NEON_CYAN = "#00D2BE"
FERRARI_YELLOW = "#FFF500"
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


def plot_historical_reliability_evolution(output_path="outputs/plots/historical_reliability_evolution.png"):
    """Plot mechanical DNF rate vs finish rate across 7 Formula 1 eras (1950–2026)."""
    engine = F1HistoricalEngine()
    rel = engine.compute_reliability_evolution()

    fig, ax1 = plt.subplots(figsize=(13, 5.5), dpi=300)
    apply_f1_theme(ax1, fig)

    eras = [str(e).replace(" (", "\n(") for e in rel["era"]]
    x = np.arange(len(eras))
    width = 0.35

    # Bars for Mechanical DNF %
    bars1 = ax1.bar(
        x - width/2,
        rel["mech_dnf_pct"],
        width,
        label="Mechanical DNF Rate (%) [Engine/Gearbox/Hydraulics]",
        color=F1_RED,
        alpha=0.9
    )

    # Bars for Finish %
    bars2 = ax1.bar(
        x + width/2,
        rel["finish_pct"],
        width,
        label="Finishing & Classified Rate (%)",
        color="#10B981",
        alpha=0.9
    )

    # Annotate mechanical DNF drops
    for i, v in enumerate(rel["mech_dnf_pct"]):
        ax1.text(x[i] - width/2, v + 1.2, f"{v:.1f}%", ha="center", color="#FFAAAA", fontsize=8.5, fontweight="bold")
    for i, v in enumerate(rel["finish_pct"]):
        ax1.text(x[i] + width/2, v + 1.2, f"{v:.1f}%", ha="center", color="#B0F5D5", fontsize=8.5, fontweight="bold")

    ax1.set_title(
        "F1 RELIABILITY REVOLUTION (1950–2026) — MECHANICAL ATTRITION COLLAPSE",
        color=TEXT_WHITE,
        fontsize=12.5,
        fontweight="bold",
        pad=14
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(eras, color=TEXT_WHITE, fontsize=8.5)
    ax1.set_ylabel("Percentage of Race Grid Entries (%)", color=TEXT_WHITE, fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_historical_pitstop_evolution(output_path="outputs/plots/historical_pitstop_evolution.png"):
    """Plot pit stop timing evolution across the modern era (2012–2026)."""
    engine = F1HistoricalEngine()
    pit_df = engine.compute_pitstop_evolution()
    if pit_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    apply_f1_theme(ax, fig)

    ax.plot(
        pit_df["year"],
        pit_df["median_duration_s"],
        color=NEON_CYAN,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Median Pit Lane Transit Time (s)"
    )
    ax.plot(
        pit_df["year"],
        pit_df["fastest_stop_s"],
        color=FERRARI_YELLOW,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="Fastest Pit Lane Stop Record (s)"
    )

    ax.set_title(
        "PIT STOP PRECISION & TIME COMPRESSION ACROSS RECENT SEASONS (2012–2026)",
        color=TEXT_WHITE,
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Championship Season", color=TEXT_WHITE, fontsize=10)
    ax.set_ylabel("Pit Lane Transit Duration (seconds)", color=TEXT_WHITE, fontsize=10)
    ax.legend(facecolor=DARK_GRAY, edgecolor="#555566", labelcolor=TEXT_WHITE, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_current_vs_historical_changes(output_path="outputs/plots/current_vs_historical_changes.png"):
    """
    Comparison bar chart contrasting the 3 defining eras across 5 key technical metrics:
    - 1980s Turbo Monster Era
    - 2000s V10 High-Rev Era
    - 2022–2026 Current Ground-Effect Era
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=300)
    for ax_row in axes:
        for ax in ax_row:
            apply_f1_theme(ax, fig)

    eras = ["1980s Turbo", "2000s V10", "Current 2026 Game"]
    colors = [F1_RED, FERRARI_YELLOW, NEON_CYAN]

    # 1. Thermal Efficiency
    thermal_eff = [28.0, 31.0, 52.5]
    axes[0, 0].bar(eras, thermal_eff, color=colors, width=0.5)
    axes[0, 0].set_title("Powertrain Thermal Efficiency (%)", color=TEXT_WHITE, fontsize=10, fontweight="bold")
    axes[0, 0].set_ylim(0, 60)
    for i, v in enumerate(thermal_eff): axes[0, 0].text(i, v + 1.2, f"{v}%", ha="center", color=TEXT_WHITE, fontweight="bold")

    # 2. Minimum Car Weight (kg)
    weight = [540, 605, 798]
    axes[0, 1].bar(eras, weight, color=colors, width=0.5)
    axes[0, 1].set_title("Minimum Vehicle Mass (kg)", color=TEXT_WHITE, fontsize=10, fontweight="bold")
    axes[0, 1].set_ylim(0, 920)
    for i, v in enumerate(weight): axes[0, 1].text(i, v + 18, f"{v} kg", ha="center", color=TEXT_WHITE, fontweight="bold")

    # 3. Mechanical DNF Failure Rate (%)
    dnf_rate = [47.5, 36.1, 8.9]
    axes[0, 2].bar(eras, dnf_rate, color=colors, width=0.5)
    axes[0, 2].set_title("Mechanical DNF Failure Rate (%)", color=TEXT_WHITE, fontsize=10, fontweight="bold")
    axes[0, 2].set_ylim(0, 55)
    for i, v in enumerate(dnf_rate): axes[0, 2].text(i, v + 1.2, f"{v}%", ha="center", color=TEXT_WHITE, fontweight="bold")

    # 4. Stationary Pit Stop Duration (s)
    pitstop_s = [25.0, 8.5, 2.2]
    axes[1, 0].bar(eras, pitstop_s, color=colors, width=0.5)
    axes[1, 0].set_title("Stationary Pit Stop Duration (s)", color=TEXT_WHITE, fontsize=10, fontweight="bold")
    axes[1, 0].set_ylim(0, 30)
    for i, v in enumerate(pitstop_s): axes[1, 0].text(i, v + 0.8, f"{v}s", ha="center", color=TEXT_WHITE, fontweight="bold")

    # 5. Downforce Loss When Following at 10m (%)
    wake_loss = [45.0, 52.0, 18.0]
    axes[1, 1].bar(eras, wake_loss, color=colors, width=0.5)
    axes[1, 1].set_title("Downforce Loss Following at 10m (%)", color=TEXT_WHITE, fontsize=10, fontweight="bold")
    axes[1, 1].set_ylim(0, 60)
    for i, v in enumerate(wake_loss): axes[1, 1].text(i, v + 1.2, f"{v}%", ha="center", color=TEXT_WHITE, fontweight="bold")

    # 6. Sensor Density & Telemetry Channels per Car
    sensors = [12, 110, 350]
    axes[1, 2].bar(eras, sensors, color=colors, width=0.5)
    axes[1, 2].set_title("Onboard Sensor Count per Car", color=TEXT_WHITE, fontsize=10, fontweight="bold")
    axes[1, 2].set_ylim(0, 420)
    for i, v in enumerate(sensors): axes[1, 2].text(i, v + 10, f"{v}+", ha="center", color=TEXT_WHITE, fontweight="bold")

    fig.suptitle(
        "F1 CURRENT GAME TRANSFORMATION (2022–2026 vs. 1980s & 2000s)",
        color=TEXT_WHITE,
        fontsize=13.5,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_historical_reliability_evolution()
    plot_historical_pitstop_evolution()
    plot_current_vs_historical_changes()
