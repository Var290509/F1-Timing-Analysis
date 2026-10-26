"""
Generate custom, high-resolution F1 graphics for the dashboard:
1. F1 Pit Wall Hero Banner (1200x260)
2. Circuit of The Americas (COTA) Track Map with Sectors, DRS Zones, and Turns (800x600)
3. 5-Model Hybrid Ensemble & Adaptive Threshold Pipeline Architecture Diagram (1000x420)
4. Constructor Team Grid Card (1000x260)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("app/assets", exist_ok=True)

# -------------------------------------------------------------------------
# 1. F1 PIT WALL HERO BANNER
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 3.2), facecolor="#0E0E15")
ax.set_facecolor("#0E0E15")

# Speed stripes / carbon accents
for i in range(12):
    poly = patches.Polygon([
        [i * 1.2, 0], [i * 1.2 + 0.6, 0], [i * 1.2 + 1.6, 3.2], [i * 1.2 + 1.0, 3.2]
    ], closed=True, facecolor="#181824", alpha=0.45)
    ax.add_patch(poly)

# Red racing ribbon
ax.axhspan(0.1, 0.25, facecolor="#E10600", alpha=0.9)
poly_red = patches.Polygon([[0, 2.7], [2.2, 2.7], [3.2, 0], [1.0, 0]], closed=True, facecolor="#E10600", alpha=0.9)
ax.add_patch(poly_red)

# Telemetry waveform overlay in background
x_wave = np.linspace(5.0, 14.0, 200)
y_wave = 1.0 + 0.35 * np.sin(2.5 * x_wave) + 0.15 * np.cos(5.0 * x_wave)
ax.plot(x_wave, y_wave, color="#00D2BE", lw=1.8, alpha=0.7)
ax.fill_between(x_wave, 0.25, y_wave, color="#00D2BE", alpha=0.08)

# Text Callouts
ax.text(0.4, 1.8, "F1", color="#FFFFFF", fontsize=32, fontweight="bold", family="sans-serif")
ax.text(1.7, 1.8, "SENTINEL", color="#E10600", fontsize=32, fontweight="bold", family="sans-serif")
ax.text(0.4, 0.9, "PIT WALL REAL-TIME TELEMETRY MONITORING & GRID ANOMALY RADAR", color="#F3F3F3", fontsize=11, fontweight="bold", family="sans-serif")
ax.text(0.4, 0.45, "Utkrishta 2026 AI/ML Hackathon | Challenge 2: Sentinel | 76-Season Historical Telemetry Engine (1950–2026)", color="#9CA3AF", fontsize=8.5, family="sans-serif")

# Live HUD boxes on right
hud_box1 = patches.FancyBboxPatch((9.2, 1.3), 2.1, 1.4, boxstyle="round,pad=0.1", facecolor="#1A1A28", edgecolor="#00D2BE", lw=1.5)
ax.add_patch(hud_box1)
ax.text(10.25, 2.2, "SYSTEM STATUS", color="#9CA3AF", fontsize=7.5, fontweight="bold", ha="center")
ax.text(10.25, 1.6, "ONLINE // LIVE", color="#00D2BE", fontsize=11, fontweight="bold", ha="center")

hud_box2 = patches.FancyBboxPatch((11.6, 1.3), 2.1, 1.4, boxstyle="round,pad=0.1", facecolor="#1A1A28", edgecolor="#E10600", lw=1.5)
ax.add_patch(hud_box2)
ax.text(12.65, 2.2, "INFERENCE LATENCY", color="#9CA3AF", fontsize=7.5, fontweight="bold", ha="center")
ax.text(12.65, 1.6, "0.047 ms / pt", color="#FFFFFF", fontsize=11, fontweight="bold", ha="center")

ax.set_xlim(0, 14)
ax.set_ylim(0, 3.2)
ax.axis("off")
plt.tight_layout()
banner_path = "app/assets/f1_hero_banner.png"
plt.savefig(banner_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Generated: {banner_path}")

# -------------------------------------------------------------------------
# 2. CIRCUIT OF THE AMERICAS (COTA) TRACK MAP
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6.5), facecolor="#12121A")
ax.set_facecolor("#12121A")

# Approximate COTA Track Coordinates (Turns 1 to 20)
cota_x = [
    1.0, 1.8, 2.5, 3.5, 4.2, 5.0, 5.5, 6.0, 6.2, 6.0, 5.2, 4.0, 2.5, 1.5, 2.2, 3.0, 3.6, 4.0, 3.2, 1.8, 1.0
]
cota_y = [
    2.5, 5.5, 5.2, 4.6, 4.2, 3.6, 3.8, 4.2, 5.0, 5.6, 6.2, 6.5, 6.0, 4.5, 3.2, 2.8, 2.2, 1.5, 1.0, 1.2, 2.5
]

# Interpolate smooth spline
t = np.linspace(0, len(cota_x)-1, 300)
spline_x = np.interp(t, np.arange(len(cota_x)), cota_x)
spline_y = np.interp(t, np.arange(len(cota_y)), cota_y)

# Draw track shadow / boundary
ax.plot(spline_x, spline_y, color="#252538", lw=14, solid_capstyle="round")

# Sector 1 (Yellow / Red Bull), Sector 2 (Cyan / Mercedes), Sector 3 (Red / Ferrari)
n_pts = len(spline_x)
ax.plot(spline_x[:100], spline_y[:100], color="#FFB800", lw=6, label="Sector 1: High-Speed Esses & Turn 1 Climb")
ax.plot(spline_x[98:200], spline_y[98:200], color="#00D2BE", lw=6, label="Sector 2: 1.2km Back Straight & Hairpin")
ax.plot(spline_x[198:], spline_y[198:], color="#E10600", lw=6, label="Sector 3: Multi-Apex Stadium & Main Straight")

# Key Turn Callouts
ax.plot(1.8, 5.5, "o", color="#FFFFFF", markersize=9, markeredgecolor="#E10600", markeredgewidth=2)
ax.text(1.8, 5.85, "Turn 1\n(Steep Uphill)", color="#FFFFFF", fontsize=8, fontweight="bold", ha="center")

ax.plot(4.2, 4.2, "o", color="#FFFFFF", markersize=7, markeredgecolor="#FFB800", markeredgewidth=1.5)
ax.text(4.4, 4.45, "Esses (T3-T6)", color="#FFB800", fontsize=8, fontweight="bold")

ax.plot(4.0, 6.5, "o", color="#FFFFFF", markersize=8, markeredgecolor="#00D2BE", markeredgewidth=1.5)
ax.text(4.0, 6.8, "Back Straight (335 km/h)", color="#00D2BE", fontsize=8.5, fontweight="bold", ha="center")

ax.plot(4.0, 1.5, "o", color="#FFFFFF", markersize=8, markeredgecolor="#E10600", markeredgewidth=1.5)
ax.text(4.2, 1.25, "Turn 16-18 Carousel", color="#E10600", fontsize=8, fontweight="bold")

# Start/Finish Line
ax.plot([0.9, 1.1], [2.5, 2.5], color="#FFFFFF", lw=4)
ax.text(0.5, 2.45, "START / FINISH", color="#FFFFFF", fontsize=7.5, fontweight="bold", ha="right")

ax.set_title("CIRCUIT OF THE AMERICAS (AUSTIN, TEXAS) — TELEMETRY SECTOR MAP", color="#FFFFFF", fontsize=10, fontweight="bold", pad=12)
ax.legend(loc="lower left", facecolor="#181824", edgecolor="#2D2D3F", labelcolor="#E0E0E0", fontsize=7.5)
ax.axis("off")
plt.tight_layout()
cota_path = "app/assets/cota_circuit_map.png"
plt.savefig(cota_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Generated: {cota_path}")

# -------------------------------------------------------------------------
# 3. 5-DETECTOR HYBRID ENSEMBLE PIPELINE ARCHITECTURE DIAGRAM
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="#12121A")
ax.set_facecolor("#12121A")

# Inputs
box_input = patches.FancyBboxPatch((0.4, 1.5), 1.8, 1.8, boxstyle="round,pad=0.15", facecolor="#1F2937", edgecolor="#00D2BE", lw=2)
ax.add_patch(box_input)
ax.text(1.3, 2.7, "TELEMETRY", color="#00D2BE", fontsize=9, fontweight="bold", ha="center")
ax.text(1.3, 2.4, "STREAM", color="#00D2BE", fontsize=9, fontweight="bold", ha="center")
ax.text(1.3, 1.9, "14 Causal\nFeatures\n(W=30)", color="#E5E7EB", fontsize=8, ha="center")

# 5 Models
models_info = [
    {"name": "Z-Score Baseline", "sub": "Rolling Distribution", "color": "#10B981", "y": 3.8},
    {"name": "Isolation Forest", "sub": "100 Decision Trees", "color": "#3B82F6", "y": 3.0},
    {"name": "Local Outlier Factor", "sub": "Novelty Density k=20", "color": "#8B5CF6", "y": 2.2},
    {"name": "One-Class SVM", "sub": "RBF Kernel Support", "color": "#F59E0B", "y": 1.4},
    {"name": "PyTorch Autoencoder", "sub": "Bottleneck Reconstruction", "color": "#EC4899", "y": 0.6}
]

for m in models_info:
    # Model box
    box_m = patches.FancyBboxPatch((3.2, m["y"] - 0.35), 2.5, 0.7, boxstyle="round,pad=0.1", facecolor="#1E1E2E", edgecolor=m["color"], lw=1.8)
    ax.add_patch(box_m)
    ax.text(4.45, m["y"] + 0.1, m["name"], color="#FFFFFF", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(4.45, m["y"] - 0.15, m["sub"], color=m["color"], fontsize=7.2, ha="center")
    
    # Arrows from input
    ax.annotate("", xy=(3.2, m["y"]), xytext=(2.2, 2.4), arrowprops=dict(arrowstyle="->", color="#4B5563", lw=1.2))
    # Arrows to ensemble
    ax.annotate("", xy=(6.7, 2.4), xytext=(5.7, m["y"]), arrowprops=dict(arrowstyle="->", color=m["color"], lw=1.2))

# Ensemble Fusion Box
box_ens = patches.FancyBboxPatch((6.7, 1.4), 2.2, 2.0, boxstyle="round,pad=0.15", facecolor="#2A1B3D", edgecolor="#A855F7", lw=2)
ax.add_patch(box_ens)
ax.text(7.8, 2.8, "HYBRID ENSEMBLE", color="#C084FC", fontsize=9, fontweight="bold", ha="center")
ax.text(7.8, 2.4, "Weighted Consensus\nFusion", color="#E9D5FF", fontsize=7.8, ha="center")
ax.text(7.8, 1.7, "Score S ∈ [0, 1]", color="#FFFFFF", fontsize=8.5, fontweight="bold", ha="center")

# Adaptive Threshold Box
box_thresh = patches.FancyBboxPatch((9.7, 1.4), 2.0, 2.0, boxstyle="round,pad=0.15", facecolor="#2B1515", edgecolor="#E10600", lw=2)
ax.add_patch(box_thresh)
ax.text(10.7, 2.8, "ADAPTIVE THRESHOLD", color="#F87171", fontsize=8.5, fontweight="bold", ha="center")
ax.text(10.7, 2.3, "μ_trailing + k·σ\nRegime Shift Recal", color="#FECACA", fontsize=7.5, ha="center")
ax.text(10.7, 1.7, "Outlier Resistant", color="#FFB800", fontsize=7.8, fontweight="bold", ha="center")

# Arrow from ensemble to threshold
ax.annotate("", xy=(9.7, 2.4), xytext=(8.9, 2.4), arrowprops=dict(arrowstyle="->", color="#A855F7", lw=2))

# Output Box
box_out = patches.FancyBboxPatch((12.3, 1.4), 1.9, 2.0, boxstyle="round,pad=0.15", facecolor="#064E3B", edgecolor="#10B981", lw=2)
ax.add_patch(box_out)
ax.text(13.25, 2.8, "OUTPUT VERDICT", color="#34D399", fontsize=8.5, fontweight="bold", ha="center")
ax.text(13.25, 2.3, "NORMAL / ANOMALY\nEvent F1 Benchmark", color="#A7F3D0", fontsize=7.5, ha="center")
ax.text(13.25, 1.7, "Telemetry Evidence", color="#FFFFFF", fontsize=8, fontweight="bold", ha="center")

# Arrow from threshold to output
ax.annotate("", xy=(12.3, 2.4), xytext=(11.7, 2.4), arrowprops=dict(arrowstyle="->", color="#E10600", lw=2))

ax.set_xlim(0, 14.5)
ax.set_ylim(0, 4.6)
ax.axis("off")
plt.title("F1 SENTINEL — MULTI-MODEL ADAPTIVE ANOMALY DETECTION ENGINE PIPELINE", color="#FFFFFF", fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
arch_path = "app/assets/model_architecture_pipeline.png"
plt.savefig(arch_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Generated: {arch_path}")

# -------------------------------------------------------------------------
# 4. F1 CONSTRUCTOR TEAM IDENTITY & COLOR CARDS
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(13, 2.5), facecolor="#12121A")
teams = [
    {"name": "MERCEDES-AMG", "drivers": "ANT (#12) / RUS (#63)", "color": "#00D2BE", "bg": "#0D2624"},
    {"name": "SCUDERIA FERRARI", "drivers": "LEC (#16) / HAM (#44)", "color": "#E10600", "bg": "#2B0B0B"},
    {"name": "ORACLE RED BULL", "drivers": "VER (#1) / HAD (#6)", "color": "#FFB800", "bg": "#29210B"},
    {"name": "MCLAREN F1", "drivers": "NOR (#4) / PIA (#81)", "color": "#FF8700", "bg": "#29190B"},
    {"name": "ASTON MARTIN", "drivers": "ALO (#14) / STR (#18)", "color": "#00594F", "bg": "#0A211D"}
]

for ax, t in zip(axes, teams):
    ax.set_facecolor(t["bg"])
    ax.spines['bottom'].set_color(t["color"])
    ax.spines['top'].set_color(t["color"])
    ax.spines['right'].set_color(t["color"])
    ax.spines['left'].set_color(t["color"])
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    ax.text(0.5, 0.72, t["name"], color=t["color"], fontsize=9.5, fontweight="bold", ha="center")
    ax.text(0.5, 0.42, t["drivers"], color="#FFFFFF", fontsize=8.5, ha="center")
    ax.text(0.5, 0.18, "ACTIVE IN RUNNING GP", color="#9CA3AF", fontsize=7, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("FORMULA 1 2026 WORLD CHAMPIONSHIP GRID — TELEMETRY CONSTRUCTORS", color="#FFFFFF", fontsize=10, fontweight="bold", y=1.03)
plt.tight_layout()
teams_path = "app/assets/f1_team_grid.png"
plt.savefig(teams_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Generated: {teams_path}")
