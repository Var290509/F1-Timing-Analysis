"""
Download and generate visual assets for the F1 Sentinel Dashboard:
- F1 official logo & track maps from Wikimedia
- High-quality telemetry sensor map diagram (matplotlib / PIL)
- Driver & tyre graphics
"""

import os
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("app/assets", exist_ok=True)

# 1. Download official vector/raster public domain graphics from Wikimedia
urls = {
    "f1_logo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/640px-F1.svg.png",
    "cota_circuit.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Circuit_of_the_Americas_track_map.svg/800px-Circuit_of_the_Americas_track_map.svg.png",
    "f1_steering_wheel.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/BMW_Sauber_F1.07_Steering_Wheel.jpg/800px-BMW_Sauber_F1.07_Steering_Wheel.jpg",
}

for filename, url in urls.items():
    filepath = os.path.join("app/assets", filename)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(filepath, "wb") as f:
                f.write(response.read())
        print(f"Downloaded: {filepath} ({os.path.getsize(filepath):,} bytes)")
    except Exception as e:
        print(f"Notice: Could not download {url}: {e}")

# 2. Generate a custom, high-tech F1 Car Telemetry Sensor Map Graphic
fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#12121A")
ax.set_facecolor("#12121A")

# Draw F1 car schematic outline
# Body
car_body = patches.Polygon([
    [1.5, 2.2], [3.0, 2.3], [4.5, 2.4], [6.5, 2.3], [8.0, 2.1], 
    [9.2, 2.0], [9.2, 1.8], [8.0, 1.7], [6.5, 1.5], [4.5, 1.4], 
    [3.0, 1.5], [1.5, 1.6]
], closed=True, facecolor="#1E1E2E", edgecolor="#E10600", linewidth=2.5)
ax.add_patch(car_body)

# Cockpit & Halo
cockpit = patches.Polygon([[4.8, 2.35], [5.8, 2.3], [5.6, 2.0], [4.6, 2.0]], closed=True, facecolor="#3B82F6", alpha=0.6, edgecolor="#60A5FA", linewidth=1.5)
ax.add_patch(cockpit)

# Front Wing
front_wing = patches.Rectangle((0.6, 1.4), 0.8, 1.0, facecolor="#E10600", edgecolor="#FFFFFF", linewidth=1.2)
ax.add_patch(front_wing)

# Rear Wing
rear_wing = patches.Rectangle((9.0, 1.5), 0.6, 1.2, facecolor="#E10600", edgecolor="#FFFFFF", linewidth=1.2)
ax.add_patch(rear_wing)

# Wheels (Front & Rear)
fl_wheel = patches.Rectangle((2.0, 2.45), 1.4, 0.45, facecolor="#000000", edgecolor="#EF4444", linewidth=2)
fr_wheel = patches.Rectangle((2.0, 0.9), 1.4, 0.45, facecolor="#000000", edgecolor="#EF4444", linewidth=2)
rl_wheel = patches.Rectangle((7.4, 2.45), 1.5, 0.55, facecolor="#000000", edgecolor="#EF4444", linewidth=2)
rr_wheel = patches.Rectangle((7.4, 0.8), 1.5, 0.55, facecolor="#000000", edgecolor="#EF4444", linewidth=2)

for w in [fl_wheel, fr_wheel, rl_wheel, rr_wheel]:
    ax.add_patch(w)

# Sensor nodes & callouts
sensors = [
    {"name": "TEL-ENG-01\nICE Coolant & Head Temp", "pos": (6.0, 1.9), "xytext": (6.0, 3.4), "color": "#00D2BE"},
    {"name": "TEL-TYR-FL\nTyre Pressure & Bulk Temp", "pos": (2.7, 2.7), "xytext": (1.0, 3.4), "color": "#FFB800"},
    {"name": "TEL-LAP-01\nGPS Sector & Timing Loop", "pos": (5.2, 2.15), "xytext": (3.5, 0.3), "color": "#3B82F6"},
    {"name": "TEL-HYB-01\nERS-K & MGU Battery State", "pos": (7.0, 1.9), "xytext": (8.5, 3.4), "color": "#10B981"},
    {"name": "TEL-BRK-RR\nCarbon-Carbon Disc Temp", "pos": (8.1, 1.1), "xytext": (7.5, 0.3), "color": "#EF4444"}
]

for s in sensors:
    ax.plot(s["pos"][0], s["pos"][1], "o", color=s["color"], markersize=9, markeredgecolor="#FFFFFF", markeredgewidth=1.5)
    ax.annotate(
        s["name"],
        xy=s["pos"],
        xytext=s["xytext"],
        arrowprops=dict(facecolor=s["color"], edgecolor=s["color"], arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#181824", edgecolor=s["color"], lw=1.2),
        color="#FFFFFF",
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center"
    )

ax.set_xlim(0, 10.5)
ax.set_ylim(0, 4.0)
ax.axis("off")
plt.title("FORMULA 1 ONBOARD TELEMETRY SENSOR ARCHITECTURE & CHANNEL MAPPING", color="#FFFFFF", fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
sensor_map_path = os.path.join("app/assets", "f1_sensor_architecture.png")
plt.savefig(sensor_map_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Generated: {sensor_map_path} ({os.path.getsize(sensor_map_path):,} bytes)")

# 3. Generate Pirelli Tyre Compounds Graphic
fig, axes = plt.subplots(1, 5, figsize=(10, 2.2), facecolor="#12121A")
compounds = [
    {"name": "HARD (C1/C2)", "color": "#FFFFFF", "stripe": "#FFFFFF", "dur": "Highest (40+ Laps)", "grip": "Baseline"},
    {"name": "MEDIUM (C3)", "color": "#FFD700", "stripe": "#FFD700", "dur": "Balanced (25-35 Laps)", "grip": "+0.5s/lap"},
    {"name": "SOFT (C4/C5)", "color": "#EF4444", "stripe": "#EF4444", "dur": "Low (15-22 Laps)", "grip": "+1.2s/lap"},
    {"name": "INTERMEDIATE", "color": "#10B981", "stripe": "#10B981", "dur": "Wet / Damp Track", "grip": "30L/sec water"},
    {"name": "WET", "color": "#3B82F6", "stripe": "#3B82F6", "dur": "Standing Water", "grip": "85L/sec water"}
]

for i, (ax, comp) in enumerate(zip(axes, compounds)):
    ax.set_facecolor("#12121A")
    # Outer tire
    tire = patches.Circle((0.5, 0.6), 0.32, facecolor="#1E1E24", edgecolor=comp["color"], linewidth=3.5)
    inner = patches.Circle((0.5, 0.6), 0.16, facecolor="#0B0B0F", edgecolor="#4A4A5A", linewidth=1.5)
    ax.add_patch(tire)
    ax.add_patch(inner)
    ax.text(0.5, 0.6, "P ZERO", color=comp["color"], fontsize=6, fontweight="black", ha="center", va="center")
    ax.text(0.5, 0.18, comp["name"], color=comp["color"], fontsize=8.5, fontweight="bold", ha="center")
    ax.text(0.5, 0.04, comp["dur"], color="#9CA3AF", fontsize=6.5, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

plt.suptitle("PIRELLI F1 TYRE SPECIFICATIONS & ANOMALY CLIFF PROFILES", color="#FFFFFF", fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout()
tyre_map_path = os.path.join("app/assets", "pirelli_tyre_compounds.png")
plt.savefig(tyre_map_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"Generated: {tyre_map_path} ({os.path.getsize(tyre_map_path):,} bytes)")
