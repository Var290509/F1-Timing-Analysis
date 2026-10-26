"""
Generate an F1 Team Radio HUD / Audio Wave graphic for the live feed view
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("app/assets", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 2.2), facecolor="#141522")
ax.set_facecolor("#141522")

# Audio wave bars
n_bars = 48
x_bars = np.linspace(0.5, 9.5, n_bars)
np.random.seed(42)
heights = 0.2 + 0.6 * np.abs(np.sin(np.linspace(0, 3*np.pi, n_bars))) + 0.2 * np.random.rand(n_bars)

for x, h in zip(x_bars, heights):
    color = "#E10600" if x > 6.0 else ("#FFB800" if x > 3.0 else "#00D2BE")
    ax.plot([x, x], [1.1 - h/2, 1.1 + h/2], color=color, lw=3.5, solid_capstyle="round")

# Team Radio Label
badge = patches.FancyBboxPatch((0.5, 1.7), 2.2, 0.4, boxstyle="round,pad=0.08", facecolor="#E10600", edgecolor="none")
ax.add_patch(badge)
ax.text(1.6, 1.88, "TEAM RADIO // LIVE", color="#FFFFFF", fontsize=8.5, fontweight="bold", ha="center")

ax.text(0.5, 0.22, "CH 16 // PIT WALL <-> DRIVER // REAL-TIME STRATEGIC COMMS", color="#9CA3AF", fontsize=7.5, fontweight="bold")
ax.text(9.5, 0.22, "AUDIO FREQ: 468.250 MHz", color="#6B7280", fontsize=7.5, ha="right")

ax.set_xlim(0, 10)
ax.set_ylim(0, 2.2)
ax.axis("off")
plt.tight_layout()
out_path = "app/assets/f1_radio_comms.png"
plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Generated: {out_path}")
