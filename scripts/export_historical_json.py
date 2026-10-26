"""
Exports complete historical F1 benchmark metrics and current game transformation
pillars to outputs/f1_historical_comparison.json.
"""

import os
import sys
import json

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.historical_engine import F1HistoricalEngine, F1_ERAS


def export_historical_metrics(output_path="outputs/f1_historical_comparison.json"):
    engine = F1HistoricalEngine()
    rel_df = engine.compute_reliability_evolution()
    pit_df = engine.compute_pitstop_evolution()
    pillars = engine.get_current_game_transformation_pillars()

    data = {
        "eras_overview": F1_ERAS,
        "reliability_evolution_by_era": rel_df.to_dict(orient="records"),
        "pitstop_evolution_by_year": pit_df.to_dict(orient="records") if not pit_df.empty else [],
        "current_game_transformation_pillars": pillars
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    export_historical_metrics()
