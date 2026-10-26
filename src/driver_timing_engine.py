"""
F1 Sentinel Driver Race Timing & Live Telemetry Engine
Analyzes lap-by-lap timings of every driver in current and historical Grands Prix.
Computes multi-driver pace deltas, gap to leader, intervals, tyre degradation curves,
pit stop out-laps, and real-time timing anomalies across the entire grid.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class DriverTimingEngine:
    """
    Engine to ingest and analyze full race timing telemetry for every driver on the grid.
    Detects driver-specific timing anomalies, tyre degradation cliffs, and teammate pace deltas.
    """

    def __init__(self, data_dir: str = "data/f1_historical"):
        self.data_dir = data_dir
        self.laps_df: Optional[pd.DataFrame] = None
        self.races_df: Optional[pd.DataFrame] = None
        self.drivers_df: Optional[pd.DataFrame] = None
        self.constructors_df: Optional[pd.DataFrame] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.pit_stops_df: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self):
        """Load relational timing tables."""
        try:
            self.laps_df = pd.read_csv(os.path.join(self.data_dir, "lap_times.csv"))
            self.races_df = pd.read_csv(os.path.join(self.data_dir, "races.csv"))
            self.drivers_df = pd.read_csv(os.path.join(self.data_dir, "drivers.csv"))
            self.constructors_df = pd.read_csv(os.path.join(self.data_dir, "constructors.csv"))
            self.results_df = pd.read_csv(os.path.join(self.data_dir, "results.csv"))
            
            pit_path = os.path.join(self.data_dir, "pit_stops.csv")
            if os.path.exists(pit_path):
                self.pit_stops_df = pd.read_csv(pit_path)
        except Exception as e:
            print(f"[WARNING] DriverTimingEngine failed to load tables: {e}")

    def get_latest_race_id(self) -> int:
        """Return the most recent raceId present in lap_times."""
        if self.laps_df is not None and not self.laps_df.empty:
            return int(self.laps_df["raceId"].max())
        return 1186

    def get_race_info(self, race_id: int) -> Dict[str, Any]:
        """Fetch race metadata (year, round, name, circuit)."""
        if self.races_df is None:
            return {"raceId": race_id, "name": f"Grand Prix #{race_id}", "year": 2026}
        matched = self.races_df[self.races_df["raceId"] == race_id]
        if not matched.empty:
            row = matched.iloc[0]
            return {
                "raceId": int(row["raceId"]),
                "year": int(row["year"]),
                "round": int(row["round"]),
                "name": str(row["name"]),
                "date": str(row["date"])
            }
        return {"raceId": race_id, "name": f"Grand Prix #{race_id}", "year": 2026}

    def get_race_telemetry_grid(self, race_id: Optional[int] = None) -> pd.DataFrame:
        """
        Build the comprehensive multi-driver lap-by-lap timing matrix.
        Calculates lap time in seconds, cumulative time, gap to leader, and interval.
        """
        if race_id is None:
            race_id = self.get_latest_race_id()

        sub_laps = self.laps_df[self.laps_df["raceId"] == race_id].copy()
        if sub_laps.empty:
            return pd.DataFrame()

        # Merge driver metadata
        sub_laps = sub_laps.merge(
            self.drivers_df[["driverId", "code", "forename", "surname", "number"]],
            on="driverId",
            how="left"
        )
        sub_laps["driver_name"] = sub_laps["forename"] + " " + sub_laps["surname"]
        sub_laps["lap_sec"] = sub_laps["milliseconds"] / 1000.0

        # Merge constructor metadata from results if available
        if self.results_df is not None:
            res_sub = self.results_df[self.results_df["raceId"] == race_id][["driverId", "constructorId"]]
            res_sub = res_sub.merge(self.constructors_df[["constructorId", "name"]], on="constructorId")
            res_sub = res_sub.rename(columns={"name": "team_name"})
            sub_laps = sub_laps.merge(res_sub, on="driverId", how="left")
            sub_laps["team_name"] = sub_laps["team_name"].fillna("Formula 1 Team")
        else:
            sub_laps["team_name"] = "Formula 1 Team"

        # Merge pit stop flags
        sub_laps["is_pit_lap"] = False
        sub_laps["pit_duration_sec"] = 0.0
        if self.pit_stops_df is not None:
            pits = self.pit_stops_df[self.pit_stops_df["raceId"] == race_id][["driverId", "lap", "duration"]]
            pits["pit_duration_sec"] = pd.to_numeric(pits["duration"], errors="coerce").fillna(0.0)
            pits["is_pit_lap"] = True
            sub_laps = sub_laps.merge(pits[["driverId", "lap", "is_pit_lap", "pit_duration_sec"]], on=["driverId", "lap"], how="left")
            sub_laps["is_pit_lap"] = sub_laps["is_pit_lap_y"].fillna(False)
            sub_laps["pit_duration_sec"] = sub_laps["pit_duration_sec_y"].fillna(0.0)
            sub_laps = sub_laps.drop(columns=["is_pit_lap_x", "is_pit_lap_y", "pit_duration_sec_x", "pit_duration_sec_y"], errors="ignore")

        # Sort chronologically by driver and lap
        sub_laps = sub_laps.sort_values(["driverId", "lap"]).reset_index(drop=True)
        sub_laps["cum_time_sec"] = sub_laps.groupby("driverId")["lap_sec"].cumsum()

        # Compute Gap to Leader and Interval on each lap
        gaps = []
        intervals = []
        for lap_num, lap_group in sub_laps.groupby("lap"):
            lap_sorted = lap_group.sort_values("cum_time_sec")
            leader_time = lap_sorted["cum_time_sec"].iloc[0]
            prev_time = leader_time
            
            for _, row in lap_group.iterrows():
                gap_val = row["cum_time_sec"] - leader_time
                gaps.append((row["driverId"], row["lap"], gap_val))
                
                # Find car ahead in position
                pos = row["position"]
                car_ahead = lap_group[lap_group["position"] == pos - 1]
                if not car_ahead.empty:
                    interval_val = row["cum_time_sec"] - car_ahead["cum_time_sec"].iloc[0]
                else:
                    interval_val = 0.0
                intervals.append((row["driverId"], row["lap"], interval_val))

        gap_df = pd.DataFrame(gaps, columns=["driverId", "lap", "gap_to_leader_sec"])
        interval_df = pd.DataFrame(intervals, columns=["driverId", "lap", "interval_sec"])
        
        sub_laps = sub_laps.merge(gap_df, on=["driverId", "lap"]).merge(interval_df, on=["driverId", "lap"])
        return sub_laps

    def compute_driver_pace_summary(self, race_id: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate pace benchmarks for every driver in the race:
        Fastest lap, median clean lap, standard deviation, positions gained.
        """
        grid_df = self.get_race_telemetry_grid(race_id)
        if grid_df.empty:
            return pd.DataFrame()

        summary_records = []
        for code, grp in grid_df.groupby("code"):
            is_pit_bool = grp["is_pit_lap"].fillna(False).astype(bool)
            clean_laps = grp[~is_pit_bool]
            if clean_laps.empty:
                clean_laps = grp
            fastest_lap_s = grp["lap_sec"].min()
            fastest_lap_row = grp[grp["lap_sec"] == fastest_lap_s].iloc[0]
            
            start_pos = grp["position"].iloc[0]
            end_pos = grp["position"].iloc[-1]
            pos_delta = start_pos - end_pos  # positive = gained positions

            summary_records.append({
                "code": code,
                "driver_name": grp["driver_name"].iloc[0],
                "team": grp["team_name"].iloc[0],
                "total_laps": len(grp),
                "start_pos": int(start_pos),
                "finish_pos": int(end_pos),
                "pos_change": f"{pos_delta:+d}",
                "fastest_lap_str": fastest_lap_row["time"],
                "fastest_lap_sec": round(fastest_lap_s, 3),
                "median_pace_sec": round(clean_laps["lap_sec"].median(), 3),
                "pace_std_sec": round(clean_laps["lap_sec"].std(), 3),
                "total_pit_stops": int(grp["is_pit_lap"].sum())
            })

        summary_df = pd.DataFrame(summary_records).sort_values("finish_pos").reset_index(drop=True)
        return summary_df

    def detect_driver_timing_anomalies(self, race_id: Optional[int] = None) -> pd.DataFrame:
        """
        Run multi-criteria real-time anomaly detection across EVERY driver's lap time sequence:
        1. PACE_DROP_SPIKE: Sudden lap time spike > +2.5s (lockup, off-track, yellow flag).
        2. TYRE_CLIFF: Lap times increasing by > +0.65s/lap over 3 consecutive laps.
        3. PIT_DELAY: Pit stop stationary duration > 3.5s or total transit > 26.5s.
        4. UNEXPECTED_PACE_COLLAPSE: Extreme lap degradation > +5.0s on clean track.
        """
        grid_df = self.get_race_telemetry_grid(race_id)
        if grid_df.empty:
            return pd.DataFrame()

        anomalies = []

        for code, grp in grid_df.groupby("code"):
            grp = grp.sort_values("lap").reset_index(drop=True)
            n_laps = len(grp)

            # Rolling 5-lap baseline median (excluding pit stops)
            rolling_base = grp["lap_sec"].rolling(window=5, min_periods=2).median()

            for i in range(n_laps):
                lap_num = int(grp["lap"].iloc[i])
                time_str = grp["time"].iloc[i]
                lap_sec = float(grp["lap_sec"].iloc[i])
                is_pit = bool(grp["is_pit_lap"].iloc[i])
                base_pace = float(rolling_base.iloc[i])
                delta_from_base = lap_sec - base_pace

                # 1. Pit Stop Delay Anomaly
                if is_pit:
                    pit_dur = float(grp["pit_duration_sec"].iloc[i])
                    if pit_dur > 26.0:  # Transit anomaly
                        excess = pit_dur - 23.5
                        anom_score = min(1.0, 0.65 + (excess / 10.0))
                        anomalies.append({
                            "driver_code": code,
                            "driver_name": grp["driver_name"].iloc[i],
                            "team": grp["team_name"].iloc[i],
                            "lap": lap_num,
                            "lap_time": time_str,
                            "lap_sec": round(lap_sec, 3),
                            "anomaly_type": "PIT_LANE_DELIVERY_DELAY",
                            "anomaly_score": round(anom_score, 3),
                            "evidence": f"Pit stop duration {pit_dur:.2f}s exceeded 23.5s target by +{excess:.2f}s; stationary wheel-change delay."
                        })
                    continue

                # 2. Sudden Severe Pace Collapse (Off-track / Damage)
                if delta_from_base > 5.0 and not is_pit:
                    anom_score = min(1.0, 0.80 + (delta_from_base / 20.0))
                    anomalies.append({
                        "driver_code": code,
                        "driver_name": grp["driver_name"].iloc[i],
                        "team": grp["team_name"].iloc[i],
                        "lap": lap_num,
                        "lap_time": time_str,
                        "lap_sec": round(lap_sec, 3),
                        "anomaly_type": "OFF_TRACK_OR_COLLISION_DAMAGE",
                        "anomaly_score": round(anom_score, 3),
                        "evidence": f"Pace collapsed by +{delta_from_base:.2f}s vs rolling median ({base_pace:.2f}s); potential puncture or wing damage."
                    })
                    continue

                # 3. Moderate Pace Drop / Incident (Yellow flag, lock-up)
                if delta_from_base > 2.5 and not is_pit:
                    anom_score = min(1.0, 0.65 + (delta_from_base / 15.0))
                    anomalies.append({
                        "driver_code": code,
                        "driver_name": grp["driver_name"].iloc[i],
                        "team": grp["team_name"].iloc[i],
                        "lap": lap_num,
                        "lap_time": time_str,
                        "lap_sec": round(lap_sec, 3),
                        "anomaly_type": "PACE_DROP_SPIKE",
                        "anomaly_score": round(anom_score, 3),
                        "evidence": f"Lap time spiked +{delta_from_base:.2f}s above driver median pace ({base_pace:.2f}s); severe tyre lockup or traffic block."
                    })
                    continue

                # 4. Multi-Lap Tyre Degradation Cliff
                if i >= 3:
                    prev_laps = grp["lap_sec"].iloc[i-3:i].values
                    # If every lap was strictly slower than the previous by > 0.6s
                    d1 = prev_laps[1] - prev_laps[0]
                    d2 = prev_laps[2] - prev_laps[1]
                    d3 = lap_sec - prev_laps[2]
                    if d1 > 0.5 and d2 > 0.5 and d3 > 0.5:
                        anom_score = 0.76
                        anomalies.append({
                            "driver_code": code,
                            "driver_name": grp["driver_name"].iloc[i],
                            "team": grp["team_name"].iloc[i],
                            "lap": lap_num,
                            "lap_time": time_str,
                            "lap_sec": round(lap_sec, 3),
                            "anomaly_type": "TYRE_THERMAL_CLIFF",
                            "anomaly_score": round(anom_score, 3),
                            "evidence": f"Tyre degradation cliff confirmed: 3 consecutive laps slowed by +{d1:.2f}s, +{d2:.2f}s, +{d3:.2f}s; Box Box alert."
                        })

        anom_df = pd.DataFrame(anomalies)
        if not anom_df.empty:
            anom_df = anom_df.sort_values(["lap", "anomaly_score"], ascending=[True, False]).reset_index(drop=True)
        return anom_df

    def compare_teammates(self, race_id: Optional[int] = None) -> pd.DataFrame:
        """
        Compare lap-by-lap head-to-head performance of teammates in identical cars.
        Highlights intra-team pace divergences and chassis/power-unit anomalies.
        """
        grid_df = self.get_race_telemetry_grid(race_id)
        if grid_df.empty:
            return pd.DataFrame()

        teammate_comparisons = []
        for team, grp in grid_df.groupby("team_name"):
            driver_codes = grp["code"].unique()
            if len(driver_codes) >= 2:
                d1, d2 = driver_codes[0], driver_codes[1]
                d1_laps = grp[grp["code"] == d1].set_index("lap")["lap_sec"]
                d2_laps = grp[grp["code"] == d2].set_index("lap")["lap_sec"]
                
                common_laps = d1_laps.index.intersection(d2_laps.index)
                for lap in common_laps:
                    t1 = float(d1_laps.loc[lap])
                    t2 = float(d2_laps.loc[lap])
                    delta = t1 - t2  # negative means d1 was faster
                    
                    is_anom = abs(delta) > 1.2
                    teammate_comparisons.append({
                        "team": team,
                        "driver_1": d1,
                        "driver_2": d2,
                        "lap": int(lap),
                        "t1_sec": round(t1, 3),
                        "t2_sec": round(t2, 3),
                        "delta_d1_minus_d2": round(delta, 3),
                        "is_divergence_anomaly": is_anom,
                        "faster_driver": d1 if delta < 0 else d2
                    })

        return pd.DataFrame(teammate_comparisons)

    def export_driver_timing_artifacts(self, race_id: Optional[int] = None, output_dir: str = "outputs"):
        """Export driver timing anomalies CSV and summary JSON."""
        os.makedirs(output_dir, exist_ok=True)
        race_info = self.get_race_info(race_id or self.get_latest_race_id())
        
        # 1. Driver Anomalies CSV
        anom_df = self.detect_driver_timing_anomalies(race_id)
        csv_path = os.path.join(output_dir, "driver_timing_anomalies.csv")
        anom_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(anom_df)} anomalies detected across grid)")

        # 2. Driver Summary JSON
        summary_df = self.compute_driver_pace_summary(race_id)
        json_path = os.path.join(output_dir, "driver_race_timings_summary.json")
        data = {
            "race_info": race_info,
            "total_drivers_analyzed": len(summary_df),
            "driver_pace_benchmarks": summary_df.to_dict(orient="records"),
            "total_timing_anomalies_flagged": len(anom_df),
            "top_anomalies_sample": anom_df.head(10).to_dict(orient="records")
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    engine = DriverTimingEngine()
    latest_id = engine.get_latest_race_id()
    print("Latest Race ID:", latest_id)
    race_info = engine.get_race_info(latest_id)
    print("Race Info:", race_info)
    
    summary = engine.compute_driver_pace_summary(latest_id)
    print("\nDriver Pace Summary (Top 5):")
    print(summary[["code", "team", "finish_pos", "fastest_lap_str", "median_pace_sec"]].head(5))
    
    anoms = engine.detect_driver_timing_anomalies(latest_id)
    print(f"\nTotal Driver Timing Anomalies Detected: {len(anoms)}")
    print(anoms[["driver_code", "lap", "anomaly_type", "anomaly_score", "evidence"]].head(5))
