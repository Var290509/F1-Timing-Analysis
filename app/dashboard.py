"""
🏎️ F1 SENTINEL — REAL-TIME TELEMETRY ANOMALY DETECTION & HISTORICAL ENGINE (1950–2026)
Utkrishta 2026 AI/ML Hackathon | Challenge 2: Sentinel
Pit Wall Real-Time Telemetry Monitoring & 76-Season Historical Evolution Console
"""

import os
import sys
import time
import json
from typing import Dict, Any

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from src.data_loader import load_nab_dataset, get_available_channels
from src.feature_engine import extract_window_features, StreamingFeatureBuffer
from src.models.ensemble import HybridEnsembleDetector
from src.adaptive_threshold import AdaptiveThresholdManager
from src.evidence_generator import EvidenceGenerator
from src.historical_engine import F1HistoricalEngine, F1_ERAS
from src.driver_timing_engine import DriverTimingEngine


# Page Configuration
st.set_page_config(
    page_title="F1 Sentinel | Pit Wall & Era Evolution Console",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Formula 1 Carbon Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #12121A;
        color: #F3F3F3;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .f1-header {
        background: linear-gradient(90deg, #E10600 0%, #B00400 40%, #1E1E2D 100%);
        padding: 18px 24px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(225, 6, 0, 0.35);
    }
    .f1-title {
        font-size: 26px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: 1px;
    }
    .f1-subtitle {
        font-size: 13px;
        color: #E0E0E0;
        margin: 4px 0 0 0;
        font-weight: 500;
    }
    
    .alert-box {
        background-color: #2D0A0A;
        border-left: 6px solid #E10600;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(225, 6, 0, 0.4);
    }
    .alert-title {
        color: #FF3333;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .alert-evidence {
        color: #FFD4D4;
        font-size: 13px;
        font-family: monospace;
        line-height: 1.4;
    }
    
    .nominal-box {
        background-color: #0A201A;
        border-left: 6px solid #00D2BE;
        border-radius: 6px;
        padding: 12px 18px;
        margin-bottom: 20px;
    }
    .nominal-title {
        color: #00D2BE;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .nominal-text {
        color: #B0EADF;
        font-size: 12px;
        font-family: monospace;
    }

    .metric-card {
        background-color: #1A1A26;
        border: 1px solid #2E2E42;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 800;
        color: #FFFFFF;
    }
    .metric-lbl {
        font-size: 11px;
        color: #8E8EA8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .pillar-card {
        background-color: #1A1A26;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .pillar-title {
        font-size: 15px;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 6px;
    }
    .pillar-body {
        font-size: 12.5px;
        color: #D1D5DB;
        line-height: 1.5;
    }
    .pillar-tag {
        font-size: 11.5px;
        color: #34D399;
        font-weight: 600;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_prepare_channel(channel_key: str):
    """
    Cached data loading and ensemble model training for the selected telemetry stream.
    """
    df, meta, gt_windows = load_nab_dataset(channel_key)
    feats_df = extract_window_features(df["value"], window_size=30)
    X_all = feats_df.values
    series_all = df["value"]

    train_pts = min(2000, int(len(df) * 0.25))
    ensemble = HybridEnsembleDetector()
    ensemble.fit(X_all[:train_pts], series_all.iloc[:train_pts])

    model_scores = ensemble.score_all_models(X_all, series_all)
    ensemble_scores = model_scores["Hybrid_Ensemble"]

    thresh_mgr = AdaptiveThresholdManager(
        trailing_window=150,
        base_k=3.0,
        min_threshold=0.68,
        max_threshold=0.88,
        regime_detection=True
    )
    dynamic_threshs, binary_preds, regime_metas = thresh_mgr.compute_series(
        ensemble_scores,
        series_all.values
    )

    evidence_gen = EvidenceGenerator(channel_name=meta["channel_name"], unit=meta["unit"])
    evidence_list = []
    ae_mse = ensemble.ae_model.get_raw_reconstruction_error(X_all)

    for i in range(len(df)):
        ev = evidence_gen.generate(
            anomaly_score=float(ensemble_scores[i]),
            threshold=float(dynamic_threshs[i]),
            label="ANOMALOUS" if binary_preds[i] else "NORMAL",
            current_value=float(df["value"].iloc[i]),
            model_scores={k: float(v[i]) for k, v in model_scores.items()},
            feature_dict={
                "window_zscore": float(feats_df["window_zscore"].iloc[i]),
                "rate_of_change": float(feats_df["rate_of_change"].iloc[i]),
                "ae_reconstruction_error": float(ae_mse[i])
            },
            regime_meta=regime_metas[i]
        )
        evidence_list.append(ev)

    return {
        "df": df,
        "meta": meta,
        "gt_windows": gt_windows,
        "feats_df": feats_df,
        "model_scores": model_scores,
        "ensemble_scores": ensemble_scores,
        "dynamic_threshs": dynamic_threshs,
        "binary_preds": binary_preds,
        "evidence_list": evidence_list,
        "regime_metas": regime_metas
    }


@st.cache_resource
def get_historical_engine():
    return F1HistoricalEngine()


@st.cache_resource
def get_driver_engine():
    return DriverTimingEngine()


def main():
    channels = get_available_channels()
    hist_engine = get_historical_engine()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
        st.markdown("### 🎛️ Navigation & Control")
        app_mode = st.radio(
            "Dashboard View:",
            options=[
                "🏎️ Real-Time Telemetry Monitor",
                "⏱️ Multi-Driver Live Timing & Anomaly Matrix",
                "🏛️ 1950–2026 Historical Evolution & Current Game"
            ]
        )

        st.markdown("---")
        st.markdown("### 🛠️ Telemetry Channel")
        channel_names = {k: f"{v['telemetry_code']} - {v['channel_name']}" for k, v in channels.items()}
        selected_key = st.selectbox(
            "Live Telemetry Stream:",
            options=list(channels.keys()),
            format_func=lambda k: channel_names[k]
        )

        st.markdown("---")
        st.markdown("### ⏱️ Live Simulation Controls")

        if "stream_idx" not in st.session_state:
            st.session_state.stream_idx = 100
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False
        if "current_channel" not in st.session_state or st.session_state.current_channel != selected_key:
            st.session_state.current_channel = selected_key
            st.session_state.stream_idx = 100

        col_play, col_pause, col_reset = st.columns(3)
        with col_play:
            if st.button("▶ Play", use_container_width=True):
                st.session_state.is_playing = True
        with col_pause:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.is_playing = False
        with col_reset:
            if st.button("⏮ Reset", use_container_width=True):
                st.session_state.stream_idx = 100
                st.session_state.is_playing = False

        playback_speed = st.slider("Streaming Pace (steps/frame):", min_value=1, max_value=20, value=3)

        st.markdown("---")
        st.markdown("### 🎯 Jump to Racing Incident")
        data_bundle = load_and_prepare_channel(selected_key)
        gt_windows = data_bundle["gt_windows"]
        df = data_bundle["df"]

        if gt_windows:
            jump_options = [f"Incident {i+1}: {w[0].strftime('%b %d %H:%M')}" for i, w in enumerate(gt_windows)]
            selected_incident = st.selectbox("Select Anomaly Window:", options=list(range(len(jump_options))), format_func=lambda i: jump_options[i])
            if st.button("🚀 Jump to Incident", use_container_width=True):
                target_time = gt_windows[selected_incident][0]
                matching_indices = df[df["timestamp"] >= target_time].index
                if len(matching_indices) > 0:
                    st.session_state.stream_idx = max(50, matching_indices[0] - 30)
                    st.rerun()

        st.markdown("---")
        st.markdown(
            "**Utkrishta 2026 AI/ML Hackathon**  \n"
            "*Challenge 2: Sentinel (Adaptive Time-Series)*  \n"
            "Domain: Formula 1 Racing (1950–2026 Archive)"
        )

    # Load current channel data
    data_bundle = load_and_prepare_channel(selected_key)
    df = data_bundle["df"]
    meta = data_bundle["meta"]
    gt_windows = data_bundle["gt_windows"]
    scores = data_bundle["ensemble_scores"]
    thresholds = data_bundle["dynamic_threshs"]
    preds = data_bundle["binary_preds"]
    evidences = data_bundle["evidence_list"]
    model_scores = data_bundle["model_scores"]

    if st.session_state.is_playing:
        st.session_state.stream_idx = min(len(df) - 1, st.session_state.stream_idx + playback_speed)

    curr_idx = st.session_state.stream_idx
    curr_time = df["timestamp"].iloc[curr_idx]
    curr_val = df["value"].iloc[curr_idx]
    curr_score = scores[curr_idx]
    curr_thresh = thresholds[curr_idx]
    curr_pred = preds[curr_idx]
    curr_evidence = evidences[curr_idx]

    # --- VIEW 1: LIVE REAL-TIME TELEMETRY MONITOR ---
    if app_mode == "🏎️ Real-Time Telemetry Monitor":
        st.markdown(f"""
        <div class="f1-header">
            <h1 class="f1-title">🏎️ F1 SENTINEL — REAL-TIME TELEMETRY ANOMALY DETECTION</h1>
            <p class="f1-subtitle">PIT WALL MONITORING & TACTICAL ANOMALY DIAGNOSTIC CONSOLE | CHANNEL: {meta['telemetry_code']} - {meta['channel_name']}</p>
        </div>
        """, unsafe_allow_html=True)

        if curr_pred:
            st.markdown(f"""
            <div class="alert-box">
                <div class="alert-title">🚨 RED ALERT: TELEMETRY ANOMALY DETECTED ON CAR #16</div>
                <div class="alert-evidence">{curr_evidence}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="nominal-box">
                <div class="nominal-title">🟢 NOMINAL RACE TELEMETRY — ALL SYSTEMS HEALTHY</div>
                <div class="nominal-text">{curr_evidence}</div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #00D2BE;">{curr_val:.2f} {meta['unit']}</div>
                <div class="metric-lbl">Live Sensor Value</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            score_color = "#E10600" if curr_score >= curr_thresh else "#3B82F6"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {score_color};">{curr_score:.3f}</div>
                <div class="metric-lbl">Ensemble Anomaly Score</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFB800;">{curr_thresh:.3f}</div>
                <div class="metric-lbl">Adaptive Dynamic Threshold</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            status_text = "ANOMALOUS" if curr_pred else "NOMINAL"
            status_color = "#E10600" if curr_pred else "#10B981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {status_color};">{status_text}</div>
                <div class="metric-lbl">Sentinel Status</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            progress_pct = (curr_idx / len(df)) * 100.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFFFFF;">{progress_pct:.1f}%</div>
                <div class="metric-lbl">Session Ingested</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns([2, 1])
        window_pts = 180
        start_history = max(0, curr_idx - window_pts)
        slice_df = df.iloc[start_history: curr_idx + 1].copy()
        slice_df["score"] = scores[start_history: curr_idx + 1]
        slice_df["threshold"] = thresholds[start_history: curr_idx + 1]
        slice_df["flag"] = preds[start_history: curr_idx + 1]

        with col_chart1:
            st.markdown("#### 📈 Real-Time Telemetry Stream & Detection Flags")
            base = alt.Chart(slice_df).encode(x=alt.X("timestamp:T", title="Session Time"))
            line_sensor = base.mark_line(color="#00D2BE", strokeWidth=2.2).encode(
                y=alt.Y("value:Q", title=f"Sensor Value ({meta['unit']})", scale=alt.Scale(zero=False))
            )
            anom_points = base.filter("datum.flag == true").mark_point(
                color="#E10600",
                size=90,
                filled=True
            ).encode(y="value:Q")

            chart1 = (line_sensor + anom_points).properties(height=260).configure_view(
                strokeWidth=0
            ).configure_axis(
                gridColor="#2D2D3F",
                labelColor="#A0A0B0",
                titleColor="#FFFFFF"
            )
            st.altair_chart(chart1, use_container_width=True)

            st.markdown("#### ⚡ Anomaly Score vs. Dynamic Adaptive Threshold")
            line_score = base.mark_line(color="#3B82F6", strokeWidth=2.0).encode(
                y=alt.Y("score:Q", title="Anomaly Score / Threshold", scale=alt.Scale(domain=[0, 1]))
            )
            line_thresh = base.mark_line(color="#EF4444", strokeDash=[4, 4], strokeWidth=2.0).encode(
                y="threshold:Q"
            )
            chart2 = (line_score + line_thresh).properties(height=220).configure_view(
                strokeWidth=0
            ).configure_axis(
                gridColor="#2D2D3F",
                labelColor="#A0A0B0",
                titleColor="#FFFFFF"
            )
            st.altair_chart(chart2, use_container_width=True)

        with col_chart2:
            st.markdown("#### 🤖 Model Consensus Breakdown")
            models_display = {
                "Z-Score Statistical": float(model_scores["ZScore_Detector"][curr_idx]),
                "Isolation Forest": float(model_scores["Isolation_Forest"][curr_idx]),
                "Local Outlier Factor": float(model_scores["Local_Outlier_Factor"][curr_idx]),
                "One-Class SVM": float(model_scores["OneClass_SVM"][curr_idx]),
                "PyTorch Autoencoder": float(model_scores["PyTorch_Autoencoder"][curr_idx]),
                "Hybrid Ensemble": float(curr_score)
            }

            breakdown_df = pd.DataFrame({
                "Detector": list(models_display.keys()),
                "Score": list(models_display.values())
            })

            bar_chart = alt.Chart(breakdown_df).mark_bar().encode(
                y=alt.Y("Detector:N", sort=None, title=""),
                x=alt.X("Score:Q", title="Anomaly Confidence", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(
                    alt.datum.Score >= curr_thresh,
                    alt.value("#E10600"),
                    alt.value("#3B82F6")
                )
            ).properties(height=260).configure_view(strokeWidth=0).configure_axis(
                gridColor="#2D2D3F",
                labelColor="#FFFFFF",
                titleColor="#FFFFFF"
            )
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("#### 🏛️ Live vs. Historical Era Benchmark")
            hist_bench = hist_engine.benchmark_current_telemetry(meta["channel_name"], curr_val, curr_score)
            st.info(
                f"**Current Game Baseline**: `{hist_bench['current_game_attrition_baseline']}`  \n"
                f"**1980s Turbo Baseline**: `{hist_bench['historical_turbo_attrition_baseline']}`  \n"
                f"**Tactical Verdict**: {hist_bench['tactical_verdict']}"
            )

        # Recent Incident Log
        st.markdown("---")
        st.markdown("### 📜 Past Detected Telemetry Incidents (Event Log)")
        past_preds = preds[:curr_idx + 1]
        if np.any(past_preds):
            incident_indices = np.where(past_preds)[0]
            events_start = []
            for i, idx in enumerate(incident_indices):
                if i == 0 or idx != incident_indices[i - 1] + 1:
                    events_start.append(idx)

            recent_events = events_start[-8:]
            event_rows = []
            for ev_idx in reversed(recent_events):
                event_rows.append({
                    "Timestamp": str(df["timestamp"].iloc[ev_idx]),
                    "Value": f"{df['value'].iloc[ev_idx]:.2f} {meta['unit']}",
                    "Anomaly Score": f"{scores[ev_idx]:.3f}",
                    "Threshold": f"{thresholds[ev_idx]:.3f}",
                    "Diagnostic Evidence": evidences[ev_idx]
                })
            st.dataframe(pd.DataFrame(event_rows), use_container_width=True)
        else:
            st.markdown("*No telemetry anomalies flagged yet in current session playback.*")

        if st.session_state.is_playing and curr_idx < len(df) - 1:
            time.sleep(0.08)
            st.rerun()

    # --- VIEW 2: MULTI-DRIVER LIVE TIMING & ANOMALY MATRIX ---
    elif app_mode == "⏱️ Multi-Driver Live Timing & Anomaly Matrix":
        driver_engine = get_driver_engine()
        latest_id = driver_engine.get_latest_race_id()
        race_info = driver_engine.get_race_info(latest_id)
        grid_df = driver_engine.get_race_telemetry_grid(latest_id)
        driver_summary = driver_engine.compute_driver_pace_summary(latest_id)
        driver_anoms = driver_engine.detect_driver_timing_anomalies(latest_id)

        st.markdown(f"""
        <div class="f1-header">
            <h1 class="f1-title">⏱️ MULTI-DRIVER LIVE TIMING & RACE ANOMALY MATRIX</h1>
            <p class="f1-subtitle">SESSION: {race_info['year']} {race_info['name'].upper()} | ROUND {race_info['round']} | {len(driver_summary)} DRIVERS ANALYZED</p>
        </div>
        """, unsafe_allow_html=True)

        # KPI Row
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            winner = driver_summary.iloc[0]["code"] if not driver_summary.empty else "P1"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFB800;">{winner}</div>
                <div class="metric-lbl">Race Winner (P1)</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            fastest_row = driver_summary.sort_values("fastest_lap_sec").iloc[0] if not driver_summary.empty else None
            fastest_str = f"{fastest_row['fastest_lap_str']} ({fastest_row['code']})" if fastest_row is not None else "1:37.9"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #00D2BE; font-size: 19px;">{fastest_str}</div>
                <div class="metric-lbl">Fastest Race Lap</div>
            </div>
            """, unsafe_allow_html=True)
        with d3:
            median_p = f"{driver_summary['median_pace_sec'].median():.2f}s" if not driver_summary.empty else "100.4s"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #3B82F6;">{median_p}</div>
                <div class="metric-lbl">Grid Median Pace</div>
            </div>
            """, unsafe_allow_html=True)
        with d4:
            total_anom = len(driver_anoms)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #E10600;">{total_anom}</div>
                <div class="metric-lbl">Grid Timing Anomalies Flagged</div>
            </div>
            """, unsafe_allow_html=True)
        with d5:
            max_laps = grid_df["lap"].max() if not grid_df.empty else 56
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFFFFF;">{max_laps} Laps</div>
                <div class="metric-lbl">Total Race Distance</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for Timing Analysis
        tab_board, tab_pace, tab_anom, tab_team = st.tabs([
            "🏆 Official F1 Timing Leaderboard",
            "📈 Multi-Driver Lap Pace Comparison",
            "🚨 Grid Timing Anomalies Incident Log",
            "⚔️ Intra-Team Head-to-Head Battle"
        ])

        with tab_board:
            st.markdown("### 🏆 Race Classification & Driver Timing Summary")
            st.markdown("Pace metrics, fastest laps, median race pace, and pit stop counts for every driver:")
            st.dataframe(driver_summary, use_container_width=True)

        with tab_pace:
            st.markdown("### 📈 Driver Lap-by-Lap Pace & Stint Degradation")
            driver_options = sorted(grid_df["code"].unique())
            default_selection = [c for c in ["ANT", "LEC", "VER", "HAM", "NOR", "PIA"] if c in driver_options]
            selected_drivers = st.multiselect(
                "Select Drivers to Compare:",
                options=driver_options,
                default=default_selection
            )

            if selected_drivers:
                sub_compare = grid_df[grid_df["code"].isin(selected_drivers)].copy()
                chart_pace = alt.Chart(sub_compare).mark_line(strokeWidth=2.2).encode(
                    x=alt.X("lap:Q", title="Lap Number"),
                    y=alt.Y("lap_sec:Q", title="Lap Time (seconds)", scale=alt.Scale(domain=[95, 125])),
                    color=alt.Color("code:N", title="Driver Code")
                ).properties(height=380).configure_view(strokeWidth=0).configure_axis(
                    gridColor="#2D2D3F",
                    labelColor="#FFFFFF",
                    titleColor="#FFFFFF"
                )
                st.altair_chart(chart_pace, use_container_width=True)

            plot_pace_path = "outputs/plots/driver_lap_pace_comparison.png"
            if os.path.exists(plot_pace_path):
                st.image(plot_pace_path, use_container_width=True)

        with tab_anom:
            st.markdown(f"### 🚨 Grid Timing Anomalies ({len(driver_anoms)} Incidents Detected)")
            st.markdown(
                "Real-time timing anomalies across all drivers: sudden pace spikes, tyre degradation cliffs, "
                "and pit lane transit delays with explainable diagnostic evidence."
            )
            st.dataframe(driver_anoms, use_container_width=True)

            plot_scatter_path = "outputs/plots/driver_timing_anomalies_scatter.png"
            if os.path.exists(plot_scatter_path):
                st.image(plot_scatter_path, use_container_width=True)

        with tab_team:
            st.markdown("### ⚔️ Intra-Team Head-to-Head Performance Deltas")
            st.markdown("Lap-by-lap comparison of teammates driving identical machinery to detect chassis or power-unit performance divergence:")
            teammates_df = driver_engine.compare_teammates(latest_id)
            if not teammates_df.empty:
                teams_list = sorted(teammates_df["team"].unique())
                selected_team = st.selectbox("Select Constructor Pairing:", options=teams_list)
                team_sub = teammates_df[teammates_df["team"] == selected_team].sort_values("lap")
                
                chart_team = alt.Chart(team_sub).mark_line(color="#E10600", strokeWidth=2.0).encode(
                    x=alt.X("lap:Q", title="Lap Number"),
                    y=alt.Y("delta_d1_minus_d2:Q", title=f"Lap Delta: {team_sub['driver_1'].iloc[0]} vs {team_sub['driver_2'].iloc[0]} (s)", scale=alt.Scale(domain=[-5, 5]))
                ).properties(height=320).configure_view(strokeWidth=0).configure_axis(
                    gridColor="#2D2D3F",
                    labelColor="#FFFFFF",
                    titleColor="#FFFFFF"
                )
                st.altair_chart(chart_team, use_container_width=True)

            plot_team_path = "outputs/plots/teammate_pace_deltas.png"
            if os.path.exists(plot_team_path):
                st.image(plot_team_path, use_container_width=True)

    # --- VIEW 3: 1950–2026 HISTORICAL EVOLUTION & CURRENT GAME TRANSFORMATION ---
    else:
        st.markdown("""
        <div class="f1-header">
            <h1 class="f1-title">🏛️ FORMULA 1 HISTORICAL EVOLUTION (1950–2026)</h1>
            <p class="f1-subtitle">INTEGRATING 76 SEASONS, 1,172 GRANDS PRIX, AND 27,600+ RACES TO BENCHMARK THE CURRENT RUNNING GAME</p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFFFFF;">1,172</div>
                <div class="metric-lbl">Grands Prix (1950–2026)</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-val" style="color: #00D2BE;">8.9%</div>
                <div class="metric-lbl">Current Mech DNF (vs 47.5% in 1980s)</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFB800;">52.5%</div>
                <div class="metric-lbl">Thermal Efficiency (vs 18% in 1950)</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-val" style="color: #3B82F6;">1.80s</div>
                <div class="metric-lbl">Record Pit Stop (vs 60s in 1950)</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-val" style="color: #E10600;">798 kg</div>
                <div class="metric-lbl">Current Minimum Car Mass</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed analysis
        tab_rel, tab_pit, tab_pillars, tab_eras = st.tabs([
            "📉 Reliability & DNF Evolution",
            "⏱️ Pit Stop Precision Evolution",
            "🚀 What Changed in Current Game?",
            "📚 7 Eras of Formula 1 (1950–2026)"
        ])

        with tab_rel:
            st.markdown("### 📉 The Mechanical Reliability Revolution")
            st.markdown(
                "How Formula 1 transitioned from a sport where **half the cars broke down** every Sunday "
                "to an aerospace-grade competition where an engine failure is a catastrophic rarity."
            )
            rel_df = hist_engine.compute_reliability_evolution()
            if not rel_df.empty:
                st.dataframe(rel_df, use_container_width=True)

            plot_path = "outputs/plots/historical_reliability_evolution.png"
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)

        with tab_pit:
            st.markdown("### ⏱️ Pit Stop Precision & Time Compression")
            st.markdown(
                "From manual copper wheel-mallet changes in the 1950s (40–80s), to dangerous fuel-rig stops (1994–2009, 8–12s), "
                "to modern sub-2-second synchronous tyre choreography."
            )
            pit_path = "outputs/plots/historical_pitstop_evolution.png"
            if os.path.exists(plot_path):
                st.image(pit_path, use_container_width=True)

        with tab_pillars:
            st.markdown("### 🚀 The 6 Transformative Pillars of the Current Running Game (2022–2026)")
            st.markdown(
                "What makes modern Formula 1 fundamentally distinct from any prior era in racing history?"
            )
            pillars = hist_engine.get_current_game_transformation_pillars()
            for p in pillars:
                st.markdown(f"""
                <div class="pillar-card">
                    <div class="pillar-title">{p['pillar']}</div>
                    <div class="pillar-body"><b>Historical Baseline:</b> {p['historical_state']}</div>
                    <div class="pillar-body" style="margin-top: 4px;"><b>Current Game Reality:</b> {p['current_game_state']}</div>
                    <div class="pillar-tag">⚡ AI Anomaly Detection Consequence: {p['telemetry_implication']}</div>
                </div>
                """, unsafe_allow_html=True)

            trans_plot = "outputs/plots/current_vs_historical_changes.png"
            if os.path.exists(trans_plot):
                st.markdown("#### 📊 Quantitative Pillar Comparison")
                st.image(trans_plot, use_container_width=True)

        with tab_eras:
            st.markdown("### 📚 The 7 Distinct Eras of Formula 1 World Championship Racing")
            for era_key, era_meta in F1_ERAS.items():
                with st.expander(f"🏁 {era_meta['name']} ({era_meta['years'][0]}–{era_meta['years'][1]})"):
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.write(f"**Engine Architecture:** {era_meta['engine_spec']}")
                        st.write(f"**Minimum Vehicle Mass:** {era_meta['weight_min_kg']} kg")
                        st.write(f"**Thermal Efficiency:** ~{era_meta['thermal_efficiency_pct']}%")
                    with c_b:
                        st.write(f"**Aerodynamic Design:** {era_meta['aero_type']}")
                        st.write(f"**Key Era Dynamics:** {era_meta['key_characteristics']}")


if __name__ == "__main__":
    main()
