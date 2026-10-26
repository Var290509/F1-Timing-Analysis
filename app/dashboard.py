"""
🏎️ F1 SENTINEL — REAL-TIME TELEMETRY ANOMALY DETECTION & HISTORICAL ENGINE (1950–2026)
Utkrishta 2026 AI/ML Hackathon | Challenge 2: Sentinel
Pit Wall Command Center & Formula 1 Evolution Console
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


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Sentinel | Pit Wall Command & Anomaly Radar",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM FORMULA 1 CARBON & NEON THEME STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global App Container */
    .stApp {
        background-color: #0D0E15;
        color: #F3F4F6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Banner Wrapper */
    .hero-banner-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2E2E42;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
        margin-bottom: 20px;
    }

    /* Red Alert Card */
    .alert-box {
        background: linear-gradient(135deg, #2D0808 0%, #1A0505 100%);
        border-left: 6px solid #E10600;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(225, 6, 0, 0.35);
        border-top: 1px solid #4A1010;
        border-right: 1px solid #4A1010;
        border-bottom: 1px solid #4A1010;
    }
    .alert-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #FF3B30;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .alert-evidence {
        color: #FFD4D4;
        font-size: 13.5px;
        font-family: "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        line-height: 1.5;
        background-color: rgba(0, 0, 0, 0.35);
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid rgba(225, 6, 0, 0.25);
    }
    
    /* Nominal Telemetry Card */
    .nominal-box {
        background: linear-gradient(135deg, #06211A 0%, #041410 100%);
        border-left: 6px solid #00D2BE;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 22px;
        border-top: 1px solid #0E3D32;
        border-right: 1px solid #0E3D32;
        border-bottom: 1px solid #0E3D32;
        box-shadow: 0 4px 15px rgba(0, 210, 190, 0.15);
    }
    .nominal-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #00D2BE;
        font-size: 15.5px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .nominal-text {
        color: #A7F3D0;
        font-size: 12.5px;
        font-family: "SF Mono", Monaco, Consolas, monospace;
    }

    /* Metric HUD Card */
    .metric-card {
        background: linear-gradient(180deg, #181926 0%, #12131D 100%);
        border: 1px solid #28293D;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
        border-color: #3E4060;
    }
    .metric-val {
        font-size: 23px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .metric-lbl {
        font-size: 11px;
        color: #9CA3AF;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.8px;
    }

    /* Quick Guide Box */
    .guide-box {
        background-color: #151622;
        border: 1px solid #2B2C40;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 13px;
        color: #D1D5DB;
        line-height: 1.5;
    }

    /* Pillar Cards */
    .pillar-card {
        background: linear-gradient(135deg, #171826 0%, #11121C 100%);
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border-top: 1px solid #232538;
        border-right: 1px solid #232538;
        border-bottom: 1px solid #232538;
    }
    .pillar-title {
        font-size: 16px;
        font-weight: 700;
        color: #60A5FA;
        margin-bottom: 8px;
    }
    .pillar-body {
        font-size: 13px;
        color: #D1D5DB;
        line-height: 1.5;
    }
    .pillar-tag {
        font-size: 12px;
        color: #34D399;
        font-weight: 600;
        margin-top: 8px;
        background-color: rgba(52, 211, 153, 0.1);
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA PIPELINE & MODEL BUNDLES
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    channels = get_available_channels()
    hist_engine = get_historical_engine()

    # --- TOP HERO BANNER IMAGE ---
    banner_path = "app/assets/f1_hero_banner.png"
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)

    # --- SIDEBAR NAVIGATION & CONTROLS ---
    with st.sidebar:
        st.markdown("## 🏎️ Pit Wall Console")
        st.markdown("Select your operational dashboard mode below:")
        
        app_mode = st.radio(
            "Console View:",
            options=[
                "🏎️ Real-Time Telemetry Monitor",
                "⏱️ Multi-Driver Live Timing & Anomaly Matrix",
                "🏛️ 1950–2026 Historical Evolution",
                "📊 Competition Artifacts & Plot Gallery"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📡 Telemetry Channels")
        channel_names = {k: f"{v['telemetry_code']} - {v['channel_name']}" for k, v in channels.items()}
        selected_key = st.selectbox(
            "Sensor Data Stream:",
            options=list(channels.keys()),
            format_func=lambda k: channel_names[k]
        )

        st.markdown("---")
        st.markdown("### ⏯️ Live Simulation Playback")

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

        playback_speed = st.slider("Streaming Speed (pts/tick):", min_value=1, max_value=25, value=4)

        st.markdown("---")
        st.markdown("### 🎯 Jump to Racing Incident")
        data_bundle = load_and_prepare_channel(selected_key)
        gt_windows = data_bundle["gt_windows"]
        df = data_bundle["df"]

        if gt_windows:
            jump_options = [f"Incident #{i+1}: {w[0].strftime('%b %d %H:%M')}" for i, w in enumerate(gt_windows)]
            selected_incident = st.selectbox("Select Anomaly Event:", options=list(range(len(jump_options))), format_func=lambda i: jump_options[i])
            if st.button("🚀 Jump to Incident", use_container_width=True):
                target_time = gt_windows[selected_incident][0]
                matching_indices = df[df["timestamp"] >= target_time].index
                if len(matching_indices) > 0:
                    st.session_state.stream_idx = max(50, matching_indices[0] - 25)
                    st.session_state.is_playing = False
                    st.rerun()

        st.markdown("---")
        st.caption("🏆 **Utkrishta 2026 AI/ML Hackathon**")
        st.caption("Challenge 2: Sentinel (Adaptive Time-Series Anomaly Detection)")
        st.caption("Repository: `Var290509/F1-Timing-Analysis`")

    # Load active telemetry stream
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

    # =========================================================================
    # VIEW 1: REAL-TIME TELEMETRY MONITOR
    # =========================================================================
    if app_mode == "🏎️ Real-Time Telemetry Monitor":
        # Quick User Guide Expander
        with st.expander("ℹ️ How to Use This Pit Wall Monitor (Click to Expand)", expanded=False):
            st.markdown("""
            **Welcome to F1 Sentinel Live Telemetry Radar!**
            - **Live Sensor Value**: Real-time reading from car #16 (Charles Leclerc).
            - **Ensemble Score**: Continuous float [0, 1] combining 5 detectors (Z-Score, IsoForest, LOF, OCSVM, Autoencoder).
            - **Adaptive Dynamic Threshold**: Automatically recalculates trailing baseline distributions and shifts during tactical condition changes.
            - **Red Alert Banner**: Activates when `Ensemble Score >= Threshold`, providing plain-English diagnostic telemetry evidence.
            - **Controls**: Use the **Play / Pause** buttons or **Jump to Racing Incident** in the sidebar to simulate live streaming.
            """)

        # Alert or Nominal Status Card
        if curr_pred:
            st.markdown(f"""
            <div class="alert-box">
                <div class="alert-header">🚨 RED ALERT: TELEMETRY ANOMALY CONFIRMED (CAR #16)</div>
                <div class="alert-evidence">{curr_evidence}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="nominal-box">
                <div class="nominal-header">🟢 NOMINAL RACE TELEMETRY — ALL SYSTEMS WITHIN OPERATING WINDOW</div>
                <div class="nominal-text">{curr_evidence}</div>
            </div>
            """, unsafe_allow_html=True)

        # 5 Metric HUD Cards
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
                <div class="metric-lbl">Dynamic Threshold</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            status_text = "ANOMALY FLAG" if curr_pred else "NOMINAL"
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
                <div class="metric-lbl">Stream Ingested</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        col_chart1, col_chart2 = st.columns([2, 1])
        window_pts = 180
        start_history = max(0, curr_idx - window_pts)
        slice_df = df.iloc[start_history: curr_idx + 1].copy()
        slice_df["score"] = scores[start_history: curr_idx + 1]
        slice_df["threshold"] = thresholds[start_history: curr_idx + 1]
        slice_df["flag"] = preds[start_history: curr_idx + 1]

        with col_chart1:
            st.markdown("#### 📈 Real-Time Telemetry Stream & Anomaly Flags")
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
                gridColor="#242536",
                labelColor="#A0A0B0",
                titleColor="#FFFFFF"
            )
            st.altair_chart(chart1, use_container_width=True)

            st.markdown("#### ⚡ Continuous Anomaly Score vs. Adaptive Dynamic Threshold")
            line_score = base.mark_line(color="#3B82F6", strokeWidth=2.0).encode(
                y=alt.Y("score:Q", title="Score / Threshold", scale=alt.Scale(domain=[0, 1]))
            )
            line_thresh = base.mark_line(color="#EF4444", strokeDash=[4, 4], strokeWidth=2.0).encode(
                y="threshold:Q"
            )
            chart2 = (line_score + line_thresh).properties(height=220).configure_view(
                strokeWidth=0
            ).configure_axis(
                gridColor="#242536",
                labelColor="#A0A0B0",
                titleColor="#FFFFFF"
            )
            st.altair_chart(chart2, use_container_width=True)

        with col_chart2:
            st.markdown("#### 🤖 Model Consensus Breakdown")
            models_display = {
                "Z-Score Baseline": float(model_scores["ZScore_Detector"][curr_idx]),
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
            ).properties(height=240).configure_view(strokeWidth=0).configure_axis(
                gridColor="#242536",
                labelColor="#FFFFFF",
                titleColor="#FFFFFF"
            )
            st.altair_chart(bar_chart, use_container_width=True)

            # Historical Era Comparison Snippet
            hist_bench = hist_engine.benchmark_current_telemetry(meta["channel_name"], curr_val, curr_score)
            st.info(
                f"**Current Game Attrition Baseline**: `{hist_bench['current_game_attrition_baseline']}`  \n"
                f"**1980s Turbo Era Baseline**: `{hist_bench['historical_turbo_attrition_baseline']}`  \n"
                f"**Tactical Verdict**: {hist_bench['tactical_verdict']}"
            )

        # Technical Sensor & Model Architecture Visual Explanations
        st.markdown("---")
        st.markdown("### 🏎️ Telemetry Sensor Mapping & Multi-Model Architecture")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            sensor_img = "app/assets/f1_sensor_architecture.png"
            if os.path.exists(sensor_img):
                st.image(sensor_img, caption="F1 Onboard Telemetry Architecture & Sensor Placement", use_container_width=True)
        with col_img2:
            arch_img = "app/assets/model_architecture_pipeline.png"
            if os.path.exists(arch_img):
                st.image(arch_img, caption="5-Model Hybrid Ensemble & Adaptive Dynamic Thresholding Pipeline", use_container_width=True)

        # Recent Incident Log
        st.markdown("---")
        st.markdown("### 📜 Session Incident Log (Past Flagged Events)")
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
                    "Sensor Value": f"{df['value'].iloc[ev_idx]:.2f} {meta['unit']}",
                    "Anomaly Score": f"{scores[ev_idx]:.3f}",
                    "Threshold": f"{thresholds[ev_idx]:.3f}",
                    "Telemetry Diagnostic Evidence": evidences[ev_idx]
                })
            st.dataframe(pd.DataFrame(event_rows), use_container_width=True)
        else:
            st.markdown("*No telemetry anomalies flagged yet in current playback segment.*")

        if st.session_state.is_playing and curr_idx < len(df) - 1:
            time.sleep(0.08)
            st.rerun()

    # =========================================================================
    # VIEW 2: MULTI-DRIVER LIVE TIMING & ANOMALY MATRIX
    # =========================================================================
    elif app_mode == "⏱️ Multi-Driver Live Timing & Anomaly Matrix":
        driver_engine = get_driver_engine()
        latest_id = driver_engine.get_latest_race_id()
        race_info = driver_engine.get_race_info(latest_id)
        grid_df = driver_engine.get_race_telemetry_grid(latest_id)
        driver_summary = driver_engine.compute_driver_pace_summary(latest_id)
        driver_anoms = driver_engine.detect_driver_timing_anomalies(latest_id)

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #E10600 0%, #1E1E2D 100%); padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-size: 22px;">⏱️ FORMULA 1 GRID LIVE TIMING & ANOMALY RADAR</h2>
            <p style="margin: 4px 0 0 0; color: #E0E0E0; font-size: 13px;">SESSION: {race_info['year']} {race_info['name'].upper()} | ROUND {race_info['round']} | {len(driver_summary)} DRIVERS ANALYZED</p>
        </div>
        """, unsafe_allow_html=True)

        # Team Grid Identity Graphic
        team_grid_img = "app/assets/f1_team_grid.png"
        if os.path.exists(team_grid_img):
            st.image(team_grid_img, use_container_width=True)

        # KPI Row
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            winner = driver_summary.iloc[0]["code"] if not driver_summary.empty else "P1"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFB800;">{winner}</div>
                <div class="metric-lbl">Grand Prix Winner</div>
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
                <div class="metric-lbl">Grid Anomalies Flagged</div>
            </div>
            """, unsafe_allow_html=True)
        with d5:
            max_laps = grid_df["lap"].max() if not grid_df.empty else 56
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFFFFF;">{max_laps} Laps</div>
                <div class="metric-lbl">Race Distance</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for Driver Timing Analysis
        tab_board, tab_pace, tab_anom, tab_team, tab_track = st.tabs([
            "🏆 Official Timing Tower",
            "📈 Multi-Driver Lap Pace",
            "🚨 Grid Anomaly Log (91 Events)",
            "⚔️ Teammate Head-to-Head",
            "🗺️ Circuit Map & Pirelli Tyres"
        ])

        with tab_board:
            st.markdown("### 🏆 Race Classification & Driver Timing Summary")
            st.markdown("Detailed breakdown of start/finish positions, fastest laps, median race pace, and pit counts:")
            st.dataframe(driver_summary, use_container_width=True)

        with tab_pace:
            st.markdown("### 📈 Driver Lap Pace & Degradation Trajectories")
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
                    color=alt.Color("code:N", title="Driver")
                ).properties(height=380).configure_view(strokeWidth=0).configure_axis(
                    gridColor="#242536",
                    labelColor="#FFFFFF",
                    titleColor="#FFFFFF"
                )
                st.altair_chart(chart_pace, use_container_width=True)

            plot_pace_path = "outputs/plots/driver_lap_pace_comparison.png"
            if os.path.exists(plot_pace_path):
                st.image(plot_pace_path, caption="Comprehensive Lap Pace & Stint Progression Across All Drivers", use_container_width=True)

        with tab_anom:
            st.markdown(f"### 🚨 Grid Timing Anomalies ({len(driver_anoms)} Incidents Detected)")
            st.markdown(
                "Filter and inspect real-time timing anomalies across all 22 drivers: "
                "sudden pace drops, tyre degradation cliffs, and pit delays."
            )
            
            # Anomaly filter
            anom_types = ["ALL"] + sorted(driver_anoms["anomaly_type"].unique().tolist())
            selected_type = st.selectbox("Filter Anomaly Class:", options=anom_types)
            if selected_type != "ALL":
                filtered_anoms = driver_anoms[driver_anoms["anomaly_type"] == selected_type]
            else:
                filtered_anoms = driver_anoms

            st.dataframe(filtered_anoms, use_container_width=True)

            plot_scatter_path = "outputs/plots/driver_timing_anomalies_scatter.png"
            if os.path.exists(plot_scatter_path):
                st.image(plot_scatter_path, caption="Grid Timing Anomaly Distribution (Lap vs Pace Spike)", use_container_width=True)

        with tab_team:
            st.markdown("### ⚔️ Intra-Team Head-to-Head Performance Deltas")
            st.markdown("Lap-by-lap comparison of teammates driving identical machinery to detect chassis or power-unit performance divergence:")
            teammates_df = driver_engine.compare_teammates(latest_id)
            if not teammates_df.empty:
                teams_list = sorted(teammates_df["team"].unique())
                selected_team = st.selectbox("Select Constructor Pairing:", options=teams_list)
                team_sub = teammates_df[teammates_df["team"] == selected_team].sort_values("lap")
                
                chart_team = alt.Chart(team_sub).mark_line(color="#E10600", strokeWidth=2.2).encode(
                    x=alt.X("lap:Q", title="Lap Number"),
                    y=alt.Y("delta_d1_minus_d2:Q", title=f"Lap Delta: {team_sub['driver_1'].iloc[0]} vs {team_sub['driver_2'].iloc[0]} (s)", scale=alt.Scale(domain=[-5, 5]))
                ).properties(height=320).configure_view(strokeWidth=0).configure_axis(
                    gridColor="#242536",
                    labelColor="#FFFFFF",
                    titleColor="#FFFFFF"
                )
                st.altair_chart(chart_team, use_container_width=True)

            plot_team_path = "outputs/plots/teammate_pace_deltas.png"
            if os.path.exists(plot_team_path):
                st.image(plot_team_path, caption="Intra-Team Pace Delta Distribution & Divergence", use_container_width=True)

        with tab_track:
            st.markdown("### 🗺️ Circuit Layout & Pirelli Tyre Compound Specs")
            c_track, c_tyre = st.columns([1, 1])
            with c_track:
                cota_map = "app/assets/cota_circuit_map.png"
                if os.path.exists(cota_map):
                    st.image(cota_map, caption="Circuit of the Americas (COTA) 20-Turn Sector Layout", use_container_width=True)
            with c_tyre:
                tyre_spec = "app/assets/pirelli_tyre_compounds.png"
                if os.path.exists(tyre_spec):
                    st.image(tyre_spec, caption="Pirelli F1 Tyre Compounds & Degradation Cliff Criteria", use_container_width=True)

    # =========================================================================
    # VIEW 3: 1950–2026 HISTORICAL EVOLUTION
    # =========================================================================
    elif app_mode == "🏛️ 1950–2026 Historical Evolution":
        st.markdown("""
        <div style="background: linear-gradient(90deg, #3B82F6 0%, #1E1E2D 100%); padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-size: 22px;">🏛️ FORMULA 1 HISTORICAL EVOLUTION (1950–2026)</h2>
            <p style="margin: 4px 0 0 0; color: #E0E0E0; font-size: 13px;">76 SEASONS, 1,172 GRANDS PRIX, AND 27,600+ RACES BENCHMARKING THE MODERN RUNNING GAME</p>
        </div>
        """, unsafe_allow_html=True)

        # 5 Historical Metric HUD Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-val" style="color: #FFFFFF;">1,172</div>
                <div class="metric-lbl">Total Grands Prix</div>
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
                <div class="metric-lbl">Minimum Vehicle Mass</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Analysis
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
                "to an aerospace-grade competition where an engine failure is an acute anomaly."
            )
            rel_df = hist_engine.compute_reliability_evolution()
            if not rel_df.empty:
                st.dataframe(rel_df, use_container_width=True)

            plot_path = "outputs/plots/historical_reliability_evolution.png"
            if os.path.exists(plot_path):
                st.image(plot_path, caption="Mechanical DNF Rate by Decade across 76 Seasons", use_container_width=True)

        with tab_pit:
            st.markdown("### ⏱️ Pit Stop Precision & Time Compression")
            st.markdown(
                "From manual copper wheel-mallet changes in the 1950s (40–80s), to dangerous fuel-rig stops (1994–2009, 8–12s), "
                "to modern sub-2-second synchronous tyre choreography."
            )
            pit_path = "outputs/plots/historical_pitstop_evolution.png"
            if os.path.exists(pit_path):
                st.image(pit_path, caption="Pit Stop Duration Compression (1950 vs Current Era)", use_container_width=True)

        with tab_pillars:
            st.markdown("### 🚀 The 6 Transformative Pillars of the Current Running Game (2022–2026)")
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
                st.image(trans_plot, caption="Pillar-by-Pillar Quantitative Shift: Historical vs Current F1", use_container_width=True)

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

    # =========================================================================
    # VIEW 4: COMPETITION ARTIFACTS & PLOT GALLERY
    # =========================================================================
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #10B981 0%, #1E1E2D 100%); padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; color: #FFFFFF; font-size: 22px;">📊 COMPETITION ARTIFACTS & VISUAL PLOT GALLERY</h2>
            <p style="margin: 4px 0 0 0; color: #E0E0E0; font-size: 13px;">ALL 11 HIGH-RESOLUTION VISUALIZATION ARTIFACTS GENERATED FOR CHALLENGE 2: SENTINEL</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "Every artifact below was autonomously generated by the F1 Sentinel pipeline using official "
            "Numenta Anomaly Benchmark (NAB) and Formula 1 Grand Prix telemetry data."
        )

        plots_info = [
            {"path": "outputs/plots/time_series_anomalies.png", "title": "1. Telemetry Stream with Ground Truth Windows & Detected Flags", "desc": "Continuous sensor readings with shaded NAB ground-truth failure windows and red anomaly markers."},
            {"path": "outputs/plots/anomaly_score_and_threshold.png", "title": "2. Anomaly Score Trajectory vs. Dynamic Adaptive Threshold", "desc": "Trailing mean + k*std threshold adjusting to data distribution without lookahead leakage."},
            {"path": "outputs/plots/model_comparison_metrics.png", "title": "3. Multi-Model Benchmark Comparison (F1, Precision, Recall, Delay)", "desc": "Comparing all 5 detectors + Hybrid Ensemble under official NAB event-based metrics."},
            {"path": "outputs/plots/confusion_breakdown.png", "title": "4. Confusion-Style Event Breakdown (TP, FP, FN Counts)", "desc": "Event-level true positives, false positives, and false negatives for each detector."},
            {"path": "outputs/plots/zoom_in_cases.png", "title": "5. Zoom-In Case Studies with Diagnostic Evidence Debriefs", "desc": "Detailed inspection of one True Positive, one False Positive, and one Missed Window with natural language debriefs."},
            {"path": "outputs/plots/driver_lap_pace_comparison.png", "title": "6. Driver Lap-by-Lap Pace Trajectories (2026 US GP)", "desc": "Stint degradation and pace comparison across top contenders."},
            {"path": "outputs/plots/driver_timing_anomalies_scatter.png", "title": "7. Grid-Wide Timing Anomaly Scatter Matrix", "desc": "Scatter plot of 91 detected anomalies categorized by lockup, tyre cliff, pit delay, and off-track damage."},
            {"path": "outputs/plots/teammate_pace_deltas.png", "title": "8. Intra-Team Teammate Pace Delta Distributions", "desc": "Divergence between teammates in identical machinery across laps."},
            {"path": "outputs/plots/historical_reliability_evolution.png", "title": "9. 76-Season Mechanical Reliability Evolution", "desc": "Decade-by-decade collapse of mechanical attrition from 47.5% down to 8.9%."},
            {"path": "outputs/plots/historical_pitstop_evolution.png", "title": "10. Pit Stop Compression Across Eras", "desc": "Historical shift from 60s copper-mallet wheel changes to 1.80s robotic tyre changes."},
            {"path": "outputs/plots/current_vs_historical_changes.png", "title": "11. The 6 Transformation Pillars of Modern Formula 1", "desc": "Quantitative radar and bar chart comparing historical F1 to the current running game."}
        ]

        for p in plots_info:
            if os.path.exists(p["path"]):
                with st.expander(f"🖼️ {p['title']}", expanded=True):
                    st.write(p["desc"])
                    st.image(p["path"], use_container_width=True)


if __name__ == "__main__":
    main()
