"""
🏎️ F1 SENTINEL — REAL-TIME TELEMETRY ANOMALY DETECTION DASHBOARD
Utkrishta 2026 AI/ML Hackathon | Challenge 2: Sentinel
Pit Wall Real-Time Telemetry Monitoring & Anomaly Diagnostic Console
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


# Page Configuration
st.set_page_config(
    page_title="F1 Sentinel | Pit Wall Telemetry Monitor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Formula 1 Carbon Dark Theme CSS
st.markdown("""
<style>
    /* Main container and dark theme */
    .stApp {
        background-color: #12121A;
        color: #F3F3F3;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header bar */
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
    
    /* Red Alert Box */
    .alert-box {
        background-color: #2D0A0A;
        border-left: 6px solid #E10600;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 20px;
        animation: pulse 1.8s infinite;
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
    
    /* Nominal Box */
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

    /* Metric cards */
    .metric-card {
        background-color: #1A1A26;
        border: 1px solid #2E2E42;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: 800;
        color: #FFFFFF;
    }
    .metric-lbl {
        font-size: 11px;
        color: #8E8EA8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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

    # Train on first 2,000 baseline points
    train_pts = min(2000, int(len(df) * 0.25))
    ensemble = HybridEnsembleDetector()
    ensemble.fit(X_all[:train_pts], series_all.iloc[:train_pts])

    # Pre-score for smooth streaming playback
    model_scores = ensemble.score_all_models(X_all, series_all)
    ensemble_scores = model_scores["Hybrid_Ensemble"]

    # Trailing dynamic threshold
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

    # Evidence generator
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


def main():
    channels = get_available_channels()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=120)
        st.markdown("### 🛠️ Telemetry Control Unit")

        channel_names = {k: f"{v['telemetry_code']} - {v['channel_name']}" for k, v in channels.items()}
        selected_key = st.selectbox(
            "Select Telemetry Stream:",
            options=list(channels.keys()),
            format_func=lambda k: channel_names[k]
        )

        st.markdown("---")
        st.markdown("### ⏱️ Live Simulation Controls")

        # Session state for playback index
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

        # Quick Jump to Anomaly Window
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
                # Find matching row index
                matching_indices = df[df["timestamp"] >= target_time].index
                if len(matching_indices) > 0:
                    st.session_state.stream_idx = max(50, matching_indices[0] - 30)
                    st.rerun()

        st.markdown("---")
        st.markdown(
            "**Utkrishta 2026 AI/ML Hackathon**  \n"
            "*Challenge 2: Sentinel (Adaptive Time-Series)*  \n"
            "Domain Framing: Formula 1 High-Frequency Timing"
        )

    # Load channel data
    data_bundle = load_and_prepare_channel(selected_key)
    df = data_bundle["df"]
    meta = data_bundle["meta"]
    gt_windows = data_bundle["gt_windows"]
    scores = data_bundle["ensemble_scores"]
    thresholds = data_bundle["dynamic_threshs"]
    preds = data_bundle["binary_preds"]
    evidences = data_bundle["evidence_list"]
    model_scores = data_bundle["model_scores"]

    # Playback index progression
    if st.session_state.is_playing:
        st.session_state.stream_idx = min(len(df) - 1, st.session_state.stream_idx + playback_speed)

    curr_idx = st.session_state.stream_idx
    curr_time = df["timestamp"].iloc[curr_idx]
    curr_val = df["value"].iloc[curr_idx]
    curr_score = scores[curr_idx]
    curr_thresh = thresholds[curr_idx]
    curr_pred = preds[curr_idx]
    curr_evidence = evidences[curr_idx]

    # --- TOP HEADER BANNER ---
    st.markdown(f"""
    <div class="f1-header">
        <h1 class="f1-title">🏎️ F1 SENTINEL — REAL-TIME TELEMETRY ANOMALY DETECTION</h1>
        <p class="f1-subtitle">PIT WALL MONITORING & TACTICAL ANOMALY DIAGNOSTIC CONSOLE | CHANNEL: {meta['telemetry_code']} - {meta['channel_name']}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- REAL-TIME ALERT BANNER ---
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

    # --- TOP KPI METRICS CARDS ---
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
            <div class="metric-lbl">Session Telemetry Ingested</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHARTS SECTION (2 COLUMNS) ---
    col_chart1, col_chart2 = st.columns([2, 1])

    # History window to display (last 180 points)
    window_pts = 180
    start_history = max(0, curr_idx - window_pts)
    slice_df = df.iloc[start_history: curr_idx + 1].copy()
    slice_df["score"] = scores[start_history: curr_idx + 1]
    slice_df["threshold"] = thresholds[start_history: curr_idx + 1]
    slice_df["flag"] = preds[start_history: curr_idx + 1]

    with col_chart1:
        st.markdown("#### 📈 Real-Time Telemetry Stream & Detection Flags")
        # Altair line chart for smooth interactive display
        base = alt.Chart(slice_df).encode(x=alt.X("timestamp:T", title="Session Time"))
        
        line_sensor = base.mark_line(color="#00D2BE", strokeWidth=2.2).encode(
            y=alt.Y("value:Q", title=f"Sensor Value ({meta['unit']})", scale=alt.Scale(zero=False))
        )
        
        # Overlay anomaly flags as red circles
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
        
        # Current scores across all 5 models
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

        # Bar chart of model scores
        bar_chart = alt.Chart(breakdown_df).mark_bar().encode(
            y=alt.Y("Detector:N", sort=None, title=""),
            x=alt.X("Score:Q", title="Anomaly Confidence", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum.Score >= curr_thresh,
                alt.value("#E10600"),  # Triggered
                alt.value("#3B82F6")   # Normal
            )
        ).properties(height=260).configure_view(strokeWidth=0).configure_axis(
            gridColor="#2D2D3F",
            labelColor="#FFFFFF",
            titleColor="#FFFFFF"
        )
        st.altair_chart(bar_chart, use_container_width=True)

        st.markdown("#### 📋 Diagnostic Telemetry Telebrief")
        st.info(
            f"**Timestamp**: `{curr_time.strftime('%Y-%m-%d %H:%M:%S')}`  \n"
            f"**Delta from Prev**: `{float(data_bundle['feats_df']['rate_of_change'].iloc[curr_idx]):+.2f} {meta['unit']}`  \n"
            f"**Z-Score**: `{float(data_bundle['feats_df']['window_zscore'].iloc[curr_idx]):.2f}`  \n"
            f"**Tactical Context**: `{data_bundle['regime_metas'][curr_idx]['regime']}`"
        )

    # --- RECENT INCIDENT LOG ---
    st.markdown("---")
    st.markdown("### 📜 Past Detected Telemetry Incidents (Event Log)")

    # Find past anomalous segments up to current index
    past_preds = preds[:curr_idx + 1]
    if np.any(past_preds):
        incident_indices = np.where(past_preds)[0]
        # Group into contiguous events
        events_start = []
        for i, idx in enumerate(incident_indices):
            if i == 0 or idx != incident_indices[i - 1] + 1:
                events_start.append(idx)

        recent_events = events_start[-8:]  # Last 8 events
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

    # Auto-rerun loop if playing
    if st.session_state.is_playing and curr_idx < len(df) - 1:
        time.sleep(0.08)
        st.rerun()


if __name__ == "__main__":
    main()
