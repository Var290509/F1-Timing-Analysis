"""
F1 Sentinel Explainable Evidence Generator
Translates multi-model detector signals, statistical deviations, and autoencoder reconstruction
errors into concise, interpretable race engineer telemetry debriefs.
"""

from typing import Dict, Any, Optional, List
import numpy as np


class EvidenceGenerator:
    """
    Generates human-interpretable telemetry diagnostic evidence for flagged anomalies and nominal states.
    """

    def __init__(self, channel_name: str = "ICE Coolant Temp", unit: str = "°C"):
        self.channel_name = channel_name
        self.unit = unit

    def generate(
        self,
        anomaly_score: float,
        threshold: float,
        label: str,
        current_value: float,
        model_scores: Optional[Dict[str, float]] = None,
        feature_dict: Optional[Dict[str, float]] = None,
        regime_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Produce a concise, professional diagnostic evidence string.
        """
        if label != "ANOMALOUS":
            margin = threshold - anomaly_score
            return (
                f"Nominal telemetry: Score {anomaly_score:.2f} within adaptive threshold {threshold:.2f} "
                f"(safety margin {margin:+.2f}); {self.channel_name} = {current_value:.2f}{self.unit}."
            )

        evidence_parts = []

        # 1. Anomaly score & threshold breach
        delta_score = anomaly_score - threshold
        evidence_parts.append(
            f"Anomaly Score = {anomaly_score:.2f} (exceeds dynamic threshold {threshold:.2f} by +{delta_score:.2f})"
        )

        # 2. Base model consensus
        if model_scores:
            flagged_detectors = []
            for name, score in model_scores.items():
                if name == "Hybrid_Ensemble":
                    continue
                # Individual model threshold check
                if score >= 0.50:
                    clean_name = name.replace("_Detector", "").replace("PyTorch_", "").replace("_", " ")
                    flagged_detectors.append(f"{clean_name} ({score:.2f})")

            num_flagged = len(flagged_detectors)
            total_models = len([k for k in model_scores if k != "Hybrid_Ensemble"])
            if num_flagged > 0:
                det_list = ", ".join(flagged_detectors)
                evidence_parts.append(f"Consensus: {num_flagged}/{total_models} models triggered [{det_list}]")

        # 3. Statistical and Physical feature telemetry
        if feature_dict:
            z = feature_dict.get("window_zscore", 0.0)
            roc = feature_dict.get("rate_of_change", 0.0)
            ae_mse = feature_dict.get("ae_reconstruction_error", None)

            feat_details = []
            if abs(z) >= 2.0:
                feat_details.append(f"Z-score = {z:.2f}")
            if abs(roc) > 0.01:
                feat_details.append(f"Delta = {roc:+.2f}{self.unit}/step")
            if ae_mse is not None:
                feat_details.append(f"Reconstruction MSE = {ae_mse:.3f}")

            if feat_details:
                evidence_parts.append("; ".join(feat_details))

        # 4. Tactical Regime condition
        if regime_meta and regime_meta.get("regime_shift_active", False):
            regime = regime_meta.get("regime", "SHIFT")
            evidence_parts.append(f"Tactical Context: {regime} active")

        return "; ".join(evidence_parts) + "."
