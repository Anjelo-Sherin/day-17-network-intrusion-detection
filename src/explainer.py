"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Analyst-Friendly Explainability Engine powered by SHAP (TreeExplainer).
Transforms black-box multi-class predictions into plain-English SOC triage justifications.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import shap

from src.data_loader import ATTACK_CLASSES, FEATURE_COLUMNS

# Analyst diagnostic dictionary translating feature metrics to cybersecurity telemetry
FEATURE_CYBER_CONTEXT = {
    "flow_duration": "Flow Lifespan",
    "tot_fwd_pkts": "Forward Packet Count",
    "tot_bwd_pkts": "Inbound Response Packets",
    "tot_len_fwd_pkts": "Total Outbound Payload Volume",
    "tot_len_bwd_pkts": "Total Inbound Payload Volume",
    "fwd_pkt_len_mean": "Mean Outbound Packet Payload Size",
    "bwd_pkt_len_mean": "Mean Inbound Packet Payload Size",
    "flow_bytes_per_s": "Bandwidth Consumption (Bytes/sec)",
    "flow_pkts_per_s": "Packet Flooding Rate (Packets/sec)",
    "flow_iat_mean": "Packet Inter-Arrival Timing (IAT)",
    "flow_iat_std": "Timing Jitter Regularity",
    "fwd_iat_tot": "Outbound Inter-Arrival Time",
    "bwd_iat_tot": "Inbound Inter-Arrival Time",
    "fin_flag_cnt": "FIN Termination Flag",
    "syn_flag_cnt": "SYN Synchronization Flag",
    "rst_flag_cnt": "RST Connection Abort Flag",
    "psh_flag_cnt": "PSH Immediate Push Flag",
    "ack_flag_cnt": "ACK Acknowledgement Flag",
    "urg_flag_cnt": "URG Urgent Pointer Flag",
    "down_up_ratio": "Asymmetric Download/Upload Ratio",
    "pkt_size_avg": "Average Packet MTU Size",
    "init_win_bytes_fwd": "Client Initial TCP Window Buffer",
    "init_win_bytes_bwd": "Server Initial TCP Window Buffer",
    "dst_port": "Target Destination Service Port",
    "protocol": "Transport Protocol Type",
}


def explain_feature_context(feat_name: str, raw_val: float, shap_val: float, attack_type: str) -> str:
    """Provides plain English contextual rationale for a SHAP feature contribution."""
    human_name = FEATURE_CYBER_CONTEXT.get(feat_name, feat_name)

    if feat_name == "syn_flag_cnt" and raw_val > 0:
        return f"{human_name} ({int(raw_val)}): Unacknowledged SYN packet sequence indicates SYN flood or TCP stealth scan."
    elif feat_name == "flow_pkts_per_s" and raw_val > 5000:
        return f"{human_name} ({raw_val:,.0f} pkts/s): Extreme packet transmission rate exceeding typical human/API bounds."
    elif feat_name == "flow_bytes_per_s" and raw_val > 100_000:
        return f"{human_name} ({raw_val:,.0f} B/s): Heavy sustained bandwidth saturation pattern."
    elif feat_name == "flow_iat_mean" and raw_val < 50:
        return f"{human_name} ({raw_val:.1f} µs): Ultra-tight inter-arrival interval typical of scripted packet generators."
    elif feat_name == "flow_iat_std" and raw_val < 100:
        return f"{human_name} (low jitter): Machine-like periodic beaconing interval typical of C2 Botnets."
    elif feat_name == "rst_flag_cnt" and raw_val > 0:
        return f"{human_name} ({int(raw_val)}): Aborted connection reset indicates closed port scanning or service evasion."
    elif feat_name == "dst_port":
        return f"{human_name} ({int(raw_val)}): Traffic directed at specific service port with known vulnerability vector."
    elif feat_name == "down_up_ratio" and raw_val < 0.2:
        return f"{human_name} ({raw_val:.2f}): Heavy one-way transmission with negligible response (server unresponsiveness)."
    elif feat_name == "init_win_bytes_fwd" and raw_val <= 1024:
        return f"{human_name} ({int(raw_val)}): Constrained initial TCP window buffer characteristic of crafted attack tools."
    else:
        direction = "elevated" if raw_val > 0 else "suppressed"
        return f"{human_name} ({raw_val:.2f}): Value contributed +{shap_val:.3f} toward {attack_type} classification."


class IDSExplainer:
    """Manages TreeSHAP computation and translation for SOC analysts."""

    def __init__(self, classifier, label_to_idx: Optional[Dict[str, int]] = None):
        self.classifier = classifier
        self.label_to_idx = label_to_idx or {cls: i for i, cls in enumerate(ATTACK_CLASSES)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.classifier)

    def explain_flow(
        self,
        scaled_vector: np.ndarray,
        predicted_class: str,
        raw_features: Dict[str, float],
        top_k: int = 5,
    ) -> Dict:
        """
        Computes exact SHAP values for the predicted class and generates:
        - Top K positive driving features with impact scores
        - Human-readable narrative reasons
        - Full waterfall plot data
        """
        class_idx = self.label_to_idx.get(predicted_class, 0)

        # Reshape to (1, n_features)
        X_sample = np.array(scaled_vector).reshape(1, -1)
        shap_values = self.explainer.shap_values(X_sample)

        # Handle different SHAP output formats (list of arrays vs 3D ndarray)
        if isinstance(shap_values, list):
            # List of arrays per class
            class_shaps = shap_values[class_idx][0]
            base_value = float(self.explainer.expected_value[class_idx])
        elif len(shap_values.shape) == 3:
            # (1, n_features, n_classes)
            class_shaps = shap_values[0, :, class_idx]
            base_value = float(self.explainer.expected_value[class_idx])
        else:
            class_shaps = shap_values[0]
            base_value = float(self.explainer.expected_value)

        # Rank features by positive contribution toward predicted class
        ranked_indices = np.argsort(class_shaps)[::-1]

        top_drivers = []
        narratives = []
        for idx in ranked_indices[:top_k]:
            feat_name = FEATURE_COLUMNS[idx]
            shap_val = float(class_shaps[idx])
            raw_val = float(raw_features.get(feat_name, 0.0))

            context_text = explain_feature_context(feat_name, raw_val, shap_val, predicted_class)
            narratives.append(context_text)

            top_drivers.append({
                "feature": feat_name,
                "display_name": FEATURE_CYBER_CONTEXT.get(feat_name, feat_name),
                "raw_value": raw_val,
                "shap_value": shap_val,
                "percentage_impact": round(abs(shap_val) * 100, 1),
            })

        # Data formatted for waterfall/bar charts
        feature_impacts = [
            {
                "feature": FEATURE_CYBER_CONTEXT.get(FEATURE_COLUMNS[i], FEATURE_COLUMNS[i]),
                "raw_name": FEATURE_COLUMNS[i],
                "shap_value": float(class_shaps[i]),
                "raw_value": float(raw_features.get(FEATURE_COLUMNS[i], 0.0)),
            }
            for i in range(len(FEATURE_COLUMNS))
        ]
        # Sort by absolute SHAP impact
        feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "predicted_class": predicted_class,
            "base_value": base_value,
            "top_drivers": top_drivers,
            "narratives": narratives,
            "sorted_impacts": feature_impacts[:10],
        }
