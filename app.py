"""
NetSentinel SOC — Day 17: Network Intrusion Detection Dashboard
Redesigned Next-Gen Cyber Defense Command Center with:
- Dual-Engine Threat Detection (XGBoost Classifier + Neural Autoencoder for Zero-Days)
- Real-Time Live Flow Streamer & Packet Replay
- TreeSHAP Feature Attribution & Plain-English Forensics
- MITRE ATT&CK Mapping & Automated Mitigation Playbooks (iptables, ufw, Suricata)
- Global Attack Geo Map & Threat Analytics
- Interactive Custom Packet Crafter Sandbox
"""

from __future__ import annotations

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/matplotlib", exist_ok=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.alert_service import AlertDispatcher
from src.data_loader import (
    ATTACK_CLASSES,
    FEATURE_COLUMNS,
    RealTimeFlowStreamer,
    generate_flow_record,
)
from src.explainer import IDSExplainer
from src.models import DualIDSModelEngine
from src.threat_intel import MITRE_ATTCK_MAPPINGS, generate_mitigation_playbook

# Setup Page Configuration
st.set_page_config(
    page_title="NetSentinel SOC — AI Network Intrusion Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark SOC Cyber HUD Design System CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    /* Global Dark Cyber Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E2E8F0;
        background-color: #080C14;
    }
    .stApp {
        background: radial-gradient(circle at 15% 10%, #0F172A 0%, #080C14 100%);
    }

    /* Code & Telemetry Font */
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    /* Cyber Command Header */
    .soc-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 0 12px rgba(56, 189, 248, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .soc-title-area h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .soc-title-area p {
        margin: 0.25rem 0 0 0;
        font-size: 0.92rem;
        color: #94A3B8;
    }
    .soc-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.45rem 0.9rem;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.2);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1.1rem;
        backdrop-filter: blur(12px);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }
    .metric-card .title {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 0.35rem 0;
        color: #F8FAFC;
    }
    .metric-card .sub {
        font-size: 0.78rem;
        color: #64748B;
    }

    /* Severity Badges */
    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.15);
        border: 1px solid #F97316;
        color: #FB923C;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-benign {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-zeroday {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid #A855F7;
        color: #C084FC;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }

    /* Custom Buttons Styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background-color: #1E293B;
        color: #F8FAFC;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #38BDF8;
        background-color: #0F172A;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25);
    }

    /* Triage Box */
    .triage-box {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Models & Explainer (Cached)
@st.cache_resource(show_spinner=False)
def load_soc_resources():
    models_dir = Path(__file__).resolve().parent / "models"
    engine = DualIDSModelEngine.load_artifacts(models_dir)
    explainer = IDSExplainer(engine.classifier, engine.label_to_idx)
    dispatcher = AlertDispatcher()
    streamer = RealTimeFlowStreamer(random_state=int(datetime.datetime.now().timestamp()) % 10000)
    return engine, explainer, dispatcher, streamer

engine, explainer, dispatcher, streamer = load_soc_resources()

# Initialize Session State Buffer for Live Flows
if "flow_buffer" not in st.session_state:
    # Seed buffer with initial 12 flows
    initial_flows = []
    rng = np.random.default_rng(42)
    # Mix normal and various attacks
    seed_types = [
        "BENIGN", "BENIGN", "DoS / DDoS", "BENIGN", "PortScan",
        "BENIGN", "BruteForce", "BENIGN", "Botnet (C2)", "BENIGN", "Web Attack", "BENIGN"
    ]
    for stype in seed_types:
        f = generate_flow_record(stype, rng)
        pred = engine.predict_flow(f)
        f.update(pred)
        initial_flows.append(f)
    st.session_state.flow_buffer = initial_flows

if "selected_flow_idx" not in st.session_state:
    st.session_state.selected_flow_idx = 2  # Default to the DoS sample

if "webhook_url" not in st.session_state:
    st.session_state.webhook_url = ""

# Sidebar: Controls, Telemetry & Webhook Settings
with st.sidebar:
    st.markdown("### 🎛️ SOC Operator Console")

    st.markdown("**Live Ingestion Feed**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("▶ Ingest 1 Flow", use_container_width=True):
            f = streamer.get_flow()
            pred = engine.predict_flow(f)
            f.update(pred)
            st.session_state.flow_buffer.insert(0, f)
            if len(st.session_state.flow_buffer) > 60:
                st.session_state.flow_buffer.pop()
            st.session_state.selected_flow_idx = 0
            st.rerun()

    with col_s2:
        if st.button("⏩ Ingest 10 Flows", use_container_width=True):
            for _ in range(10):
                f = streamer.get_flow()
                pred = engine.predict_flow(f)
                f.update(pred)
                st.session_state.flow_buffer.insert(0, f)
            st.session_state.flow_buffer = st.session_state.flow_buffer[:60]
            st.session_state.selected_flow_idx = 0
            st.rerun()

    if st.button("🧹 Flush Flow Buffer", use_container_width=True):
        st.session_state.flow_buffer = []
        st.session_state.selected_flow_idx = 0
        st.rerun()

    st.markdown("---")
    st.markdown("**⚡ Live Adversary Injection**")
    st.caption("Inject controlled attacks to evaluate real-time triage:")
    inj_attack = st.selectbox(
        "Attack Vector",
        ["DoS / DDoS", "PortScan", "BruteForce", "Botnet (C2)", "Web Attack", "BENIGN"],
        index=0,
    )
    if st.button(f"🚨 Inject {inj_attack}", use_container_width=True):
        f = streamer.get_flow(forced_attack=inj_attack)
        pred = engine.predict_flow(f)
        f.update(pred)
        st.session_state.flow_buffer.insert(0, f)
        st.session_state.selected_flow_idx = 0
        st.success(f"Injected {inj_attack} into live packet pipeline!")
        st.rerun()

    st.markdown("---")
    st.markdown("**🔔 Webhook Dispatcher**")
    st.session_state.webhook_url = st.text_input(
        "Slack / Telegram / Discord URL",
        value=st.session_state.webhook_url,
        placeholder="https://hooks.slack.com/services/...",
        type="password",
    )
    if st.button("Test Dispatch Alert", use_container_width=True):
        if st.session_state.flow_buffer:
            flow_to_alert = st.session_state.flow_buffer[st.session_state.selected_flow_idx]
            res = dispatcher.dispatch(flow_to_alert, target_url=st.session_state.webhook_url)
            if res["status"] in ["delivered", "simulated"]:
                st.success(f"Dispatched! ({res['status']})")
            else:
                st.error(f"Error: {res.get('message', res)}")

    st.markdown("---")
    st.markdown("**ℹ️ System Architecture**")
    st.caption(
        "• **Supervised Model:** XGBoost Multi-Class\n"
        "• **Zero-Day Model:** Neural Autoencoder (MSE error)\n"
        "• **Explainability:** TreeSHAP (Local Attributions)\n"
        "• **Standard:** CIC-IDS2017 & NSL-KDD"
    )

# Top Cyber Header
st.markdown(
    """
    <div class="soc-header">
        <div class="soc-title-area">
            <h1>🛡️ NetSentinel SOC <span style="font-size: 1rem; color: #38BDF8; font-weight: 500;">// Day 17</span></h1>
            <p>AI-Powered Autonomous Network Intrusion Detection & Real-Time Incident Forensics</p>
        </div>
        <div class="soc-status-badge">
            <span class="pulse-dot"></span> DUAL DEFENSE ENGINE ACTIVE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI Metrics Calculation from Current Buffer
buffer_df = pd.DataFrame(st.session_state.flow_buffer) if st.session_state.flow_buffer else pd.DataFrame()

if not buffer_df.empty:
    total_flows = len(buffer_df)
    total_intrusions = int((buffer_df["verdict"] != "BENIGN").sum())
    threat_rate = (total_intrusions / total_flows) * 100
    zero_day_cnt = int(buffer_df.get("is_novel_anomaly", pd.Series([False])).sum())
    top_port = buffer_df["dst_port"].mode().iloc[0] if "dst_port" in buffer_df else 80
    avg_risk = int(buffer_df["risk_score"].mean()) if "risk_score" in buffer_df else 0
else:
    total_flows = 0
    total_intrusions = 0
    threat_rate = 0.0
    zero_day_cnt = 0
    top_port = 80
    avg_risk = 0

# KPI Cards Row
col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
with col_k1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">Ingested Flows</div>
            <div class="value">{total_flows}</div>
            <div class="sub">Active Rolling Window</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_k2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">Attacks Flagged</div>
            <div class="value" style="color: #F87171;">{total_intrusions} <span style="font-size: 0.95rem; font-weight: 500;">({threat_rate:.1f}%)</span></div>
            <div class="sub">XGBoost Supervised</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_k3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">Zero-Day Anomalies</div>
            <div class="value" style="color: #C084FC;">{zero_day_cnt}</div>
            <div class="sub">Autoencoder Reconstruction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_k4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">Target Port #1</div>
            <div class="value" style="color: #38BDF8;">:{top_port}</div>
            <div class="sub">Most Probed Destination</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_k5:
    risk_color = "#34D399" if avg_risk < 30 else ("#FBBF24" if avg_risk < 65 else "#EF4444")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="title">SOC Threat Index</div>
            <div class="value" style="color: {risk_color};">{avg_risk}<span style="font-size: 1rem; color: #94A3B8;">/100</span></div>
            <div class="sub">Composite Severity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Main Tabs Navigation
tab_stream, tab_forensics, tab_mitre, tab_geomap, tab_zeroday, tab_sandbox = st.tabs([
    "📡 Live Triage Feed",
    "🔬 Forensics & SHAP Attribution",
    "🛡️ MITRE & Mitigation Playbook",
    "🌍 Threat Map & Geo Analytics",
    "🧪 Zero-Day Anomaly Lab",
    "🛠️ Packet Crafter Sandbox",
])

# ==============================================================================
# TAB 1: LIVE TRIAGE FEED
# ==============================================================================
with tab_stream:
    st.markdown("#### Real-Time Ingested Packet Flows & Incident Triage")
    if buffer_df.empty:
        st.info("Buffer empty. Click 'Ingest 1 Flow' or inject an attack from the sidebar.")
    else:
        # Search and Filtering row
        fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
        with fcol1:
            search_query = st.text_input("🔍 Filter by IP, Country, or Port", placeholder="e.g. 185.220 or 443")
        with fcol2:
            filter_class = st.selectbox("Filter Attack Type", ["ALL"] + ATTACK_CLASSES)
        with fcol3:
            filter_sev = st.selectbox("Filter Severity", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "INFO"])

        # Filter dataframe
        filtered_df = buffer_df.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["src_ip"].astype(str).str.contains(search_query, case=False) |
                filtered_df["country"].astype(str).str.contains(search_query, case=False) |
                filtered_df["dst_port"].astype(str).str.contains(search_query, case=False)
            ]
        if filter_class != "ALL":
            filtered_df = filtered_df[filtered_df["verdict"] == filter_class]
        if filter_sev != "ALL":
            filtered_df = filtered_df[filtered_df["severity"] == filter_sev]

        # Table Display
        display_cols = [
            "timestamp", "src_ip", "dst_port", "protocol", "flow_pkts_per_s",
            "verdict", "confidence", "reconstruction_error", "severity", "risk_score"
        ]
        
        # Format dataframe for display
        table_view = filtered_df[display_cols].copy()
        table_view["flow_pkts_per_s"] = table_view["flow_pkts_per_s"].apply(lambda x: f"{x:,.0f} p/s")
        table_view["confidence"] = table_view["confidence"].apply(lambda x: f"{x*100:.1f}%")
        table_view["reconstruction_error"] = table_view["reconstruction_error"].apply(lambda x: f"{x:.3f}")
        table_view["risk_score"] = table_view["risk_score"].apply(lambda x: f"{x}/100")

        st.dataframe(
            table_view,
            use_container_width=True,
            height=320,
        )

        st.markdown("##### 🔍 Drill-Down Forensic Selector")
        flow_options = [
            f"#{i} [{f['timestamp']}] — {f['verdict']} | Src: {f['src_ip']} → Port: {f['dst_port']} (Risk: {f['risk_score']})"
            for i, f in enumerate(st.session_state.flow_buffer)
        ]
        sel_idx = st.selectbox(
            "Select Flow to Inspect in Detail:",
            range(len(flow_options)),
            format_func=lambda x: flow_options[x],
            index=min(st.session_state.selected_flow_idx, len(flow_options) - 1),
        )
        st.session_state.selected_flow_idx = sel_idx

        # Export Feed Data
        st.download_button(
            label="📥 Download Ingested Flows CSV",
            data=buffer_df.to_csv(index=False),
            file_name=f"netsentinel_flows_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 2: FORENSICS & SHAP ATTRIBUTION INSPECTOR
# ==============================================================================
with tab_forensics:
    if not st.session_state.flow_buffer:
        st.warning("No flows available. Ingest flows in the sidebar first.")
    else:
        curr_flow = st.session_state.flow_buffer[st.session_state.selected_flow_idx]
        
        # Forensic Overview Banner
        sev = curr_flow.get("severity", "INFO")
        verdict = curr_flow.get("verdict", "BENIGN")
        risk = curr_flow.get("risk_score", 0)
        conf = curr_flow.get("confidence", 0.0)
        recon_err = curr_flow.get("reconstruction_error", 0.0)
        thresh = curr_flow.get("anomaly_threshold", 0.28)
        
        badge_style = "badge-critical" if sev == "CRITICAL" else ("badge-high" if sev == "HIGH" else ("badge-medium" if sev == "MEDIUM" else "badge-benign"))

        st.markdown(
            f"""
            <div class="triage-box">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span class="{badge_style}">{sev}</span>
                        <h2 style="margin: 0.4rem 0 0.2rem 0; font-size: 1.5rem; color: #F8FAFC;">
                            {verdict} <span style="font-size: 1rem; color: #94A3B8;">({conf*100:.1f}% Model Confidence)</span>
                        </h2>
                        <p style="margin: 0; color: #94A3B8; font-size: 0.88rem;">
                            Source IP: <code style="color: #38BDF8;">{curr_flow['src_ip']}</code> ({curr_flow.get('country', 'Unknown')}) &nbsp;•&nbsp; 
                            Target: <code style="color: #A78BFA;">{curr_flow['dst_ip']}:{curr_flow['dst_port']}</code> &nbsp;•&nbsp; 
                            ASN: <code style="color: #94A3B8;">{curr_flow.get('asn', 'N/A')}</code>
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Composite Risk</div>
                        <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono'; color: {'#EF4444' if risk > 70 else ('#FBBF24' if risk > 40 else '#34D399')};">
                            {risk}/100
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # Two columns: SHAP Waterfall / Feature Drivers & Dual Engine Consensus
        col_f1, col_f2 = st.columns([3, 2])

        with col_f1:
            st.markdown("##### 🧠 TreeSHAP Feature Attribution (Why Was This Flagged?)")
            st.caption("Individual feature impacts calculated via Shapley values on XGBoost trees:")
            
            # Compute SHAP explanation
            explanation = explainer.explain_flow(
                curr_flow["scaled_vector"],
                curr_flow["predicted_class"],
                curr_flow["raw_features"],
                top_k=5,
            )

            # Bar chart of top feature drivers
            drivers = explanation["sorted_impacts"][:7]
            features = [d["feature"] for d in drivers][::-1]
            shap_vals = [d["shap_value"] for d in drivers][::-1]
            colors = ["#EF4444" if v > 0 else "#38BDF8" for v in shap_vals]

            fig_shap = go.Figure(
                go.Bar(
                    x=shap_vals,
                    y=features,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in shap_vals],
                    textposition="auto",
                )
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                xaxis=dict(title="SHAP Value (Contribution toward Attack)", color="#94A3B8", zerolinecolor="#64748B"),
                yaxis=dict(color="#CBD5E1"),
                font=dict(family="Inter", color="#E2E8F0"),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Plain English Explanations
            st.markdown("###### 📝 Analyst Triage Narrative:")
            for narrative in explanation["narratives"][:3]:
                st.markdown(f"• <span style='font-size: 0.88rem; color: #E2E8F0;'>{narrative}</span>", unsafe_allow_html=True)

        with col_f2:
            st.markdown("##### ⚖️ Dual-Engine Consensus")
            
            # 1. Supervised Probabilities
            st.markdown("**Supervised XGBoost Probabilities:**")
            probs = curr_flow.get("class_probabilities", {})
            prob_df = pd.DataFrame([{"Class": k, "Probability": v} for k, v in probs.items()])
            
            fig_prob = px.bar(
                prob_df,
                x="Probability",
                y="Class",
                orientation="h",
                color="Probability",
                color_continuous_scale="Viridis",
                range_x=[0, 1.0],
            )
            fig_prob.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                coloraxis_showscale=False,
                xaxis=dict(color="#94A3B8"),
                yaxis=dict(color="#CBD5E1"),
                font=dict(family="Inter", color="#E2E8F0"),
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # 2. Unsupervised Autoencoder Gauge
            st.markdown("**Autoencoder Reconstruction Error:**")
            err_ratio = min(recon_err / (thresh + 1e-5), 2.5)
            is_anomaly = recon_err > thresh
            
            st.markdown(
                f"""
                <div style="background: rgba(15,23,42,0.7); border: 1px solid {'#A855F7' if is_anomaly else 'rgba(255,255,255,0.1)'}; border-radius: 8px; padding: 0.8rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94A3B8;">
                        <span>Reconstruction Error: <b style="color: {'#C084FC' if is_anomaly else '#34D399'}; font-family: 'JetBrains Mono';">{recon_err:.4f}</b></span>
                        <span>Threshold (τ): <b style="color: #94A3B8; font-family: 'JetBrains Mono';">{thresh:.4f}</b></span>
                    </div>
                    <div style="width: 100%; height: 8px; background: #334155; border-radius: 4px; margin: 0.5rem 0; overflow: hidden;">
                        <div style="width: {min(int(err_ratio * 40), 100)}%; height: 100%; background: {'#A855F7' if is_anomaly else '#10B981'};"></div>
                    </div>
                    <div style="font-size: 0.8rem; color: {'#C084FC' if is_anomaly else '#34D399'};">
                        {'⚠️ Novel Anomaly Detected: reconstruction error exceeds 96.5th percentile baseline!' if is_anomaly else '✓ Conforms to normal baseline reconstruction profile.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Raw Telemetry Expandable View
        with st.expander("📊 Inspect Full Raw Flow Telemetry (25 Features)"):
            raw_f = curr_flow.get("raw_features", {})
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            r_items = list(raw_f.items())
            chunk = len(r_items) // 4 + 1
            for col, items in zip([r_col1, r_col2, r_col3, r_col4], [r_items[i*chunk:(i+1)*chunk] for i in range(4)]):
                with col:
                    for k, v in items:
                        st.markdown(f"**`{k}`**: `{v:,.2f}`" if isinstance(v, float) else f"**`{k}`**: `{v}`")

# ==============================================================================
# TAB 3: MITRE ATT&CK & MITIGATION PLAYBOOK
# ==============================================================================
with tab_mitre:
    if not st.session_state.flow_buffer:
        st.warning("No flow selected.")
    else:
        curr_flow = st.session_state.flow_buffer[st.session_state.selected_flow_idx]
        verdict = curr_flow.get("verdict", "BENIGN")
        mitre_data = MITRE_ATTCK_MAPPINGS.get(verdict, MITRE_ATTCK_MAPPINGS["BENIGN"])

        st.markdown("#### MITRE ATT&CK® Enterprise Alignment")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"**Tactic:**\n`{mitre_data['tactic']}`")
        with m_col2:
            st.markdown(f"**Technique ID:**\n`{mitre_data['technique_id']}`")
        with m_col3:
            st.markdown(f"**Technique Name:**\n`{mitre_data['technique_name']}`")
        with m_col4:
            st.markdown(f"**Base CVSS Score:**\n`{mitre_data['cvss_score']} / 10.0`")

        st.info(f"**Threat Description:** {mitre_data['description']}")

        st.markdown("---")
        st.markdown("#### Automated SOC Mitigation Playbook")
        st.caption("Executable firewall rules and active threat countermeasures synthesized for this flow:")

        playbook = generate_mitigation_playbook(
            attack_type=verdict,
            src_ip=curr_flow["src_ip"],
            dst_ip=curr_flow["dst_ip"],
            dst_port=curr_flow["dst_port"],
            protocol=curr_flow.get("protocol", 6),
        )

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("**1. Linux iptables Rule:**")
            st.code(playbook["iptables"], language="bash")
            
            st.markdown("**2. UFW Command:**")
            st.code(playbook["ufw"], language="bash")

        with p_col2:
            st.markdown("**3. Snort / Suricata IDS Signature:**")
            st.code(playbook["suricata_rule"], language="text")

            st.markdown("**4. L1/L2 Incident Response Actions:**")
            for step in playbook["mitigation_steps"]:
                st.markdown(f"<span style='font-size: 0.88rem; color: #CBD5E1;'>{step}</span>", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: THREAT MAP & GEO ANALYTICS
# ==============================================================================
with tab_geomap:
    st.markdown("#### Global Attacker Geolocation & Threat Trajectory")
    if buffer_df.empty:
        st.info("No flow data available to plot.")
    else:
        # Separate attacking flows
        threats_df = buffer_df[buffer_df["verdict"] != "BENIGN"].copy()
        
        if threats_df.empty:
            st.success("All current flows in buffer are BENIGN. Inject attacks to view geographical origins!")
        else:
            # Map of attacking IPs
            fig_map = px.scatter_geo(
                threats_df,
                lat="lat",
                lon="lon",
                color="verdict",
                hover_name="src_ip",
                hover_data=["city", "country", "asn", "risk_score", "dst_port"],
                size="risk_score",
                size_max=18,
                projection="natural earth",
                color_discrete_map={
                    "DoS / DDoS": "#EF4444",
                    "PortScan": "#F59E0B",
                    "BruteForce": "#FB923C",
                    "Botnet (C2)": "#A855F7",
                    "Web Attack": "#EC4899",
                    "Zero-Day Anomaly": "#8B5CF6",
                },
            )
            fig_map.update_geos(
                bgcolor="rgba(8, 12, 20, 1)",
                showcoastlines=True,
                coastlinecolor="#334155",
                showland=True,
                landcolor="#0F172A",
                showocean=True,
                oceancolor="#080C14",
                showcountries=True,
                countrycolor="#1E293B",
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=380,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#CBD5E1"),
                ),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown("##### 📍 Top Attacking Origin Countries")
                country_counts = threats_df["country"].value_counts().reset_index()
                country_counts.columns = ["Country", "Incidents"]
                fig_country = px.bar(
                    country_counts,
                    x="Incidents",
                    y="Country",
                    orientation="h",
                    color="Incidents",
                    color_continuous_scale="Reds",
                )
                fig_country.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                    coloraxis_showscale=False,
                    xaxis=dict(color="#94A3B8"),
                    yaxis=dict(color="#CBD5E1"),
                )
                st.plotly_chart(fig_country, use_container_width=True)

            with g_col2:
                st.markdown("##### 🎯 Targeted Destination Ports")
                port_counts = threats_df["dst_port"].value_counts().reset_index()
                port_counts.columns = ["Port", "Count"]
                port_counts["Port"] = port_counts["Port"].apply(lambda p: f":{p}")
                fig_ports = px.pie(
                    port_counts,
                    names="Port",
                    values="Count",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Plasma,
                )
                fig_ports.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=240,
                    font=dict(color="#CBD5E1"),
                )
                st.plotly_chart(fig_ports, use_container_width=True)

# ==============================================================================
# TAB 5: ZERO-DAY ANOMALY LAB
# ==============================================================================
with tab_zeroday:
    st.markdown("#### Unsupervised Deep Neural Autoencoder — Anomaly Research")
    st.markdown(
        """
        The autoencoder is trained **exclusively on benign baseline traffic** with a bottleneck compression 
        architecture `(25 → 14 → 8 → 14 → 25)`.
        When an uncharacterized zero-day attack arrives, the reconstruction error $MSE(x, \\hat{x})$ spikes 
        because the bottleneck has never learned its compressed representation.
        """
    )

    # Threshold adjustment slider
    cur_thresh = engine.anomaly_threshold
    new_thresh = st.slider(
        "Interactive Anomaly Detection Threshold (τ)",
        min_value=0.05,
        max_value=1.50,
        value=float(cur_thresh),
        step=0.01,
        help="Lower threshold catches more subtle anomalies but increases false positives. Higher threshold flags only extreme outliers.",
    )
    if new_thresh != cur_thresh:
        engine.anomaly_threshold = new_thresh
        st.success(f"Updated dynamic threshold to τ = {new_thresh:.4f}")

    if not buffer_df.empty:
        errors = buffer_df["reconstruction_error"].values
        labels = buffer_df["verdict"].values

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=[e for e, l in zip(errors, labels) if l == "BENIGN"],
                name="BENIGN Flows",
                marker_color="#10B981",
                opacity=0.75,
                nbinsx=20,
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=[e for e, l in zip(errors, labels) if l != "BENIGN"],
                name="Attacks / Anomalies",
                marker_color="#EF4444",
                opacity=0.75,
                nbinsx=20,
            )
        )
        # Vertical line for threshold
        fig_hist.add_vline(
            x=new_thresh,
            line_width=3,
            line_dash="dash",
            line_color="#A855F7",
            annotation_text=f"Threshold τ = {new_thresh:.3f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
            xaxis=dict(title="Reconstruction Error (MSE)", color="#94A3B8"),
            yaxis=dict(title="Flow Count", color="#94A3B8"),
            font=dict(color="#CBD5E1"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Load offline evaluation report if available
    eval_file = Path(__file__).resolve().parent / "models" / "eval_metrics.json"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_data = json.load(f)
        
        st.markdown("##### 📈 Offline Model Validation Benchmark Metrics")
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            st.metric("Total Benchmark Dataset", f"{eval_data.get('total_samples', 0):,}")
        with e_col2:
            st.metric("Supervised Macro F1", f"{eval_data.get('classification_metrics', {}).get('val_f1_macro', 0.0):.4f}")
        with e_col3:
            st.metric("Autoencoder Anomaly Detection", f"{eval_data.get('autoencoder_tpr', 0.0)*100:.1f}%")
        with e_col4:
            st.metric("Benign False Positive Rate", f"{eval_data.get('autoencoder_fpr', 0.0)*100:.2f}%")

# ==============================================================================
# TAB 6: PACKET CRAFTER SANDBOX
# ==============================================================================
with tab_sandbox:
    st.markdown("#### Interactive Custom Flow Generator & AI Diagnostic Sandbox")
    st.caption("Craft arbitrary network flow characteristics to test the dual ML detection engine in real time:")

    with st.form("custom_packet_form"):
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            s_dst_port = st.number_input("Target Port (dst_port)", min_value=1, max_value=65535, value=80)
            s_proto = st.selectbox("Protocol", [6, 17, 1], format_func=lambda x: "TCP (6)" if x==6 else ("UDP (17)" if x==17 else "ICMP (1)"))
            s_duration = st.number_input("Flow Duration (µs)", min_value=1.0, value=250_000.0, step=1000.0)
            s_flow_pkts_per_s = st.number_input("Flow Packets/s", min_value=0.0, value=450.0, step=50.0)

        with p_c2:
            s_tot_fwd = st.number_input("Total Forward Packets", min_value=1, value=12)
            s_tot_bwd = st.number_input("Total Backward Packets", min_value=0, value=15)
            s_fwd_len = st.number_input("Mean Fwd Packet Size (bytes)", min_value=0.0, value=250.0)
            s_flow_bytes = st.number_input("Flow Bytes/s", min_value=0.0, value=45_000.0)

        with p_c3:
            s_syn = st.selectbox("SYN Flag", [0, 1], index=0)
            s_rst = st.selectbox("RST Flag", [0, 1], index=0)
            s_fin = st.selectbox("FIN Flag", [0, 1], index=1)
            s_iat = st.number_input("Flow IAT Mean (µs)", min_value=0.0, value=1200.0)

        submit_test = st.form_submit_button("🧪 Test Flow Against Dual AI Defense Engine")

    if submit_test:
        test_record = {
            "flow_duration": float(s_duration),
            "tot_fwd_pkts": int(s_tot_fwd),
            "tot_bwd_pkts": int(s_tot_bwd),
            "tot_len_fwd_pkts": float(s_tot_fwd * s_fwd_len),
            "tot_len_bwd_pkts": float(s_tot_bwd * 500.0),
            "fwd_pkt_len_mean": float(s_fwd_len),
            "bwd_pkt_len_mean": 500.0,
            "flow_bytes_per_s": float(s_flow_bytes),
            "flow_pkts_per_s": float(s_flow_pkts_per_s),
            "flow_iat_mean": float(s_iat),
            "flow_iat_std": 500.0,
            "fwd_iat_tot": float(s_duration * 0.9),
            "bwd_iat_tot": float(s_duration * 0.8),
            "fin_flag_cnt": int(s_fin),
            "syn_flag_cnt": int(s_syn),
            "rst_flag_cnt": int(s_rst),
            "psh_flag_cnt": 1,
            "ack_flag_cnt": 1,
            "urg_flag_cnt": 0,
            "down_up_ratio": float(s_tot_bwd / max(s_tot_fwd, 1)),
            "pkt_size_avg": float(s_fwd_len),
            "init_win_bytes_fwd": 29200,
            "init_win_bytes_bwd": 29200,
            "dst_port": int(s_dst_port),
            "protocol": int(s_proto),
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "country": "Sandbox Lab",
            "city": "Virtual",
            "asn": "AS0 Lab Test",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        test_pred = engine.predict_flow(test_record)
        test_record.update(test_pred)

        st.markdown("##### 🔬 AI Diagnostic Result:")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric("Classification Verdict", test_pred["verdict"])
        with t_col2:
            st.metric("Supervised Confidence", f"{test_pred['confidence']*100:.1f}%")
        with t_col3:
            st.metric("Autoencoder Error", f"{test_pred['reconstruction_error']:.4f}")

        # Compute SHAP
        exp = explainer.explain_flow(
            test_pred["scaled_vector"],
            test_pred["predicted_class"],
            test_pred["raw_features"],
            top_k=3,
        )
        st.markdown("**Top SHAP Contributors:**")
        for narrative in exp["narratives"]:
            st.markdown(f"• {narrative}")
