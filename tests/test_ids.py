"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Comprehensive Test Suite covering Data Generation, Dual Model Inference,
TreeSHAP Feature Explanations, MITRE ATT&CK Mapping, and Alerting.
"""

import os
import sys
from pathlib import Path

# Ensure matplotlib does not attempt to write outside sandbox
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/matplotlib", exist_ok=True)

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_service import AlertDispatcher
from src.data_loader import (
    ATTACK_CLASSES,
    FEATURE_COLUMNS,
    clean_and_prepare_features,
    generate_benchmark_dataset,
    generate_flow_record,
)
from src.explainer import IDSExplainer
from src.models import DualIDSModelEngine
from src.threat_intel import MITRE_ATTCK_MAPPINGS, generate_mitigation_playbook


@pytest.fixture(scope="session")
def trained_engine():
    models_dir = PROJECT_ROOT / "models"
    assert (models_dir / "xgboost_ids.joblib").exists(), "Model artifacts must exist"
    return DualIDSModelEngine.load_artifacts(models_dir)


def test_feature_columns_and_generation():
    rng = np.random.default_rng(123)
    for attack in ATTACK_CLASSES:
        rec = generate_flow_record(attack, rng)
        assert rec["label"] == attack
        assert "src_ip" in rec and "dst_ip" in rec
        for col in FEATURE_COLUMNS:
            assert col in rec, f"Missing {col} in record"
            assert not np.isnan(rec[col]), f"{col} should not be NaN"


def test_data_cleaning_and_scaling():
    df = generate_benchmark_dataset(n_samples=50, random_state=42)
    # Inject an infinity and NaN to verify robust handling
    df.loc[0, "flow_bytes_per_s"] = np.inf
    df.loc[1, "flow_iat_std"] = np.nan

    X_scaled, X_raw, scaler = clean_and_prepare_features(df)
    assert X_scaled.shape == (50, len(FEATURE_COLUMNS))
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()


def test_supervised_and_autoencoder_predictions(trained_engine):
    rng = np.random.default_rng(42)

    # 1. Benign flow should be classified as BENIGN with high confidence
    benign_flow = generate_flow_record("BENIGN", rng)
    benign_pred = trained_engine.predict_flow(benign_flow)
    assert benign_pred["predicted_class"] == "BENIGN"
    assert benign_pred["risk_score"] < 50

    # 2. Attack flows should be classified properly
    for attack in ["DoS / DDoS", "PortScan", "BruteForce", "Botnet (C2)", "Web Attack"]:
        attack_flow = generate_flow_record(attack, rng)
        pred = trained_engine.predict_flow(attack_flow)
        assert pred["predicted_class"] == attack
        assert pred["risk_score"] >= 50
        assert pred["severity"] in ["MEDIUM", "HIGH", "CRITICAL"]


def test_shap_explainer_attribution(trained_engine):
    explainer = IDSExplainer(trained_engine.classifier, trained_engine.label_to_idx)
    rng = np.random.default_rng(999)
    dos_flow = generate_flow_record("DoS / DDoS", rng)
    pred = trained_engine.predict_flow(dos_flow)

    explanation = explainer.explain_flow(
        pred["scaled_vector"],
        pred["predicted_class"],
        pred["raw_features"],
        top_k=3,
    )

    assert explanation["predicted_class"] == "DoS / DDoS"
    assert len(explanation["top_drivers"]) == 3
    assert len(explanation["narratives"]) == 3
    assert "display_name" in explanation["top_drivers"][0]
    assert "shap_value" in explanation["top_drivers"][0]


def test_threat_intel_and_mitigation_playbook():
    for attack in ATTACK_CLASSES:
        mitre = MITRE_ATTCK_MAPPINGS.get(attack)
        assert mitre is not None, f"Missing MITRE mapping for {attack}"
        assert "tactic" in mitre and "technique_id" in mitre

    playbook = generate_mitigation_playbook(
        attack_type="DoS / DDoS",
        src_ip="185.220.101.5",
        dst_ip="10.0.0.15",
        dst_port=80,
    )
    assert "iptables -I INPUT -s 185.220.101.5" in playbook["iptables"]
    assert "ufw insert 1 deny from 185.220.101.5" in playbook["ufw"]
    assert "alert" in playbook["suricata_rule"]
    assert len(playbook["mitigation_steps"]) >= 3


def test_alert_dispatcher():
    dispatcher = AlertDispatcher()
    incident = {
        "verdict": "DoS / DDoS",
        "severity": "CRITICAL",
        "risk_score": 95,
        "src_ip": "185.220.101.5",
        "dst_ip": "10.0.0.42",
        "dst_port": 80,
        "confidence": 0.98,
        "country": "Russia",
        "asn": "AS49453 HostKey",
    }
    res = dispatcher.dispatch(incident)
    assert res["status"] == "simulated"
    assert "text" in res["payload"]
    assert "185.220.101.5" in res["payload"]["text"]
