"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Offline Training Pipeline for Dual Model IDS (XGBoost Classifier + Neural Autoencoder).
Generates benchmark dataset, trains models, evaluates metrics, and exports artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Set matplotlib config dir to prevent sandbox write errors
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.makedirs("/tmp/matplotlib", exist_ok=True)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import (
    ATTACK_CLASSES,
    FEATURE_COLUMNS,
    clean_and_prepare_features,
    generate_benchmark_dataset,
)
from src.models import DualIDSModelEngine


def main():
    print("=" * 60)
    print("🛡️  NETSENTINEL SOC — IDS MODEL TRAINING PIPELINE")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "cic_ids_benchmark.csv"
    if not csv_path.exists():
        print("📦 Generating realistic CIC-IDS2017 benchmark flow dataset (15,000 samples)...")
        df = generate_benchmark_dataset(n_samples=15_000, random_state=42)
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved benchmark dataset to {csv_path}")
    else:
        print(f"📂 Loading existing dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

    print(f"📊 Dataset shape: {df.shape}")
    print("📈 Class distribution:")
    print(df["label"].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

    # Map labels to integers
    label_to_idx = {cls: i for i, cls in enumerate(ATTACK_CLASSES)}
    y = df["label"].map(label_to_idx).values

    # Preprocess & scale features
    X_scaled, X_raw, scaler = clean_and_prepare_features(df)

    # Train/test split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    engine = DualIDSModelEngine(scaler=scaler, label_to_idx=label_to_idx)

    # 1. Train Supervised Classifier
    print("\n[1/2] 🧠 Training Supervised Multi-class XGBoost Classifier...")
    cls_metrics = engine.train_supervised_classifier(X_train, y_train, X_val, y_val)
    print(f"  • Validation Macro F1: {cls_metrics.get('val_f1_macro', 0.0):.4f}")
    print(f"  • Validation Macro Precision: {cls_metrics.get('val_precision_macro', 0.0):.4f}")
    print(f"  • Validation Macro Recall: {cls_metrics.get('val_recall_macro', 0.0):.4f}")

    # Generate full report
    y_pred = engine.classifier.predict(X_val)
    report = classification_report(
        y_val, y_pred, target_names=ATTACK_CLASSES, output_dict=True
    )
    print("\nDetailed Per-Class Classification Report:")
    print(classification_report(y_val, y_pred, target_names=ATTACK_CLASSES))

    # 2. Train Unsupervised Autoencoder on BENIGN flows only
    print("\n[2/2] 🔍 Training Unsupervised Neural Autoencoder on Benign Flows...")
    benign_idx = label_to_idx["BENIGN"]
    benign_mask = y_train == benign_idx
    X_benign_train = X_train[benign_mask]

    threshold = engine.train_autoencoder(X_benign_train, percentile_threshold=96.5)
    print(f"  • Autoencoder Bottleneck Topology: (25 -> 14 -> 8 -> 14 -> 25)")
    print(f"  • Dynamic Anomaly Threshold (96.5% percentile): {threshold:.4f}")

    # Evaluate zero-day detection on attack samples vs benign
    benign_val = X_val[y_val == benign_idx]
    attack_val = X_val[y_val != benign_idx]

    err_benign = engine.compute_reconstruction_error(benign_val)
    err_attack = engine.compute_reconstruction_error(attack_val)
    fpr = float(np.mean(err_benign > threshold))
    tpr = float(np.mean(err_attack > threshold))
    print(f"  • Autoencoder Benign False Positive Rate: {fpr:.2%}")
    print(f"  • Autoencoder Attack Anomaly Detection Rate: {tpr:.2%}")

    # Save artifacts
    print(f"\n💾 Saving model artifacts to {models_dir}...")
    engine.save_artifacts(models_dir)

    metrics_payload = {
        "classification_metrics": cls_metrics,
        "per_class_report": report,
        "autoencoder_threshold": threshold,
        "autoencoder_fpr": fpr,
        "autoencoder_tpr": tpr,
        "total_samples": len(df),
        "benign_samples": int(np.sum(y == benign_idx)),
        "attack_samples": int(np.sum(y != benign_idx)),
    }
    with open(models_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("🎉 Training pipeline complete! All artifacts ready for NetSentinel SOC.")
    print("=" * 60)


if __name__ == "__main__":
    main()
