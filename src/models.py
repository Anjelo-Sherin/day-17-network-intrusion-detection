"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Dual-Engine Model Architecture:
1. Supervised Multi-class XGBoost Classifier (Known Attacks)
2. Unsupervised Neural Autoencoder (Zero-Day Anomaly Detection)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_loader import ATTACK_CLASSES, FEATURE_COLUMNS


class DualIDSModelEngine:
    """
    Coordinates both Supervised Multi-class Detection (XGBoost)
    and Unsupervised Reconstruction-based Anomaly Detection (Autoencoder).
    """

    def __init__(
        self,
        classifier: Optional[XGBClassifier] = None,
        autoencoder: Optional[MLPRegressor] = None,
        scaler: Optional[StandardScaler] = None,
        anomaly_threshold: float = 0.85,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.scaler = scaler
        self.anomaly_threshold = anomaly_threshold
        self.label_to_idx = label_to_idx or {cls: i for i, cls in enumerate(ATTACK_CLASSES)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def train_supervised_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Trains the multi-class XGBoost classifier with class balancing."""
        self.classifier = XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=len(ATTACK_CLASSES),
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.classifier.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        metrics = {}
        if X_val is not None and y_val is not None:
            y_pred = self.classifier.predict(X_val)
            metrics["val_f1_macro"] = float(f1_score(y_val, y_pred, average="macro"))
            metrics["val_precision_macro"] = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
            metrics["val_recall_macro"] = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        return metrics

    def train_autoencoder(
        self,
        X_benign_train: np.ndarray,
        percentile_threshold: float = 96.5,
    ) -> float:
        """
        Trains an unsupervised neural autoencoder (compressing 25 features to 8 bottleneck neurons)
        exclusively on normal BENIGN network flows.
        """
        # Bottleneck architecture: 25 -> 14 -> 8 -> 14 -> 25
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(14, 8, 14),
            activation="relu",
            solver="adam",
            max_iter=75,
            alpha=1e-4,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=8,
        )
        self.autoencoder.fit(X_benign_train, X_benign_train)

        # Compute reconstruction error on benign samples to establish dynamic threshold
        X_recon = self.autoencoder.predict(X_benign_train)
        errors = np.mean(np.square(X_benign_train - X_recon), axis=1)
        self.anomaly_threshold = float(np.percentile(errors, percentile_threshold))
        return self.anomaly_threshold

    def compute_reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calculates mean squared reconstruction error per sample."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder is not trained or loaded.")
        X_recon = self.autoencoder.predict(X_scaled)
        errors = np.mean(np.square(X_scaled - X_recon), axis=1)
        return errors

    def predict_flow(self, flow_dict: Dict) -> Dict:
        """
        Processes a single flow record and produces:
        - Supervised attack classification + class probabilities
        - Autoencoder reconstruction anomaly score & zero-day alert flag
        - Unified consensus verdict & risk score (0-100)
        """
        # Extract features in required order using DataFrame to preserve feature names
        raw_df = pd.DataFrame([[flow_dict.get(col, 0.0) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        raw_df = raw_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.scaler is not None:
            scaled_vals = self.scaler.transform(raw_df)
        else:
            scaled_vals = raw_df.values

        # 1. Supervised prediction
        probs = self.classifier.predict_proba(scaled_vals)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.idx_to_label[pred_idx]
        confidence = float(probs[pred_idx])

        # 2. Unsupervised autoencoder anomaly score
        recon_error = float(self.compute_reconstruction_error(scaled_vals)[0])
        is_novel_anomaly = recon_error > self.anomaly_threshold

        # 3. Combined SOC verdict & risk assessment
        # Determine threat level: CRITICAL, HIGH, MEDIUM, LOW, INFO
        if pred_label == "BENIGN":
            if is_novel_anomaly:
                verdict = "Zero-Day Anomaly"
                severity = "HIGH"
                risk_score = min(int(50 + (recon_error / max(self.anomaly_threshold, 1e-4)) * 30), 95)
            else:
                verdict = "BENIGN"
                severity = "INFO"
                risk_score = int(max(0, (1.0 - confidence) * 20))
        else:
            verdict = pred_label
            if pred_label in ["DoS / DDoS", "Botnet (C2)"]:
                severity = "CRITICAL"
                risk_score = int(80 + confidence * 20)
            elif pred_label in ["BruteForce", "Web Attack"]:
                severity = "HIGH"
                risk_score = int(70 + confidence * 25)
            else:  # PortScan
                severity = "MEDIUM"
                risk_score = int(50 + confidence * 35)

        return {
            "verdict": verdict,
            "predicted_class": pred_label,
            "confidence": confidence,
            "class_probabilities": {self.idx_to_label[i]: float(p) for i, p in enumerate(probs)},
            "reconstruction_error": recon_error,
            "anomaly_threshold": self.anomaly_threshold,
            "is_novel_anomaly": bool(is_novel_anomaly),
            "severity": severity,
            "risk_score": risk_score,
            "raw_features": {col: flow_dict[col] for col in FEATURE_COLUMNS},
            "scaled_vector": scaled_vals[0].tolist(),
        }

    def save_artifacts(self, directory: Union[str, Path]) -> None:
        """Saves model weights, scaler, and threshold metadata."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.classifier, dir_path / "xgboost_ids.joblib")
        joblib.dump(self.autoencoder, dir_path / "autoencoder_ids.joblib")
        joblib.dump(self.scaler, dir_path / "scaler.joblib")

        meta = {
            "anomaly_threshold": self.anomaly_threshold,
            "feature_columns": FEATURE_COLUMNS,
            "attack_classes": ATTACK_CLASSES,
            "label_to_idx": self.label_to_idx,
        }
        with open(dir_path / "model_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_artifacts(cls, directory: Union[str, Path]) -> "DualIDSModelEngine":
        """Loads serialized model artifacts from disk."""
        dir_path = Path(directory)
        classifier = joblib.load(dir_path / "xgboost_ids.joblib")
        autoencoder = joblib.load(dir_path / "autoencoder_ids.joblib")
        scaler = joblib.load(dir_path / "scaler.joblib")

        with open(dir_path / "model_metadata.json", "r") as f:
            meta = json.load(f)

        return cls(
            classifier=classifier,
            autoencoder=autoencoder,
            scaler=scaler,
            anomaly_threshold=meta["anomaly_threshold"],
            label_to_idx=meta.get("label_to_idx"),
        )
