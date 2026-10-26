"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Data Ingestion, CICFlowMeter Feature Definitions, and Real-time Flow Streamer.
Based on the CIC-IDS2017 and NSL-KDD benchmark standards.
"""

from __future__ import annotations

import datetime
import random
import time
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Core CICFlowMeter feature names selected for network intrusion detection
FEATURE_COLUMNS = [
    "flow_duration",         # Duration of the flow in microseconds
    "tot_fwd_pkts",          # Total packets in forward direction
    "tot_bwd_pkts",          # Total packets in backward direction
    "tot_len_fwd_pkts",      # Total size of forward packets (bytes)
    "tot_len_bwd_pkts",      # Total size of backward packets (bytes)
    "fwd_pkt_len_mean",      # Mean size of forward packets
    "bwd_pkt_len_mean",      # Mean size of backward packets
    "flow_bytes_per_s",      # Flow throughput in bytes/s
    "flow_pkts_per_s",       # Flow packet rate in packets/s
    "flow_iat_mean",         # Mean time between two packets
    "flow_iat_std",          # Standard deviation of inter-arrival time
    "fwd_iat_tot",           # Total time between packets sent forward
    "bwd_iat_tot",           # Total time between packets sent backward
    "fin_flag_cnt",          # FIN flag count
    "syn_flag_cnt",          # SYN flag count
    "rst_flag_cnt",          # RST flag count
    "psh_flag_cnt",          # PUSH flag count
    "ack_flag_cnt",          # ACK flag count
    "urg_flag_cnt",          # URGENT flag count
    "down_up_ratio",         # Download / upload ratio
    "pkt_size_avg",          # Average packet size
    "init_win_bytes_fwd",    # Bytes sent in initial forward window
    "init_win_bytes_bwd",    # Bytes sent in initial backward window
    "dst_port",              # Destination port (e.g. 80, 443, 22, 53, 21)
    "protocol",              # 6=TCP, 17=UDP, 1=ICMP
]

ATTACK_CLASSES = [
    "BENIGN",
    "DoS / DDoS",
    "PortScan",
    "BruteForce",
    "Botnet (C2)",
    "Web Attack",
]

# Geolocation simulation coordinates & metadata for threat mapping
GEO_IP_PROFILES = {
    "BENIGN": [
        {"ip": "10.0.1.15", "country": "United States", "city": "San Jose", "lat": 37.3382, "lon": -121.8863, "asn": "AS15169 Google LLC"},
        {"ip": "192.168.1.105", "country": "United States", "city": "Seattle", "lat": 47.6062, "lon": -122.3321, "asn": "AS8075 Microsoft"},
        {"ip": "172.16.0.42", "country": "Germany", "city": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "asn": "AS24940 Hetzner"},
        {"ip": "10.2.0.88", "country": "United Kingdom", "city": "London", "lat": 51.5074, "lon": -0.1278, "asn": "AS5089 Virgin Media"},
    ],
    "DoS / DDoS": [
        {"ip": "185.220.101.5", "country": "Russia", "city": "Moscow", "lat": 55.7558, "lon": 37.6173, "asn": "AS49453 HostKey"},
        {"ip": "45.154.255.89", "country": "Netherlands", "city": "Amsterdam", "lat": 52.3676, "lon": 4.9041, "asn": "AS44592 Tor Exit"},
        {"ip": "103.251.167.22", "country": "China", "city": "Beijing", "lat": 39.9042, "lon": 116.4074, "asn": "AS4134 Chinanet"},
    ],
    "PortScan": [
        {"ip": "194.26.29.112", "country": "Romania", "city": "Bucharest", "lat": 44.4268, "lon": 26.1025, "asn": "AS202425 IP Volume"},
        {"ip": "79.124.62.82", "country": "Bulgaria", "city": "Sofia", "lat": 42.6977, "lon": 23.3219, "asn": "AS49981 WorldStream"},
        {"ip": "193.32.161.4", "country": "Poland", "city": "Warsaw", "lat": 52.2297, "lon": 21.0122, "asn": "AS200019 VDS"},
    ],
    "BruteForce": [
        {"ip": "218.92.0.187", "country": "China", "city": "Jiangsu", "lat": 32.0617, "lon": 118.7632, "asn": "AS4134 Chinanet"},
        {"ip": "178.128.85.12", "country": "Singapore", "city": "Singapore", "lat": 1.3521, "lon": 103.8198, "asn": "AS14061 DigitalOcean"},
        {"ip": "185.191.171.14", "country": "Seychelles", "city": "Victoria", "lat": -4.6191, "lon": 55.4513, "asn": "AS59711 Hivelocity"},
    ],
    "Botnet (C2)": [
        {"ip": "91.240.118.231", "country": "Lithuania", "city": "Vilnius", "lat": 54.6872, "lon": 25.2797, "asn": "AS47583 Hostinger"},
        {"ip": "195.123.245.18", "country": "Ukraine", "city": "Kyiv", "lat": 50.4501, "lon": 30.5234, "asn": "AS39798 Serverian"},
        {"ip": "141.98.11.90", "country": "Panama", "city": "Panama City", "lat": 8.9824, "lon": -79.5199, "asn": "AS40824 Scalaxy"},
    ],
    "Web Attack": [
        {"ip": "198.51.100.77", "country": "Brazil", "city": "São Paulo", "lat": -23.5505, "lon": -46.6333, "asn": "AS27699 Telecom"},
        {"ip": "82.102.23.45", "country": "Turkey", "city": "Istanbul", "lat": 41.0082, "lon": 28.9784, "asn": "AS9121 TTNet"},
        {"ip": "179.43.149.20", "country": "Switzerland", "city": "Zurich", "lat": 47.3769, "lon": 8.5417, "asn": "AS51852 PrivateLayer"},
    ],
}


def generate_flow_record(attack_type: str, rng: np.random.Generator) -> Dict:
    """Generates a realistic single network flow record with CICFlowMeter telemetry."""
    if attack_type == "BENIGN":
        protocol = rng.choice([6, 17], p=[0.85, 0.15])
        dst_port = rng.choice([80, 443, 53, 22, 8080], p=[0.25, 0.60, 0.10, 0.03, 0.02])
        duration = float(rng.exponential(scale=350_000) + 1_000)
        tot_fwd = int(rng.poisson(lam=12) + 2)
        tot_bwd = int(rng.poisson(lam=15) + 2)
        fwd_len_mean = float(np.clip(rng.normal(loc=280, scale=80), 40, 1460))
        bwd_len_mean = float(np.clip(rng.normal(loc=720, scale=200), 40, 1460))
        tot_len_fwd = tot_fwd * fwd_len_mean
        tot_len_bwd = tot_bwd * bwd_len_mean
        flow_bytes_per_s = (tot_len_fwd + tot_len_bwd) / (duration / 1e6 + 1e-5)
        flow_pkts_per_s = (tot_fwd + tot_bwd) / (duration / 1e6 + 1e-5)
        flow_iat_mean = duration / max(tot_fwd + tot_bwd - 1, 1)
        flow_iat_std = float(rng.uniform(100, 5000))
        fwd_iat_tot = duration * rng.uniform(0.7, 0.95)
        bwd_iat_tot = duration * rng.uniform(0.6, 0.90)
        fin_flags = 1 if rng.random() > 0.4 else 0
        syn_flags = 1 if rng.random() > 0.3 else 0
        rst_flags = 0
        psh_flags = 1 if rng.random() > 0.5 else 0
        ack_flags = 1
        urg_flags = 0
        down_up = tot_bwd / max(tot_fwd, 1)
        pkt_size_avg = (tot_len_fwd + tot_len_bwd) / max(tot_fwd + tot_bwd, 1)
        win_fwd = int(rng.choice([29200, 64240, 65535, 14600]))
        win_bwd = int(rng.choice([29200, 64240, 65535, 14600]))

    elif attack_type == "DoS / DDoS":
        # Massive packet burst, high SYN, low duration or infinite byte rates, tiny packets
        protocol = rng.choice([6, 17], p=[0.7, 0.3])
        dst_port = rng.choice([80, 443, 8080], p=[0.6, 0.35, 0.05])
        duration = float(rng.uniform(100, 15_000)) # Very short lived bursts
        tot_fwd = int(rng.uniform(150, 2000))
        tot_bwd = int(rng.uniform(0, 5)) # Server overwhelmed, few responses
        fwd_len_mean = float(rng.uniform(40, 70)) # Tiny SYN packets
        bwd_len_mean = float(rng.uniform(0, 40))
        tot_len_fwd = tot_fwd * fwd_len_mean
        tot_len_bwd = tot_bwd * bwd_len_mean
        flow_bytes_per_s = (tot_len_fwd + tot_len_bwd) / (duration / 1e6 + 1e-5)
        flow_pkts_per_s = (tot_fwd + tot_bwd) / (duration / 1e6 + 1e-5)
        flow_iat_mean = float(rng.uniform(2, 50)) # Ultra tight interval
        flow_iat_std = float(rng.uniform(1, 15))
        fwd_iat_tot = duration * 0.98
        bwd_iat_tot = duration * 0.1
        fin_flags = 0
        syn_flags = 1 # Classic SYN flood signature
        rst_flags = int(rng.choice([0, 1], p=[0.7, 0.3]))
        psh_flags = 0
        ack_flags = int(rng.choice([0, 1], p=[0.8, 0.2]))
        urg_flags = 0
        down_up = tot_bwd / max(tot_fwd, 1)
        pkt_size_avg = fwd_len_mean
        win_fwd = int(rng.choice([1024, 2048, 4096, 0])) # Small or zero window
        win_bwd = 0

    elif attack_type == "PortScan":
        # Rapid scanning across varied destination ports, tiny payloads, high RST/FIN
        protocol = 6
        dst_port = int(rng.choice([21, 22, 23, 25, 53, 80, 110, 139, 443, 445, 1433, 3306, 3389, 8080]))
        duration = float(rng.uniform(500, 50_000))
        tot_fwd = int(rng.choice([1, 2, 3]))
        tot_bwd = int(rng.choice([0, 1]))
        fwd_len_mean = float(rng.uniform(40, 60))
        bwd_len_mean = 0.0 if tot_bwd == 0 else float(rng.uniform(40, 60))
        tot_len_fwd = tot_fwd * fwd_len_mean
        tot_len_bwd = tot_bwd * bwd_len_mean
        flow_bytes_per_s = (tot_len_fwd + tot_len_bwd) / (duration / 1e6 + 1e-5)
        flow_pkts_per_s = (tot_fwd + tot_bwd) / (duration / 1e6 + 1e-5)
        flow_iat_mean = duration / max(tot_fwd + tot_bwd - 1, 1)
        flow_iat_std = float(rng.uniform(5, 50))
        fwd_iat_tot = duration
        bwd_iat_tot = 0.0
        fin_flags = int(rng.choice([0, 1], p=[0.8, 0.2]))
        syn_flags = 1
        rst_flags = int(rng.choice([0, 1], p=[0.4, 0.6])) # RST responses from closed ports
        psh_flags = 0
        ack_flags = int(rng.choice([0, 1], p=[0.5, 0.5]))
        urg_flags = 0
        down_up = tot_bwd / max(tot_fwd, 1)
        pkt_size_avg = fwd_len_mean
        win_fwd = int(rng.choice([1024, 2048, 512]))
        win_bwd = 0

    elif attack_type == "BruteForce":
        # Repeated authentication attempts on SSH/FTP/RDP with fixed payload bursts
        protocol = 6
        dst_port = int(rng.choice([22, 21, 3389, 445], p=[0.55, 0.25, 0.15, 0.05]))
        duration = float(rng.uniform(40_000, 300_000))
        tot_fwd = int(rng.uniform(15, 45))
        tot_bwd = int(rng.uniform(12, 40))
        fwd_len_mean = float(np.clip(rng.normal(loc=120, scale=15), 60, 250))
        bwd_len_mean = float(np.clip(rng.normal(loc=180, scale=20), 80, 300))
        tot_len_fwd = tot_fwd * fwd_len_mean
        tot_len_bwd = tot_bwd * bwd_len_mean
        flow_bytes_per_s = (tot_len_fwd + tot_len_bwd) / (duration / 1e6 + 1e-5)
        flow_pkts_per_s = (tot_fwd + tot_bwd) / (duration / 1e6 + 1e-5)
        flow_iat_mean = float(rng.uniform(800, 3500))
        flow_iat_std = float(rng.uniform(200, 1200))
        fwd_iat_tot = duration * 0.95
        bwd_iat_tot = duration * 0.92
        fin_flags = 1
        syn_flags = 1
        rst_flags = int(rng.choice([0, 1], p=[0.6, 0.4])) # Reset on auth failure
        psh_flags = 1
        ack_flags = 1
        urg_flags = 0
        down_up = tot_bwd / max(tot_fwd, 1)
        pkt_size_avg = (tot_len_fwd + tot_len_bwd) / max(tot_fwd + tot_bwd, 1)
        win_fwd = 65535
        win_bwd = 29200

    elif attack_type == "Botnet (C2)":
        # Periodic C2 beaconing, fixed intervals, distinct payloads, non-standard high ports
        protocol = rng.choice([6, 17], p=[0.75, 0.25])
        dst_port = int(rng.choice([6667, 8000, 8080, 4444, 9999, 53], p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]))
        duration = float(rng.uniform(600_000, 4_500_000)) # Long lived periodic flow
        tot_fwd = int(rng.uniform(4, 16))
        tot_bwd = int(rng.uniform(4, 16))
        fwd_len_mean = float(rng.uniform(64, 180)) # Small command pulses
        bwd_len_mean = float(rng.uniform(64, 250))
        tot_len_fwd = tot_fwd * fwd_len_mean
        tot_len_bwd = tot_bwd * bwd_len_mean
        flow_bytes_per_s = (tot_len_fwd + tot_len_bwd) / (duration / 1e6 + 1e-5)
        flow_pkts_per_s = (tot_fwd + tot_bwd) / (duration / 1e6 + 1e-5)
        flow_iat_mean = float(rng.uniform(50_000, 300_000)) # Regular pulse
        flow_iat_std = float(rng.uniform(100, 2500)) # Extremely low jitter
        fwd_iat_tot = duration * 0.98
        bwd_iat_tot = duration * 0.98
        fin_flags = 0
        syn_flags = 1
        rst_flags = 0
        psh_flags = 1
        ack_flags = 1
        urg_flags = 0
        down_up = 1.0
        pkt_size_avg = (tot_len_fwd + tot_len_bwd) / max(tot_fwd + tot_bwd, 1)
        win_fwd = 8192
        win_bwd = 8192

    elif attack_type == "Web Attack":
        # SQL Injection / XSS on web server ports with abnormal URI payloads & high byte counts
        protocol = 6
        dst_port = int(rng.choice([80, 443, 8080, 8443]))
        duration = float(rng.uniform(25_000, 450_000))
        tot_fwd = int(rng.uniform(6, 25))
        tot_bwd = int(rng.uniform(8, 30))
        fwd_len_mean = float(rng.uniform(450, 1200)) # Heavy SQLi / XSS string payload
        bwd_len_mean = float(rng.uniform(800, 2400)) # Server error trace or dump
        tot_len_fwd = tot_fwd * fwd_len_mean
        tot_len_bwd = tot_bwd * bwd_len_mean
        flow_bytes_per_s = (tot_len_fwd + tot_len_bwd) / (duration / 1e6 + 1e-5)
        flow_pkts_per_s = (tot_fwd + tot_bwd) / (duration / 1e6 + 1e-5)
        flow_iat_mean = float(rng.uniform(1500, 18_000))
        flow_iat_std = float(rng.uniform(800, 7500))
        fwd_iat_tot = duration * 0.9
        bwd_iat_tot = duration * 0.95
        fin_flags = 1
        syn_flags = 1
        rst_flags = int(rng.choice([0, 1], p=[0.75, 0.25]))
        psh_flags = 1
        ack_flags = 1
        urg_flags = 0
        down_up = tot_bwd / max(tot_fwd, 1)
        pkt_size_avg = (tot_len_fwd + tot_len_bwd) / max(tot_fwd + tot_bwd, 1)
        win_fwd = 29200
        win_bwd = 64240
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # Select random geo profile
    geo = random.choice(GEO_IP_PROFILES[attack_type])
    src_ip = geo["ip"]
    country = geo["country"]
    city = geo["city"]
    lat = geo["lat"] + rng.uniform(-0.05, 0.05)
    lon = geo["lon"] + rng.uniform(-0.05, 0.05)
    asn = geo["asn"]

    dst_ip = f"10.0.0.{rng.integers(2, 250)}"
    src_port = int(rng.integers(1024, 65535))

    return {
        "flow_duration": max(float(duration), 1.0),
        "tot_fwd_pkts": int(tot_fwd),
        "tot_bwd_pkts": int(tot_bwd),
        "tot_len_fwd_pkts": float(tot_len_fwd),
        "tot_len_bwd_pkts": float(tot_len_bwd),
        "fwd_pkt_len_mean": float(fwd_len_mean),
        "bwd_pkt_len_mean": float(bwd_len_mean),
        "flow_bytes_per_s": float(flow_bytes_per_s),
        "flow_pkts_per_s": float(flow_pkts_per_s),
        "flow_iat_mean": float(flow_iat_mean),
        "flow_iat_std": float(flow_iat_std),
        "fwd_iat_tot": float(fwd_iat_tot),
        "bwd_iat_tot": float(bwd_iat_tot),
        "fin_flag_cnt": int(fin_flags),
        "syn_flag_cnt": int(syn_flags),
        "rst_flag_cnt": int(rst_flags),
        "psh_flag_cnt": int(psh_flags),
        "ack_flag_cnt": int(ack_flags),
        "urg_flag_cnt": int(urg_flags),
        "down_up_ratio": float(down_up),
        "pkt_size_avg": float(pkt_size_avg),
        "init_win_bytes_fwd": int(win_fwd),
        "init_win_bytes_bwd": int(win_bwd),
        "dst_port": int(dst_port),
        "protocol": int(protocol),
        # Contextual metadata
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "country": country,
        "city": city,
        "lat": float(lat),
        "lon": float(lon),
        "asn": asn,
        "label": attack_type,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_benchmark_dataset(
    n_samples: int = 12_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates a full CIC-IDS2017 benchmark dataset replicating realistic class imbalance:
    - ~75% BENIGN (Normal)
    - ~10% DoS / DDoS
    - ~7% PortScan
    - ~4% BruteForce
    - ~2.5% Botnet
    - ~1.5% Web Attack
    """
    rng = np.random.default_rng(random_state)
    class_probs = [0.75, 0.10, 0.07, 0.04, 0.025, 0.015]

    records = []
    for _ in range(n_samples):
        attack_type = rng.choice(ATTACK_CLASSES, p=class_probs)
        records.append(generate_flow_record(attack_type, rng))

    df = pd.DataFrame(records)
    return df


def clean_and_prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Cleans infinite and NaN values (standard in network flow parsing)
    and standardizes continuous numerical features.
    """
    X = df[FEATURE_COLUMNS].copy()
    # Replace inf with high values, fill NaNs with 0
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.values, scaler


class RealTimeFlowStreamer:
    """Simulates a live streaming network flow capture engine (e.g. Suricata/Zeek feed)."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.default_rng(random_state)
        self.running = False

    def get_flow(self, forced_attack: Optional[str] = None) -> Dict:
        """Returns a single real-time network flow record."""
        if forced_attack and forced_attack in ATTACK_CLASSES:
            attack_type = forced_attack
        else:
            # Typical live operational mix (90% normal, 10% active attacks)
            attack_type = self.rng.choice(
                ATTACK_CLASSES,
                p=[0.88, 0.04, 0.03, 0.02, 0.015, 0.015],
            )
        return generate_flow_record(attack_type, self.rng)
