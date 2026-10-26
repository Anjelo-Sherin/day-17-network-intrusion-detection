"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Real-Time Alert Dispatcher supporting Slack, Discord, and Telegram Webhooks.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

import requests


class AlertDispatcher:
    """Manages instant SOC incident dispatching to webhooks or message queues."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url

    def build_alert_payload(self, incident: Dict) -> Dict:
        """Formats standard security alert payload with severity and telemetry."""
        severity = incident.get("severity", "MEDIUM")
        verdict = incident.get("verdict", "Unknown")
        src_ip = incident.get("src_ip", "0.0.0.0")
        dst_ip = incident.get("dst_ip", "0.0.0.0")
        dst_port = incident.get("dst_port", 0)
        risk = incident.get("risk_score", 50)
        country = incident.get("country", "Unknown")
        asn = incident.get("asn", "Unknown")

        # Markdown formatted text for Slack / Discord / Telegram
        text = (
            f"🚨 *[NETSENTINEL ALERT] — {severity} THREAT DETECTED*\n"
            f"• *Verdict*: `{verdict}`\n"
            f"• *Risk Score*: `{risk}/100`\n"
            f"• *Source IP*: `{src_ip}` ({country} — {asn})\n"
            f"• *Target*: `{dst_ip}:{dst_port}`\n"
            f"• *Confidence / Anomaly*: `{incident.get('confidence', 0.0):.2%}`\n"
            f"• *MITRE ATT&CK*: `{incident.get('mitre_id', 'T1498')}`\n"
        )

        return {
            "text": text,
            "incident": incident,
        }

    def dispatch(self, incident: Dict, target_url: Optional[str] = None) -> Dict:
        """Dispatches alert to external webhook endpoint."""
        url = target_url or self.webhook_url
        payload = self.build_alert_payload(incident)

        if not url:
            return {
                "status": "simulated",
                "message": "No webhook configured. Alert simulated locally in SOC buffer.",
                "payload": payload,
            }

        try:
            resp = requests.post(url, json={"text": payload["text"]}, timeout=3.0)
            return {
                "status": "delivered" if resp.status_code in [200, 204] else "failed",
                "code": resp.status_code,
                "response": resp.text[:200],
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }
