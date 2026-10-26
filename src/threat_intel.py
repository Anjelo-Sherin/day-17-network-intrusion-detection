"""
NetSentinel SOC — Day 17: Network Intrusion Detection
Threat Intelligence Enrichment, MITRE ATT&CK Mapping, and SOC Automated Playbook Generator.
"""

from __future__ import annotations

from typing import Dict, List, Optional


MITRE_ATTCK_MAPPINGS = {
    "DoS / DDoS": {
        "tactic": "Impact (TA0040)",
        "technique_id": "T1498.001",
        "technique_name": "Network Denial of Service: Direct Network Flood",
        "severity": "CRITICAL",
        "cvss_score": 8.6,
        "description": "Adversary is flooding network interfaces with spoofed/crafted packets to disrupt service availability and exhaust state tables.",
    },
    "PortScan": {
        "tactic": "Discovery (TA0007)",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "severity": "MEDIUM",
        "cvss_score": 5.3,
        "description": "Adversary is probing IP ports across the perimeter to enumerate listening daemons, open services, and OS fingerprints.",
    },
    "BruteForce": {
        "tactic": "Credential Access (TA0006)",
        "technique_id": "T1110.001",
        "technique_name": "Brute Force: Password Guessing",
        "severity": "HIGH",
        "cvss_score": 7.8,
        "description": "Systematic automated credential guessing against remote authentication services (SSH, FTP, RDP) using dictionary attacks.",
    },
    "Botnet (C2)": {
        "tactic": "Command and Control (TA0011)",
        "technique_id": "T1071.001",
        "technique_name": "Application Layer Protocol: Web/Custom Protocols",
        "severity": "CRITICAL",
        "cvss_score": 9.1,
        "description": "Infected host is communicating with external command and control (C2) infrastructure via periodic beacon pulses.",
    },
    "Web Attack": {
        "tactic": "Initial Access / Execution (TA0001)",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "severity": "HIGH",
        "cvss_score": 8.2,
        "description": "Input injection attempt (SQLi, XSS, Path Traversal) directed against public web applications to extract data or execute commands.",
    },
    "Zero-Day Anomaly": {
        "tactic": "Defense Evasion / Unknown (TA0005)",
        "technique_id": "T1027",
        "technique_name": "Obfuscated / Uncharacterized Network Anomaly",
        "severity": "HIGH",
        "cvss_score": 7.5,
        "description": "Network telemetry deviates significantly from baseline autoencoder model. Potential unknown zero-day or covert channel.",
    },
    "BENIGN": {
        "tactic": "Normal Operations",
        "technique_id": "N/A",
        "technique_name": "Legitimate Traffic",
        "severity": "INFO",
        "cvss_score": 0.0,
        "description": "Flow telemetry conforms to standard expected protocol baselines.",
    },
}


def generate_mitigation_playbook(
    attack_type: str,
    src_ip: str,
    dst_ip: str,
    dst_port: int,
    protocol: int = 6,
) -> Dict:
    """
    Generates automated SOC remediation actions including copy-pasteable firewall commands
    and Snort/Suricata IDS rules.
    """
    proto_str = "tcp" if protocol == 6 else ("udp" if protocol == 17 else "ip")

    if attack_type == "DoS / DDoS":
        iptables_cmd = f"iptables -I INPUT -s {src_ip} -p {proto_str} --dport {dst_port} -j DROP"
        ufw_cmd = f"ufw insert 1 deny from {src_ip} to any port {dst_port}"
        suricata_rule = (
            f'alert {proto_str} {src_ip} any -> any {dst_port} '
            f'(msg:"NETSENTINEL DoS Flood Detected from {src_ip}"; flow:stateless; threshold:type both, track by_src, count 200, seconds 2; sid:1000171; rev:1;)'
        )
        actions = [
            f"1. Immediately apply edge firewall DROP rule on source IP {src_ip}",
            f"2. Engage upstream BGP Blackhole / Cloudflare Magic Transit if volumetric load exceeds 1 Gbps",
            f"3. Enable SYN cookies on target {dst_ip} (sysctl -w net.ipv4.tcp_syncookies=1)",
            f"4. Inspect upstream gateway connection rate limiter.",
        ]

    elif attack_type == "PortScan":
        iptables_cmd = f"iptables -A INPUT -s {src_ip} -m recent --set --name PORTSCAN -j DROP"
        ufw_cmd = f"ufw deny from {src_ip}"
        suricata_rule = (
            f'alert tcp {src_ip} any -> any any '
            f'(msg:"NETSENTINEL Rapid Port Scan from {src_ip}"; flags:S; threshold:type both, track by_src, count 15, seconds 3; sid:1000172; rev:1;)'
        )
        actions = [
            f"1. Rate-limit or null-route source IP {src_ip} for 24 hours",
            f"2. Ensure internal non-public services are bound to 127.0.0.1 or behind VPN",
            f"3. Deploy honeypot port listener (e.g. Endlessh / Cowrie) to capture scanner tooling signatures.",
        ]

    elif attack_type == "BruteForce":
        iptables_cmd = f"iptables -I INPUT -p tcp --dport {dst_port} -s {src_ip} -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP"
        ufw_cmd = f"ufw deny from {src_ip} to any port {dst_port}"
        suricata_rule = (
            f'alert tcp {src_ip} any -> {dst_ip} {dst_port} '
            f'(msg:"NETSENTINEL Repeated Authentication Failures from {src_ip}"; flow:established,to_server; content:"Failed password"; sid:1000173; rev:1;)'
        )
        actions = [
            f"1. Block {src_ip} via Fail2ban / CrowdSec jail immediately",
            f"2. Enforce SSH key-only authentication; disable PasswordAuthentication in sshd_config",
            f"3. Verify if target account on {dst_ip} was compromised; audit `/var/log/auth.log`.",
        ]

    elif attack_type == "Botnet (C2)":
        iptables_cmd = f"iptables -I FORWARD -d {src_ip} -j REJECT --reject-with icmp-host-prohibited"
        ufw_cmd = f"ufw deny out to {src_ip}"
        suricata_rule = (
            f'alert tcp any any -> {src_ip} {dst_port} '
            f'(msg:"NETSENTINEL Potential Botnet C2 Beacon to {src_ip}"; flow:established,to_server; threshold:type limit, track by_src, count 1, seconds 300; sid:1000174; rev:1;)'
        )
        actions = [
            f"1. Quarantine local host {dst_ip} from the LAN immediately (isolate switch port)",
            f"2. Sinkhole outbound DNS resolution for malicious domain associated with {src_ip}",
            f"3. Perform memory and endpoint forensics (check crontabs, running process trees, rootkits).",
        ]

    elif attack_type == "Web Attack":
        iptables_cmd = f"iptables -I INPUT -s {src_ip} -p tcp --dport {dst_port} -j DROP"
        ufw_cmd = f"ufw deny from {src_ip} to any port {dst_port}"
        suricata_rule = (
            f'alert http {src_ip} any -> any {dst_port} '
            f'(msg:"NETSENTINEL Web Application Injection Attack from {src_ip}"; flow:to_server; content:"UNION SELECT"; nocase; sid:1000175; rev:1;)'
        )
        actions = [
            f"1. Blacklist IP {src_ip} on Web Application Firewall (ModSecurity / Cloudflare WAF)",
            f"2. Check web server access logs for HTTP 200/500 response codes on vulnerable endpoints",
            f"3. Validate parameter parameterized queries and output escaping in backend code.",
        ]

    elif attack_type == "Zero-Day Anomaly":
        iptables_cmd = f"iptables -I INPUT -s {src_ip} -j LOG --log-prefix 'NETSENTINEL_ZERO_DAY: '"
        ufw_cmd = f"ufw deny from {src_ip}"
        suricata_rule = (
            f'alert ip {src_ip} any -> any any '
            f'(msg:"NETSENTINEL High Reconstruction Error Anomaly from {src_ip}"; sid:1000176; rev:1;)'
        )
        actions = [
            f"1. Enable full packet capture (PCAP) for all traffic from {src_ip}",
            f"2. Escalate flow telemetry to Tier 3 Threat Hunting & Forensics team",
            f"3. Sandboxing: replay flow against isolated honeypot container.",
        ]
    else:
        iptables_cmd = f"# No action required for {attack_type} traffic"
        ufw_cmd = "# No firewall modification required"
        suricata_rule = "# No IDS rule needed"
        actions = ["Flow is classified as benign. Continue passive monitoring."]

    return {
        "iptables": iptables_cmd,
        "ufw": ufw_cmd,
        "suricata_rule": suricata_rule,
        "mitigation_steps": actions,
    }
