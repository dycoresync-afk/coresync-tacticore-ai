#!/usr/bin/env python3
"""
TactiCore AI Demo: Fusion with XAI explanation.
Run: python src/main.py
"""

from tacticore.data_fusion import DataFusion
from tacticore.explainable_ai import ExplainableAI

def main():
    print("=== TactiCore AI Alpha Demo ===")
    fusion = DataFusion()
    intel_data = {"humint": 0.7, "sigint": 0.6, "osint": 0.8, "cctv": 0.5}
    reliability = {"humint": 0.95, "osint": 0.85}
    result = fusion.fuse_intel(intel_data, reliability)
    xai = ExplainableAI()
    threat_score, explanation = xai.explain_threat(intel_data, reliability)
    print(f"Fused Threat Score: {result['fused_score'].item():.2f}")
    print("XAI Explanation:", explanation["interpretation"])

if __name__ == "__main__":
    main()
