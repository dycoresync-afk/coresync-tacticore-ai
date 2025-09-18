#!/usr/bin/env python3
"""
TactiCore AI Demo: Simple mission fusion simulation.
Run: python src/main.py
"""

import torch
from tacticore.data_fusion import fuse_intel  # Import stub

def main():
    print("=== TactiCore AI Alpha Demo ===")
    # Simulate real-time fusion
    intel_data = {"osint": torch.tensor([0.8]), "cctv": torch.tensor([0.6])}
    fused = fuse_intel(intel_data)
    print(f"Fused Threat Score: {fused.item():.2f}")
    print("XAI Explanation: High OSINT confidence drives alert.")
    print("Web3 Audit Log: Pending Solana integration.")

if __name__ == "__main__":
    main()
