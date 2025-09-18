"""
Explainable AI (XAI) Module for TactiCore AI
- Provides SHAP-based explanations for threat assessment predictions.
- Ensures GDPR/DoD compliance with auditable insights.
- Alpha stage: Integrates with data_fusion.py for real-time fusion outputs.
"""

import torch
import torch.nn as nn
import shap
from tacticore.data_fusion import fuse_intel  # Import your fusion stub
import numpy as np

# Simple Neural Network Model (placeholder for threat prediction)
class ThreatModel(nn.Module):
    def __init__(self, input_size=2):
        super(ThreatModel, self).__init__()
        self.layer = nn.Linear(input_size, 1)  # 2 inputs (e.g., OSINT, CCTV)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))  # Output: Threat probability (0-1)

# XAI Module
class ExplainableAI:
    def __init__(self):
        # Initialize model and train with dummy data (replace with real training later)
        self.model = ThreatModel()
        self._train_model()
        self.explainer = None  # SHAP explainer initialized on first use

    def _train_model(self):
        """Train a simple model with dummy fusion data for demo purposes."""
        # Dummy training data: OSINT and CCTV scores
        X_train = torch.tensor([[0.8, 0.6], [0.7, 0.5], [0.9, 0.7]], dtype=torch.float32)
        y_train = torch.tensor([[0.85], [0.70], [0.90]], dtype=torch.float32)  # Threat labels
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for _ in range(100):  # Quick training loop
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    def explain_threat(self, intel_data: dict) -> tuple[float, dict]:
        """
        Generate threat score and SHAP explanation.
        Args:
            intel_data (dict): e.g., {"osint": 0.8, "cctv": 0.6}
        Returns:
            tuple: (threat_score, explanation_dict)
        """
        # Convert intel to model input
        input_data = torch.tensor(
            [[intel_data.get("osint", 0.0), intel_data.get("cctv", 0.0)]],
            dtype=torch.float32
        )
        threat_score = self.model(input_data).item()  # Predict threat probability

        # Initialize SHAP explainer if not done
        if self.explainer is None:
            background = torch.tensor([[0.5, 0.5]], dtype=torch.float32)  # Baseline
            self.explainer = shap.KernelExplainer(self.model, background)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(input_data.numpy())[0]
        explanation = {
            "threat_score": f"{threat_score:.2f}",
            "osint_contribution": f"{shap_values[0]:.2f}",
            "cctv_contribution": f"{shap_values[1]:.2f}",
            "interpretation": (
                "Higher values indicate stronger influence on threat score."
                f" OSINT: {shap_values[0]:+.2f}, CCTV: {shap_values[1]:+.2f}"
            )
        }

        return threat_score, explanation

# Example usage (for testing)
if __name__ == "__main__":
    xai = ExplainableAI()
    intel = {"osint": 0.8, "cctv": 0.6}  # Sample data
    score, explanation = xai.explain_threat(intel)
    print(f"Threat Score: {score:.2f}")
    print("Explanation:", explanation)
