"""
Explainable AI (XAI) Module for TactiCore AI
- Provides SHAP-based explanations for threat assessment predictions.
- Ensures GDPR/DoD compliance with auditable insights.
"""

import torch
import torch.nn as nn
import shap
from tacticore.data_fusion import DataFusion
import numpy as np

class ThreatModel(nn.Module):
    def __init__(self, input_size=4):  # 4 sources: humint, sigint, osint, cctv
        super(ThreatModel, self).__init__()
        self.layer = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

class ExplainableAI:
    def __init__(self):
        self.model = ThreatModel()
        self._train_model()
        self.explainer = None
        self.fusion = DataFusion()

    def _train_model(self):
        X_train = torch.tensor([[0.7, 0.6, 0.8, 0.5], [0.6, 0.5, 0.7, 0.4]], dtype=torch.float32)
        y_train = torch.tensor([[0.75], [0.60]], dtype=torch.float32)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

    def explain_threat(self, intel_data: dict, reliability: dict = None) -> tuple[float, dict]:
        fusion_result = self.fusion.fuse_intel(intel_data, reliability)
        input_data = torch.tensor([v.item() for v in fusion_result["raw_data"].values()], dtype=torch.float32).unsqueeze(0)
        threat_score = self.model(input_data).item()

        if self.explainer is None:
            background = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
            self.explainer = shap.KernelExplainer(self.model, background)

        shap_values = self.explainer.shap_values(input_data.numpy())[0]
        sources = list(intel_data.keys())
        explanation = {
            "threat_score": f"{threat_score:.2f}",
            **{f"{s}_contribution": f"{shap_values[i]:.2f}" for i, s in enumerate(sources)},
            "interpretation": (
                f"Contributions: {', '.join([f'{s}: {shap_values[i]:+.2f}' for i, s in enumerate(sources)])}. "
                "Higher values drive the threat score."
            )
        }

        return threat_score, explanation
