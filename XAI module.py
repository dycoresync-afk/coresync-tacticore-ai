"""
XAI Module for TactiCore AI
- Explains decisions on route planning and risk assessments using SHAP.
- Integrates with data_fusion.py for inputs.
- Alpha stage: Uses a simple neural net model for demonstration.
"""

import torch
import torch.nn as nn
import shap
import numpy as np
from typing import Dict, Tuple
from tacticore.data_fusion import DataFusion  # Assume existing fusion module

# Simple Neural Net for Route Planning and Risk Assessment
class TactiCoreModel(nn.Module):
    def __init__(self, input_size=4):  # e.g., humint, sigint, osint, cctv
        super(TactiCoreModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.route_output = nn.Linear(8, 3)  # e.g., 3 route options scores
        self.risk_output = nn.Linear(8, 1)   # Risk probability (0-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        route_scores = torch.softmax(self.route_output(x), dim=1)  # Route recommendations
        risk_score = torch.sigmoid(self.risk_output(x))  # Risk assessment
        return route_scores, risk_score

class XAIExplainer:
    def __init__(self):
        self.model = TactiCoreModel()
        self._train_model()  # Dummy training for alpha demo
        self.fusion = DataFusion()  # From data_fusion.py
        self.explainer_route = None
        self.explainer_risk = None

    def _train_model(self):
        """Dummy training with sample data."""
        X_train = torch.tensor([[0.7, 0.6, 0.8, 0.5], [0.6, 0.5, 0.7, 0.4]], dtype=torch.float32)
        y_route_train = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32)  # Route scores
        y_risk_train = torch.tensor([[0.75], [0.60]], dtype=torch.float32)  # Risk labels
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            route_out, risk_out = self.model(X_train)
            loss = criterion(route_out, y_route_train) + criterion(risk_out, y_risk_train)
            loss.backward()
            optimizer.step()

    def explain_decisions(self, intel_data: Dict[str, float], reliability: Dict[str, float] = None) -> Dict:
        """
        Explain route planning and risk assessment decisions using SHAP.
        Args:
            intel_data: Raw intel (e.g., {"humint": 0.7, "sigint": 0.6, "osint": 0.8, "cctv": 0.5})
            reliability: Optional reliability scores.
        Returns:
            Dict with route/risk scores and SHAP explanations.
        """
        fused = self.fusion.fuse_intel(intel_data, reliability)
        input_data = torch.tensor([v.item() for v in fused["raw_data"].values()], dtype=torch.float32).unsqueeze(0)
        route_scores, risk_score = self.model(input_data)

        # Initialize SHAP explainers
        if self.explainer_route is None or self.explainer_risk is None:
            background = torch.tensor([[0.5] * 4], dtype=torch.float32)  # Baseline
            self.explainer_route = shap.KernelExplainer(lambda x: self.model(torch.tensor(x))[0], background)
            self.explainer_risk = shap.KernelExplainer(lambda x: self.model(torch.tensor(x))[1], background)

        # SHAP for route planning
        shap_route = self.explainer_route.shap_values(input_data.numpy())
        route_explanation = {
            "scores": route_scores.squeeze().tolist(),  # [Route A, B, C probabilities]
            "contributions": {f"Route_{i+1}": shap_route[i][0].tolist() for i in range(3)},  # Per route, per feature
            "interpretation": "SHAP values show feature impact on each route's score; positive boosts selection."
        }

        # SHAP for risk assessment
        shap_risk = self.explainer_risk.shap_values(input_data.numpy())[0]
        risk_explanation = {
            "score": risk_score.item(),
            "contributions": shap_risk.tolist(),  # Per feature contribution to risk
            "interpretation": "SHAP values indicate how each intel source influences risk; higher positive means increased threat."
        }

        return {
            "route_planning": route_explanation,
            "risk_assessment": risk_explanation,
            "metadata": fused["metadata"]  # For auditability
        }

# Example usage
if __name__ == "__main__":
    xai = XAIExplainer()
    intel = {"humint": 0.7, "sigint": 0.6, "osint": 0.8, "cctv": 0.5}
    reliability = {"humint": 0.95, "osint": 0.85}
    explanations = xai.explain_decisions(intel, reliability)
    print("Route Planning Explanation:", explanations["route_planning"])
    print("Risk Assessment Explanation:", explanations["risk_assessment"])
