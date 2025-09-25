"""
Integrated XAI Module for TactiCore AI
- Explains route planning and risk assessment decisions.
- Integrates with data_fusion.py and document-based analysis.
- Standalone run-able: Test with sample intel/documents.
- Requirements: pip install torch shap numpy transformers
"""

import torch
import shap
import numpy as np
from typing import Dict, Tuple
from tacticore.data_fusion import DataFusion  # Assume existing

class TactiCoreModel(torch.nn.Module):
    def __init__(self, input_size=4):  # e.g., humint, sigint, osint, cctv
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.route_output = torch.nn.Linear(8, 3)  # 3 route options
        self.risk_output = torch.nn.Linear(8, 1)   # Risk score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        route_scores = torch.softmax(self.route_output(x), dim=1)
        risk_score = torch.sigmoid(self.risk_output(x))
        return route_scores, risk_score

class XAIExplainer:
    def __init__(self):
        self.model = TactiCoreModel()
        self._train_model()  # Dummy training
        self.fusion = DataFusion()
        self.explainer_route = None
        self.explainer_risk = None

    def _train_model(self):
        """Dummy training for alpha demo."""
        X_train = torch.tensor([[0.7, 0.6, 0.8, 0.5], [0.6, 0.5, 0.7, 0.4]], dtype=torch.float32)
        y_route = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32)
        y_risk = torch.tensor([[0.75], [0.6]], dtype=torch.float32)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            route_out, risk_out = self.model(X_train)
            loss = criterion(route_out, y_route) + criterion(risk_out, y_risk)
            loss.backward()
            optimizer.step()

    def explain_decisions(self, intel_data: Dict[str, float], reliability: Dict[str, float] = None) -> Dict:
        """Explain route planning and risk assessment with SHAP."""
        fused = self.fusion.fuse_intel(intel_data, reliability)
        input_data = torch.tensor([v.item() for v in fused["raw_data"].values()], dtype=torch.float32).unsqueeze(0)
        route_scores, risk_score = self.model(input_data)

        # Initialize SHAP
        if self.explainer_route is None:
            background = torch.tensor([[0.5] * input_data.shape[1]])
            def route_predict(x):
                return self.model(torch.tensor(x))[0].detach().numpy()
            self.explainer_route = shap.KernelExplainer(route_predict, background.numpy())
            def risk_predict(x):
                return self.model(torch.tensor(x))[1].detach().numpy()
            self.explainer_risk = shap.KernelExplainer(risk_predict, background.numpy())

        # SHAP values
        shap_route = self.explainer_route.shap_values(input_data.numpy())
        shap_risk = self.explainer_risk.shap_values(input_data.numpy())

        sources = list(intel_data.keys())
        explanation = {
            "route_planning": {
                "scores": route_scores.squeeze().tolist(),
                "shap_contributions": [shap_route[i][0].tolist() for i in range(3)],
                "interpretation": f"SHAP shows impact on routes: {', '.join([f'{s}: {shap_route[0][0][j]:+.2f}' for j, s in enumerate(sources)])}"
            },
            "risk_assessment": {
                "score": risk_score.item(),
                "shap_contributions": shap_risk[0].tolist(),
                "interpretation": f"SHAP shows risk impact: {', '.join([f'{s}: {shap_risk[0][j]:+.2f}' for j, s in enumerate(sources)])}"
            },
            "metadata": fused["metadata"]
        }

        return explanation

# Standalone Test
if __name__ == "__main__":
    xai = XAIExplainer()
    intel = {"humint": 0.7, "sigint": 0.6, "osint": 0.8, "cctv": 0.5}
    reliability = {"humint": 0.95, "osint": 0.85}
    result = xai.explain_decisions(intel, reliability)
    print("Explanation Result:")
    print(result)
