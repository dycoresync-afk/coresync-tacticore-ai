"""
Real-time Multi-Domain Data Fusion Module for TactiCore AI
- Fuses HUMINT, SIGINT, OSINT, CCTV, and other intel sources into a unified threat score.
- Adaptive weighting based on source reliability and timestamp freshness.
- Alpha stage: Integrates with XAI and planned Web3 for auditable outputs.
"""

import torch
import logging
from datetime import datetime
import numpy as np
from typing import Dict, Optional

# Set up logging for auditability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFusion:
    def __init__(self, default_weights: Dict[str, float] = None):
        """
        Initialize fusion module with configurable weights.
        Args:
            default_weights (dict): Initial weights for intel sources (e.g., {'osint': 0.3, 'cctv': 0.2}).
        """
        self.weights = default_weights or {
            "humint": 0.25,  # Human intelligence
            "sigint": 0.25,  # Signals intelligence
            "osint": 0.25,   # Open-source intelligence
            "cctv": 0.25     # Closed-circuit TV
        }
        self.last_update = datetime.now()
        self.reliability_scores = {k: 0.9 for k in self.weights.keys()}  # Initial reliability

    def _adjust_weights(self, source: str, reliability: float) -> None:
        """Dynamically adjust weights based on source reliability (0-1)."""
        if 0 <= reliability <= 1:
            self.reliability_scores[source] = reliability
            total_reliability = sum(self.reliability_scores.values())
            if total_reliability > 0:
                for key in self.weights:
                    self.weights[key] = (self.reliability_scores[key] / total_reliability) * sum(self.weights.values())
            logger.info(f"Adjusted weight for {source} to {self.weights[source]:.2f} based on reliability {reliability}")

    def _validate_input(self, data: Dict) -> bool:
        """Validate input data for consistency and non-null values."""
        if not data or not isinstance(data, dict):
            logger.error("Invalid input: Data must be a non-empty dictionary")
            return False
        for k, v in data.items():
            if not isinstance(v, (int, float, np.number)) or v < 0 or v > 1:
                logger.error(f"Invalid value for {k}: Must be 0-1 numeric")
                return False
        return True

    def fuse_intel(self, data: Dict[str, float], reliability: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Fuse multiple intel sources into a unified threat score with metadata.
        Args:
            data (dict): Raw intel values (e.g., {"osint": 0.8, "cctv": 0.6}).
            reliability (dict): Optional reliability scores (0-1) for each source.
        Returns:
            dict: Fused score, raw data, and metadata (timestamp, weights).
        """
        try:
            if not self._validate_input(data):
                raise ValueError("Invalid input data")

            # Update weights based on reliability if provided
            if reliability:
                for source, rel in reliability.items():
                    self._adjust_weights(source, rel)

            # Convert to tensor and apply weighted fusion
            input_tensor = torch.tensor([data.get(k, 0.0) for k in self.weights.keys()], dtype=torch.float32)
            weighted_sum = torch.sum(input_tensor * torch.tensor(list(self.weights.values()), dtype=torch.float32))
            fused_score = torch.clamp(weighted_sum, 0.0, 1.0)  # Normalize to [0, 1]

            # Metadata for auditability
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "weights": {k: f"{v:.2f}" for k, v in self.weights.items()},
                "reliability": {k: f"{v:.2f}" for k, v in self.reliability_scores.items()}
            }
            logger.info(f"Fused score: {fused_score.item():.2f} with metadata: {metadata}")

            return {
                "fused_score": fused_score,
                "raw_data": {k: torch.tensor(v) for k, v in data.items()},
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Fusion failed: {str(e)}")
            return {"fused_score": torch.tensor(0.0), "raw_data": {}, "metadata": {"error": str(e)}}

# Example usage (for testing)
if __name__ == "__main__":
    fusion = DataFusion()
    intel_data = {"humint": 0.7, "sigint": 0.6, "osint": 0.8, "cctv": 0.5}
    reliability_data = {"humint": 0.95, "osint": 0.85}  # Higher reliability for HUMINT/OSINT
    result = fusion.fuse_intel(intel_data, reliability_data)
    print(f"Fused Threat Score: {result['fused_score'].item():.2f}")
    print("Metadata:", result["metadata"])
