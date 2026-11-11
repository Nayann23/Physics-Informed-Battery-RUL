import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import joblib

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger


class ArtifactManager:
    """Centralized manager for serializing models, saving evaluation metrics,
    and managing pipeline artifacts on disk.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )

    def save_model(self, model: Any, filename: str = "battery_rul_model.joblib") -> str:
        """Saves a trained model to the configured artifacts/models directory."""
        model_dir = self.config["artifacts"].get("model_dir", "artifacts/models")
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, filename)

        joblib.dump(model, file_path)
        self.logger.info(f"Model artifact successfully saved to: {file_path}")
        return file_path

    def load_model(self, filename: str = "battery_rul_model.joblib") -> Any:
        """Loads a saved model from disk."""
        model_dir = self.config["artifacts"].get("model_dir", "artifacts/models")
        file_path = os.path.join(model_dir, filename)

        if not os.path.exists(file_path):
            self.logger.error(f"Model file not found at: {file_path}")
            raise FileNotFoundError(f"Model artifact missing: {file_path}")

        model = joblib.load(file_path)
        self.logger.info(f"Model artifact successfully loaded from: {file_path}")
        return model

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "evaluation_metrics.json") -> str:
        """Saves evaluation performance metrics dictionary as a formatted JSON file."""
        metrics_dir = self.config["artifacts"].get("metrics_dir", "artifacts/metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        file_path = os.path.join(metrics_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        self.logger.info(f"Evaluation metrics JSON persisted to: {file_path}")
        return file_path

    def load_metrics(self, filename: str = "evaluation_metrics.json") -> Dict[str, Any]:
        """Loads evaluation metrics from disk."""
        metrics_dir = self.config["artifacts"].get("metrics_dir", "artifacts/metrics")
        file_path = os.path.join(metrics_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metrics file missing: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        return metrics


if __name__ == "__main__":
    import yaml

    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    manager = ArtifactManager(config)

    # Test saving sample metrics JSON
    sample_metrics = {
        "rmse": 4.12,
        "mae": 2.85,
        "r2_score": 0.96,
        "evaluated_cell": "B0018"
    }
    saved_path = manager.save_metrics(sample_metrics, "test_run_metrics.json")
    loaded = manager.load_metrics("test_run_metrics.json")
    print("\nArtifactManager Test Succeeded:")
    print(f"Loaded from disk: {loaded}")