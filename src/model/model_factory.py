import sys
from pathlib import Path
from typing import Dict, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger

# Attempt LightGBM import with graceful Scikit-learn fallback
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class ModelFactory:
    """Instantiates configured ML regression models using the Factory Design Pattern."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"]
        )

    def get_model(self) -> BaseEstimator:
        """Retrieves and initializes the model specified in config.yaml.

        Returns:
            BaseEstimator: Initialized Scikit-learn compatible regression estimator.
        """
        model_name = self.config["model"].get("name", "lightgbm").lower()
        model_params = self.config["model"].get("params", {})

        self.logger.info(f"Initializing regression model engine: '{model_name}'")

        if model_name == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                self.logger.warning(
                    "LightGBM is not installed in the environment! Falling back to RandomForestRegressor."
                )
                return RandomForestRegressor(
                    n_estimators=model_params.get("n_estimators", 100),
                    max_depth=model_params.get("max_depth", 5),
                    random_state=model_params.get("random_state", 42),
                    n_jobs=-1
                )
            
            # Fast, cache-friendly gradient boosting
            return LGBMRegressor(
                n_estimators=model_params.get("n_estimators", 120),
                learning_rate=model_params.get("learning_rate", 0.05),
                max_depth=model_params.get("max_depth", 4),
                num_leaves=model_params.get("num_leaves", 15),
                random_state=model_params.get("random_state", 42),
                n_jobs=model_params.get("n_jobs", -1),
                verbose=-1
            )

        elif model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=model_params.get("n_estimators", 100),
                max_depth=model_params.get("max_depth", 5),
                random_state=model_params.get("random_state", 42),
                n_jobs=-1
            )

        elif model_name == "ridge":
            return Ridge(
                alpha=model_params.get("alpha", 1.0),
                random_state=model_params.get("random_state", 42)
            )

        else:
            self.logger.error(f"Unsupported model requested: '{model_name}'")
            raise ValueError(f"Unknown model name '{model_name}'. Choose 'lightgbm', 'random_forest', or 'ridge'.")


if __name__ == "__main__":
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    factory = ModelFactory(config)
    estimator = factory.get_model()
    print(f"\nModel Factory Successfully Initialized: {estimator.__class__.__name__}")
    print(f"Hyperparameters: {estimator.get_params()}")