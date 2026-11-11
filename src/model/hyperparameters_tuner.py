import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.model.model_factory import ModelFactory


class HyperparameterTuner:
    """Performs CPU-efficient hyperparameter search using RandomizedSearchCV

    calibrated specifically for low-overhead execution on dual-core hardware.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )

    def tune(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Dict[str, Any], Any]:
        """Runs a lightweight cross-validated search for optimal boosting parameters.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Target RUL series.

        Returns:
            Tuple[Dict[str, Any], Any]: (best_params dictionary, best_estimator)
        """
        self.logger.info("Initializing lightweight hyperparameter search space...")

        # Small, high-impact search grid tuned for tabular gradient boosting
        param_distributions = {
            "n_estimators": [80, 120, 160],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [3, 4, 5],
            "num_leaves": [12, 15, 20],
        }

        factory = ModelFactory(self.config)
        base_model = factory.get_model()

        # 3-Fold CV with 6 randomized iterations = 18 fits total (executes in ~3 seconds)
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=6,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        self.logger.info("Executing parameter optimization...")
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_score = -random_search.best_score_

        self.logger.info(f"Hyperparameter search complete.")
        self.logger.info(f"Best CV RMSE: {best_score:.4f}")
        self.logger.info(f"Optimal Parameters: {best_params}")

        return best_params, random_search.best_estimator_


if __name__ == "__main__":
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_path = config["data"]["final_train_path"]
    df_train = pd.read_csv(train_path)

    target_col = config["split"].get("target_column", "rul")
    drop_cols = {"battery_id", target_col}
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    tuner = HyperparameterTuner(config)
    best_params, best_model = tuner.tune(X_train, y_train)

    print("\nTuning Summary:")
    print(f"Optimal Hyperparameters found: {best_params}")