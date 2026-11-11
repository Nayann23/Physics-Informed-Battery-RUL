import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.model.model_factory import ModelFactory


class ModelTrainer:
    """Orchestrates model fitting, feature importance logging,

    and serialization of the trained regression estimator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )
        self.model_factory = ModelFactory(self.config)
        self.target_column = self.config["split"].get("target_column", "rul")

    def train(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseEstimator:
        """Trains the configured regression model on provided features and target.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Target RUL series.

        Returns:
            BaseEstimator: Fitted model object.
        """
        self.logger.info(
            f"Initiating model training on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
        )

        model = self.model_factory.get_model()

        # Fit estimator
        model.fit(X_train, y_train)
        self.logger.info("Model fitting successfully completed.")

        # Log Feature Importances if available (Tree-based estimators)
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)

            self.logger.info("Top degradation predictors identified by model:")
            for feat, score in importances.items():
                self.logger.info(f"  - {feat:<20}: {score}")

        # Save model artifact to disk
        model_dir = self.config["artifacts"]["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "battery_rul_model.joblib")

        joblib.dump(model, model_path)
        self.logger.info(f"Trained model artifact exported to: {model_path}")

        return model


if __name__ == "__main__":
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_path = config["data"]["final_train_path"]
    if not os.path.exists(train_path):
        print(f"File {train_path} not found. Run train_test_splitter first.")
        sys.exit(1)

    df_train = pd.read_csv(train_path)

    # Separate X and y
    target_col = config["split"].get("target_column", "rul")
    drop_cols = {"battery_id", target_col}
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    trainer = ModelTrainer(config)
    fitted_model = trainer.train(X_train, y_train)

    print("\nTraining verification complete.")