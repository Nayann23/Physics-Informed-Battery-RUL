import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager


class ModelEvaluator:
    """Evaluates the trained model on unseen test battery cycles

    and generates quantitative regression performance metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )
        self.artifact_manager = ArtifactManager(self.config)
        self.target_column = self.config["split"].get("target_column", "rul")

    def evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_battery_id: str = "B0018",
    ) -> Dict[str, Any]:
        """Calculates RMSE, MAE, and R2 on out-of-sample battery cycles.

        Args:
            model: Trained regression model.
            X_test (pd.DataFrame): Test feature matrix.
            y_test (pd.Series): Actual RUL ground truth.
            test_battery_id (str): Target cell evaluated.

        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        self.logger.info(
            f"Evaluating model on {len(X_test)} unseen cycles from cell '{test_battery_id}'..."
        )

        # Generate predictions
        y_pred = model.predict(X_test)

        # Compute regression error metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {
            "evaluated_cell": test_battery_id,
            "test_sample_count": int(len(X_test)),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
        }

        self.logger.info("Out-of-sample Evaluation Results:")
        self.logger.info(
            f"  - RMSE:     {metrics['rmse']} cycles (average deviation penalty)"
        )
        self.logger.info(
            f"  - MAE:      {metrics['mae']} cycles (mean absolute error)"
        )
        self.logger.info(
            f"  - R2 Score: {metrics['r2_score']} (explained degradation variance)"
        )

        # Save metrics JSON via ArtifactManager
        self.artifact_manager.save_metrics(
            metrics, "battery_rul_evaluation.json"
        )

        return metrics


if __name__ == "__main__":
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load test features
    test_path = config["data"]["final_test_path"]
    if not os.path.exists(test_path):
        print(f"File {test_path} not found. Run train_test_splitter first.")
        sys.exit(1)

    df_test = pd.read_csv(test_path)
    target_col = config["split"].get("target_column", "rul")

    # Drop non-feature columns
    drop_cols = {"battery_id", target_col}
    feature_cols = [c for c in df_test.columns if c not in drop_cols]

    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # 2. Load trained model artifact
    manager = ArtifactManager(config)
    trained_model = manager.load_model("battery_rul_model.joblib")

    # 3. Evaluate
    evaluator = ModelEvaluator(config)
    eval_metrics = evaluator.evaluate(
        trained_model, X_test, y_test, test_battery_id="B0018"
    )

    # Sample predictions comparison
    preds = trained_model.predict(X_test)
    comparison = pd.DataFrame(
        {
            "Actual_RUL": y_test.values[:10],
            "Predicted_RUL": np.round(preds[:10], 1),
            "Absolute_Error": np.round(np.abs(y_test.values[:10] - preds[:10]), 1),
        }
    )
    print("\nFirst 10 Cycles Prediction Snapshot (Unseen B0018):")
    print(comparison)