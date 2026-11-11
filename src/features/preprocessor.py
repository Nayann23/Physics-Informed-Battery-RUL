import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger


class BatteryPreprocessor:
    """Handles feature sanitization, leakage column removal, and robust scaling

    without causing data leakage across train and test partitions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_column = self.config["split"].get("target_column", "rul")
        self.leakage_columns = self.config["data"].get("leakage_columns", [])

    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops ground truth leakage columns and replaces inf/-inf with boundary values."""
        df_clean = df.copy()

        # 1. Drop known leakage columns
        cols_to_drop = [
            c for c in self.leakage_columns if c in df_clean.columns
        ]
        if cols_to_drop:
            self.logger.info(
                f"Dropping ground-truth leakage columns: {cols_to_drop}"
            )
            df_clean = df_clean.drop(columns=cols_to_drop)

        # 2. Handle potential Inf values from feature ratios
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )

        # Fill any residual NaN with column forward fill then median
        if df_clean[numeric_cols].isnull().sum().sum() > 0:
            df_clean[numeric_cols] = (
                df_clean[numeric_cols]
                .ffill()
                .fillna(df_clean[numeric_cols].median())
            )

        return df_clean

    def fit(self, train_df: pd.DataFrame) -> "BatteryPreprocessor":
        """Fits the scaler exclusively on training battery features to prevent leakage."""
        self.logger.info("Fitting preprocessor scaler on training set...")
        clean_train = self._sanitize(train_df)

        # Exclude metadata and target from scaling
        non_feature_cols = {"battery_id", self.target_column}
        self.feature_columns = [
            c for c in clean_train.columns if c not in non_feature_cols
        ]

        self.logger.info(
            f"Active feature set for scaling ({len(self.feature_columns)} features): {self.feature_columns}"
        )

        self.scaler.fit(clean_train[self.feature_columns])
        self.logger.info("Scaler fitted successfully on training distribution.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms features using previously fitted scaling statistics."""
        if not self.feature_columns:
            raise ValueError(
                "Preprocessor has not been fitted yet! Call .fit() first."
            )

        df_clean = self._sanitize(df)
        df_scaled = df_clean.copy()

        # Scale only feature columns
        scaled_array = self.scaler.transform(df_clean[self.feature_columns])
        df_scaled[self.feature_columns] = scaled_array

        self.logger.info(
            f"Transformed dataset. Result shape: {df_scaled.shape}"
        )
        return df_scaled

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits on train dataframe and returns transformed output."""
        return self.fit(train_df).transform(train_df)

    def save(self, filepath: str):
        """Persists the fitted preprocessor artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {"scaler": self.scaler, "feature_columns": self.feature_columns},
            filepath,
        )
        self.logger.info(f"Preprocessor artifact saved to: {filepath}")

    def load(self, filepath: str) -> "BatteryPreprocessor":
        """Loads a saved preprocessor artifact from disk."""
        artifact = joblib.load(filepath)
        self.scaler = artifact["scaler"]
        self.feature_columns = artifact["feature_columns"]
        self.logger.info(f"Preprocessor artifact loaded from: {filepath}")
        return self


if __name__ == "__main__":
    from src.features.feature_engineering import FeatureEngineer

    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocessed_path = config["data"]["preprocessed_path"]
    df_clean = pd.read_csv(preprocessed_path)

    # Feature engineering step
    engineer = FeatureEngineer(config)
    df_feat = engineer.engineer_features(df_clean)

    # Preprocessing test
    preprocessor = BatteryPreprocessor(config)
    df_transformed = preprocessor.fit_transform(df_feat)

    print("\nColumns remaining after leakage drop:")
    print(list(df_transformed.columns))

    print(
        "\nTransformed Scaled Sample (Mean ~0, Std ~1 for numerical features):"
    )
    print(df_transformed[["battery_id", "voltage", "temp_diff", "rul"]].head(3))