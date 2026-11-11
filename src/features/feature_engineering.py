import os
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger


class FeatureEngineer:
    """Transforms raw voltage, temperature, and cycle time-series into

    physics-informed tabular degradation features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes lag deltas, rolling smoothing, thermal stress,

        and impedance proxy ratios grouped by battery_id.

        Args:
            df (pd.DataFrame): Validated clean dataframe.

        Returns:
            pd.DataFrame: Dataframe enriched with engineered features.
        """
        self.logger.info("Starting physics-informed feature engineering...")
        df_feats = df.copy()

        # Sort values chronologically per battery to ensure sequential integrity
        df_feats = df_feats.sort_values(
            by=["battery_id", "cycle"]
        ).reset_index(drop=True)

        # 1. First-Order Difference (Per-cycle Rate of Change)
        self.logger.info("Computing first-order rate-of-change deltas...")
        df_feats["voltage_diff"] = df_feats.groupby("battery_id")[
            "voltage"
        ].diff()
        df_feats["temp_diff"] = df_feats.groupby("battery_id")[
            "temperature"
        ].diff()

        # Initial cycle per battery will be NaN for diff; fill with 0.0
        df_feats["voltage_diff"] = df_feats["voltage_diff"].fillna(0.0)
        df_feats["temp_diff"] = df_feats["temp_diff"].fillna(0.0)

        # 2. Short-window Rolling Aggregations (Noise filtering & local volatility)
        self.logger.info("Extracting rolling statistics (window = 3)...")
        grouped = df_feats.groupby("battery_id")

        df_feats["voltage_roll_mean_3"] = grouped["voltage"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_feats["temp_roll_mean_3"] = grouped["temperature"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_feats["temp_roll_std_3"] = (
            grouped["temperature"]
            .transform(lambda x: x.rolling(window=3, min_periods=1).std())
            .fillna(0.0)
        )

        # 3. Electro-Thermal Degradation Ratio (V / T)
        # As internal resistance climbs, V drops and T climbs -> ratio drops steeply
        self.logger.info(
            "Synthesizing Electro-Thermal ratio (V / T impedance proxy)..."
        )
        df_feats["v_t_ratio"] = df_feats["voltage"] / (
            df_feats["temperature"] + 1e-5
        )

        # 4. Thermal Cumulative Stress History (Total heat exposure across lifespan)
        self.logger.info("Accumulating thermal fatigue history...")
        df_feats["thermal_cum_stress"] = grouped["temperature"].cumsum()

        self.logger.info(
            f"Feature engineering completed. Shape: {df_feats.shape[0]} rows, "
            f"{df_feats.shape[1]} columns ({len(df_feats.columns) - len(df.columns)} new features added)."
        )

        return df_feats


if __name__ == "__main__":
    # Test script independently using preprocessed data
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocessed_path = config["data"]["preprocessed_path"]
    if not os.path.exists(preprocessed_path):
        print(f"File {preprocessed_path} not found. Run data_validator first.")
        sys.exit(1)

    df_clean = pd.read_csv(preprocessed_path)
    engineer = FeatureEngineer(config)
    df_engineered = engineer.engineer_features(df_clean)

    print("\nFeature Engineering Snapshot (B0005):")
    sample_cols = [
        "battery_id",
        "cycle",
        "voltage",
        "voltage_diff",
        "temp_roll_mean_3",
        "v_t_ratio",
        "thermal_cum_stress",
        "rul",
    ]
    print(df_engineered[sample_cols].head(5))