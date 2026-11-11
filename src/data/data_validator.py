import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger


class DataValidator:
    """Validates data schema, structural completeness, and domain physics bounds

    before letting data pass to the feature engineering pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )
        self.expected_columns = [
            "battery_id",
            "cycle",
            "voltage",
            "temperature",
            "capacity",
            "soh",
            "rul",
        ]

    def validate(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """Runs end-to-end validation checks on the loaded dataset.

        Args:
            df (pd.DataFrame): Raw ingested dataframe.

        Returns:
            Tuple[bool, pd.DataFrame]: (is_valid status, cleaned/validated df).
        """
        self.logger.info("Starting data validation sequence...")

        # 1. Schema Check: Missing required columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            self.logger.error(
                f"Validation Failed: Missing critical columns: {missing_cols}"
            )
            raise ValueError(f"Schema mismatch! Missing: {missing_cols}")
        self.logger.info(
            "Schema validation PASSED: All expected columns are present."
        )

        # 2. Null / NaN Value Check
        null_counts = df[self.expected_columns].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            self.logger.warning(
                f"Found {total_nulls} null entries. Dropping rows with missing values."
            )
            df = df.dropna(subset=self.expected_columns).reset_index(drop=True)
        else:
            self.logger.info(
                "Completeness check PASSED: Zero null values found."
            )

        # 3. Physics Range Validation (Config Driven)
        v_min = self.config["data"]["validation_ranges"]["voltage_min"]
        v_max = self.config["data"]["validation_ranges"]["voltage_max"]
        t_min = self.config["data"]["validation_ranges"]["temp_min"]
        t_max = self.config["data"]["validation_ranges"]["temp_max"]

        initial_len = len(df)
        valid_mask = (
            (df["voltage"] >= v_min)
            & (df["voltage"] <= v_max)
            & (df["temperature"] >= t_min)
            & (df["temperature"] <= t_max)
        )

        outliers_count = initial_len - valid_mask.sum()
        if outliers_count > 0:
            self.logger.warning(
                f"Detected {outliers_count} anomalous rows violating physical bounds "
                f"(Voltage [{v_min}, {v_max}], Temp [{t_min}, {t_max}]). Filtering them out."
            )
            df = df[valid_mask].reset_index(drop=True)
        else:
            self.logger.info(
                f"Physics bounds validation PASSED: All readings within safe ranges."
            )

        # 4. Low-Cycle Battery Filtering (Audit Insight)
        # Aborted tests (1-5 cycles) ko drop karna taaki model noise se bache
        min_cycles = self.config["data"].get("min_cycle_threshold", 50)
        cycle_counts = df.groupby("battery_id")["cycle"].max()
        valid_batteries = cycle_counts[cycle_counts >= min_cycles].index

        dropped_batteries = set(df["battery_id"].unique()) - set(
            valid_batteries
        )
        if dropped_batteries:
            self.logger.info(
                f"Filtering out incomplete/aborted battery tests (< {min_cycles} cycles): "
                f"{len(dropped_batteries)} batteries removed."
            )
            df = df[df["battery_id"].isin(valid_batteries)].reset_index(
                drop=True
            )

        self.logger.info(
            f"Validation complete. Clean dataset shape: {df.shape[0]} rows, {df.shape[1]} columns."
        )

        # Preprocessed folder me validated data save karna
        preprocessed_path = self.config["data"]["preprocessed_path"]
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        df.to_csv(preprocessed_path, index=False)
        self.logger.info(
            f"Saved validated intermediate dataset to: {preprocessed_path}"
        )

        return True, df


if __name__ == "__main__":
    from src.data.data_loader import DataLoader

    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(config)
    raw_df = loader.load_data()

    validator = DataValidator(config)
    is_valid, validated_df = validator.validate(raw_df)
    print(f"\nFinal Validated Batteries: {validated_df['battery_id'].unique()}")