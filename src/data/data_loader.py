import os
from typing import Dict, Any
import pandas as pd
import yaml

from src.utils.logger import get_logger

class DataLoader:
    """Handles raw data ingestion from disk with configurable debug slicing

    and validation of file existence.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )

    def load_data(self) -> pd.DataFrame:
        """Loads the raw battery cycle dataset from the configured path.

        Applies sequential battery filtering if debug mode is active.

        Returns:
            pd.DataFrame: Loaded tabular dataframe.
        """
        raw_path = self.config["data"]["raw_path"]
        debug_mode = self.config["pipeline"].get("debug_mode", False)

        self.logger.info(f"Initiating data ingestion from: {raw_path}")

        # 1. Existence check to fail fast with a descriptive error
        if not os.path.exists(raw_path):
            self.logger.error(f"Dataset file not found at: {raw_path}")
            raise FileNotFoundError(
                f"Dataset not found at {raw_path}. Verify folder path."
            )

        # 2. Ingest raw CSV
        df = pd.read_csv(raw_path)
        self.logger.info(
            f"Successfully read raw CSV. Shape: {df.shape[0]} rows, {df.shape[1]} columns"
        )

        # 3. Sequential debug mode (Preserves time-series continuity)
        if debug_mode:
            debug_battery = self.config["pipeline"].get(
                "debug_battery_id", "B0005"
            )
            self.logger.warning(
                f"Debug mode is ENABLED. Isolating single battery '{debug_battery}' to maintain cycle sequence."
            )

            df = (
                df[df["battery_id"] == debug_battery]
                .sort_values(by="cycle")
                .reset_index(drop=True)
            )

            self.logger.info(
                f"Filtered to battery '{debug_battery}' with shape: {df.shape}"
            )

        self.logger.info(f"Data loading stage complete. Output shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Independent module validation
    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(config)
    data = loader.load_data()
    print("\nPreview:")
    print(data.head(3))