import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger


class TrainTestSplitter:
    """Partitions the dataset strictly by battery cell entities to test

    true out-of-sample physical generalization on unseen batteries.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(
            name=self.__class__.__name__,
            log_dir=self.config["logging"]["log_dir"],
            log_level=self.config["logging"]["log_level"],
        )
        self.train_batteries = self.config["split"]["train_batteries"]
        self.test_batteries = self.config["split"]["test_batteries"]
        self.target_column = self.config["split"].get("target_column", "rul")

    def split_by_battery(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataframe into training and testing partitions

        based on battery IDs defined in configuration.

        Args:
            df (pd.DataFrame): Preprocessed dataframe containing 'battery_id'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        self.logger.info("Executing battery-level entity split...")

        train_mask = df["battery_id"].isin(self.train_batteries)
        test_mask = df["battery_id"].isin(self.test_batteries)

        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)

        if train_df.empty:
            self.logger.error(
                f"Training set is empty! None of {self.train_batteries} found in dataset."
            )
            raise ValueError("Training partition contains 0 samples.")

        if test_df.empty:
            self.logger.error(
                f"Testing set is empty! None of {self.test_batteries} found in dataset."
            )
            raise ValueError("Testing partition contains 0 samples.")

        self.logger.info(
            f"Train split prepared: {train_df.shape[0]} cycles from cells {self.train_batteries}"
        )
        self.logger.info(
            f"Test split prepared:  {test_df.shape[0]} cycles from unseen cell {self.test_batteries}"
        )

        return train_df, test_df

    def extract_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separates the feature matrix (X) from the target series (y).

        Removes non-predictive metadata ('battery_id').

        Args:
            df (pd.DataFrame): Dataframe partition.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        ignore_cols = {"battery_id", self.target_column}
        feature_cols = [c for c in df.columns if c not in ignore_cols]

        X = df[feature_cols].copy()
        y = df[self.target_column].copy()

        return X, y

    def save_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Saves the train and test partitions to the final data folder."""
        train_path = self.config["data"]["final_train_path"]
        test_path = self.config["data"]["final_test_path"]

        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        self.logger.info(f"Saved final train partition to: {train_path}")
        self.logger.info(f"Saved final test partition to:  {test_path}")


if __name__ == "__main__":
    from src.features.feature_engineering import FeatureEngineer
    from src.features.preprocessor import BatteryPreprocessor

    config_path = "config/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load validated data
    preprocessed_path = config["data"]["preprocessed_path"]
    df_clean = pd.read_csv(preprocessed_path)

    # 2. Feature engineering
    engineer = FeatureEngineer(config)
    df_feat = engineer.engineer_features(df_clean)

    # 3. Battery-Level Split
    splitter = TrainTestSplitter(config)
    train_raw, test_raw = splitter.split_by_battery(df_feat)

    # 4. Fit Preprocessor ONLY on Train, Transform Both (MLOps Golden Rule)
    preprocessor = BatteryPreprocessor(config)
    train_final = preprocessor.fit_transform(train_raw)
    test_final = preprocessor.transform(test_raw)

    # 5. Persist final model-ready datasets
    splitter.save_splits(train_final, test_final)

    X_train, y_train = splitter.extract_features_and_target(train_final)
    X_test, y_test = splitter.extract_features_and_target(test_final)

    print("\n--- Final Dataset Summary ---")
    print(f"X_train shape: {X_train.shape}, y_train samples: {len(y_train)}")
    print(f"X_test shape:  {X_test.shape},  y_test samples:  {len(y_test)}")
    print(f"Features: {list(X_train.columns)}")