import os
import sys
import time
from pathlib import Path
import yaml

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.logger import get_logger
from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.feature_engineering import FeatureEngineer
from src.features.preprocessor import BatteryPreprocessor
from src.data.train_test_splitter import TrainTestSplitter
from src.model.model_trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator


def run_pipeline(config_path: str = "config/config.yaml"):
    """Orchestrates the entire Battery RUL Machine Learning pipeline."""
    # 0. Load Configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = get_logger(
        name="PipelineOrchestrator",
        log_dir=config["logging"]["log_dir"],
        log_level=config["logging"]["log_level"],
    )

    start_time = time.time()
    logger.info("=" * 65)
    logger.info(f"STARTING END-TO-END PIPELINE: {config['project']['name']} v{config['project']['version']}")
    logger.info("=" * 65)

    # 1. Ingestion Stage
    logger.info(">>> STAGE 1: INGESTION")
    loader = DataLoader(config)
    raw_df = loader.load_data()

    # 2. Validation Stage
    logger.info(">>> STAGE 2: VALIDATION & FILTERING")
    validator = DataValidator(config)
    _, clean_df = validator.validate(raw_df)

    # 3. Feature Engineering Stage
    logger.info(">>> STAGE 3: PHYSICS FEATURE ENGINEERING")
    engineer = FeatureEngineer(config)
    featured_df = engineer.engineer_features(clean_df)

    # 4. Entity Splitting Stage
    logger.info(">>> STAGE 4: BATTERY-LEVEL DATA SPLITTING")
    splitter = TrainTestSplitter(config)
    train_raw, test_raw = splitter.split_by_battery(featured_df)

    # 5. Preprocessing & Scaling Stage (Fit exclusively on Train)
    logger.info(">>> STAGE 5: LEAKAGE-FREE PREPROCESSING & SCALING")
    preprocessor = BatteryPreprocessor(config)
    train_final = preprocessor.fit_transform(train_raw)
    test_final = preprocessor.transform(test_raw)

    # Save fitted preprocessor artifact
    scaler_dir = config["artifacts"].get("preprocessor_dir", "artifacts/preprocessors")
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, "battery_preprocessor.joblib")
    preprocessor.save(scaler_path)

    # Save final datasets & isolate X, y
    splitter.save_splits(train_final, test_final)
    X_train, y_train = splitter.extract_features_and_target(train_final)
    X_test, y_test = splitter.extract_features_and_target(test_final)

    # 6. Model Training Stage
    logger.info(">>> STAGE 6: MODEL TRAINING")
    trainer = ModelTrainer(config)
    fitted_model = trainer.train(X_train, y_train)

    # 7. Out-of-Sample Evaluation Stage
    logger.info(">>> STAGE 7: OUT-OF-SAMPLE EVALUATION")
    evaluator = ModelEvaluator(config)
    test_battery_id = config["split"]["test_batteries"][0]
    metrics = evaluator.evaluate(fitted_model, X_test, y_test, test_battery_id=test_battery_id)

    elapsed_time = round(time.time() - start_time, 2)
    logger.info("=" * 65)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_time} SECONDS")
    logger.info(f"Final Model Metrics on unseen '{test_battery_id}': RMSE={metrics['rmse']} | MAE={metrics['mae']} | R2={metrics['r2_score']}")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_pipeline()