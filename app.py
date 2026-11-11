import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import yaml

# Root directory path configuration
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.logger import get_logger
from src.utils.artifact_manager import ArtifactManager

# Load project configuration
CONFIG_PATH_XY8 = "config/config.yaml"
with open(CONFIG_PATH_XY8, "r", encoding="utf-8") as f_cfg_q9:
    SYSTEM_CONFIG_V9 = yaml.safe_load(f_cfg_q9)

app_logger_k4 = get_logger(
    name="FastApiBmsBackend_z7",
    log_dir=SYSTEM_CONFIG_V9["logging"]["log_dir"],
    log_level=SYSTEM_CONFIG_V9["logging"]["log_level"],
)

# FastAPI application initialization
app = FastAPI(
    title="Battery RUL & Health Prognostics API",
    description="Lightweight physics-informed API for Lithium-Ion Remaining Useful Life forecasting.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory artifact holders
cached_estimator_m3 = None
cached_preprocessor_w1 = None
cached_feature_names_j5 = []


@app.on_event("startup")
def bootstrap_model_artifacts_x7():
    """Loads pre-trained LightGBM model and fitted Scaler into memory on startup."""
    global cached_estimator_m3, cached_preprocessor_w1, cached_feature_names_j5
    try:
        art_mgr_p2 = ArtifactManager(SYSTEM_CONFIG_V9)

        # 1. Load Model
        cached_estimator_m3 = art_mgr_p2.load_model("battery_rul_model.joblib")
        app_logger_k4.info("Successfully loaded trained regression estimator.")

        # 2. Load Fitted Preprocessor
        scaler_dir_v4 = SYSTEM_CONFIG_V9["artifacts"].get("preprocessor_dir", "artifacts/preprocessors")
        scaler_file_n8 = os.path.join(scaler_dir_v4, "battery_preprocessor.joblib")
        prep_payload_u6 = joblib.load(scaler_file_n8)
        cached_preprocessor_w1 = prep_payload_u6["scaler"]
        cached_feature_names_j5 = prep_payload_u6["feature_columns"]

        app_logger_k4.info(f"Preprocessor loaded. Expected features: {cached_feature_names_j5}")
    except Exception as exc_err_b9:
        app_logger_k4.error(f"Failed to bootstrap model artifacts: {str(exc_err_b9)}")
        raise RuntimeError(f"Startup failed: {str(exc_err_b9)}")


# Request Schema with strict validation
class BatteryTelemetryInput_r9x(BaseModel):
    cycle: int = Field(..., ge=1, le=1000, description="Current operational cycle index")
    voltage: float = Field(..., ge=2.0, le=4.5, description="Operating plateau discharge voltage (V)")
    temperature: float = Field(..., ge=5.0, le=60.0, description="Surface temperature reading (°C)")
    voltage_diff: Optional[float] = Field(None, description="Delta V from previous cycle")
    temp_diff: Optional[float] = Field(None, description="Delta T from previous cycle")
    voltage_roll_mean_3: Optional[float] = Field(None, description="3-cycle rolling voltage average")
    temp_roll_mean_3: Optional[float] = Field(None, description="3-cycle rolling temperature average")
    temp_roll_std_3: Optional[float] = Field(None, description="3-cycle rolling temperature std dev")
    thermal_cum_stress: Optional[float] = Field(None, description="Cumulative life thermal exposure")


class BatteryPredictionResponse_t2z(BaseModel):
    predicted_rul_cycles: float
    battery_health_status: str
    risk_level: str
    feature_payload_received: Dict[str, float]


def synthesize_missing_signals_q3w(payload_dict_d1: Dict[str, Any]) -> Dict[str, float]:
    """Auto-approximates rolling and differential signals if client only provides core sensor telemetry."""
    v_val_o7 = payload_dict_d1["voltage"]
    t_val_e4 = payload_dict_d1["temperature"]
    c_val_k2 = payload_dict_d1["cycle"]

    processed_payload_m8 = dict(payload_dict_d1)

    if processed_payload_m8.get("voltage_diff") is None:
        processed_payload_m8["voltage_diff"] = -0.0012
    if processed_payload_m8.get("temp_diff") is None:
        processed_payload_m8["temp_diff"] = 0.05
    if processed_payload_m8.get("voltage_roll_mean_3") is None:
        processed_payload_m8["voltage_roll_mean_3"] = v_val_o7
    if processed_payload_m8.get("temp_roll_mean_3") is None:
        processed_payload_m8["temp_roll_mean_3"] = t_val_e4
    if processed_payload_m8.get("temp_roll_std_3") is None:
        processed_payload_m8["temp_roll_std_3"] = 0.025
    if processed_payload_m8.get("thermal_cum_stress") is None:
        processed_payload_m8["thermal_cum_stress"] = t_val_e4 * c_val_k2

    processed_payload_m8["v_t_ratio"] = v_val_o7 / (t_val_e4 + 1e-5)
    return processed_payload_m8


@app.get("/", response_class=FileResponse, tags=["Dashboard"])
def render_dashboard_template_view_m4k():
    """Serves the single-page frontend directly at root URL (Flask render_template equivalent)."""
    html_file_path_v8z = "index.html"
    if not os.path.exists(html_file_path_v8z):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="index.html not found in project root."
        )
    return FileResponse(html_file_path_v8z)


@app.get("/health", tags=["System"])
def system_health_probe_u4y():
    """Health check endpoint to verify API and artifact availability."""
    return {
        "status": "online",
        "artifacts_loaded": cached_estimator_m3 is not None and cached_preprocessor_w1 is not None,
        "engine": SYSTEM_CONFIG_V9["model"]["name"],
    }


@app.get("/api/v1/metrics", tags=["Prognostics"])
def get_evaluation_metrics_k8v():
    """Retrieves out-of-sample benchmark metrics from the latest training run."""
    try:
        art_mgr_p2 = ArtifactManager(SYSTEM_CONFIG_V9)
        return art_mgr_p2.load_metrics("battery_rul_evaluation.json")
    except Exception as exc_err_b9:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation metrics report not found: {str(exc_err_b9)}",
        )


@app.post("/api/v1/predict", response_model=BatteryPredictionResponse_t2z, tags=["Prognostics"])
def predict_battery_rul_g6k(telemetry: BatteryTelemetryInput_r9x):
    """Predicts Remaining Useful Life cycles from physical battery telemetry inputs."""
    if cached_estimator_m3 is None or cached_preprocessor_w1 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not ready yet.",
        )

    # 1. Synthesize missing rolling/diff signals if omitted by client
    full_signals_map_w9 = synthesize_missing_signals_q3w(telemetry.dict())

    # 2. Arrange features in exact training order with DataFrame to prevent feature name warnings
    try:
        input_feature_df_n6 = pd.DataFrame([full_signals_map_w9])[cached_feature_names_j5]
    except KeyError as key_err_z2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing feature computation: {str(key_err_z2)}",
        )

    # 3. Apply standard scaling via persisted preprocessor preserving column names
    scaled_array_s4 = cached_preprocessor_w1.transform(input_feature_df_n6)
    scaled_feature_df_w2 = pd.DataFrame(scaled_array_s4, columns=cached_feature_names_j5)

    # 4. Infer RUL with LightGBM
    raw_prediction_v5 = float(cached_estimator_m3.predict(scaled_feature_df_w2)[0])
    calibrated_rul_cycle_q1 = max(0.0, round(raw_prediction_v5, 1))

    # 5. Diagnostic classification
    if calibrated_rul_cycle_q1 > 80:
        health_status = "Optimal Health (Nominal)"
        risk_level = "LOW"
    elif calibrated_rul_cycle_q1 > 30:
        health_status = "Moderate Degradation"
        risk_level = "MODERATE"
    else:
        health_status = "Critical Wear (Retirement Imminent)"
        risk_level = "HIGH"

    return BatteryPredictionResponse_t2z(
        predicted_rul_cycles=calibrated_rul_cycle_q1,
        battery_health_status=health_status,
        risk_level=risk_level,
        feature_payload_received=full_signals_map_w9,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)