from pathlib import Path
from typing import Dict, Any

import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================

try:
    rul_model = joblib.load(MODELS_DIR / "rul_gradient_boosting_model.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    engineering_reference = joblib.load(MODELS_DIR / "engineering_reference.pkl")
    informative_sensors = joblib.load(MODELS_DIR / "informative_sensors.pkl")
    correlation_lookup = joblib.load(MODELS_DIR / "correlation_lookup.pkl")
    business_thresholds = joblib.load(MODELS_DIR / "business_thresholds.pkl")
    health_score_reference = joblib.load(MODELS_DIR / "health_score_reference.pkl")

    with open(MODELS_DIR / "model_metadata.json", "r", encoding="utf-8") as file:
        model_metadata = json.load(file)

    MODEL_LOADED = True

except Exception as error:
    MODEL_LOADED = False
    LOAD_ERROR = str(error)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SONATRACH Predictive Maintenance API",
    description=(
        "AI-powered predictive maintenance API for critical oil and gas "
        "industrial assets. The API predicts Remaining Useful Life, failure "
        "risk level, maintenance priority and operational recommendation."
    ),
    version="1.0.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class AssetSensorInput(BaseModel):
    asset_id: str = Field(..., example="COMP-102")
    time_in_cycles: int = Field(..., ge=1, example=180)

    operational_setting_1: float = Field(..., example=0.0012)
    operational_setting_2: float = Field(..., example=0.0003)

    sensor_2: float = Field(..., example=642.3)
    sensor_3: float = Field(..., example=1590.5)
    sensor_4: float = Field(..., example=1412.8)
    sensor_6: float = Field(..., example=21.61)
    sensor_7: float = Field(..., example=553.0)
    sensor_8: float = Field(..., example=2388.12)
    sensor_9: float = Field(..., example=9065.2)
    sensor_11: float = Field(..., example=47.6)
    sensor_12: float = Field(..., example=521.4)
    sensor_13: float = Field(..., example=2388.10)
    sensor_14: float = Field(..., example=8144.3)
    sensor_15: float = Field(..., example=8.45)
    sensor_17: float = Field(..., example=393.0)
    sensor_20: float = Field(..., example=38.8)
    sensor_21: float = Field(..., example=23.3)


# ============================================================
# BUSINESS DECISION FUNCTIONS
# ============================================================

def assign_failure_risk_from_rul(rul: float, thresholds: Dict[str, float]) -> str:
    if rul <= thresholds["critical"]:
        return "Critical Risk"
    elif rul <= thresholds["high"]:
        return "High Risk"
    elif rul <= thresholds["medium"]:
        return "Medium Risk"
    else:
        return "Low Risk"


def assign_maintenance_priority(failure_risk: str) -> str:
    if failure_risk == "Critical Risk":
        return "Immediate Intervention"
    elif failure_risk == "High Risk":
        return "Urgent Maintenance"
    elif failure_risk == "Medium Risk":
        return "Planned Inspection"
    else:
        return "Normal Monitoring"


def assign_recommendation(failure_risk: str) -> str:
    if failure_risk == "Critical Risk":
        return "Immediate shutdown inspection or emergency maintenance is recommended."
    elif failure_risk == "High Risk":
        return "Schedule urgent maintenance within the next operational window."
    elif failure_risk == "Medium Risk":
        return "Plan inspection and increase monitoring frequency."
    else:
        return "Continue normal monitoring under standard maintenance schedule."


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_industrial_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the same industrial feature engineering used during training.
    """
    engineered_df = input_df.copy()

    degradation_components = []
    thermal_components = []
    efficiency_components = []

    for sensor in informative_sensors:
        mean_value = engineering_reference[sensor]["mean"]
        std_value = engineering_reference[sensor]["std"]

        if std_value == 0 or pd.isna(std_value):
            std_value = 1.0

        z_sensor = (engineered_df[sensor] - mean_value) / std_value

        if correlation_lookup[sensor] < 0:
            degradation_signal = z_sensor
            thermal_components.append(degradation_signal)
        else:
            degradation_signal = -z_sensor
            efficiency_components.append(degradation_signal)

        degradation_components.append(degradation_signal)

    engineered_df["degradation_index"] = np.vstack(
        [component.to_numpy() for component in degradation_components]
    ).mean(axis=0)

    engineered_df["thermal_stress_index"] = np.vstack(
        [component.to_numpy() for component in thermal_components]
    ).mean(axis=0)

    engineered_df["efficiency_loss_index"] = np.vstack(
        [component.to_numpy() for component in efficiency_components]
    ).mean(axis=0)

    engineered_df["cycle_log"] = np.log1p(engineered_df["time_in_cycles"])

    engineered_df["cycle_ratio"] = (
        engineered_df["time_in_cycles"] / engineering_reference["cycle_max"]
    )

    engineered_df["operational_setting_magnitude"] = np.sqrt(
        engineered_df["operational_setting_1"] ** 2
        + engineered_df["operational_setting_2"] ** 2
    )

    deg_p05 = health_score_reference["deg_p05"]
    deg_p95 = health_score_reference["deg_p95"]

    engineered_df["asset_health_score"] = 100 * (
        1 - (engineered_df["degradation_index"] - deg_p05) / (deg_p95 - deg_p05)
    )

    engineered_df["asset_health_score"] = engineered_df["asset_health_score"].clip(0, 100)

    return engineered_df


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "message": "SONATRACH Predictive Maintenance API is running.",
        "status": "online",
        "project": "Predictive Maintenance for Critical Industrial Assets",
        "docs": "/docs"
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    if MODEL_LOADED:
        return {
            "api_status": "healthy",
            "model_status": "loaded",
            "model_name": model_metadata.get("model_name", "unknown"),
            "threshold_strategy": model_metadata.get("final_threshold_strategy", "unknown")
        }

    return {
        "api_status": "unhealthy",
        "model_status": "not_loaded",
        "error": LOAD_ERROR
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail=LOAD_ERROR)

    return {
        "project_name": model_metadata.get("project_name"),
        "model_name": model_metadata.get("model_name"),
        "dataset": model_metadata.get("dataset"),
        "target": model_metadata.get("target"),
        "rul_cap": model_metadata.get("rul_cap"),
        "number_of_features": model_metadata.get("number_of_features"),
        "validation_mae": model_metadata.get("validation_mae"),
        "external_test_last_cycle_mae": model_metadata.get("external_test_last_cycle_mae"),
        "external_test_last_cycle_r2": model_metadata.get("external_test_last_cycle_r2"),
        "final_threshold_strategy": model_metadata.get("final_threshold_strategy"),
        "business_thresholds": model_metadata.get("business_thresholds")
    }


@app.post("/predict-maintenance-risk")
def predict_maintenance_risk(payload: AssetSensorInput) -> Dict[str, Any]:
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail=LOAD_ERROR)

    try:
        try:
            payload_dict = payload.model_dump()
        except AttributeError:
            payload_dict = payload.dict()

        asset_id = payload_dict.pop("asset_id")

        input_df = pd.DataFrame([payload_dict])

        engineered_df = add_industrial_features(input_df)

        missing_features = [
            feature for feature in feature_columns
            if feature not in engineered_df.columns
        ]

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        X_input = engineered_df[feature_columns]

        predicted_rul = float(rul_model.predict(X_input)[0])

        rul_cap = model_metadata.get("rul_cap", 125)
        predicted_rul = float(np.clip(predicted_rul, 0, rul_cap))

        failure_risk = assign_failure_risk_from_rul(
            predicted_rul,
            business_thresholds
        )

        maintenance_priority = assign_maintenance_priority(failure_risk)
        recommendation = assign_recommendation(failure_risk)

        return {
            "asset_id": asset_id,
            "input_cycle": payload_dict["time_in_cycles"],
            "predicted_rul_cycles": round(predicted_rul, 2),
            "failure_risk": failure_risk,
            "maintenance_priority": maintenance_priority,
            "recommendation": recommendation,
            "asset_health_score": round(float(engineered_df["asset_health_score"].iloc[0]), 2),
            "degradation_index": round(float(engineered_df["degradation_index"].iloc[0]), 4),
            "threshold_strategy": model_metadata.get("final_threshold_strategy"),
            "model_name": model_metadata.get("model_name")
        }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))