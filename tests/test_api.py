from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["api_status"] == "healthy"
    assert data["model_status"] == "loaded"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    
    data = response.json()
    assert data["model_name"] == "Gradient Boosting Regressor"
    assert data["dataset"] == "NASA C-MAPSS FD001"
    assert data["final_threshold_strategy"] == "Conservative_1"


def test_predict_maintenance_risk_endpoint():
    payload = {
        "asset_id": "TEST_ASSET_001",
        "time_in_cycles": 31,
        "operational_setting_1": -0.0006,
        "operational_setting_2": 0.0004,
        "sensor_2": 642.58,
        "sensor_3": 1581.22,
        "sensor_4": 1398.91,
        "sensor_6": 21.61,
        "sensor_7": 554.42,
        "sensor_8": 2388.08,
        "sensor_9": 9056.4,
        "sensor_11": 47.23,
        "sensor_12": 521.79,
        "sensor_13": 2388.06,
        "sensor_14": 8130.11,
        "sensor_15": 8.4024,
        "sensor_17": 393,
        "sensor_20": 38.81,
        "sensor_21": 23.3552
    }
    
    response = client.post("/predict-maintenance-risk", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    assert data["asset_id"] == "TEST_ASSET_001"
    assert "predicted_rul_cycles" in data
    assert "failure_risk" in data
    assert "maintenance_priority" in data
    assert "recommendation" in data
    assert "asset_health_score" in data
    assert "degradation_index" in data
    
    assert data["failure_risk"] in [
        "Low Risk",
        "Medium Risk",
        "High Risk",
        "Critical Risk"
    ]