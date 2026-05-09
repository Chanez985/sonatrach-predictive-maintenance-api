# SONATRACH Predictive Maintenance API

AI-powered predictive maintenance API for critical industrial assets.

This project predicts the Remaining Useful Life (RUL) of industrial equipment using sensor data from the NASA C-MAPSS FD001 turbofan degradation dataset. The model output is converted into an operational maintenance decision: failure risk level, maintenance priority, and recommended action.

## Project Objective

The objective is to build a production-style machine learning API that can support maintenance teams in monitoring critical oil and gas equipment.

The API receives operational and sensor measurements and returns:

- predicted Remaining Useful Life;
- failure risk level;
- maintenance priority;
- asset health score;
- operational recommendation.

## Business Context

In an industrial environment such as oil and gas operations, unexpected equipment failures can cause production downtime, high maintenance costs, and safety risks.

Predictive maintenance helps decision-makers anticipate failures and prioritize interventions before critical breakdowns occur.

## Dataset

Dataset used:

- NASA C-MAPSS FD001
- Run-to-failure simulation data
- Multivariate time-series sensor measurements
- 100 training assets and 100 test assets

The target variable is Remaining Useful Life, capped at 125 cycles to stabilize early-life degradation modeling.

## Machine Learning Pipeline

The project includes:

1. Data loading and inspection
2. RUL target creation
3. Exploratory data analysis
4. Constant feature removal
5. Industrial feature engineering
6. Asset-level validation split
7. Regression modeling
8. External test evaluation
9. Failure risk classification
10. Business threshold calibration
11. FastAPI deployment layer
12. Automated API testing

## Feature Engineering

Several industrial indicators were created:

- degradation index;
- thermal stress index;
- efficiency loss index;
- asset health score;
- cycle ratio;
- cycle log;
- operational setting magnitude.

These features transform raw sensor readings into interpretable maintenance indicators.

## Final Model

Final regression model:

- Gradient Boosting Regressor

Performance:

| Metric | Value |
|---|---:|
| Validation MAE | 10.48 cycles |
| External Test MAE | 11.54 cycles |
| External Test R² | 0.839 |

## Failure Risk Decision Layer

The predicted RUL is converted into a maintenance risk level using the selected business strategy:

| Risk Level | RUL Threshold | Maintenance Priority |
|---|---:|---|
| Critical Risk | RUL ≤ 20 | Immediate Intervention |
| High Risk | RUL ≤ 45 | Urgent Maintenance |
| Medium Risk | RUL ≤ 85 | Planned Inspection |
| Low Risk | RUL > 85 | Normal Monitoring |

Final risk classification performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.77 |
| Weighted F1-score | 0.781 |
| Critical Risk Recall | 1.00 |
| Dangerous Underestimations | 0 |

## API Endpoints

The FastAPI service exposes:

```text
GET /
GET /health
GET /model-info
POST /predict-maintenance-risk
Example API Request
{
  "asset_id": "COMP-102",
  "time_in_cycles": 180,
  "operational_setting_1": 0.0012,
  "operational_setting_2": 0.0003,
  "sensor_2": 642.3,
  "sensor_3": 1590.5,
  "sensor_4": 1412.8,
  "sensor_6": 21.61,
  "sensor_7": 553.0,
  "sensor_8": 2388.12,
  "sensor_9": 9065.2,
  "sensor_11": 47.6,
  "sensor_12": 521.4,
  "sensor_13": 2388.1,
  "sensor_14": 8144.3,
  "sensor_15": 8.45,
  "sensor_17": 393,
  "sensor_20": 38.8,
  "sensor_21": 23.3
}
Example API Response
{
  "asset_id": "COMP-102",
  "input_cycle": 180,
  "predicted_rul_cycles": 92.95,
  "failure_risk": "Low Risk",
  "maintenance_priority": "Normal Monitoring",
  "recommendation": "Continue normal monitoring under standard maintenance schedule.",
  "asset_health_score": 56.33,
  "degradation_index": 0.0625,
  "threshold_strategy": "Conservative_1",
  "model_name": "Gradient Boosting Regressor"
}
How to Run Locally

Install dependencies:

pip install -r requirements.txt

Run the API:

python -m uvicorn api.main:app --host 127.0.0.1 --port 8001

Open the API documentation:

http://127.0.0.1:8001/docs

Health check:

http://127.0.0.1:8001/health
Run Tests
python -m pytest -q

Expected result:

3 passed
Project Structure
sonatrach-predictive-maintenance-api/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── rul_gradient_boosting_model.pkl
│   ├── feature_columns.pkl
│   ├── engineering_reference.pkl
│   ├── informative_sensors.pkl
│   ├── correlation_lookup.pkl
│   ├── business_thresholds.pkl
│   ├── health_score_reference.pkl
│   └── model_metadata.json
│
├── notebook/
│   └── sonatrach_predictive_maintenance.ipynb
│
├── reports/
├── src/
├── tests/
│   └── test_api.py
│
├── README.md
├── requirements.txt
├── Dockerfile
└── .gitignore
Technologies Used
Python
Pandas
NumPy
Scikit-learn
FastAPI
Pydantic
Joblib
Pytest
Uvicorn
Swagger UI
Author

Chanez Benidir

Data Science and Statistics student interested in machine learning, predictive maintenance, industrial AI, and decision-support systems.
```text
