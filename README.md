# India Flight Delay Predictor

This project trains a flight delay classifier, serves predictions with FastAPI, and includes an India-themed frontend for testing domestic route delay risk.

## Project Files

- `flight_delay_model.py`: model training pipeline and predictor object
- `predict_api.py`: FastAPI scoring service (`/health`, `/predict`)
- `index.html`: India-themed frontend UI for delay prediction
- `weather_features.py`: optional OpenWeather feature enrichment
- `predict.py`: compatibility launcher for training
- `flights.csv`: source dataset
- `flight_delay_model.joblib`: trained model artifact

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The default trainer uses RandomForest for stable local Windows execution.

Optional: XGBoost training

```bash
pip install xgboost
python flight_delay_model.py --model-type xgboost
```

Optional: weather features

1. Create `.env` from `.env.example`
2. Add your OpenWeather API key

## Train Model

Default training uses the first `250000` rows to keep runtime manageable:

```bash
python flight_delay_model.py
```

Common training commands:

```bash
python flight_delay_model.py --max-rows 500000
python flight_delay_model.py --max-rows 0
python flight_delay_model.py --model-output flight_delay_model.joblib
```

## Run API

Start FastAPI on the same port used by the frontend:

```bash
uvicorn predict_api:app --host 127.0.0.1 --port 8001 --reload
```

Open API docs:

- `http://127.0.0.1:8001/docs`

API title shown in docs: `India Flight Delay Predictor API`

## Frontend

Open `index.html` in your browser after the API is running.

The UI is customized for Indian flights with:

- Indian airline options (e.g., `6E`, `AI`, `UK`, `SG`, `QP`)
- Indian airport suggestions (e.g., `DEL`, `BOM`, `BLR`, `HYD`, `MAA`)
- Delay probability visualization and risk verdict

## Example Prediction Request (India Route)

```json
{
  "MONTH": 3,
  "DAY": 17,
  "DAY_OF_WEEK": 1,
  "AIRLINE": "6E",
  "ORIGIN_AIRPORT": "DEL",
  "DESTINATION_AIRPORT": "BOM",
  "DEPARTURE_TIME": 745,
  "DISTANCE": 1148
}
```

## Example Verified Response

```json
{
  "delay_probability": 0.30056777499888,
  "prediction": "On-Time",
  "threshold": 0.5499999999999999,
  "weather_used": false
}
```

## Weather Modeling Note

Live weather APIs return current conditions, not historical weather for older flight records. For statistically valid training, join weather observations that match each flight timestamp and location.

Recommended hackathon workflow:

1. Train baseline model from flight history.
2. Add weather features only when historical weather is aligned with training records.
3. Add explainability (for example SHAP) after finalizing your model.