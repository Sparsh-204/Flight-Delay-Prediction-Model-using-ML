import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from flight_delay_model import MODEL_OUTPUT_PATH


class FlightInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MONTH: int = Field(ge=1, le=12)
    DAY: int = Field(ge=1, le=31)
    DAY_OF_WEEK: int = Field(ge=1, le=7)
    AIRLINE: str = Field(min_length=2, max_length=5)
    ORIGIN_AIRPORT: str = Field(min_length=3, max_length=5)
    DESTINATION_AIRPORT: str = Field(min_length=3, max_length=5)
    DEPARTURE_TIME: int = Field(ge=0, le=2359)
    DISTANCE: int = Field(gt=0)


model_path = os.getenv("MODEL_PATH", MODEL_OUTPUT_PATH)
predictor = None
if os.path.exists(model_path):
    predictor = joblib.load(model_path)

app = FastAPI(
    title="India Flight Delay Predictor API",
    description="Predict delay probability for Indian domestic flight routes.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "india-flight-delay-predictor",
        "model_loaded": predictor is not None,
        "model_path": model_path,
    }


@app.post("/predict")
def predict_delay(data: FlightInput, include_weather: bool = False):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model file not found. Train the India flight model first with: python flight_delay_model.py",
        )

    input_df = pd.DataFrame([data.model_dump()])
    try:
        probability = float(predictor.predict_proba(input_df, include_weather=include_weather)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "delay_probability": probability,
        "prediction": "Delayed" if probability >= predictor.threshold else "On-Time",
        "threshold": predictor.threshold,
        "weather_used": bool(include_weather and predictor.use_weather_features),
    }