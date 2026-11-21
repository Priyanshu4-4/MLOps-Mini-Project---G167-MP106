
from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict


# Locate and load the trained model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(title="MP106 Milk Collection API")


# JSON logger
logger = logging.getLogger("mp106")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '{"level":"%(levelname)s","message":"%(message)s"}'
))

if not logger.handlers:
    logger.addHandler(handler)


# Request schema and client validation rules
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cows: int = Field(..., le=350)
    buffaloes: float = Field(..., ge=2.5)
    fodder_price_rs: float
    temp_c: float
    vet_visits: int
    feed_kg: float
    breed_mix: str
    village_cluster: str


# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_data)[0]

    logger.info("Prediction completed successfully")

    return {"prediction": float(prediction)}
