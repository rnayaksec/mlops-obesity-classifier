"""
api.py
------
FastAPI application exposing a /predict endpoint for obesity classification.
Run with: uvicorn src.api:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.train import MODEL_PATH, load_model

# --- Request / Response schemas ---


class PatientFeatures(BaseModel):
    gender: str
    family_history_with_overweight: str
    favc: str
    caec: str
    smoke: str
    scc: str
    calc: str
    mtrans: str
    age: float
    height: float
    weight: float
    fcvc: float
    ncp: float
    ch2o: float
    faf: float
    tue: float


class PredictionResponse(BaseModel):
    is_obese: bool
    probability: float


# --- App setup ---

model_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_bundle["model"], model_bundle["dv"] = load_model(MODEL_PATH)
    print("Model loaded.")
    yield
    model_bundle.clear()


app = FastAPI(
    title="Obesity Classifier",
    description="Predicts obesity risk from lifestyle features.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Endpoints ---


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    model = model_bundle["model"]
    dv = model_bundle["dv"]

    features = dv.transform([patient.model_dump()])
    probability = float(model.predict_proba(features)[0, 1])
    is_obese = probability >= 0.5

    return PredictionResponse(is_obese=is_obese, probability=round(probability, 4))
