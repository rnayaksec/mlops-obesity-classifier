# Lab 4 — REST API Model Serving with FastAPI

**Prerequisites:** Lab 3 complete. `models/model.pkl` exists (run `python -m src.train` if needed).

**What you'll build:** A FastAPI app that loads your trained model and exposes a `POST /predict` endpoint. You'll test it via the auto-generated Swagger UI and add an API integration test.

**New concepts:** Model serving, REST APIs for ML, request/response schemas, integration testing.

**Time:** ~1 hour

---

## Background

A model sitting in a `.pkl` file is only useful to people who can run Python. Wrapping it in a REST API means any application — a web frontend, a mobile app, a downstream service — can call it over HTTP. FastAPI is the modern Python standard for this: it's fast, generates automatic documentation, and validates request data.

---

## Step 1 — Install FastAPI

```bash
pip install fastapi uvicorn
```

Add to `requirements.txt`:
```
fastapi>=0.110
uvicorn>=0.27
```

---

## Step 2 — Create the API

Create `src/api.py`:

```python
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
```

---

## Step 3 — Run the API locally

Make sure you have a trained model first:

```bash
python -m src.train
```

Then start the API:

```bash
uvicorn src.api:app --reload
```

Open `http://localhost:8000/docs` in your browser. You'll see the auto-generated Swagger UI with both endpoints documented.

Click **POST /predict → Try it out**, paste this example body, and click **Execute**:

```json
{
  "gender": "Female",
  "family_history_with_overweight": "yes",
  "favc": "yes",
  "caec": "Sometimes",
  "smoke": "no",
  "scc": "no",
  "calc": "Sometimes",
  "mtrans": "Public_Transportation",
  "age": 25,
  "height": 1.65,
  "weight": 95,
  "fcvc": 2.0,
  "ncp": 3.0,
  "ch2o": 2.0,
  "faf": 0.5,
  "tue": 1.0
}
```

You should get back a response like:
```json
{
  "is_obese": true,
  "probability": 0.9823
}
```

---

## Step 4 — Add an integration test

Create `tests/test_api.py`:

```python
"""Integration tests for src/api.py"""
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "gender": "Female",
    "family_history_with_overweight": "yes",
    "favc": "yes",
    "caec": "Sometimes",
    "smoke": "no",
    "scc": "no",
    "calc": "Sometimes",
    "mtrans": "Public_Transportation",
    "age": 25,
    "height": 1.65,
    "weight": 95,
    "fcvc": 2.0,
    "ncp": 3.0,
    "ch2o": 2.0,
    "faf": 0.5,
    "tue": 1.0,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200


def test_predict_response_schema():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    data = response.json()
    assert "is_obese" in data
    assert "probability" in data
    assert isinstance(data["is_obese"], bool)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_invalid_input_returns_422():
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422
```

Run the full test suite:

```bash
pytest tests/ -v
```

All tests including the API tests should pass. Commit:

```bash
git add src/api.py tests/test_api.py requirements.txt
git commit -m "feat: add FastAPI prediction endpoint with integration tests"
git push
```

Your CI pipeline runs the API tests automatically on every push. ✅

---

## Key takeaways

- FastAPI uses Pydantic models for request/response validation — invalid inputs return a `422` automatically
- The `lifespan` context manager loads the model once at startup, not on every request
- `TestClient` from `fastapi.testclient` lets you write integration tests without starting a real server
- The Swagger UI at `/docs` is generated automatically from your type annotations — no extra work

---

**Next:** [Lab 5 — Docker Containerisation](lab-05-docker-containerisation.md)
