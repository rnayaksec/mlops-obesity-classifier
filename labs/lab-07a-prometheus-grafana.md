# Lab 7a — Prometheus + Grafana Monitoring

**Objective:** Add live observability to the deployed model container by exposing a `/metrics` endpoint and wiring it to a Prometheus + Grafana stack via Docker Compose.

**Prerequisites:** Lab 7 complete. `dvc repro` runs successfully, `docker build` produces a working image.

**Concepts introduced:**
- Prometheus scrape model and metrics exposition format
- `prometheus_client` Python library
- Docker Compose multi-service orchestration
- Grafana data sources and dashboards
- Custom ML metrics (prediction count, latency, accuracy gauge)

**Time:** ~1.5 hours

---

## Background

Right now the model container serves predictions but you have no visibility into how it's performing after deployment. Prometheus solves this by periodically pulling a `/metrics` endpoint from your container and storing the time-series data. Grafana reads that data and renders it as dashboards. Together they let you answer: "Is my model responding? How long does inference take? Has accuracy drifted?"

---

## Step 1 — Add metrics instrumentation to the FastAPI app

Install the Prometheus client library:

```bash
pip install prometheus-fastapi-instrumentator>=6.0
```

It's already in `requirements.txt` from Lab 6. Verify it's there:

```
prometheus-fastapi-instrumentator>=6.0
```

Open `api.py` and add instrumentation. The `Instrumentator` middleware automatically exposes a `/metrics` endpoint with HTTP request counts and latency histograms. Add three custom metrics on top: prediction count by class, inference latency, and a live accuracy gauge.

Replace the contents of `api.py` with:

```python
import time
import json
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# ── Custom metrics ────────────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "obesity_predictions_total",
    "Total number of predictions made",
    ["predicted_class"],
)
INFERENCE_LATENCY = Histogram(
    "obesity_inference_seconds",
    "Time spent running model inference",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5],
)
ACCURACY_GAUGE = Gauge(
    "obesity_model_accuracy",
    "Most recently measured model accuracy on validation set",
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Obesity Classifier", version="1.0")
Instrumentator().instrument(app).expose(app)


# ── Load model artefacts ──────────────────────────────────────────────────────
MODEL_PATH = Path("models/model.pkl")
with open(MODEL_PATH, "rb") as f:
    artefacts = pickle.load(f)

model = artefacts["model"]
dv = artefacts["dict_vectorizer"]

# Seed the accuracy gauge from metrics.json if available
METRICS_PATH = Path("metrics.json")
if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        saved = json.load(f)
    ACCURACY_GAUGE.set(saved.get("accuracy", 0.0))


# ── Request / response schemas ────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = dv.transform([request.model_dump()])

    start = time.perf_counter()
    proba = model.predict_proba(features)[0][1]
    elapsed = time.perf_counter() - start

    prediction = int(proba >= 0.5)

    # Record metrics
    PREDICTION_COUNTER.labels(predicted_class=str(prediction)).inc()
    INFERENCE_LATENCY.observe(elapsed)

    return PredictionResponse(prediction=prediction, probability=round(float(proba), 4))


@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## Step 2 — Create prometheus.yml

Create `prometheus.yml` in the project root:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: obesity-classifier
    static_configs:
      - targets:
          - model:8000
    metrics_path: /metrics
```

The `model` hostname matches the Docker Compose service name defined in the next step.

---

## Step 3 — Create docker-compose.yml

Create `docker-compose.yml` in the project root:

```yaml
services:
  model:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s

  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
    depends_on:
      model:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:9090/-/healthy"]
      interval: 10s
      timeout: 5s
      retries: 3

  grafana:
    image: grafana/grafana:10.4.2
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=mlops
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-storage:/var/lib/grafana
    depends_on:
      prometheus:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/api/health"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  grafana-storage:
```

---

## Step 4 — Rebuild and start the stack

```bash
docker compose up --build
```

Wait for all three containers to report healthy. You can check:

```bash
docker compose ps
```

All three services should show `(healthy)`.

---

## Step 5 — Set up Grafana

1. Open `http://localhost:3000` — log in with `admin` / `mlops`
2. Go to **Connections → Data sources → Add data source → Prometheus**
3. Set URL to `http://prometheus:9090` and click **Save & test**
4. Go to **Dashboards → New → New dashboard → Add visualisation**

Add these panels:

**Panel 1 — Prediction rate**
- Query: `rate(obesity_predictions_total[1m])`
- Title: `Predictions per second`
- Visualization: Time series

**Panel 2 — Inference latency (p95)**
- Query: `histogram_quantile(0.95, rate(obesity_inference_seconds_bucket[5m]))`
- Title: `Inference latency p95`
- Unit: `seconds`
- Visualization: Time series

**Panel 3 — Live accuracy**
- Query: `obesity_model_accuracy`
- Title: `Model accuracy`
- Visualization: Stat
- Thresholds: green ≥ 0.90, yellow ≥ 0.85, red < 0.85

5. Save the dashboard as `Obesity Classifier`.

---

## Step 6 — Generate some traffic

In a second terminal, send a few predictions:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male", "Age": 30, "Height": 1.75, "Weight": 95,
    "family_history_with_overweight": "yes", "FAVC": "yes",
    "FCVC": 2, "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no",
    "CH2O": 2, "SCC": "no", "FAF": 0, "TUE": 1,
    "CALC": "Sometimes", "MTRANS": "Public_Transportation"
  }'
```

Run it 10–20 times, then refresh your Grafana dashboard. You should see the prediction counter increment and latency histogram populate.

---

## Step 7 — Commit the stack

```bash
git add api.py prometheus.yml docker-compose.yml
git commit -m "feat: add Prometheus + Grafana observability stack"
git push
```

---

## Verification

| Check | Command | Expected output |
|---|---|---|
| Model serving | `curl http://localhost:8000/health` | `{"status":"ok"}` |
| Metrics exposed | `curl http://localhost:8000/metrics` | Lines containing `obesity_predictions_total` |
| Prometheus scraping | Open `http://localhost:9090/targets` | `obesity-classifier` state = `UP` |
| Grafana connected | Open `http://localhost:3000` | Dashboard shows live data after sending requests |

---

## How this connects to Lab 7b

You now have live metrics flowing from the model container into Prometheus. Lab 7b takes the next step: instead of watching a dashboard manually, you automate the accuracy check inside GitHub Actions. After every Docker push, a new job pulls the deployed image, runs it against the held-out validation set, logs the result to MLflow, and fails the pipeline if accuracy has dropped — turning your dashboard observation into an automated safety net.

**Next:** [Lab 7b — Drift Detection in GitHub Actions](lab-07b-drift-detection.md)
