# Lab 5 — Docker Containerisation

**Prerequisites:** Lab 4 complete. Docker Desktop installed and running.

**What you'll build:** A Docker image that packages your FastAPI app and trained model. You'll run the container locally, then push the image to Docker Hub.

**New concepts:** Dockerfile, image layers, container runtime, Docker Hub, image tagging.

**Time:** ~1 hour

---

## Background

Your API works on your machine. Docker makes it work *identically* on any machine — a colleague's laptop, a cloud VM, a Kubernetes cluster — by packaging your code, dependencies, and model together into a single portable image.

---

## Step 1 — Install Docker Desktop

Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) and start it. Verify:

```bash
docker --version
```

---

## Step 2 — Create the Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
# Use a slim Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model
COPY src/ ./src/
COPY models/ ./models/

# Expose the port FastAPI will listen on
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 3 — Create a .dockerignore

Create `.dockerignore` to keep the image lean:

```
.git
.venv
__pycache__
*.pyc
notebooks/
tests/
data/
mlruns/
labs/
*.md
.dvc
```

---

## Step 4 — Train the model

The model must exist before building the image (it gets copied in at build time):

```bash
python -m src.train
```

---

## Step 5 — Build the image

```bash
docker build -t obesity-classifier:v1 .
```

Watch the layers build. The `pip install` layer will be cached on subsequent builds unless `requirements.txt` changes.

Verify the image exists:

```bash
docker images obesity-classifier
```

---

## Step 6 — Run the container

```bash
docker run -p 8000:8000 obesity-classifier:v1
```

Open `http://localhost:8000/docs` — your API is now running inside the container. Test a prediction exactly as you did in Lab 4.

Press `Ctrl+C` to stop the container.

---

## Step 7 — Push to Docker Hub

1. Create a free account at [hub.docker.com](https://hub.docker.com)
2. Create a new public repository called `obesity-classifier`
3. Log in from the terminal:

```bash
docker login
```

Tag and push:

```bash
# Tag with your Docker Hub username
docker tag obesity-classifier:v1 YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1
docker tag obesity-classifier:v1 YOUR_DOCKERHUB_USERNAME/obesity-classifier:latest

# Push both tags
docker push YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1
docker push YOUR_DOCKERHUB_USERNAME/obesity-classifier:latest
```

4. Visit `hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/obesity-classifier` — your image is publicly available. Anyone can now run it with:

```bash
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/obesity-classifier:latest
```

---

## Step 8 — Commit the Dockerfile

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile for FastAPI app"
git push
```

---

## Key takeaways

- A `Dockerfile` is a recipe: each `RUN`, `COPY`, `FROM` line creates a cached layer
- Order matters: put `COPY requirements.txt` + `RUN pip install` *before* `COPY src/` so that changing your code doesn't invalidate the dependency layer
- `.dockerignore` keeps large/irrelevant files out of the build context
- Docker Hub is the public registry — equivalent to GitHub but for container images
- In Lab 6 you'll automate building and pushing this image via GitHub Actions

---
---

## Extension — Local Observability Stack with Docker Compose

**What you'll add:** A `docker-compose.yml` that runs your API alongside Prometheus (metrics scraping) and Grafana (dashboards). You'll instrument your FastAPI app to expose a `/metrics` endpoint and visualise live request data in a browser.

**New concepts:** docker-compose, multi-container networking, Prometheus scrape config, Grafana data sources, FastAPI instrumentation.

**Time:** ~45 minutes

---

### Background

Running a single container with `docker run` works for one service. In practice your model server doesn't live alone — it needs monitoring, logging, and eventually a database. `docker-compose` lets you define and start multiple containers together from a single file, wiring them onto a shared network automatically.

The stack you'll build:

```
localhost:8000  →  FastAPI model server  (your image)
localhost:9090  →  Prometheus            (scrapes /metrics every 15s)
localhost:3000  →  Grafana               (visualises Prometheus data)
```

---

### Step 9 — Instrument your FastAPI app

Install the instrumentation library:

```bash
pip install prometheus-fastapi-instrumentator==6.1.0
```

Add it to `requirements.txt`:

```
prometheus-fastapi-instrumentator==6.1.0
```

Update `src/api.py`. Order matters — instrument **after** `app` is created:

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()  # app must exist first

# Adds /metrics endpoint with request count, latency, and in-progress gauges
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    # your existing inference logic
    ...
```

Verify locally before containerising:

```bash
uvicorn src.api:app --reload
curl http://localhost:8000/metrics   # should return Prometheus text format
```

---

### Step 10 — Create the Prometheus config

Create the `monitoring/` directory and add `prometheus.yml`:

```
your-project/
├── docker-compose.yml
├── Dockerfile
├── monitoring/
│   └── prometheus.yml    ← create this
├── src/
└── models/
```

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "model-api"
    static_configs:
      - targets: ["api:8000"]   # "api" is the service name in docker-compose
    metrics_path: /metrics
```

> **Note:** Inside the Docker network, containers reach each other by **service name**, not `localhost`. Prometheus talks to your API at `api:8000`, not `localhost:8000`.

---

### Step 11 — Create docker-compose.yml

Create `docker-compose.yml` in your project root:

```yaml
services:

  api:
    build: .
    image: YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1
    container_name: model-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    networks:
      - monitoring
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - monitoring
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:

networks:
  monitoring:
    driver: bridge
```

---

### Step 12 — Start the stack

```bash
docker-compose up --build
```

`--build` rebuilds your API image to pick up the instrumentation changes. Prometheus and Grafana use pre-built images from Docker Hub so they download once and are cached.

Verify all three containers are running:

```bash
docker-compose ps
```

All three should show status **Up**. If any show **Exit**, check logs:

```bash
docker-compose logs prometheus
docker-compose logs grafana
```

---

### Step 13 — Test each service

**API** — unchanged from before:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics   # raw Prometheus output
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"your": "payload"}'
```

**Prometheus** — open `http://localhost:9090`:

1. Go to **Status → Targets** — `model-api` should show state **UP**
2. Go to the **Graph** tab and run a query:

```
http_requests_total
```

Make a few `/predict` requests and watch the counter increment.

**Grafana** — open `http://localhost:3000`, login with `admin` / `admin`:

1. Go to **Connections → Data Sources → Add data source**
2. Select **Prometheus**
3. Set the URL to `http://prometheus:9090` (service name, not localhost)
4. Click **Save & Test** — you should see "Data source is working"
5. Go to **Explore** and try these queries:

```promql
# Total requests by endpoint
http_requests_total

# Request rate per second (last 1 minute)
rate(http_requests_total[1m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

---

### How docker-compose relates to your existing workflow

Your tag and push commands are **unchanged** — compose only affects local development:

```bash
# 1. Train (unchanged)
python -m src.train

# 2. Build (unchanged — compose calls this internally too)
docker build -t obesity-classifier:v1 .

# 3. Tag and push to Docker Hub (unchanged)
docker tag obesity-classifier:v1 YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1
docker push YOUR_DOCKERHUB_USERNAME/obesity-classifier:v1

# 4. Run full local stack (new)
docker-compose up --build
```

`docker-compose up --build` calls `docker build` internally for services that have `build: .`. For services using a pre-built image (Prometheus, Grafana) it pulls from Docker Hub instead.

---

### Useful compose commands

```bash
docker-compose up --build      # build and start everything
docker-compose up -d           # start in background (detached)
docker-compose ps              # check container status
docker-compose logs api        # logs for a specific service
docker-compose logs -f         # follow all logs live
docker-compose down            # stop and remove containers
docker-compose down -v         # also delete volumes (wipes Grafana/Prometheus data)
```

---

### Key takeaways

- `docker-compose` orchestrates multiple containers locally using a single YAML file — it does not replace `docker build`, `docker tag`, or `docker push`
- Containers on the same compose network reach each other by **service name** (`api`, `prometheus`, `grafana`), not `localhost`
- `depends_on` controls start order but not readiness — Grafana starts after Prometheus but doesn't wait for it to be healthy
- Volumes (`prometheus_data`, `grafana_data`) persist data between `docker-compose down` and `up` cycles
- This pattern — app + metrics scraper + dashboard — is the foundation of production observability; in Kubernetes the same three components run as separate deployments

---

**Next:** [Lab 6 — Advanced CI/CD Pipeline](lab-06-advanced-cicd.md)




**Next:** [Lab 6 — Advanced CI/CD Pipeline](lab-06-advanced-cicd.md)
