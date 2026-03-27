# Lab 0 — MLOps Foundations

**Prerequisites:** None. Read before starting Lab 1.

**What you'll learn:** The ML lifecycle, why MLOps exists, what the tooling in this series does, and how the labs connect.

**New concepts:** ML lifecycle, MLOps definition, versioning vs tracking, CI/CD for ML, the role of each tool.

**Time:** ~30 minutes (reading + discussion)

---

## What is MLOps?

MLOps (Machine Learning Operations) is the practice of applying software engineering discipline to machine learning systems. It bridges the gap between data scientists who build models and the infrastructure needed to run those models reliably in production.

Without MLOps you typically end up with:
- Models that can't be reproduced ("it worked on my laptop")
- No record of which hyperparameters produced which results
- Manual, error-prone deployment steps
- No visibility into how the model performs after it ships

MLOps fixes each of these with tooling and automation.

---

## The ML lifecycle

Every ML project passes through the same stages regardless of the model type:

```
Data collection
      │
      ▼
Data preparation & versioning   ← DVC
      │
      ▼
Experiment & model training     ← MLflow
      │
      ▼
Model evaluation & gating       ← GitHub Actions quality gate
      │
      ▼
Packaging & deployment          ← Docker + Docker Hub
      │
      ▼
Monitoring & retraining         ← Prometheus, Grafana, scheduled CI
```

In this lab series you build each layer from scratch, so by the end you understand not just *how* to use the tools but *why* they sit where they do.

---

## Roles in a typical MLOps team

| Role | Responsibility |
|---|---|
| Data Engineer | Data pipelines, storage, versioning |
| ML Engineer / Data Scientist | Model design, training, evaluation |
| MLOps Engineer | CI/CD, infrastructure, monitoring, automation |

In a small team (like this course) one person covers all three. The tooling exists to make that manageable.

---

## What each tool does

| Tool | Problem it solves | First introduced |
|---|---|---|
| **GitHub Actions** | Automates testing, training, and deployment on every code change | Lab 1 |
| **DVC** | Versions datasets and model artefacts alongside code in git | Lab 2 |
| **MLflow** | Records every training run — parameters, metrics, model artefacts | Lab 3 |
| **FastAPI** | Serves the trained model as a REST API | Lab 4 |
| **Docker** | Packages the API and its dependencies into a portable image | Lab 5 |
| **Prometheus** | Scrapes and stores time-series metrics from the running model | Lab 7a |
| **Grafana** | Visualises Prometheus metrics in dashboards | Lab 7a |

---

## Two important distinctions

### Versioning vs Tracking

These sound similar but solve different problems:

- **Versioning** (DVC) — "what data and model artefacts were used at a given commit?" Answers the reproducibility question: can you reconstruct exactly what was shipped?
- **Tracking** (MLflow) — "across all my experiments, which hyperparameters produced the best accuracy?" Answers the experimentation question: can you compare runs and make informed decisions?

You need both. DVC without MLflow gives you reproducibility but no experiment history. MLflow without DVC gives you experiment history but the model artefacts are floating loose with no data lineage.

### CI vs CD

- **CI (Continuous Integration)** — runs automatically on every push or PR. Validates that the code is correct: linting, unit tests. Introduced in Lab 1.
- **CD (Continuous Deployment)** — runs automatically on push to `main`. Trains the model, gates on accuracy, builds the Docker image, and pushes it to Docker Hub. Introduced in Lab 6.

---

## How the labs connect

```
Lab 0 ── foundations (this lab)
Lab 1 ── CI pipeline (GitHub Actions, pytest, flake8)
Lab 2 ── DVC (data + model versioning)
Lab 3 ── MLflow (experiment tracking)
Lab 4 ── FastAPI (model serving endpoint)
Lab 5 ── Docker (containerise the API)
Lab 6 ── CD pipeline (train → gate → build → push)
Lab 7 ── DVC pipeline (dvc.yaml stages, params.yaml, dvc repro in CD)
Lab 7a ─ Prometheus + Grafana (live monitoring)
Lab 7b ─ Drift detection (post-deploy accuracy gate in Actions)
Lab 7c ─ Continuous training (scheduled dvc repro trigger)
```

Each lab builds directly on the previous one. Do not skip labs — each introduces infrastructure the next lab depends on.

---

## Before starting Lab 1

Complete the [quickstart](../README.md#quickstart) to verify your environment works:

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m src.train
python -m src.evaluate
pytest tests/ -v
```

All tests should pass. The model at `models/model.pkl` should achieve ~97% accuracy on the validation set.

---

**Next:** [Lab 1 — GitHub Actions: Your First CI Pipeline](lab-01-github-actions-ci.md)
