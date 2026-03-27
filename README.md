[![CI](https://github.com/rnayaksec/mlops-obesity-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/rnayaksec/mlops-obesity-classifier/actions/workflows/ci.yml)

# mlops-obesity-classifier

A hands-on MLOps starter project built around a logistic regression model that classifies obesity risk. This repo is the **prerequisite starting point** for the [MLOps Labs](#labs) — clone it, run it, then work through the labs to progressively add MLOps tooling.

---

## What this project does

Trains a logistic regression model on the [Obesity Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) to classify whether a person is obese (binary: 0/1). It achieves ~97% accuracy on the validation set.

The model pipeline is broken into three reusable modules:

| Module              | Responsibility                                         |
| ------------------- | ------------------------------------------------------ |
| `src/preprocess.py` | Load data, create binary target, split, one-hot encode |
| `src/train.py`      | Train logistic regression, save model as `.pkl`        |
| `src/evaluate.py`   | Compute accuracy/precision/recall, save `metrics.json` |

---

## Prerequisites

- Python 3.11+
- Git

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/rnayaksec/mlops-obesity-classifier.git
cd mlops-obesity-classifier

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Mac / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
python -m src.train

# 5. Evaluate the model
python -m src.evaluate

# 6. Run the test suite
pytest tests/ -v
```

After step 4 you'll have a trained model at `models/model.pkl`.
After step 5 you'll have evaluation results at `metrics.json`.
All tests in step 6 should pass. ✅

---

## Project structure

```
mlops-obesity-classifier/
├── .github/
│   └── workflows/          ← GitHub Actions workflows (added in Lab 1)
├── data/
│   └── ObesityDataSet_Original.csv
├── notebooks/
│   └── obesity_classification.ipynb   ← original exploratory notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_evaluate.py
├── models/                 ← saved model artefacts (git-ignored)
├── labs/                   ← step-by-step lab guides (see below)
├── .flake8
├── pyproject.toml          ← black config
├── requirements.txt
└── README.md
```

---

## Labs

The labs build progressively on this starter project, each introducing a new MLOps concept.

| Lab                                                      | Topic                           | Key tools                          |
| -------------------------------------------------------- | ------------------------------- | ---------------------------------- |
| [Lab 0](labs/lab-00-mlops-foundations.md)               | MLOps foundations               | Concepts only — read first         |
| [Lab 1](labs/lab-01-github-actions-ci.md)               | GitHub Actions — CI pipeline    | GitHub Actions                     |
| [Lab 2](labs/lab-02-dvc-versioning.md)                  | Data & model versioning         | DVC                                |
| [Lab 3](labs/lab-03-mlflow-experiment-tracking.md)      | Experiment tracking             | MLflow                             |
| [Lab 4](labs/lab-04-fastapi-model-serving.md)           | REST API model serving          | FastAPI                            |
| [Lab 5](labs/lab-05-docker-containerisation.md)         | Containerisation                | Docker                             |
| [Lab 6](labs/lab-06-advanced-cicd.md)                   | Advanced CI/CD pipeline         | GitHub Actions + Docker Hub        |
| [Lab 7](labs/lab-07-full-mlops-pipeline.md)             | DVC pipeline + MLflow in CI     | DVC + MLflow + GitHub Actions      |
| [Lab 7a](labs/lab-07a-prometheus-grafana.md)            | Live monitoring                 | Prometheus + Grafana + Docker Compose |
| [Lab 7b](labs/lab-07b-drift-detection.md)               | Drift detection                 | GitHub Actions + MLflow            |
| [Lab 7c](labs/lab-07c-continuous-training.md)           | Continuous training             | GitHub Actions (scheduled)         |

Start with [Lab 0](labs/lab-00-mlops-foundations.md).

---

## Branching strategy

`main` always reflects the latest completed lab — it moves forward as you work through the course.

Each completed lab is also saved as a permanent reference branch so you can see the exact state of the project at any point:

| Branch             | State                                                |
| ------------------ | ---------------------------------------------------- |
| `main`             | Starter project — prereqs complete, no labs started  |
| `lab-01-solution`  | After Lab 1: GitHub Actions CI pipeline              |
| `lab-02-solution`  | After Lab 2: DVC data & model versioning             |
| `lab-03-solution`  | After Lab 3: MLflow experiment tracking              |
| `lab-04-solution`  | After Lab 4: FastAPI model serving                   |
| `lab-05-solution`  | After Lab 5: Docker containerisation                 |
| `lab-06-solution`  | After Lab 6: Advanced CI/CD pipeline                 |
| `lab-07-solution`  | After Lab 7: DVC pipeline + MLflow in CI             |
| `lab-07a-solution` | After Lab 7a: Prometheus + Grafana monitoring        |
| `lab-07b-solution` | After Lab 7b: Drift detection in GitHub Actions      |
| `lab-07c-solution` | After Lab 7c: Scheduled continuous training          |

If you get stuck on a lab, check out the previous solution branch to see the working starting point:
```bash
git checkout lab-02-solution
```

---


## Running individual modules

```bash
# Train only
python -m src.train

# Evaluate only (requires a trained model)
python -m src.evaluate

# Run tests with coverage output
pytest tests/ -v
```

---

## Companion repo

For the foundational ML concepts this project builds on, see [ml-concepts-refresher](https://github.com/rnayaksec/ml-concepts-refresher).
