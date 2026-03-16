# Lab 7 — Full MLOps Pipeline: DVC + MLflow + GitHub Actions

**Prerequisites:** Labs 1–6 complete.

**What you'll build:** A DVC pipeline that replaces direct script calls, integrates MLflow tracking inside DVC stages, and plugs into the CD workflow — so a single `dvc repro` command runs the entire ML pipeline intelligently (only re-running stages whose inputs changed).

**New concepts:** DVC pipeline stages, `dvc.yaml`, `params.yaml`, incremental execution, pipeline as code.

**Time:** ~2 hours

---

## Background

Right now the CD pipeline calls `python -m src.train` and `python -m src.evaluate` as independent commands. DVC pipelines go further: they declare the *dependencies* between stages (data → preprocess → train → evaluate) so DVC can detect exactly which stages need to re-run when something changes. Change only `params.yaml`? Only the train and evaluate stages re-run, not the data loading.

---

## Step 1 — Create params.yaml

Extract all configurable values from your code into a single parameters file. Create `params.yaml` in the project root:

```yaml
model:
  solver: liblinear
  max_iter: 100
  threshold: 0.5
  seed: 42

data:
  path: data/ObesityDataSet_Original.csv
  test_size: 0.2
  val_size: 0.33
```

---

## Step 2 — Update train.py to read from params.yaml

Add this to `src/train.py` (replace the hardcoded constants at the top of `main()`):

```python
import yaml

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    solver = params["model"]["solver"]
    max_iter = params["model"]["max_iter"]
    threshold = params["model"]["threshold"]
    seed = params["model"]["seed"]
    data_path = params["data"]["path"]
    # ... rest of main() uses these variables
```

Add PyYAML to requirements:
```
pyyaml>=6.0
```

---

## Step 3 — Create dvc.yaml

Create `dvc.yaml` in the project root:

```yaml
stages:
  train:
    cmd: python -m src.train
    deps:
      - src/train.py
      - src/preprocess.py
      - data/ObesityDataSet_Original.csv
    params:
      - model.solver
      - model.max_iter
      - model.seed
      - data.path
      - data.test_size
      - data.val_size
    outs:
      - models/model.pkl

  evaluate:
    cmd: python -m src.evaluate
    deps:
      - src/evaluate.py
      - src/preprocess.py
      - models/model.pkl
      - data/ObesityDataSet_Original.csv
    params:
      - model.threshold
    metrics:
      - metrics.json:
          cache: false
```

---

## Step 4 — Run the DVC pipeline

```bash
dvc repro
```

DVC reads `dvc.yaml`, checks which stages are stale (inputs changed since last run), and runs only those. First run: both stages execute. Run again immediately:

```bash
dvc repro
```

Output: `Stage 'train' didn't change, skipping.` — DVC is smart enough not to re-train.

Now change a parameter in `params.yaml` (e.g., set `max_iter: 200`) and run again:

```bash
dvc repro
```

Only the `train` and `evaluate` stages re-run. The data loading is skipped. ✅

---

## Step 5 — Track the DVC pipeline files in git

```bash
git add dvc.yaml params.yaml requirements.txt src/train.py
git commit -m "feat: add DVC pipeline with params.yaml"
git push
```

---

## Step 6 — Update the CD workflow to use dvc repro

Open `.github/workflows/cd.yml` and replace the train + evaluate steps:

```yaml
      # Replace these two steps:
      # - name: Train model
      #   run: python -m src.train
      # - name: Evaluate model
      #   run: python -m src.evaluate

      # With this single step:
      - name: Run DVC pipeline
        run: |
          pip install dvc
          dvc repro
```

Commit and push:

```bash
git add .github/workflows/cd.yml
git commit -m "ci: use dvc repro in CD pipeline"
git push
```

---

## Step 7 — View the complete experiment in MLflow

Because `src/train.py` still has MLflow tracking (from Lab 2) and is now called by DVC, every `dvc repro` logs a run to MLflow automatically.

```bash
mlflow ui
```

You'll see runs from every `dvc repro` execution, each tagged with its parameters. The full lineage: parameters → training run → metrics → Docker image is now tracked end-to-end.

---

## What you've built — the complete picture

```
Push to main
    │
    ▼
GitHub Actions CD
    │
    ├─ pip install (incl. dvc, mlflow)
    │
    ├─ dvc repro
    │      ├─ train stage → models/model.pkl  (logged to MLflow)
    │      └─ evaluate stage → metrics.json   (logged to MLflow)
    │
    ├─ Quality gate: accuracy > 0.90?
    │      └─ No? Pipeline fails. Image not built.
    │
    └─ docker build + push → Docker Hub
```

Every component has a job:
- **DVC** — tracks data + model versions, runs pipeline incrementally
- **MLflow** — records every experiment run with its parameters and metrics
- **GitHub Actions** — orchestrates everything on every commit to main
- **Docker Hub** — hosts the deployable artefact

---

## Key takeaways

- `dvc.yaml` declares stage dependencies explicitly — DVC uses this to skip unchanged stages
- `params.yaml` is the single source of truth for all hyperparameters — change it to trigger a new run
- DVC + MLflow complement each other: DVC tracks *what ran*, MLflow tracks *what was measured*
- The full pipeline is now automated, versioned, and reproducible from a single `git clone`

---

## Where to go next

| Topic | Tool | Resource |
|---|---|---|
| Cloud DVC remote | S3 / GCS | `dvc remote add -d myremote s3://bucket/path` |
| Model registry | MLflow Model Registry | `mlflow.register_model()` |
| Kubernetes deployment | KServe | [iam-veeramalla/mlops-zero-to-hero](https://github.com/iam-veeramalla/mlops-zero-to-hero) Module 7 |
| Cloud CI/CD | AWS SageMaker Pipelines | [iam-veeramalla/mlops-zero-to-hero](https://github.com/iam-veeramalla/mlops-zero-to-hero) Module 8 |

**Congratulations — you've completed the MLOps labs.** 🎉
