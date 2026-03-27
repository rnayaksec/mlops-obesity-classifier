# Lab 7b — Drift Detection in GitHub Actions

**Objective:** Add a post-deploy accuracy check to the CD pipeline that pulls the live Docker image, runs inference on the held-out test set, logs results to MLflow, and fails the workflow if accuracy has dropped below threshold.

**Prerequisites:** Lab 7a complete. Docker image pushed to Docker Hub. `metrics.json` produced by `dvc repro`.

**Concepts introduced:**
- Post-deploy validation as a CI/CD stage
- Running a Docker container inside GitHub Actions
- Logging drift results to MLflow from CI
- Workflow failure with a structured summary
- Reusing existing quality gate patterns across pipeline stages

**Time:** ~1 hour

---

## Background

The quality gate in Lab 6 checks accuracy *before* the image is built, using the metrics produced by `dvc repro`. That catches regressions introduced by code or parameter changes at training time. But it doesn't catch what happens after the image ships: data distribution can shift, dependencies can behave differently in the container, or a stale model can gradually drift relative to incoming data.

Lab 7b adds a second gate *after* the Docker push. It pulls the freshly pushed image, runs it against the same validation CSV used during training, and re-measures accuracy independently. If accuracy has dropped below the threshold, the workflow fails before any downstream system can consume the new image.

---

## Step 1 — Create the drift detection script

Create `scripts/detect_drift.py`:

```python
"""
Post-deploy drift detection.
Runs inference on the validation split, logs to MLflow, exits non-zero if accuracy < threshold.
"""

import json
import sys
import os
import time
import argparse
import requests
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
from src.preprocess import load_data, split_data

THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))
API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
DATA_PATH = os.getenv("DATA_PATH", "data/ObesityDataSet_Original.csv")


def wait_for_api(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    raise RuntimeError(f"API at {url} did not become healthy within {timeout}s")


def run_validation(api_url: str) -> dict:
    df = load_data(DATA_PATH)
    _, df_val, _, _, y_val, _ = split_data(df)

    preds = []
    for _, row in df_val.iterrows():
        payload = row.drop("NObeyesdad").to_dict()
        r = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        r.raise_for_status()
        preds.append(r.json()["prediction"])

    accuracy = accuracy_score(y_val, preds)
    return {"accuracy": round(accuracy, 4), "n_samples": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-tag", default="unknown", help="Docker image tag being validated")
    args = parser.parse_args()

    print(f"Waiting for API at {API_URL}...")
    wait_for_api(API_URL)

    print("Running validation inference...")
    results = run_validation(API_URL)

    accuracy = results["accuracy"]
    print(f"Validation accuracy: {accuracy:.4f} (threshold: {THRESHOLD})")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("obesity-classifier-drift")
    with mlflow.start_run(run_name=f"drift-check-{args.image_tag}"):
        mlflow.log_param("image_tag", args.image_tag)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_metric("post_deploy_accuracy", accuracy)
        mlflow.log_metric("n_validation_samples", results["n_samples"])
        mlflow.set_tag("drift_check", "pass" if accuracy >= THRESHOLD else "fail")

    if accuracy < THRESHOLD:
        print(f"DRIFT DETECTED: accuracy {accuracy:.4f} < threshold {THRESHOLD}")
        sys.exit(1)

    print(f"Drift check passed: accuracy {accuracy:.4f} >= threshold {THRESHOLD}")


if __name__ == "__main__":
    main()
```

---

## Step 2 — Add the drift detection job to cd.yml

Open `.github/workflows/cd.yml` and add a new job after the existing `deploy` job. The full updated file:

```yaml
name: CD

on:
  push:
    branches: [main]

env:
  IMAGE_NAME: ${{ secrets.DOCKERHUB_USERNAME }}/obesity-classifier
  ACCURACY_THRESHOLD: "0.90"

jobs:
  train-and-gate:
    runs-on: ubuntu-latest
    outputs:
      accuracy: ${{ steps.gate.outputs.accuracy }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run DVC pipeline
        run: dvc repro

      - name: Accuracy gate
        id: gate
        run: |
          ACCURACY=$(python -c "import json; print(json.load(open('metrics.json'))['accuracy'])")
          echo "accuracy=$ACCURACY" >> "$GITHUB_OUTPUT"
          python -c "
          acc = float('$ACCURACY')
          threshold = float('${{ env.ACCURACY_THRESHOLD }}')
          if acc < threshold:
              raise SystemExit(f'Accuracy {acc:.4f} below threshold {threshold}')
          print(f'Gate passed: {acc:.4f} >= {threshold}')
          "

  build-and-push:
    needs: train-and-gate
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=sha-
            type=raw,value=latest

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}

  drift-check:
    needs: build-and-push
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Pull and start model container
        run: |
          docker pull ${{ env.IMAGE_NAME }}:latest
          docker run -d \
            --name obesity-model \
            -p 8000:8000 \
            ${{ env.IMAGE_NAME }}:latest

      - name: Run drift detection
        env:
          MODEL_API_URL: http://localhost:8000
          MLFLOW_TRACKING_URI: ./mlruns
          DATA_PATH: data/ObesityDataSet_Original.csv
          ACCURACY_THRESHOLD: ${{ env.ACCURACY_THRESHOLD }}
        run: |
          python scripts/detect_drift.py --image-tag ${{ github.sha }}

      - name: Post summary on failure
        if: failure()
        run: |
          echo "## Drift Detection Failed" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "Post-deploy accuracy fell below the **${{ env.ACCURACY_THRESHOLD }}** threshold." >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "The Docker image was pushed but downstream consumers should not promote it." >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "Check the MLflow run in the \`obesity-classifier-drift\` experiment for details." >> $GITHUB_STEP_SUMMARY

      - name: Stop model container
        if: always()
        run: docker stop obesity-model && docker rm obesity-model
```

---

## Step 3 — Create the scripts directory and commit

```bash
mkdir -p scripts
git add scripts/detect_drift.py .github/workflows/cd.yml
git commit -m "feat: add post-deploy drift detection stage"
git push
```

---

## Verification

1. Push to `main` and open the Actions tab on GitHub
2. Observe three jobs run in sequence: `train-and-gate` → `build-and-push` → `drift-check`
3. In the `drift-check` job logs you should see:
   ```
   Waiting for API at http://localhost:8000...
   Running validation inference...
   Validation accuracy: 0.9700 (threshold: 0.90)
   Drift check passed: accuracy 0.9700 >= threshold 0.90
   ```
4. To test failure behaviour: temporarily set `ACCURACY_THRESHOLD: "0.99"` in the workflow, push, and confirm the job fails with the summary message. Revert before continuing.

**Confirm MLflow logged the run:**
```bash
mlflow ui
```
Open `http://localhost:5000`, navigate to the `obesity-classifier-drift` experiment. You should see a run tagged `drift_check=pass`.

---

## How this connects to Lab 7c

Lab 7b added a reactive check — it fires after every push to `main`. Lab 7c adds a *proactive* check: a scheduled workflow that re-runs the full training pipeline on a weekly cadence regardless of whether any code changed. This is continuous training: the model is periodically retrained on the latest data, and the same accuracy gate and drift check apply automatically.

**Next:** [Lab 7c — Scheduled Continuous Training](lab-07c-continuous-training.md)
