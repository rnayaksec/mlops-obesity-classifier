# Lab 6 — Advanced CI/CD: Train → Evaluate → Build → Push

**Prerequisites:** Labs 1–5 complete. Docker Hub account set up.

**What you'll build:** A CD (Continuous Deployment) pipeline in GitHub Actions that automatically retrains the model, gates on accuracy, builds the Docker image, and pushes it to Docker Hub — all triggered on every push to `main`.

**New concepts:** CD pipeline, quality gates, GitHub Secrets, pipeline branching (CI vs CD), environment promotion.

**Time:** ~1.5 hours

---

## Background

In Lab 1 you built a CI pipeline that runs tests. That's *Continuous Integration* — merging code safely. Now you'll add *Continuous Deployment* — automatically packaging and shipping the model whenever the main branch changes. The key addition is a **quality gate**: the pipeline will refuse to build the Docker image if the model's accuracy is below a minimum threshold.

---

## Step 1 — Add GitHub Secrets

Your CD pipeline needs to log in to Docker Hub. Never put credentials in code — use GitHub Secrets instead.

1. Go to your repo on GitHub → **Settings → Secrets and variables → Actions → New repository secret**
2. Add two secrets:
   - `DOCKERHUB_USERNAME` — your Docker Hub username
   - `DOCKERHUB_TOKEN` — create an access token at [hub.docker.com/settings/security](https://hub.docker.com/settings/security)

---

## Step 2 — Update evaluate.py to write metrics.json

The quality gate needs a file it can read. Verify `src/evaluate.py` already has this (it does from the starter):

```python
def save_metrics(metrics: dict, path: str = METRICS_PATH):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
```

And `main()` calls `save_metrics(metrics)` at the end. ✅

---

## Step 3 — Create the CD workflow

Create `.github/workflows/cd.yml`:

```yaml
name: CD

on:
  push:
    branches: [main]

jobs:
  train-evaluate-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python -m src.train

      - name: Evaluate model
        run: python -m src.evaluate

      - name: Accuracy quality gate
        run: |
          python - <<'EOF'
          import json, sys
          with open("metrics.json") as f:
              m = json.load(f)
          accuracy = m["accuracy"]
          threshold = 0.90
          print(f"Model accuracy: {accuracy} (threshold: {threshold})")
          if accuracy < threshold:
              print(f"FAILED: accuracy {accuracy} is below minimum {threshold}")
              sys.exit(1)
          print("PASSED: accuracy gate")
          EOF

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build Docker image
        run: |
          docker build -t ${{ secrets.DOCKERHUB_USERNAME }}/obesity-classifier:latest .

      - name: Push Docker image
        run: |
          docker push ${{ secrets.DOCKERHUB_USERNAME }}/obesity-classifier:latest
```

---

## Step 4 — Commit and push to main

```bash
git add .github/workflows/cd.yml
git commit -m "ci: add CD pipeline with quality gate and Docker push"
git push
```

Go to **Actions** on GitHub. You'll see both `CI` and `CD` running. The CD pipeline will:
1. Train the model from scratch on GitHub's Ubuntu runner
2. Evaluate it and write `metrics.json`
3. Check accuracy > 0.90
4. Build the Docker image
5. Push it to Docker Hub

---

## Step 5 — Test the quality gate

Simulate a bad model by temporarily lowering the threshold in `src/evaluate.py` so it *outputs* a low accuracy — or more realistically, raise the gate threshold to an unreachable value to see it fail.

In `.github/workflows/cd.yml`, change:
```python
threshold = 0.90
```
to:
```python
threshold = 0.999
```

Commit and push. Watch the pipeline fail at the quality gate step — **the Docker image is NOT built or pushed**. This is the key behaviour: broken models never reach production.

Revert the threshold and push again. Green. ✅

---

## Step 6 — Understand CI vs CD

You now have two workflow files:

| File | Trigger | Purpose |
|---|---|---|
| `ci.yml` | Every push, every branch | Fast feedback: lint + test |
| `cd.yml` | Push to `main` only | Full pipeline: train + evaluate + gate + deploy |

Feature branches run CI only. When a branch is merged to `main`, CD kicks in.

---

## Key takeaways

- GitHub Secrets keep credentials out of code — they're injected as environment variables at runtime
- A **quality gate** is just an assertion in your pipeline: fail the step if the metric is below threshold
- CI and CD are separate workflows with separate triggers — CI is fast and runs everywhere, CD is slower and only runs on `main`
- Every push to `main` now produces a fresh, validated Docker image on Docker Hub automatically

---

**Next:** [Lab 7 — Full MLOps Pipeline](lab-07-full-mlops-pipeline.md)
