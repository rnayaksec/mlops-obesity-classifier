# Lab 7c — Scheduled Continuous Training

**Objective:** Add a weekly `on: schedule` trigger to the CD workflow so the full ML pipeline reruns automatically, rebuilds the Docker image only when needed, and applies the same accuracy gate and drift check as the push-triggered pipeline.

**Prerequisites:** Lab 7b complete. The full CD pipeline (`train-and-gate` → `build-and-push` → `drift-check`) is passing on push to `main`.

**Concepts introduced:**
- `on: schedule` cron triggers in GitHub Actions
- Event-driven vs schedule-driven CI/CD in the same workflow file
- DVC early-exit behaviour when no inputs have changed
- Conditional job execution based on DVC output
- Continuous training (CT) as distinct from continuous deployment (CD)

**Time:** ~45 minutes

---

## Background

The CD pipeline you built in Labs 6–7b fires on every push to `main`. That handles *code* changes well. But models can degrade for reasons that have nothing to do with code — new data arrives, the data distribution shifts, or a third-party feature source drifts. Continuous training (CT) solves this by periodically rerunning the pipeline on the latest data, independent of any code push.

The key design insight: you don't need a separate workflow. GitHub Actions supports multiple triggers on the same workflow file. A `push` trigger handles developer-driven deploys; a `schedule` trigger handles time-driven retraining. Both go through the same quality gate and drift check, so the same safety properties apply.

---

## Step 1 — Understand the DVC early-exit behaviour

Before adding the schedule trigger, run `dvc repro` twice back-to-back locally:

```bash
dvc repro
dvc repro
```

Second run output:
```
Stage 'train' didn't change, skipping.
Stage 'evaluate' didn't change, skipping.
Data and pipelines are up to date.
```

DVC computes a hash of every declared input (source files, data files, `params.yaml` values). If nothing changed, no stage runs. This is what makes a weekly CT trigger safe: if no new data has arrived and no parameters changed, `dvc repro` exits cleanly in seconds without re-training or overwriting the model.

---

## Step 2 — Add the schedule trigger to cd.yml

Open `.github/workflows/cd.yml` and update the `on:` block:

```yaml
on:
  push:
    branches: [main]
  schedule:
    # Runs every Monday at 02:00 UTC
    - cron: "0 2 * * 1"
  workflow_dispatch:
    # Allows manual trigger from the Actions tab
```

`workflow_dispatch` is optional but highly recommended — it lets you manually trigger the scheduled pipeline to test it without waiting for Monday.

---

## Step 3 — Add a DVC change-detection step

The full pipeline (train, evaluate, build, push) is expensive to run if nothing changed. Add a step to the `train-and-gate` job that detects whether DVC has any work to do and surfaces that as a job output:

```yaml
  train-and-gate:
    runs-on: ubuntu-latest
    outputs:
      accuracy: ${{ steps.gate.outputs.accuracy }}
      pipeline_ran: ${{ steps.dvc_check.outputs.pipeline_ran }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run DVC pipeline
        id: dvc_check
        run: |
          OUTPUT=$(dvc repro 2>&1)
          echo "$OUTPUT"
          if echo "$OUTPUT" | grep -q "up to date"; then
            echo "pipeline_ran=false" >> "$GITHUB_OUTPUT"
            echo "DVC: no inputs changed — skipping downstream jobs"
          else
            echo "pipeline_ran=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Accuracy gate
        id: gate
        if: steps.dvc_check.outputs.pipeline_ran == 'true'
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
```

---

## Step 4 — Make build-and-push conditional

Update the `build-and-push` job to skip when DVC found nothing to do:

```yaml
  build-and-push:
    needs: train-and-gate
    if: needs.train-and-gate.outputs.pipeline_ran == 'true'
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    # ... rest of job unchanged
```

And likewise for `drift-check`:

```yaml
  drift-check:
    needs: build-and-push
    if: needs.train-and-gate.outputs.pipeline_ran == 'true'
    runs-on: ubuntu-latest
    # ... rest of job unchanged
```

---

## Step 5 — The complete updated cd.yml

Replace `.github/workflows/cd.yml` with:

```yaml
name: CD

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 2 * * 1"
  workflow_dispatch:

env:
  IMAGE_NAME: ${{ secrets.DOCKERHUB_USERNAME }}/obesity-classifier
  ACCURACY_THRESHOLD: "0.90"

jobs:
  train-and-gate:
    runs-on: ubuntu-latest
    outputs:
      accuracy: ${{ steps.gate.outputs.accuracy }}
      pipeline_ran: ${{ steps.dvc_check.outputs.pipeline_ran }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run DVC pipeline
        id: dvc_check
        run: |
          OUTPUT=$(dvc repro 2>&1)
          echo "$OUTPUT"
          if echo "$OUTPUT" | grep -q "up to date"; then
            echo "pipeline_ran=false" >> "$GITHUB_OUTPUT"
            echo "DVC: no inputs changed — skipping downstream jobs"
          else
            echo "pipeline_ran=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Accuracy gate
        id: gate
        if: steps.dvc_check.outputs.pipeline_ran == 'true'
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
    if: needs.train-and-gate.outputs.pipeline_ran == 'true'
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
    needs: [train-and-gate, build-and-push]
    if: needs.train-and-gate.outputs.pipeline_ran == 'true'
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

      - name: Stop model container
        if: always()
        run: docker stop obesity-model && docker rm obesity-model
```

---

## Step 6 — Commit and trigger manually

```bash
git add .github/workflows/cd.yml
git commit -m "feat: add scheduled CT trigger with DVC early-exit"
git push
```

Trigger the workflow manually to test without waiting for Monday:

1. Go to your repo on GitHub → **Actions → CD → Run workflow → Run workflow**
2. Watch `train-and-gate` — if data and params haven't changed, you'll see:
   ```
   DVC: no inputs changed — skipping downstream jobs
   ```
   And `build-and-push` and `drift-check` will show as **skipped** (grey), not failed.

---

## Step 7 — Test the full CT path

Simulate a data update to trigger the full pipeline:

```bash
# Add a comment line to the CSV to change its hash
echo "# updated $(date)" >> data/ObesityDataSet_Original.csv
git add data/ObesityDataSet_Original.csv
git commit -m "data: simulate weekly data refresh"
git push
```

Now the full pipeline runs: DVC detects the changed input, retrains, gates on accuracy, builds, pushes, and runs the drift check.

---

## Verification

| Scenario | `pipeline_ran` | `build-and-push` | `drift-check` |
|---|---|---|---|
| Push with no data/param changes | `false` | skipped | skipped |
| Push with code change (no data change) | `false` | skipped | skipped |
| Push with `params.yaml` change | `true` | runs | runs |
| Push with CSV data change | `true` | runs | runs |
| Weekly schedule, no changes | `false` | skipped | skipped |
| Weekly schedule, new data landed | `true` | runs | runs |

Confirm expected behaviour by checking the `dvc_check` step output in each scenario.

---

## Key takeaways

- `on: schedule` and `on: push` can coexist in the same workflow — the same jobs run regardless of which trigger fired
- DVC's content-addressed hashing makes scheduled CT safe: no wasted compute when nothing changed
- `workflow_dispatch` gives you a manual escape hatch to test scheduled workflows on demand
- The accuracy gate and drift check apply to scheduled runs identically to push-triggered runs — there is no separate "safe" path

---

## Congratulations — you've completed the full lab series

You have built a production-grade MLOps pipeline from scratch:

```
Code push / weekly schedule
        │
        ▼
GitHub Actions CD
        │
        ├─ dvc repro (skip if unchanged)
        │      ├─ train → model.pkl  (logged to MLflow)
        │      └─ evaluate → metrics.json
        │
        ├─ Accuracy gate: > 0.90?
        │
        ├─ docker build + push → Docker Hub
        │
        └─ Drift check: pull image, re-measure accuracy, log to MLflow
                │
                └─ Accuracy still > 0.90? ✅ Done. ❌ Fail with summary.

Monitoring (always on):
        model container → /metrics → Prometheus → Grafana
```

Every component has a job, every gate has a purpose, and the whole system runs without manual intervention.
