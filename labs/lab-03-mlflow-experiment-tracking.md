# Lab 3 — Experiment Tracking with MLflow

**Prerequisites:** Lab 2 complete. DVC initialised, dataset and model tracked.

**What you'll build:** MLflow tracking inside `train.py` so every training run automatically logs its hyperparameters, metrics, and model artefact. You'll compare runs in the MLflow UI.

**New concepts:** Experiment tracking, run logging, model registry (local), run comparison.

**Time:** ~1 hour

---

## Background

Right now every time you run `python -m src.train`, the old model is silently overwritten. You have no record of what hyperparameters were used, what the metrics were, or how this run compared to yesterday's. MLflow fixes this by giving every run a unique ID and storing everything.

---

## Step 1 — Install MLflow

```bash
pip install mlflow
```

Add it to `requirements.txt`:

```
mlflow>=2.10
```

---

## Step 2 — Add MLflow tracking to train.py

Open `src/train.py` and update the `main()` function to wrap the training in an MLflow run. Replace the existing `main()` with:

```python
import mlflow
import mlflow.sklearn

def main():
    # --- configurable hyperparameters (will log these) ---
    solver = "liblinear"
    max_iter = 100
    threshold = 0.5

    mlflow.set_experiment("obesity-classifier")

    with mlflow.start_run():
        print("Loading data...")
        df = load_data(DATA_PATH)
        df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
        X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)

        # Log hyperparameters
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("threshold", threshold)

        print("Training model...")
        model = LogisticRegression(
            solver=solver, max_iter=max_iter, random_state=SEED
        )
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        from src.evaluate import evaluate
        metrics = evaluate(model, dv, X_val, y_val, threshold=threshold)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])

        # Log the model artefact
        mlflow.sklearn.log_model(model, artifact_path="model")

        save_model(model, dv)
        print(f"Run complete. Accuracy: {metrics['accuracy']}")
```

---

## Step 3 — Run training three times with different hyperparameters

```bash
python -m src.train
```

Now edit `src/train.py`, change `solver = "liblinear"` to `solver = "saga"`, and run again:

```bash
python -m src.train
```

Change `max_iter` from `100` to `200` and run a third time:

```bash
python -m src.train
```

---

## Step 4 — Open the MLflow UI

```bash
mlflow ui
```

Open your browser at `http://localhost:5000`. You'll see:
- Three runs under the `obesity-classifier` experiment
- Each run's parameters (solver, max_iter, threshold)
- Each run's metrics (accuracy, precision, recall)
- Click any run to see the full artefact list including the saved model

Click **"Compare"** after selecting two runs to see a side-by-side diff of metrics.

---

## Step 5 — Add mlruns/ to .gitignore

MLflow stores all run data in a local `mlruns/` folder. This should not go into git:

Open `.gitignore` and verify this line is present (it already is from the starter):
```
mlruns/
```

Commit your `train.py` and `requirements.txt` changes:

```bash
git add src/train.py requirements.txt
git commit -m "feat: add MLflow experiment tracking to train.py"
git push
```

Your CI pipeline will run — it skips the MLflow UI but still validates the code runs without errors.

---

## Key takeaways

- `mlflow.set_experiment()` creates a named bucket for related runs
- `mlflow.start_run()` opens a context where everything logged is attached to that run
- `mlflow.log_param()` stores hyperparameters; `mlflow.log_metric()` stores output metrics
- `mlflow.sklearn.log_model()` saves the model artefact alongside the run
- The MLflow UI lets you compare runs visually — essential when tuning models

---

**Next:** [Lab 4 — REST API Model Serving with FastAPI](lab-04-fastapi-model-serving.md)
