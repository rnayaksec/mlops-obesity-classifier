"""
evaluate.py
-----------
Model evaluation: accuracy, precision, recall, and confusion matrix.
Saves results to metrics.json for use in CI/CD quality gates.
Run directly:
    python -m src.evaluate
"""

import json
import os

from src.preprocess import build_features, load_data, split_data
from src.train import load_model

DATA_PATH = os.path.join("data", "ObesityDataSet_Original.csv")
MODEL_PATH = os.path.join("models", "model.pkl")
METRICS_PATH = "metrics.json"
THRESHOLD = 0.5


def evaluate(model, dv, X, y, threshold: float = THRESHOLD) -> dict:
    """
    Evaluate model predictions against ground truth labels.
    Returns a dict with accuracy, precision, recall, and threshold.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = y_prob >= threshold

    tp = int(((y_pred == 1) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())

    accuracy = round((tp + tn) / len(y), 4)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def save_metrics(metrics: dict, path: str = METRICS_PATH):
    """Write metrics dict to a JSON file."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def main():
    print("Loading data and model...")
    df = load_data(DATA_PATH)
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)
    model, dv = load_model(MODEL_PATH)

    print("Evaluating on validation set...")
    metrics = evaluate(model, dv, X_val, y_val)

    print(f"  Accuracy:  {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")

    save_metrics(metrics)


if __name__ == "__main__":
    main()
