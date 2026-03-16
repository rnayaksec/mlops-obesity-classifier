"""Unit tests for src/evaluate.py"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.evaluate import evaluate, save_metrics
from src.preprocess import build_features, load_data, split_data
from src.train import train_model

DATA_PATH = "data/ObesityDataSet_Original.csv"


@pytest.fixture(scope="module")
def eval_bundle():
    df = load_data(DATA_PATH)
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)
    model = train_model(X_train, y_train)
    metrics = evaluate(model, dv, X_val, y_val)
    return metrics


# --- evaluate ---


def test_metrics_keys_present(eval_bundle):
    expected = {"accuracy", "precision", "recall", "threshold", "confusion_matrix"}
    assert expected.issubset(eval_bundle.keys())


def test_accuracy_in_range(eval_bundle):
    assert 0.0 <= eval_bundle["accuracy"] <= 1.0


def test_precision_in_range(eval_bundle):
    assert 0.0 <= eval_bundle["precision"] <= 1.0


def test_recall_in_range(eval_bundle):
    assert 0.0 <= eval_bundle["recall"] <= 1.0


def test_accuracy_above_minimum(eval_bundle):
    assert (
        eval_bundle["accuracy"] > 0.999
    ), f"Accuracy {eval_bundle['accuracy']} is below minimum threshold of 0.90"


def test_confusion_matrix_keys(eval_bundle):
    cm = eval_bundle["confusion_matrix"]
    assert {"tp", "fp", "fn", "tn"} == set(cm.keys())


def test_confusion_matrix_non_negative(eval_bundle):
    cm = eval_bundle["confusion_matrix"]
    assert all(v >= 0 for v in cm.values())


# --- save_metrics ---


def test_save_metrics_creates_file(eval_bundle):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metrics.json")
        save_metrics(eval_bundle, path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == eval_bundle["accuracy"]
