"""Unit tests for src/train.py"""

import os
import pickle
import tempfile

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocess import build_features, load_data, split_data
from src.train import load_model, save_model, train_model

DATA_PATH = "data/ObesityDataSet_Original.csv"


@pytest.fixture(scope="module")
def trained_bundle():
    df = load_data(DATA_PATH)
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)
    model = train_model(X_train, y_train)
    return model, dv, X_val, y_val


# --- train_model ---

def test_returns_logistic_regression(trained_bundle):
    model, *_ = trained_bundle
    assert isinstance(model, LogisticRegression)


def test_model_is_fitted(trained_bundle):
    model, *_ = trained_bundle
    assert hasattr(model, "coef_")


def test_validation_accuracy_above_threshold(trained_bundle):
    model, dv, X_val, y_val = trained_bundle
    y_pred = model.predict_proba(X_val)[:, 1] >= 0.5
    accuracy = (y_pred == y_val).mean()
    assert accuracy > 0.90, f"Accuracy {accuracy:.3f} is below 0.90"


def test_predict_proba_in_range(trained_bundle):
    model, dv, X_val, _ = trained_bundle
    probs = model.predict_proba(X_val)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# --- save_model / load_model ---

def test_save_and_load_model(trained_bundle):
    model, dv, X_val, y_val = trained_bundle
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        save_model(model, dv, path)
        assert os.path.exists(path)

        loaded_model, loaded_dv = load_model(path)
        assert isinstance(loaded_model, LogisticRegression)

        # predictions should be identical before and after save/load
        original_preds = model.predict_proba(X_val)[:, 1]
        loaded_preds = loaded_model.predict_proba(X_val)[:, 1]
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
