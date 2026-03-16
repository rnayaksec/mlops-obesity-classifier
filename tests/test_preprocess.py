"""Unit tests for src/preprocess.py"""

import numpy as np
import pytest

from src.preprocess import (
    build_features,
    load_data,
    split_data,
)

DATA_PATH = "data/ObesityDataSet_Original.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def split(raw_df):
    return split_data(raw_df)


# --- load_data ---

def test_load_data_shape(raw_df):
    assert raw_df.shape[0] == 2111
    assert raw_df.shape[1] == 17  # 16 features + is_obese


def test_load_data_no_nulls(raw_df):
    assert raw_df.isnull().sum().sum() == 0


def test_load_data_target_binary(raw_df):
    assert set(raw_df["is_obese"].unique()) == {0, 1}


def test_load_data_drops_nobeyesdad(raw_df):
    assert "nobeyesdad" not in raw_df.columns


# --- split_data ---

def test_split_sizes(split):
    df_train, df_val, df_test, y_train, y_val, y_test = split
    total = len(df_train) + len(df_val) + len(df_test)
    assert total == 2111


def test_split_target_not_in_features(split):
    df_train, df_val, df_test, *_ = split
    for df in [df_train, df_val, df_test]:
        assert "is_obese" not in df.columns


def test_split_label_arrays_shape(split):
    _, _, _, y_train, y_val, y_test = split
    assert y_train.ndim == 1
    assert y_val.ndim == 1
    assert y_test.ndim == 1


# --- build_features ---

def test_build_features_output_shape(split):
    df_train, df_val, df_test, *_ = split
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)
    assert X_train.shape[0] == len(df_train)
    assert X_val.shape[0] == len(df_val)
    assert X_test.shape[0] == len(df_test)


def test_build_features_same_columns(split):
    df_train, df_val, df_test, *_ = split
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


def test_build_features_no_nan(split):
    df_train, df_val, df_test, *_ = split
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
