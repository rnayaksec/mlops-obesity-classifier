"""
preprocess.py
-------------
Data loading, cleaning, target creation, train/val/test splitting,
and feature engineering for the obesity classification model.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

SEED = 42

CATEGORICAL_COLS = [
    "gender",
    "family_history_with_overweight",
    "favc",
    "caec",
    "smoke",
    "scc",
    "calc",
    "mtrans",
]

NUMERICAL_COLS = [
    "age",
    "height",
    "weight",
    "fcvc",
    "ncp",
    "ch2o",
    "faf",
    "tue",
]

FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

NON_OBESE_CATEGORIES = ["Normal_Weight", "Insufficient_Weight"]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV, standardise column names, and create binary target."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    if "nobeyesdad" not in df.columns:
        raise ValueError("Expected column 'nobeyesdad' not found in dataset.")

    df["is_obese"] = np.where(df["nobeyesdad"].isin(NON_OBESE_CATEGORIES), 0, 1)
    df = df.drop("nobeyesdad", axis=1)
    return df


def split_data(df: pd.DataFrame, seed: int = SEED):
    """
    Split into train (53%), validation (27%), test (20%).
    Returns: df_train, df_val, df_test, y_train, y_val, y_test
    """
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.33, random_state=seed
    )

    y_train = df_train["is_obese"].values
    y_val = df_val["is_obese"].values
    y_test = df_test["is_obese"].values

    df_train = df_train.drop("is_obese", axis=1)
    df_val = df_val.drop("is_obese", axis=1)
    df_test = df_test.drop("is_obese", axis=1)

    return df_train, df_val, df_test, y_train, y_val, y_test


def build_features(df_train, df_val, df_test):
    """
    One-hot encode categoricals, pass through numericals using DictVectorizer.
    Fits on training data only; transforms val and test.
    Returns: X_train, X_val, X_test, fitted DictVectorizer
    """
    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(df_train[FEATURES].to_dict(orient="records"))
    X_val = dv.transform(df_val[FEATURES].to_dict(orient="records"))
    X_test = dv.transform(df_test[FEATURES].to_dict(orient="records"))

    return X_train, X_val, X_test, dv
