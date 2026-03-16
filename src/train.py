"""
train.py
--------
Model training and persistence for the obesity classification model.
Run directly to train and save the model:
    python -m src.train
"""

import os
import pickle

from sklearn.linear_model import LogisticRegression

from src.preprocess import build_features, load_data, split_data

SEED = 42
DATA_PATH = os.path.join("data", "ObesityDataSet_Original.csv")
MODEL_PATH = os.path.join("models", "model.pkl")


def train_model(X_train, y_train, seed: int = SEED):
    """Train a logistic regression model and return the fitted model."""
    model = LogisticRegression(solver="liblinear", random_state=seed)
    model.fit(X_train, y_train)
    return model


def save_model(model, dv, path: str = MODEL_PATH):
    """Persist the trained model and DictVectorizer together as a pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "dv": dv}, f)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH):
    """Load and return the saved model bundle (model + dv)."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["dv"]


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Splitting data...")
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)

    print("Building features...")
    X_train, X_val, X_test, dv = build_features(df_train, df_val, df_test)

    print("Training model...")
    model = train_model(X_train, y_train)

    save_model(model, dv)
    print("Training complete.")


if __name__ == "__main__":
    main()
