"""
train.py
--------
Model training and persistence for the obesity classification model.
Run directly to train and save the model:
    python -m src.train
"""

import os
import pickle
import mlflow
import mlflow.sklearn
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
    # --- configurable hyperparameters (will log these) ---
    # Lab2 - trial 1
    solver = "liblinear"
    # Lab2 - trial 2
    # solver = "saga"
    # Lab2 - trial 3
    max_iter = 200
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
        model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=SEED)
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


if __name__ == "__main__":
    main()
