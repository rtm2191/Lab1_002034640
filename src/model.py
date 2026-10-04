"""
Breast cancer classification pipeline with a model quality gate.

Functions mirror the original lab's fun1-fun4 structure:
    load_data()     -> reads the dataset from the data/ folder
    train_model()   -> trains a scaled logistic regression model
    evaluate()      -> computes accuracy, precision and recall
    run_pipeline()  -> combines all of the above end to end

In this dataset, label 0 = malignant and label 1 = benign.
Missing a malignant tumor (false negative) is the costly mistake,
so the quality gate checks recall on the malignant class.
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "breast_cancer.csv")
TARGET_COLUMN = "target"
MALIGNANT = 0
RANDOM_STATE = 42

# Quality gate thresholds
MIN_ACCURACY = 0.93
MIN_MALIGNANT_RECALL = 0.93


def load_data(path=DATA_PATH):
    """Load the dataset and split it into features (X) and labels (y)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column")
    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_model(X, y, random_state=RANDOM_STATE):
    """Train a logistic regression model with feature scaling."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    model.fit(X, y)
    return model


def evaluate(model, X, y):
    """Return accuracy, precision and recall, with malignant as the positive class."""
    predictions = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "precision_malignant": round(
            float(precision_score(y, predictions, pos_label=MALIGNANT)), 4
        ),
        "recall_malignant": round(
            float(recall_score(y, predictions, pos_label=MALIGNANT)), 4
        ),
    }


def run_pipeline(path=DATA_PATH, test_size=0.2, random_state=RANDOM_STATE):
    """Load data, split it, train the model and evaluate it on held-out data."""
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = train_model(X_train, y_train, random_state=random_state)
    metrics = evaluate(model, X_test, y_test)
    return model, metrics


def passes_quality_gate(metrics):
    """Return True if the model meets the minimum accuracy and recall thresholds."""
    return (
        metrics["accuracy"] >= MIN_ACCURACY
        and metrics["recall_malignant"] >= MIN_MALIGNANT_RECALL
    )


if __name__ == "__main__":
    # Used by the GitHub Actions workflow to train, save and gate the model.
    os.makedirs("artifacts", exist_ok=True)

    trained_model, results = run_pipeline()
    joblib.dump(trained_model, os.path.join("artifacts", "model.joblib"))

    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

    if not passes_quality_gate(results):
        print(
            f"Quality gate FAILED: need accuracy >= {MIN_ACCURACY} "
            f"and malignant recall >= {MIN_MALIGNANT_RECALL}"
        )
        sys.exit(1)

    print("Quality gate PASSED")
