import numpy as np
import pandas as pd
import pytest

from src import model


@pytest.fixture(scope="module")
def data():
    return model.load_data()


@pytest.fixture(scope="module")
def pipeline_result():
    return model.run_pipeline()


# ---------- load_data ----------

def test_load_data_shape(data):
    X, y = data
    assert X.shape == (569, 30)
    assert len(y) == 569


def test_load_data_no_missing_values(data):
    X, y = data
    assert not X.isnull().values.any()
    assert not y.isnull().any()


def test_load_data_labels_are_binary(data):
    _, y = data
    assert set(y.unique()) == {0, 1}


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        model.load_data("data/does_not_exist.csv")


def test_load_data_missing_target_raises(tmp_path):
    bad_file = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad_file, index=False)
    with pytest.raises(ValueError):
        model.load_data(str(bad_file))


# ---------- train_model ----------

def test_train_model_predicts_valid_labels(data):
    X, y = data
    trained = model.train_model(X, y)
    predictions = trained.predict(X)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert len(predictions) == len(y)


def test_train_model_mismatched_lengths_raises(data):
    X, y = data
    with pytest.raises(ValueError):
        model.train_model(X, y[:-10])


# ---------- evaluate ----------

def test_evaluate_returns_expected_keys(pipeline_result):
    _, metrics = pipeline_result
    assert set(metrics) == {"accuracy", "precision_malignant", "recall_malignant"}


@pytest.mark.parametrize("metric", ["accuracy", "precision_malignant", "recall_malignant"])
def test_metrics_are_between_0_and_1(pipeline_result, metric):
    _, metrics = pipeline_result
    assert 0.0 <= metrics[metric] <= 1.0


# ---------- quality gate ----------

def test_accuracy_meets_threshold(pipeline_result):
    _, metrics = pipeline_result
    assert metrics["accuracy"] >= model.MIN_ACCURACY


def test_malignant_recall_meets_threshold(pipeline_result):
    _, metrics = pipeline_result
    assert metrics["recall_malignant"] >= model.MIN_MALIGNANT_RECALL


@pytest.mark.parametrize(
    "metrics, expected",
    [
        ({"accuracy": 0.97, "recall_malignant": 0.96}, True),
        ({"accuracy": 0.90, "recall_malignant": 0.96}, False),
        ({"accuracy": 0.97, "recall_malignant": 0.85}, False),
    ],
)
def test_passes_quality_gate(metrics, expected):
    assert model.passes_quality_gate(metrics) is expected


# ---------- reproducibility ----------

def test_pipeline_is_reproducible():
    _, first = model.run_pipeline(random_state=42)
    _, second = model.run_pipeline(random_state=42)
    assert first == second
