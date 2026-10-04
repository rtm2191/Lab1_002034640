import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import model


class TestBreastCancerModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data and run the pipeline once for all tests
        cls.X, cls.y = model.load_data()
        cls.trained_model, cls.metrics = model.run_pipeline()

    # ---------- load_data ----------

    def test_load_data_shape(self):
        self.assertEqual(self.X.shape, (569, 30))
        self.assertEqual(len(self.y), 569)

    def test_load_data_no_missing_values(self):
        self.assertFalse(self.X.isnull().values.any())
        self.assertFalse(self.y.isnull().any())

    def test_load_data_labels_are_binary(self):
        self.assertEqual(set(self.y.unique()), {0, 1})

    def test_load_data_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            model.load_data("data/does_not_exist.csv")

    def test_load_data_missing_target_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_file = os.path.join(tmp_dir, "bad.csv")
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad_file, index=False)
            with self.assertRaises(ValueError):
                model.load_data(bad_file)

    # ---------- train_model ----------

    def test_train_model_predicts_valid_labels(self):
        predictions = self.trained_model.predict(self.X)
        self.assertTrue(set(np.unique(predictions)).issubset({0, 1}))
        self.assertEqual(len(predictions), len(self.y))

    def test_train_model_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            model.train_model(self.X, self.y[:-10])

    # ---------- evaluate ----------

    def test_evaluate_returns_expected_keys(self):
        self.assertEqual(
            set(self.metrics),
            {"accuracy", "precision_malignant", "recall_malignant"},
        )

    def test_metrics_are_between_0_and_1(self):
        for name, value in self.metrics.items():
            with self.subTest(metric=name):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    # ---------- quality gate ----------

    def test_accuracy_meets_threshold(self):
        self.assertGreaterEqual(self.metrics["accuracy"], model.MIN_ACCURACY)

    def test_malignant_recall_meets_threshold(self):
        self.assertGreaterEqual(
            self.metrics["recall_malignant"], model.MIN_MALIGNANT_RECALL
        )

    def test_passes_quality_gate(self):
        self.assertTrue(model.passes_quality_gate({"accuracy": 0.97, "recall_malignant": 0.96}))
        self.assertFalse(model.passes_quality_gate({"accuracy": 0.90, "recall_malignant": 0.96}))
        self.assertFalse(model.passes_quality_gate({"accuracy": 0.97, "recall_malignant": 0.85}))

    # ---------- reproducibility ----------

    def test_pipeline_is_reproducible(self):
        _, first = model.run_pipeline(random_state=42)
        _, second = model.run_pipeline(random_state=42)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
