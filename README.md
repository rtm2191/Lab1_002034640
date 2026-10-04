# Lab 1 – Breast Cancer Model Quality Gate (MLOps IE-7374)

This lab builds on the original GitHub Actions calculator lab. Instead of basic arithmetic, `src/` contains a small machine learning pipeline, and GitHub Actions blocks any change that makes the model worse.

## What it does

The model is a logistic regression classifier trained on the Breast Cancer Wisconsin dataset (569 samples, 30 tumor measurements, labeled malignant or benign). Missing a malignant tumor is the costly mistake, so the quality gate checks **recall on the malignant class** in addition to overall accuracy.

| Gate | Threshold |
|---|---|
| Accuracy | ≥ 0.93 |
| Malignant recall | ≥ 0.93 |

## Project structure

```
data/breast_cancer.csv          dataset
src/model.py                    pipeline: load_data, train_model, evaluate, run_pipeline
test/test_pytest.py             pytest tests (fixtures, parametrize)
test/test_unittest.py           unittest tests
.github/workflows/
    pytest_action.yml           lint → tests on 3 Python versions → train + quality gate
    unittest_action.yml         unittest suite on 3 Python versions
requirements.txt
pytest.ini
```

## Running locally

```bash
python -m venv lab_01
lab_01\Scripts\activate          # Windows
source lab_01/bin/activate       # macOS/Linux
pip install -r requirements.txt

pytest -v                                                   # pytest suite
python -m unittest discover -s test -t . -p "test_unittest.py" -v   # unittest suite
python -m src.model                                         # train, save model, check gate
```

## Changes from the original lab

- Added tests for data quality, model behavior, quality thresholds, and reproducibility.
- Upgraded GitHub Actions versions (`checkout@v4`, `setup-python@v5`, `upload-artifact@v4`); the original `upload-artifact@v2` is retired.
- Workflows now run on pull requests and manually, not just on pushes to main.
- Tests run on Python 3.10, 3.11, and 3.12 using a matrix.
- Added ruff linting, pip caching, and a minimum test coverage of 80%.
- Added a `train` job that only runs after tests pass, saves the model and `metrics.json` as artifacts, and fails if the model is below the quality gate.


MIN_MALIGNANT_RECALL from 0.93 to 0.99 to test quality gate
<img width="1808" height="784" alt="image" src="https://github.com/user-attachments/assets/637860d6-4c0f-4054-80b4-fc8d2f1970b7" />
