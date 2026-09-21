# Notebooks

Exploratory trail for Project 1. The production path is `src/` + `api/`;
notebooks are documentation of how we got there.

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | First-pass profiling of the raw encounters data. |
| `02_cleaning.ipynb` | Missing-value handling, outlier checks, ICD-9 grouping. |
| `03_baseline.ipynb` | Logistic Regression baseline + sanity checks. |
| `04_tuning.ipynb` | XGBoost/LightGBM tuning and comparison. |
| `05_evaluation.ipynb` | Threshold analysis, cost curves, SHAP plots. |
| `06_tracking_tests.ipynb` | MLflow logging + pytest development. |
| `07_api_docker.ipynb` | API smoke tests and Docker build notes. |

Notebooks are not part of the deploy path — `Dockerfile` only ships
`src/`, `api/`, and `models/`.
