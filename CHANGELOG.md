# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-16

### Added
- LightGBM readmission classifier (test ROC-AUC 0.689, PR-AUC 0.238) with
  sklearn preprocessing pipeline, saved as `models/readmission_best.joblib`.
- FastAPI `/predict` endpoint returning risk score, risk category, and top-5
  SHAP features; Pydantic validation (422 on malformed input).
- SHAP explainability wired to the fitted preprocessor's one-hot feature names.
- MLflow experiment tracking (`hospital_readmission` experiment).
- pytest suite for the preprocessing pipeline (6 tests).
- Dockerized serving image (`bakr1m/readmission-api:v1`, ~1.23 GB), verified to
  return bit-identical predictions to local serving.
- Professional repo hygiene: LICENSE, CONTRIBUTING, CHANGELOG, CI workflow,
  Makefile, model card, example request payload.

### Fixed
- Missing `Pipeline` import and redundant CSV reads in `src/train.py`.
- `/predict` index error from hardcoded (49) vs actual (268) feature names.
- `PredictionResponse` schema rejecting string feature names.
- `api/main.py` import failure when run as a script / in Docker.
- Working-directory-dependent relative paths in `train.py` / `serve.py`.
