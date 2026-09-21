"""
Training script for Hospital Readmission project.

Run from anywhere:  python src/train.py   (paths resolve relative to this file)
"""

import sys
from pathlib import Path

import os

# MLflow 3.x puts the local file-store backend in maintenance mode by default;
# opt in to keep using the project's existing mlruns/ directory.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import engineer_features, get_feature_columns, build_preprocessor  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "diabetes_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "readmission_best.joblib"
# Plain directory path (not a file: URI) — avoids URI-parsing issues in MLflow 3.x.
MLRUNS_URI = str(PROJECT_ROOT / "mlruns")


def train_models():
    """Train all models and log to MLflow"""
    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["readmitted_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build preprocessor (fit only on training data to avoid leakage)
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # MLflow setup
    mlflow.set_tracking_uri(MLRUNS_URI)
    mlflow.set_experiment("hospital_readmission")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 1. Logistic Regression baseline
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)

        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(X_train_processed, y_train)

        y_proba = lr.predict_proba(X_test_processed)[:, 1]
        y_pred = lr.predict(X_test_processed)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(lr, "model")

        print(f"LR: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

    # 2. XGBoost tuned
    with mlflow.start_run(run_name="xgboost_tuned"):
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }

        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_type", "XGBoost")

        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(X_train_processed, y_train)

        y_proba = xgb_clf.predict_proba(X_test_processed)[:, 1]
        y_pred = xgb_clf.predict(X_test_processed)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.xgboost.log_model(xgb_clf, "model")

        print(f"XGBoost: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

    # 3. LightGBM tuned (best)
    with mlflow.start_run(run_name="lightgbm_tuned_best"):
        params = {
            "n_estimators": 100,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }

        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_type", "LightGBM")

        lgb_clf = lgb.LGBMClassifier(**params)
        lgb_clf.fit(X_train_processed, y_train)

        y_proba = lgb_clf.predict_proba(X_test_processed)[:, 1]
        y_pred = lgb_clf.predict(X_test_processed)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.lightgbm.log_model(lgb_clf, "model")

        print(f"LightGBM: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

    # Save best pipeline (preprocessor + LightGBM) using already-fitted artifacts
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", lgb_clf)])

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Best pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLRUNS_URI)
    train_models()
