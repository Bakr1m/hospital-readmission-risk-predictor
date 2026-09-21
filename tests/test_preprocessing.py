"""
Pytest tests for preprocessing pipeline - Project 1 Hospital Readmission.

Tests import the real functions from src.preprocessing (no logic duplication):
- engineer_features / get_feature_columns / build_preprocessor
"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor, engineer_features, get_feature_columns


@pytest.fixture(scope="module")
def df():
    """Load cleaned data and apply feature engineering."""
    df = pd.read_csv("data/diabetes_clean.csv")
    df = engineer_features(df)
    return df


@pytest.fixture(scope="module")
def feature_cols(df):
    """Feature columns (excludes IDs and targets)."""
    return get_feature_columns(df)


@pytest.fixture(scope="module")
def fitted(df, feature_cols):
    """Build the preprocessor on feature columns and fit it. Returns (preprocessor, X)."""
    X = df[feature_cols]
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)
    preprocessor.fit(X)
    return preprocessor, X


def test_engineer_features_invariants(df):
    """Engineered features stay within their expected domains."""
    assert (df["prior_admissions"] >= 0).all(), "prior_admissions has negative values"
    assert (df["med_change_count"] >= 0).all(), "med_change_count has negative values"
    assert (df["med_change_count"] <= 23).all(), "med_change_count exceeds max possible"

    valid_los = ["Short (1-3d)", "Medium (4-7d)", "Long (8-14d)", "Very Long (14+d)"]
    assert df["los_category"].isin(valid_los).all(), "Invalid los_category values"
    assert df["age_midpoint"].dropna().between(5, 95).all(), "age_midpoint out of range"


def test_get_feature_columns_excludes_ids_and_targets(df, feature_cols):
    """get_feature_columns returns a non-empty list without IDs/targets/raw diag codes."""
    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 0
    for excluded in [
        "encounter_id",
        "patient_nbr",
        "diag_1",
        "diag_2",
        "diag_3",
        "readmitted",
        "readmitted_binary",
    ]:
        assert excluded not in feature_cols, f"{excluded} should be excluded"
    # Engineered features must be present
    for expected in ["prior_admissions", "med_change_count", "los_category", "age_midpoint"]:
        assert expected in feature_cols, f"{expected} missing from feature columns"


def test_preprocessor_output_has_no_missing(fitted):
    """Fitted preprocessor output contains no NaN (imputation works)."""
    preprocessor, X = fitted
    X_processed = preprocessor.transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    assert X_processed.shape[0] == X.shape[0], "Row count changed during transform"
    assert X_processed.shape[1] > 0, "Preprocessor produced 0 features"
    assert not np.isnan(X_processed).any(), "Preprocessor output contains NaN values"


def test_known_input_known_output(fitted):
    """A single known row transforms to the expected (1, n_features) shape without NaN."""
    preprocessor, X = fitted
    n_features = preprocessor.transform(X.iloc[:5]).shape[1]
    single = preprocessor.transform(X.iloc[:1])
    if hasattr(single, "toarray"):
        single = single.toarray()
    assert single.shape == (1, n_features), f"Expected (1, {n_features}), got {single.shape}"
    assert not np.isnan(single).any(), "Known input produced NaN output"


def test_pipeline_serialization_roundtrip(fitted):
    """Preprocessor survives a save/load round-trip with identical outputs."""
    preprocessor, X = fitted
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        temp_path = f.name
    try:
        joblib.dump(preprocessor, temp_path)
        loaded = joblib.load(temp_path)
        sample = X.iloc[:5]
        original = preprocessor.transform(sample)
        reloaded = loaded.transform(sample)
        if hasattr(original, "toarray"):
            original = original.toarray()
        if hasattr(reloaded, "toarray"):
            reloaded = reloaded.toarray()
        np.testing.assert_array_almost_equal(original, reloaded, decimal=5)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_class_balance_preserved(df, feature_cols):
    """Stratified split preserves the 11.2% positive-class balance."""
    X = df[feature_cols]
    y = df["readmitted_binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    overall = y.mean()
    assert abs(y_train.mean() - overall) < 0.02, "Train split class balance drifted"
    assert abs(y_test.mean() - overall) < 0.02, "Test split class balance drifted"
