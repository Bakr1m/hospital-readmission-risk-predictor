"""
Pytest tests for preprocessing pipeline - Project 1 Hospital Readmission.

Tests import the real functions from src.preprocessing (no logic duplication):
- engineer_features / get_feature_columns / build_preprocessor

The suite is hermetic: it uses the real `data/diabetes_clean.csv` when present
(local dev) and falls back to a small synthetic frame with the same schema in
CI, where `data/` is gitignored. Both paths exercise identical assertions.
"""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocessor, engineer_features, get_feature_columns

DATA_PATH = Path("data/diabetes_clean.csv")

# Column scaffolding for the synthetic fallback (same names the real CSV has;
# values are plausible but random — only the schema matters for these tests).
_DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
_AGE_BINS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)",
             "[60-70)", "[70-80)", "[80-90)", "[90-100)"]


def make_synthetic_df(n=200, seed=42):
    """Build a small frame with the production schema for CI (no data/ needed)."""
    rng = np.random.RandomState(seed)
    pick = lambda pool, size=n: rng.choice(pool, size=size).tolist()  # noqa: E731
    rint = lambda lo, hi, size=n: rng.randint(lo, hi + 1, size=size)  # noqa: E731

    data = {
        "encounter_id": range(1, n + 1),
        "patient_nbr": rint(1, n // 2),
        "race": pick(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]),
        "gender": pick(["Male", "Female"]),
        "age": pick(_AGE_BINS),
        "weight": pick(["?", "Unknown", "[50-75)", "[75-100)"]),
        "admission_type_id": rint(1, 8),
        "discharge_disposition_id": rint(1, 28),
        "admission_source_id": rint(1, 25),
        "time_in_hospital": rint(1, 14),
        "payer_code": pick(["MC", "MD", "SP", "Unknown"]),
        "medical_specialty": pick(["InternalMedicine", "Emergency", "Cardiology", "?"]),
        "num_lab_procedures": rint(1, 100),
        "num_procedures": rint(0, 6),
        "num_medications": rint(1, 40),
        "number_outpatient": rint(0, 5),
        "number_emergency": rint(0, 5),
        "number_inpatient": rint(0, 5),
        "number_diagnoses": rint(1, 9),
        "diag_1": pick(["250.8", "401", "414", "428"]),
        "diag_2": pick(["250.8", "401", "414", "428"]),
        "diag_3": pick(["250.8", "401", "414", "428"]),
        "diag_1_cat": pick(["Circulatory", "Respiratory", "Diabetes", "Other"]),
        "diag_2_cat": pick(["Circulatory", "Respiratory", "Diabetes", "Other"]),
        "diag_3_cat": pick(["Circulatory", "Respiratory", "Diabetes", "Other"]),
        "change": pick(["No", "Ch"]),
        "diabetesMed": pick(["Yes", "No"]),
        "num_diagnoses_cat": pick(["1", "2-4", "5+"]),
        # Placeholders recomputed by engineer_features; values here are ignored.
        "prior_admissions": 0,
        "med_change_count": 0,
        "los_category": "Short (1-3d)",
        "age_midpoint": 50,
    }
    for col in _DRUG_COLS:
        data[col] = pick(["No", "Steady", "Up", "Down"])
    df = pd.DataFrame(data)
    df["readmitted"] = pick(["NO", ">30", "<30"])
    df["readmitted_binary"] = (rng.rand(n) < 0.11).astype(int)  # ~11% positive
    return df


@pytest.fixture(scope="module")
def df():
    """Real CSV when present, else synthetic frame (same schema, CI-safe)."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = make_synthetic_df()
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
