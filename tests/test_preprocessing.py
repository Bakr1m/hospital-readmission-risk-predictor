"""
Pytest tests for preprocessing pipeline - Project 1 Hospital Readmission
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# Load data and build preprocessor (shared fixture)
@pytest.fixture(scope="module")
def data_and_preprocessor():
    df = pd.read_csv('data/diabetes_clean.csv')
    
    exclude_cols = ['encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3', 
                    'readmitted', 'readmitted_binary']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df['readmitted_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }


def test_no_missing_after_transform(data_and_preprocessor):
    """Test that preprocessor output has no missing values"""
    X_test_proc = data_and_preprocessor['X_test_processed']
    if hasattr(X_test_proc, 'toarray'):
        X_test_proc = X_test_proc.toarray()
    assert not np.isnan(X_test_proc).any(), "Preprocessor output contains NaN values"


def test_known_input_known_output(data_and_preprocessor):
    """Test specific known input produces expected output shape and no errors"""
    preprocessor = data_and_preprocessor['preprocessor']
    X_train = data_and_preprocessor['X_train']
    X_train_processed = data_and_preprocessor['X_train_processed']
    
    known_input = data_and_preprocessor['X_train'].iloc[:1]
    transformed = preprocessor.transform(known_input)
    
    expected_n_features = data_and_preprocessor['X_train_processed'].shape[1]
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    assert transformed.shape == (1, expected_n_features), \
        f"Expected shape (1, {expected_n_features}), got {transformed.shape}"
    
    assert not np.isnan(transformed).any(), "Known input produced NaN output"


def test_feature_engineering_invariants(data_and_preprocessor):
    """Test that engineered features maintain expected invariants"""
    X_train = data_and_preprocessor['X_train']
    
    assert (data_and_preprocessor['X_train']['prior_admissions'] >= 0).all(), "prior_admissions has negative values"
    assert (data_and_preprocessor['X_train']['med_change_count'] >= 0).all(), "med_change_count has negative values"
    assert (data_and_preprocessor['X_train']['med_change_count'] <= 23).all(), "med_change_count exceeds max possible"
    
    valid_los = ['Short (1-3d)', 'Medium (4-7d)', 'Long (8-14d)', 'Very Long (14+d)']
    assert data_and_preprocessor['X_train']['los_category'].isin(valid_los).all(), "Invalid los_category values"
    assert data_and_preprocessor['X_train']['age_midpoint'].between(5, 95).all(), "age_midpoint out of range"


def test_pipeline_serialization(data_and_preprocessor):
    """Test that preprocessor can be saved and loaded correctly"""
    preprocessor = data_and_preprocessor['preprocessor']
    X_train = data_and_preprocessor['X_train']
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name
    
    try:
        joblib.dump(preprocessor, temp_path)
        loaded_preprocessor = joblib.load(temp_path)
        
        test_input = data_and_preprocessor['X_train'].iloc[:5]
        original_output = preprocessor.transform(test_input)
        loaded_output = loaded_preprocessor.transform(test_input)
        
        if hasattr(original_output, 'toarray'):
            original_output = original_output.toarray()
        if hasattr(loaded_output, 'toarray'):
            loaded_output = loaded_output.toarray()
        
        np.testing.assert_array_almost_equal(original_output, loaded_output, decimal=5)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_class_balance_preserved(data_and_preprocessor):
    """Test that stratified split preserves class balance"""
    y_train = data_and_preprocessor['y_train']
    y_test = data_and_preprocessor['y_test']
    
    train_prop = data_and_preprocessor['y_train'].mean()
    test_prop = data_and_preprocessor['y_test'].mean()
    overall_prop = pd.concat([y_train, y_test]).mean()
    
    assert abs(train_prop - overall_prop) < 0.02, f"Train prop {train_prop:.4f} vs overall {overall_prop:.4f}"
    assert abs(test_prop - overall_prop) < 0.02, f"Test prop {test_prop:.4f} vs overall {overall_prop:.4f}"