"""
Preprocessing pipeline for Hospital Readmission project
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def build_preprocessor(X_train):
    """Build and return fitted ColumnTransformer"""
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # Everything non-numeric is treated as categorical (version-proof across
    # pandas 2.x object-dtype and pandas 3.x str-dtype strings).
    categorical_features = [c for c in X_train.columns if c not in numeric_features]
    
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
    
    return preprocessor, numeric_features, categorical_features


def engineer_features(df):
    """Apply feature engineering to raw dataframe"""
    df = df.copy()
    
    # Prior admissions
    df['prior_admissions'] = df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']
    
    # Medication changes
    drug_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
                 'glipizide-metformin', 'glimepiride-pioglitazone',
                 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    df['med_change_count'] = 0
    for col in drug_cols:
        if col in df.columns:
            df['med_change_count'] += (df[col].isin(['Up', 'Down'])).astype(int)
    
    # Length of stay category
    df['los_category'] = pd.cut(df['time_in_hospital'],
                                 bins=[0, 3, 7, 14, 100],
                                 labels=['Short (1-3d)', 'Medium (4-7d)', 'Long (8-14d)', 'Very Long (14+d)'])
    
    # Age midpoint
    age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
               '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
               '[80-90)': 85, '[90-100)': 95}
    df['age_midpoint'] = df['age'].map(age_map)
    
    # Clean '?' values (non-numeric columns only; works on pandas 2.x and 3.x)
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if '?' in df[col].values:
            df[col] = df[col].replace('?', 'Unknown')
    df['gender'] = df['gender'].replace('Unknown/Invalid', 'Unknown')
    
    return df


def get_feature_columns(df):
    """Get feature columns excluding IDs and target"""
    exclude_cols = ['encounter_id', 'patient_nbr', 'diag_1', 'diag_2', 'diag_3',
                    'readmitted', 'readmitted_binary']
    return [c for c in df.columns if c not in exclude_cols]