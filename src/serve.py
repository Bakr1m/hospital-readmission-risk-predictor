"""
FastAPI serving script for Hospital Readmission model
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

app = FastAPI(title="Hospital Readmission Risk API")

# Load pipeline at startup (path resolves regardless of working directory)
MODEL_PATH = PROJECT_ROOT / "models" / "readmission_best.joblib"
pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['classifier']

# Build SHAP explainer
explainer = shap.TreeExplainer(model)

# Feature columns (must match training)
FEATURE_COLS = [
    'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty',
    'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',
    'number_emergency', 'number_inpatient', 'number_diagnoses',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
    'change', 'diabetesMed', 'diag_1_cat', 'diag_2_cat', 'diag_3_cat',
    'prior_admissions', 'med_change_count', 'los_category',
    'num_diagnoses_cat', 'age_midpoint'
]

# Risk category thresholds
RISK_THRESHOLDS = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.5
}


class PatientInput(BaseModel):
    """Input schema for prediction"""
    race: str
    gender: str
    age: str
    weight: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    glyburide_metformin: str = Field(alias="glyburide-metformin")
    glipizide_metformin: str = Field(alias="glipizide-metformin")
    glimepiride_pioglitazone: str = Field(alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str = Field(alias="metformin-rosiglitazone")
    metformin_pioglitazone: str = Field(alias="metformin-pioglitazone")
    change: str
    diabetesMed: str
    diag_1_cat: str
    diag_2_cat: str
    diag_3_cat: str
    prior_admissions: int
    med_change_count: int
    los_category: str
    num_diagnoses_cat: str
    age_midpoint: float


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    readmission_risk: float
    risk_category: str
    top_shap_features: List[ShapFeature]


def get_risk_category(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "high"
    elif probability >= RISK_THRESHOLDS["medium"]:
        return "medium"
    elif probability >= RISK_THRESHOLDS["low"]:
        return "low"
    return "low"


def get_top_shap_features(shap_values, feature_names, top_k=5):
    """Extract top k features by absolute SHAP value"""
    import numpy as np
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    
    # Get mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
    
    return [
        {"feature": feature_names[i], "shap_value": float(mean_abs_shap[i])}
        for i in top_indices
    ]


@app.get("/")
def root():
    return {"message": "Hospital Readmission Risk API", "status": "ready"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        # Convert to DataFrame with correct column order
        input_dict = patient.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])
        input_df.columns = FEATURE_COLS
        
        # Predict
        proba = pipeline.predict_proba(input_df)[0, 1]
        risk_cat = get_risk_category(proba)
        
        # SHAP explanation (feature names come from the fitted preprocessor,
        # so they always match the one-hot expanded model input width)
        preprocessor_step = pipeline.named_steps['preprocessor']
        input_processed = preprocessor_step.transform(input_df)
        shap_values = explainer.shap_values(input_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_names = list(preprocessor_step.get_feature_names_out())

        top_features = get_top_shap_features(shap_values, feature_names)
        
        return PredictionResponse(
            readmission_risk=float(proba),
            risk_category=risk_cat,
            top_shap_features=top_features
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)