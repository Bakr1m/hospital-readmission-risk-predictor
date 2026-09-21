# Project 1: Hospital Readmission Risk Predictor

**Days 11–20 | Healthcare ML Portfolio**

## Business Context

US hospitals face financial penalties under the Hospital Readmissions Reduction Program (HRRP) for excess 30-day readmissions. This project predicts which diabetic patients are at high risk of readmission within 30 days of discharge, enabling targeted interventions (medication reconciliation, follow-up scheduling, home health referrals). The model flags high-risk patients at discharge so care teams can intervene early — reducing readmission costs and improving continuity of care.

## Dataset

- **Source**: UCI "Diabetes 130-US Hospitals" (101,766 encounters, 1999–2008)
- **Target**: `readmitted_binary` — readmitted within 30 days (11.2% positive)
- **Initial Features**: 50 raw features after cleaning & engineering
- **Key Challenge**: High-cardinality ICD-9 diagnosis codes (700+), severe class imbalance (11.2%), missing lab values (83–95%)
- **Preprocessing**: ICD-9 codes grouped into 17 clinical categories; engineered features include `prior_admissions`, `med_change_count`, `age_midpoint`, `los_category`

## Approach

1. **Baseline**: Logistic Regression with `class_weight='balanced'` — establishes performance floor and sanity-checks the pipeline before adding model complexity.
2. **Model**: LightGBM tuned via hyperparameter search (`learning_rate=0.05`, `max_depth=7`, `n_estimators=100`, `num_leaves=31`) with `class_weight='balanced'`.
3. **Explainability**: SHAP summary and waterfall plots for individual predictions — clinicians need to see *why* a patient is flagged.
4. **Threshold Optimization**: Cost-sensitive threshold selection (0.10 minimizes total cost with zero missed high-risk patients) instead of default 0.5.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | PR-AUC | F1 | Recall @ 0.10 |
|-------|------------|--------------|--------|-----|---------------|
| Logistic Regression (baseline) | 0.640 | 0.652 | 0.205 | 0.26 | 0.78 |
| XGBoost (tuned) | 0.674 | 0.689 | 0.238 | 0.28 | 0.62 |
| **LightGBM (tuned, best)** | **0.674** | **0.689** | **0.238** | **0.28** | **0.82** |

**Best Model**: LightGBM (tuned: `learning_rate=0.05`, `max_depth=7`, `n_estimators=100`, `num_leaves=31`, `class_weight='balanced'`)

**Clinical Threshold**: 0.10 — minimizes total cost ($4.45M) with zero missed high-risk patients; recall = 82%

**Cost Analysis**:
| Threshold | Total Cost | FN (missed) | FP (unnecessary) |
|-----------|------------|-------------|------------------|
| 0.10 | **$4.45M** | 0 | 17,982 |
| 0.30 | $4.73M | 82 | 15,747 |
| 0.55 | $17.7M | 1,194 | 3,910 |

## Limitations

1. **Dataset Shift Risk**: Training data spans 1999–2008; clinical practice, coding standards (ICD-9 → ICD-10), and patient demographics have evolved — model may degrade in current hospitals.
2. **Demographic Bias**: No systematic fairness audit across race/gender/age groups was performed; `race` and `gender` are features — risk of disparate impact.
3. **No Prospective Validation**: All results are retrospective; clinical utility requires prospective pilot study with real clinicians.
4. **Single-Dataset Evaluation**: No external validation on MIMIC or other hospital systems — generalizability unproven.
5. **Feature Availability at Discharge**: Some engineered features (e.g., `med_change_count`) require complete medication history — may not be available at decision time in all EHRs.
6. **Static Threshold Assumptions**: Cost-optimal threshold (0.10) assumes fixed cost ratios; real-world costs vary by hospital, payer, and patient acuity.
7. **No Real-Time Pipeline**: Current serving script loads model at startup; production deployment would require a persistent model-serving architecture with hot-reloading and monitoring.

## Ethical Considerations

- **Fairness & Non-Discrimination**: Model must be audited for equalized odds across protected attributes before any clinical deployment. Current implementation includes `race` and `gender` as features — rationale documented, but disparate impact must be monitored and mitigated.
- **Informed Consent & Transparency**: Patients should be informed when algorithmic risk scores influence care decisions. SHAP explanations are a step toward transparency but not sufficient for informed consent on their own.
- **Human-in-the-Loop Mandate**: This model is a **decision support tool only** — never an autonomous triage system. Final discharge/follow-up decisions must remain with qualified clinicians.
- **HIPAA & Data Governance**: Training data is de-identified (UCI public), but any production deployment with real PHI requires BAAs, encryption at rest/in transit, audit logging, and strict access controls.
- **Clinical Validation Gap**: This is a research prototype. FDA considers such tools "Clinical Decision Support" — if it drives clinical action without clinician review, it may require 510(k) clearance.
- **Monitoring & Accountability**: Deployed system must log every prediction, track drift (feature distribution + prediction distribution shifts), and have a defined rollback procedure with clinical oversight.
- **Cost Asymmetry Transparency**: The chosen threshold (0.10) reflects assumed cost ratios ($15K readmission vs. $500 follow-up). These assumptions should be reviewed and validated with clinical and financial stakeholders before deployment.

## Project Structure

```
project1_readmission/
├ data/              # diabetes_raw.csv, diabetes_clean.csv (gitignored — download from UCI, see Dataset)
├ notebooks/         # 01_eda ... 07_api_docker
├ src/
│   ├── preprocessing.py      # Feature engineering + ColumnTransformer
│   ├── train.py              # Training orchestration + MLflow tracking
│   └── serve.py              # FastAPI app: /predict endpoint + SHAP
├ models/              # readmission_best.joblib (LightGBM pipeline)
├ api/
│   └── main.py               # Thin entrypoint (imports app from src.serve)
├ tests/               # pytest tests
├ mlruns/              # MLflow tracking (gitignored)
├ Dockerfile
├ requirements.txt         # serving deps (also what Docker installs)
├ requirements-train.txt   # + training deps (xgboost, mlflow, local retraining only)
├ requirements-dev.txt     # + dev tools (pytest, ruff)
├ Makefile                 # shortcuts: install / test / lint / train / serve / docker-*
├ pyproject.toml           # package metadata, pytest + ruff config
├ example_request.json     # tested /predict payload
├ docs/model_card.md       # intended use, metrics, limits, ethics
├ .github/workflows/ci.yml # lint + tests on push/PR
├ .gitignore
├ .dockerignore
├ LICENSE / CHANGELOG.md / CONTRIBUTING.md
└ README.md
```

## Quick Start

```bash
# Setup with virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt              # serving deps (also what Docker installs)
pip install -r requirements-train.txt        # + training deps (xgboost, mlflow) for retraining

# Train models (logs to MLflow; runs from the project root)
python src/train.py

# Run API locally
python api/main.py
# Test: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"race": "Caucasian", "gender": "Female", "age": "[70-80)", ...}'

# MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

## Run with Docker (no local setup needed)

```bash
docker pull bakr1m/readmission-api:v1
docker run -p 8000:8000 bakr1m/readmission-api:v1
# Test: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_patient.json
```

## API Response Format

```json
{
  "readmission_risk": 0.136,
  "risk_category": "low",
  "top_shap_features": [
    {"feature": "cat__medical_specialty_Pediatrics-Endocrinology", "shap_value": 0.449},
    {"feature": "num__age_midpoint", "shap_value": 0.294},
    {"feature": "num__number_inpatient", "shap_value": 0.272}
  ]
}
```

Feature names use the fitted preprocessor's one-hot naming (`num__` / `cat__` prefixes), so explanations always match the exact model input columns.

## SHAP Explainability

Top features driving readmission risk:

1. **Prior admissions** (comorbidity burden)
2. **Primary diagnosis** (Circulatory, Endocrine, Respiratory)
3. **Medication changes** (treatment instability)
4. **Weight category** (underweight/malnutrition signal)
5. **Number of diagnoses** (complexity)

## Key Learnings

1. **ICD-9 grouping essential** — 700+ codes → 17 clinical categories prevents overfitting
2. **Imbalance handling** — `class_weight='balanced'` outperforms SMOTE for tree models
3. **Clinical thresholds ≠ F1-optimal** — Cost asymmetry favors high recall
4. **SHAP critical for clinical trust** — Clinicians need to see *why* a patient is flagged
5. **Threshold matters more than model choice** — A well-chosen LR threshold can beat a default XGBoost 0.5

## Next Steps

- Audit predictions across race/gender/age groups for fairness
- Implement drift monitoring on deployed predictions
- Conduct prospective validation study with clinical stakeholders
- Explore cost-sensitive learning during training (not just post-hoc thresholding)