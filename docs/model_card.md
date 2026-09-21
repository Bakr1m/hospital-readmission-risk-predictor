# Model Card — Hospital Readmission Risk Predictor v1.0.0

## Model details
- **Architecture:** sklearn `Pipeline` (ColumnTransformer + LightGBM classifier).
- **Version / artifact:** `models/readmission_best.joblib`
  (LightGBM: `learning_rate=0.05`, `max_depth=7`, `n_estimators=100`,
  `num_leaves=31`, `class_weight='balanced'`).
- **Serving:** FastAPI `POST /predict` → risk probability, risk category,
  top-5 SHAP features. Docker: `bakr1m/readmission-api:v1`.

## Intended use
- **Decision support only** for flagging diabetic patients at elevated 30-day
  readmission risk at discharge, so care teams can target follow-up.
- **Out of scope:** autonomous triage, non-diabetic populations, real-time
  monitoring, any use without clinician review.

## Training data
- UCI "Diabetes 130-US hospitals" (101,766 encounters, 1999–2008),
  de-identified and public. Target: readmitted <30 days (11.2% positive).
- ICD-9 codes grouped to 17 clinical categories; engineered features:
  `prior_admissions`, `med_change_count`, `los_category`, `age_midpoint`.

## Evaluation
- Test ROC-AUC **0.689**, PR-AUC **0.238**, F1 **0.28**.
- Cost-optimal threshold **0.10** (recall 0.82, zero missed high-risk in
  evaluation at ~$4.45M total simulated cost).
- Retrospective only — no prospective or external validation.

## Limitations & ethical considerations
- Training data ends in 2008 (ICD-9 era); expect drift under ICD-10/modern practice.
- No fairness audit across race/gender/age has been performed; `race` and
  `gender` are model inputs — audit before any clinical use.
- SHAP values are log-odds attributions over one-hot features, not
  probability points; translate to plain language for clinicians.
- See `README.md` (Limitations, Ethical Considerations) for the full list.
