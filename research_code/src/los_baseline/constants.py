from __future__ import annotations

DEFAULT_FEATURE_NAMES = [
    "gender",
    "age",
    "sofa_score",
    "charlson_comorbidity_index",
    "aps3",
    "heart_rate_mean",
    "sbp_mean",
    "dbp_mean",
    "resp_rate_mean",
    "temperature_mean",
    "hemoglobin_mean",
    "platelets_mean",
    "wbc_mean",
    "bun_mean",
    "creatinine_mean",
    "glucose_mean",
    "chloride_mean",
    "sodium_mean",
    "potassium_mean",
]

OUTCOME_CLASS_LABELS = {
    0: "A_discharged_alive",
    1: "B_died",
    2: "C_still_admitted",
}

OUTCOME_CLASS_CODES = {
    label: code for code, label in OUTCOME_CLASS_LABELS.items()
}

MODEL_ORDER = [
    "decision_tree",
    "logistic_regression",
    "random_forest",
    "svm",
    "xgboost",
]

MODEL_DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "SVM RBF",
    "xgboost": "XGBoost",
}

COLUMN_DISPLAY_NAMES = {
    "gender": "Gender",
    "age": "Age (years)",
    "sofa_score": "SOFA score",
    "charlson_comorbidity_index": "Charlson Comorbidity Index",
    "aps3": "APS III",
    "heart_rate_mean": "Heart Rate (bpm)",
    "sbp_mean": "Systolic Blood Pressure (mmHg)",
    "dbp_mean": "Diastolic Blood Pressure (mmHg)",
    "resp_rate_mean": "Respiratory Rate (insp/min)",
    "temperature_mean": "Body Temperature (°C)",
    "hemoglobin_mean": "Hemoglobin (g/dL)",
    "platelets_mean": "Platelets (K/uL)",
    "wbc_mean": "White Blood Cell Count (K/uL)",
    "bun_mean": "BUN (mg/dL)",
    "creatinine_mean": "Creatinine (mg/dL)",
    "glucose_mean": "Glucose (mg/dL)",
    "chloride_mean": "Chloride (mEq/L)",
    "sodium_mean": "Sodium (mEq/L)",
    "potassium_mean": "Potassium (mEq/L)",
}
