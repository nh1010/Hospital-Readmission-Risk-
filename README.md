## Hospital Readmission Risk

Predicting hospital readmission (binary `readmitted` target) using clinical encounter features and treatment indicators. This repo contains a single notebook that loads `train.csv`, performs preprocessing and feature engineering, trains multiple models, evaluates performance, and produces optimization artifacts.

### Repository structure
- `DDDM_code.ipynb`: End‑to‑end analysis, modeling, and visualization.
- `train.csv`: Training dataset with continuous features (e.g., `time_in_hospital`, `num_procedures`) and many one‑hot encoded categorical fields (e.g., `age_[40-50)`, `race_Caucasian`, medication indicators, specialties, diagnoses).
- `optimal_treatment_plans.csv` (created by the notebook): Group‑level treatment recommendations exported during execution.

### Quick start
1) Create and activate a virtual environment (Windows PowerShell):
```
python -m venv .venv
./.venv/Scripts/Activate
```

2) Install dependencies:
```
pip install pandas numpy scikit-learn xgboost seaborn matplotlib statsmodels jupyter ipykernel
```

3) Launch Jupyter and open the notebook:
```
jupyter notebook
```
Open `DDDM_code.ipynb`, set the kernel to the `.venv` environment if prompted, and run cells top‑to‑bottom.

Optional headless run (saves outputs in-place):
```
pip install papermill
papermill DDDM_code.ipynb DDDM_code.out.ipynb
```

### Data
- Input: `train.csv` with columns including (non‑exhaustive):
  - Continuous: `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_inpatient`, `number_emergency`, `number_outpatient`, `number_diagnoses`.
  - Pre‑encoded categorical indicators: age buckets like `age_[40-50)`, race like `race_Caucasian`, payers, medical specialties, diagnosis codes, medication flags (e.g., `metformin_No`, `insulin_No`).
- Target: `readmitted` (0/1).

### Methodology (as implemented in the notebook)
- Preprocessing
  - Missing values: medians for continuous, sentinel category for categorical.
  - Scaling: `StandardScaler` on continuous fields.
  - Encoding: one‑hot for select categoricals; some columns are already one‑hot in the CSV.
  - Feature engineering:
    - `simplified_age` grouping: `40-60`, `60-90`, other.
    - `race_minority` derived from `race_Caucasian` when present.
    - Interaction term: `interaction_inpatient_diagnoses = number_inpatient * number_diagnoses`.
    - Binned `num_medications`.
- Modeling
  - Logistic Regression (with and without class weights) for baseline and interpretability.
  - StatsModels Logit for coefficient significance.
  - XGBoost (`XGBClassifier`) within a `Pipeline` with `ColumnTransformer` preprocessing.
- Evaluation
  - Confusion matrix, Accuracy, Precision, Recall, F1, ROC‑AUC.
  - Train/test split: 80/20 with `random_state=42` (stratified where used).

### Outputs
- Visualizations in‑notebook (matplotlib/seaborn).
- `optimal_treatment_plans.csv`: exported summary of lower‑readmission combinations by demographic group.

### Reproducibility notes
- Determinism: fixed random states where applicable.
- Environment: Python 3.10+ recommended; versions of `xgboost` and `scikit‑learn` should be compatible (latest stable works in most cases).

### Troubleshooting
- If `xgboost` installation fails on Windows, upgrade `pip` and try again:
```
python -m pip install --upgrade pip
pip install xgboost
```
- If the notebook references packages not installed, install them with `pip install <package>` and re‑run the affected cells.

