# Loan Default Prediction with Fairness Audit

A machine learning model that predicts loan default risk, paired with a fairness audit that checks whether the model's predictions are equally accurate across gender and age groups.

### 🔗 [Try the live app](https://loan-default-fairness.streamlit.app)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loan-default-fairness.streamlit.app)

## Why this project

Most credit-risk portfolio projects stop at "trained a model, here's the accuracy." This one goes further: it checks whether the model treats different groups of applicants fairly — a real, non-trivial ML concern that's often skipped even in production systems. The key finding (below) is that the model doesn't just reflect real-world risk differences between groups, it actively amplifies them, especially for younger applicants.

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle) — 307,511 real loan applications, 122 raw features, with an 8.07% default rate.

## Approach

1. **EDA** — quantified class imbalance, mapped missing data (67 of 122 columns had missing values, some up to 70%), and established baseline default-rate gaps across gender and age before any modeling.
2. **Cleaning** — dropped columns missing more than 50% of values, median-imputed the rest, one-hot encoded categoricals (83 → 185 columns).
3. **Modeling** — trained and honestly compared three models under class-weighted balancing to address the imbalance.
4. **Fairness audit** — used `fairlearn` to measure demographic parity and equalized odds across gender and age group, comparing the model's behavior against the raw-data baseline.
5. **Deployment** — packaged the final model into a Streamlit app with an embedded fairness disclosure.

## Key finding #1: accuracy is the wrong metric here

A first-pass Logistic Regression hit 91.9% accuracy — but 0% recall. It simply predicted "no default" for almost everyone, exploiting the fact that 92% of applicants don't default. This is a textbook and common trap in imbalanced classification. Applying `class_weight='balanced'` dropped accuracy to 68.9% but raised recall to 67.4% and ROC-AUC from 0.629 to 0.746 — a genuinely more useful model despite the "worse-looking" accuracy number.

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (balanced) | 68.9% | 16.1% | 67.4% | 25.9% | 0.746 |
| Random Forest (balanced) | 69.9% | 16.0% | 64.5% | 25.7% | 0.734 |
| **XGBoost (balanced)** | **71.1%** | **16.9%** | 66.1% | **26.9%** | **0.754** |

XGBoost was selected as the final model. Notably, Random Forest did not outperform the simpler Logistic Regression baseline — a reminder that model complexity should be validated empirically, not assumed.

## Key finding #2: the fairness audit

| Group | Raw data disparity | Model's flagging disparity | Demographic parity diff | Equalized odds diff |
|---|---|---|---|---|
| Gender | Men default 1.45x more than women | Model flags men 1.66x more often | 0.170 | 0.166 |
| Age | 18-25 default 3.3x more than 65+ | Model flags 18-25 group **11x** more often | 0.546 | 0.633 |

The model doesn't just reflect the real-world risk gap between groups — it amplifies it, especially by age. A model flagging 60% of 18-25 year-old applicants as high-risk (vs. 5.4% of 65+ applicants) raises real practical and ethical concerns: it risks systematically excluding young applicants from credit, making it harder for them to ever build the credit history needed to qualify later. This is also legally sensitive territory, since age discrimination in lending is regulated in many jurisdictions.

A mitigation attempt using `fairlearn`'s `ThresholdOptimizer` ran into a library compatibility issue that wasn't resolved within the project timeline. In a production setting, next steps would include per-group threshold adjustment, auditing which features act as proxies for age, and involving domain/legal review on acceptable fairness-accuracy tradeoffs.

## App

**Live link: [loan-default-fairness.streamlit.app](https://loan-default-fairness.streamlit.app)**

The Streamlit app takes a simplified set of applicant details (age, income, loan amount, external credit scores, employment length, gender, education) and returns a default probability, alongside a permanent fairness disclosure summarizing the findings above.

## Tech stack

Python, pandas, scikit-learn, XGBoost, fairlearn, Streamlit, joblib

## Limitations

- Hyperparameters are mostly defaults; not extensively tuned
- The deployed app uses a subset of the 181 features the underlying model was trained on, with the rest defaulted to typical values
- Fairness mitigation was attempted but not completed due to a library issue
- No SHAP/feature-importance explainability layer yet

## Project structure

```
loan-fairness-project/
├── app/
│   └── app.py              # Streamlit app
├── models/
│   ├── xgb_model.pkl        # trained XGBoost model
│   └── model_columns.pkl    # expected feature columns
├── notebooks/
│   └── 01_eda.ipynb         # EDA, cleaning, modeling, fairness audit
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone https://github.com/Prathmesh23-lang/loan-default-fairness-audit.git
cd loan-default-fairness-audit
pip install -r requirements.txt
streamlit run app/app.py

