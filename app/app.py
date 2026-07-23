import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and expected columns
model = joblib.load('models/xgb_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')

st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")
st.title("Loan Default Risk Predictor")
st.markdown("Predicts loan default risk, with a transparency note on model fairness.")

st.header("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Annual Income (AMT_INCOME_TOTAL)", min_value=25000, max_value=2000000, value=150000, step=5000)
    credit_amt = st.number_input("Loan Amount Requested (AMT_CREDIT)", min_value=45000, max_value=4000000, value=500000, step=10000)
    ext_source_2 = st.slider("External Credit Score 2 (0=worst, 1=best)", 0.0, 1.0, 0.5)

with col2:
    ext_source_3 = st.slider("External Credit Score 3 (0=worst, 1=best)", 0.0, 1.0, 0.5)
    days_employed = st.slider("Years Employed", 0, 40, 5)
    gender = st.selectbox("Gender", ["F", "M"])
    education = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"])

if st.button("Predict Default Risk", type="primary"):
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data['DAYS_BIRTH'] = -age * 365
    input_data['AMT_INCOME_TOTAL'] = income
    input_data['AMT_CREDIT'] = credit_amt
    input_data['EXT_SOURCE_2'] = ext_source_2
    input_data['EXT_SOURCE_3'] = ext_source_3
    input_data['DAYS_EMPLOYED'] = -days_employed * 365

    gender_col = f'CODE_GENDER_{gender}' if gender != 'F' else None
    if gender_col and gender_col in input_data.columns:
        input_data[gender_col] = 1

    edu_col = f'NAME_EDUCATION_TYPE_{education}'
    if edu_col in input_data.columns:
        input_data[edu_col] = 1

    input_data = input_data.astype(float)

    probability = model.predict_proba(input_data)[0][1]
    prediction = "High Risk" if probability > 0.5 else "Low Risk"

    st.header("Prediction Result")
    st.metric("Default Probability", f"{probability*100:.1f}%")

    if prediction == "High Risk":
        st.error(f"Predicted: {prediction}")
    else:
        st.success(f"Predicted: {prediction}")

    st.progress(min(float(probability), 1.0))

st.divider()
st.header("Model Fairness Notice")
st.markdown("""
This model was audited for fairness across gender and age groups. Key findings:

- **Gender**: The model flags male applicants as high-risk at ~1.66x the rate of 
  female applicants, compared to a real-world default rate gap of ~1.45x.
- **Age**: The model flags 18-25 year olds as high-risk at ~11x the rate of 65+ year 
  olds, compared to a real-world default rate gap of ~3.3x - a substantial 
  amplification that raises fairness concerns for younger applicants.

These findings suggest the model should not be used as a sole decision-maker without 
further bias mitigation and human review, particularly for younger applicants.
""")