import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

os.chdir("C:/Users/vikas/OneDrive/Desktop/vikasji")

model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to predict churn probability.")

input_data = {}

numeric_cols = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
for col in numeric_cols:
    if col in columns:
        input_data[col] = st.number_input(col, min_value=0, value=0)

if "Gender" in columns:
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    input_data["Gender"] = 0 if gender == "Male" else 1

sub_cols = [c for c in columns if c.startswith("Subscription Type_")]
if sub_cols:
    options = [c.split("_")[-1] for c in sub_cols]
    selected_sub = st.selectbox("Subscription Type", options, key="subscription_type")
    for c in sub_cols:
        input_data[c] = 1 if c.endswith(selected_sub) else 0

contract_cols = [c for c in columns if c.startswith("Contract Length_")]
if contract_cols:
    contract_options = [c.split("_")[-1] for c in contract_cols]
    selected_contract = st.selectbox("Contract Length", contract_options, key="contract_length")
    for c in contract_cols:
        input_data[c] = 1 if c.endswith(selected_contract) else 0

input_df = pd.DataFrame([input_data])

for c in columns:
    if c not in input_df.columns:
        input_df[c] = 0

input_df = input_df[columns]

numeric_input_cols = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend']
input_df[numeric_input_cols] = scaler.transform(input_df[numeric_input_cols])

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"🚨 Customer is likely to churn. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will stay loyal. (Probability: {prob:.2f})")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and LightGBM")
