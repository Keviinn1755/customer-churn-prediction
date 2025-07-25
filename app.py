import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Churn Predictor", layout="centered")

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# HEADER SECTION
# ---------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether a telecom customer is likely to churn based on their profile.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# CHART SECTION
# ---------------------------
st.subheader(" Contract Type Distribution")

df = pd.read_csv("customer churn.csv")
contract_counts = df['Contract'].value_counts()

fig, ax = plt.subplots()
ax.bar(contract_counts.index, contract_counts.values, color='#00bcd4')
ax.set_xlabel("Contract Type")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

st.markdown("---")

# ---------------------------
# INPUT FORM SECTION
# ---------------------------
st.subheader(" Enter Customer Details")

with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (in months)", min_value=0)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])

    with col2:
        total_charges = st.number_input("Total Charges", min_value=0.0)
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    submit = st.form_submit_button(" Predict")

# ---------------------------
# PREDICTION LOGIC
# ---------------------------
if submit:
    # Encode inputs
    senior_citizen = 1 if senior_citizen == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    contract_one_year = 1 if contract_type == "One year" else 0
    contract_two_year = 1 if contract_type == "Two year" else 0

    # Prepare data
    input_data = np.array([[tenure, monthly_charges, total_charges,
                            senior_citizen, partner, dependents,
                            contract_one_year, contract_two_year]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    st.markdown("---")
    st.subheader(" Prediction Result")

    if prediction[0] == 1:
        st.error(" The customer is **likely to churn**.")
    else:
        st.success(" The customer is **likely to stay**.")

