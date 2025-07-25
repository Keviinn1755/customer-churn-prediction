#  Customer Churn Prediction App

This is a machine learning and data analytics web app built using **Logistic Regression** and **Streamlit** to predict whether a telecom customer is likely to churn.

##  Features

- Predicts churn using key customer attributes
- Uses logistic regression (trained on 8 important features)
- Clean and user-friendly Streamlit UI
- Visualizes Contract Type distribution
- Instant prediction using form inputs

##  Dataset

- **Telco Customer Churn Dataset**
- [Source](https://www.kaggle.com/blastchar/telco-customer-churn)

##  Model Details

- Trained with 8 features:
  - Tenure, MonthlyCharges, TotalCharges
  - SeniorCitizen, Partner, Dependents
  - Contract Type (One year, Two year)

##  How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run streamlit_app.py
