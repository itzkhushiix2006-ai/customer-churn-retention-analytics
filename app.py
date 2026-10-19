import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and preprocessor
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict customer churn probability and identify customers "
    "who may need retention support."
)

st.info("Enter the customer's details below.")

# Customer details
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    st.subheader("Phone Services")

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

with col2:
    st.subheader("Account Details")

    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.subheader("Internet and Additional Services")

col3, col4 = st.columns(2)

with col3:
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )

with col4:
    streaming_tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )

if st.button("Predict Churn", type="primary"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    processed_customer = preprocessor.transform(customer)

    churn_index = list(model.classes_).index("Yes")
    probability = model.predict_proba(processed_customer)[0][churn_index]

    prediction = model.predict(processed_customer)[0]

    st.subheader("Prediction Result")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Churn Probability", f"{probability:.1%}")

    with metric2:
        risk = (
            "High" if probability >= 0.60
            else "Medium" if probability >= 0.30
            else "Low"
        )
        st.metric("Risk Level", risk)

    if prediction == "Yes":
        st.warning(
            "The model predicts that this customer may churn."
        )
    else:
        st.success(
            "The model predicts that this customer may stay."
        )

    st.caption(
        "This is a machine-learning estimate based on historical "
        "telecom data, not a guarantee of customer behavior."
    )