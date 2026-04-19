import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, threshold
model     = pickle.load(open('xgb_model.pkl', 'rb'))
scaler    = pickle.load(open('scaler.pkl', 'rb'))
threshold = pickle.load(open('threshold.pkl', 'rb'))

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict if they will churn")

# ── Input fields ──────────────────────────────
st.sidebar.header("Customer Details")

gender     = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior     = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner    = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure     = st.sidebar.slider("Tenure (months)", 0, 72, 1)
phone      = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
paperless  = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
monthly    = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 100.0)
multiple   = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet   = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
security   = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup     = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device     = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech       = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
tv         = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies     = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract   = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment    = st.sidebar.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])

# ── Encode inputs ─────────────────────────────
def encode(val, positive="Yes"):
    return 1 if val == positive else 0

gender_enc     = encode(gender, "Male")
senior_enc     = encode(senior, "Yes")
partner_enc    = encode(partner, "Yes")
dependents_enc = encode(dependents, "Yes")
phone_enc      = encode(phone, "Yes")
paperless_enc  = encode(paperless, "Yes")

# Scale tenure and MonthlyCharges
scaled     = scaler.transform(pd.DataFrame([[tenure, monthly]], columns=['tenure', 'MonthlyCharges']))
tenure_sc  = scaled[0][0]
monthly_sc = scaled[0][1]

# One Hot Encoding
multiple_no_phone = 1 if multiple == "No phone service" else 0
multiple_yes      = 1 if multiple == "Yes" else 0

internet_fiber = 1 if internet == "Fiber optic" else 0
internet_no    = 1 if internet == "No" else 0

security_no_internet = 1 if security == "No internet service" else 0
security_yes         = 1 if security == "Yes" else 0

backup_no_internet = 1 if backup == "No internet service" else 0
backup_yes         = 1 if backup == "Yes" else 0

device_no_internet = 1 if device == "No internet service" else 0
device_yes         = 1 if device == "Yes" else 0

tech_no_internet = 1 if tech == "No internet service" else 0
tech_yes         = 1 if tech == "Yes" else 0

tv_no_internet = 1 if tv == "No internet service" else 0
tv_yes         = 1 if tv == "Yes" else 0

movies_no_internet = 1 if movies == "No internet service" else 0
movies_yes         = 1 if movies == "Yes" else 0

contract_one = 1 if contract == "One year" else 0
contract_two = 1 if contract == "Two year" else 0

payment_credit     = 1 if payment == "Credit card (automatic)" else 0
payment_electronic = 1 if payment == "Electronic check" else 0
payment_mailed     = 1 if payment == "Mailed check" else 0

# ── Final feature array ──────────────────────
features = np.array([[
    gender_enc, senior_enc, partner_enc, dependents_enc,
    tenure_sc, phone_enc, paperless_enc, monthly_sc,
    multiple_no_phone, multiple_yes,
    internet_fiber, internet_no,
    security_no_internet, security_yes,
    backup_no_internet, backup_yes,
    device_no_internet, device_yes,
    tech_no_internet, tech_yes,
    tv_no_internet, tv_yes,
    movies_no_internet, movies_yes,
    contract_one, contract_two,
    payment_credit, payment_electronic, payment_mailed
]])

# ── Predict ───────────────────────────────────
if st.button(" Predict Churn", key="predict_button"):
    prob = model.predict_proba(features)[0][1]
    pred = 1 if prob >= threshold else 0

    st.subheader("Prediction Result")

    if pred == 1:
        st.error("Customer is likely to CHURN!")
        st.write(f"Churn Probability: **{prob*100:.1f}%**")
        st.write("### Recommended Actions:")
        st.write("- Offer loyalty discount")
        st.write("- Suggest annual contract")
        st.write("- Assign dedicated support agent")
    else:
        st.success("Customer is likely to STAY!")
        st.write(f"Churn Probability: **{prob*100:.1f}%**")

    st.progress(float(prob))