import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide"
)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model     = pickle.load(open('xgb_model.pkl', 'rb'))
    scaler    = pickle.load(open('scaler.pkl', 'rb'))
    threshold = pickle.load(open('threshold.pkl', 'rb'))
    return model, scaler, threshold

model, scaler, threshold = load_models()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title(" Customer Churn Predictor")
st.markdown("##### Telco Customer Dataset — XGBoost Model with Tuned Threshold")
st.markdown("---")

# ── SIDEBAR INFO ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(" Model Info")
    st.markdown("""
    **Model:** XGBoost  
    **Threshold:** Tuned for best F1  
    
    **Top Churn Indicators:**
    1. Month-to-month contract
    2. Fiber optic internet
    3. High monthly charges
    4. Low tenure
    5. No online security
    """)
    st.markdown("---")
    st.caption("Dataset: Telco Customer Churn")

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    gender     = st.selectbox("Gender",           ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen",   ["No", "Yes"])
    partner    = st.selectbox("Partner",           ["Yes", "No"])
    dependents = st.selectbox("Dependents",        ["No", "Yes"])
    tenure     = st.slider("Tenure (months)", 0, 72, 1)
    monthly    = st.slider("Monthly Charges ($)", 18.0, 120.0, 100.0)

with col2:
    st.markdown("**Services**")
    phone    = st.selectbox("Phone Service",     ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    backup   = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col3:
    st.markdown("**Billing & Contract**")
    tech      = st.selectbox("Tech Support",        ["No", "Yes", "No internet service"])
    tv        = st.selectbox("Streaming TV",         ["No", "Yes", "No internet service"])
    movies    = st.selectbox("Streaming Movies",     ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing",    ["Yes", "No"])
    contract  = st.selectbox("Contract",             ["Month-to-month", "One year", "Two year"])
    payment   = st.selectbox("Payment Method",       [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"])

st.markdown("---")

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
if st.button(" Predict Churn", use_container_width=True):

    def encode(val, positive="Yes"):
        return 1 if val == positive else 0

    # Encode
    gender_enc     = encode(gender, "Male")
    senior_enc     = encode(senior, "Yes")
    partner_enc    = encode(partner, "Yes")
    dependents_enc = encode(dependents, "Yes")
    phone_enc      = encode(phone, "Yes")
    paperless_enc  = encode(paperless, "Yes")

    scaled     = scaler.transform(pd.DataFrame([[tenure, monthly]], columns=['tenure', 'MonthlyCharges']))
    tenure_sc  = scaled[0][0]
    monthly_sc = scaled[0][1]

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

    prob = model.predict_proba(features)[0][1]
    pred = 1 if prob >= threshold else 0

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(" Prediction Results")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
    with r2:
        st.metric("Retention Probability", f"{(1-prob)*100:.1f}%")
    with r3:
        st.metric("Tenure", f"{tenure} months")

    st.markdown("###")

    if pred == 1:
        st.error(f"###  Customer is likely to CHURN  ({prob*100:.1f}% probability)")
        st.progress(float(prob))
        st.markdown("---")
        st.subheader(" Recommended Actions")
        a1, a2, a3 = st.columns(3)
        with a1:
            st.warning(" Offer loyalty discount")
        with a2:
            st.warning(" Suggest annual contract")
        with a3:
            st.warning(" Assign dedicated support")
    else:
        st.success(f"###  Customer is likely to STAY  ({(1-prob)*100:.1f}% retention probability)")
        st.progress(float(prob))

    # ── CUSTOMER SUMMARY ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(" Customer Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Contract Type",   contract)
    s2.metric("Internet",        internet)
    s3.metric("Monthly Charges", f"${monthly:.0f}")
    s4.metric("Tenure",          f"{tenure} months")
