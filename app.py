# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Title
st.title("🩺 Health Risk Anomaly Detection (IF + DBSCAN)")

st.markdown("This app detects potential health risks using **Isolation Forest** and **DBSCAN** logic based on your input.")

# User Input Form
with st.form("user_input_form"):
    age = st.slider("Age (years)", 18, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", 140, 210, 170)
    weight = st.number_input("Weight (kg)", 40, 150, 70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", 90, 250, 120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", 60, 160, 80)
    smoke = st.radio("Smoker?", ["No", "Yes"])
    alco = st.radio("Alcohol consumption?", ["No", "Yes"])
    active = st.radio("Physically active?", ["Yes", "No"])
    cholesterol_level = st.selectbox("Cholesterol level", [1, 2, 3])
    gluc_level = st.selectbox("Glucose level", [1, 2, 3])
    submitted = st.form_submit_button("Predict Health Risk")

if submitted:
    # Encoded Values
    gender_val = 0 if gender == "Male" else 1
    smoke_val = 1 if smoke == "Yes" else 0
    alco_val = 1 if alco == "Yes" else 0
    active_val = 1 if active == "Yes" else 0
    bmi = weight / ((height / 100) ** 2)

    # Input row
    input_row = pd.DataFrame([{
        "age": age,
        "gender": gender_val,
        "height": height,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "smoke": smoke_val,
        "alco": alco_val,
        "active": active_val,
        "BMI": bmi,
        "cholesterol_level": cholesterol_level,
        "gluc_level": gluc_level
    }])

    # Load full dataset with predictions
    df = pd.read_csv("health_risk_cases.csv")

    # Match input structure for comparison
    features = [
        "age", "gender", "height", "ap_hi", "ap_lo", "smoke",
        "alco", "active", "BMI", "cholesterol_level", "gluc_level"
    ]
    scaler = StandardScaler()
    scaler.fit(df[features])
    input_scaled = scaler.transform(input_row)

    # Approximate logic from anomaly detection
    def rule_based_reason(row):
        reasons = []
        if row['BMI'] > 30:
            reasons.append("High BMI (possible obesity)")
        if row['cholesterol_level'] > 1:
            reasons.append("High cholesterol level")
        if row['gluc_level'] > 1:
            reasons.append("High glucose level")
        if row['ap_hi'] > 140:
            reasons.append("High systolic BP")
        if row['ap_lo'] > 90:
            reasons.append("High diastolic BP")
        if row['smoke'] == 1:
            reasons.append("Smoker")
        if row['alco'] == 1:
            reasons.append("Alcohol consumption")
        if row['active'] == 0:
            reasons.append("Physically inactive")
        return reasons

    # Dummy Isolation Forest + DBSCAN logic simulation
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN

    iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_model.fit(df[features])
    iso_pred = iso_model.predict(input_scaled)[0]

    dbscan = DBSCAN(eps=1.5, min_samples=20)
    dbscan.fit(df[features])
    dbscan_pred = dbscan.fit_predict(input_scaled)[0]

    # Final label
    if iso_pred == -1 and dbscan_pred == -1:
        final_risk = "High Risk (Detected by Both)"
    elif iso_pred == -1:
        final_risk = "Isolation Forest Risk Only"
    elif dbscan_pred == -1:
        final_risk = "DBSCAN Risk Only"
    else:
        final_risk = "No Risk Detected"

    # Display results
    st.subheader(" Prediction Summary")
    st.write("**🩺 Overall Health Risk:**", f"`{final_risk}`")

    risk_factors = rule_based_reason(input_row.iloc[0])
    if risk_factors:
        st.write("!! **Risk Factors Detected:**")
        for r in risk_factors:
            st.markdown(f"- {r}")
    else:
        st.write(" No specific risk factors detected.")

    st.write(" **Model Decisions:**")
    st.write(f"- Isolation Forest: {'Anomaly' if iso_pred == -1 else 'Normal'}")
    st.write(f"- DBSCAN: {'Anomaly' if dbscan_pred == -1 else 'Normal'}")
