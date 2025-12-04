import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load model and scalers
model = load_model("nn_model.keras", compile=False, safe_mode=False)
input_scaler = joblib.load("input_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")


def predict(data):
    # Convert to numpy array in correct order
    input_data = np.array([[
        data.get("UPE_enrollement", 0),
        data.get("Primary_non_wage_conditional_grant", 0),
        data.get("Secondary_non_wage_conditional_grant", 0),
        data.get("UPE_Performance_indexL", 0),
        data.get("Approved_annual_budgetL", 0),
        data.get("UPE_releaseL", 0),
        data.get("UPE_Grant_per_enrolled_studentL", 0),
        data.get("UPE_release_efficiency", 0)
    ]])

    # Scale input
    input_scaled = input_scaler.transform(input_data)

    # Predict scaled
    pred_scaled = model.predict(input_scaled)

    # Reverse scale
    prediction = target_scaler.inverse_transform(pred_scaled)

    return float(prediction[0][0])


st.title("UPE EFFICIENCY TRACKER SYSTEM")
st.write("Enter values below and click **Predict**")

# Streamlit input fields
UPE_enrollement = st.number_input("UPE Enrollement", value=0.0)
Primary_non_wage_conditional_grant = st.number_input("Primary Non Wage Conditional Grant", value=0.0)
Secondary_non_wage_conditional_grant = st.number_input("Secondary Non Wage Conditional Grant", value=0.0)
UPE_Performance_indexL = st.number_input("UPE Performance Index L", value=0.0)
Approved_annual_budgetL = st.number_input("Approved Annual Budget L", value=0.0)
UPE_releaseL = st.number_input("UPE Release L", value=0.0)
UPE_Grant_per_enrolled_studentL = st.number_input("UPE Grant per Enrolled Student L", value=0.0)
UPE_release_efficiency = st.number_input("UPE Release Efficiency", value=0.0)

# Predict button
if st.button("Predict"):
    payload = {
        "UPE_enrollement": UPE_enrollement,
        "Primary_non_wage_conditional_grant": Primary_non_wage_conditional_grant,
        "Secondary_non_wage_conditional_grant": Secondary_non_wage_conditional_grant,
        "UPE_Performance_indexL": UPE_Performance_indexL,
        "Approved_annual_budgetL": Approved_annual_budgetL,
        "UPE_releaseL": UPE_releaseL,
        "UPE_Grant_per_enrolled_studentL": UPE_Grant_per_enrolled_studentL,
        "UPE_release_efficiency": UPE_release_efficiency
    }

    result = predict(payload)

    st.success(f"Predicted Value: {result:.4f}")



