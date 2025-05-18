import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scalers
model = load_model('pepper_disease_model.keras')
X_scaler = joblib.load('pepper_input_scaler.pkl')
Y_scaler = joblib.load('pepper_output_scaler.pkl')

# Streamlit app
st.set_page_config(page_title="Pepper Disease Predictor", layout="centered")
st.title(" Black Pepper Disease Severity Predictor")
st.write("Enter last week's weather data to predict the severity of four major black pepper diseases.")

# Input fields
max_temp = st.number_input(" Max Temperature (°C)", min_value=10.0, max_value=50.0, value=32.0)
min_temp = st.number_input("Min Temperature (°C)", min_value=5.0, max_value=40.0, value=23.0)
rh1 = st.number_input("Morning Relative Humidity (%)", min_value=0.0, max_value=100.0, value=85.0)
rh2 = st.number_input("Evening Relative Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
sunshine = st.number_input("Sunshine Hours", min_value=0.0, max_value=15.0, value=6.5)
rainfall = st.number_input(" Rainfall (mm)", min_value=0.0, max_value=500.0, value=20.0)

# Prediction button
if st.button("Predict Disease Severity"):
    # Prepare input
    input_data = np.array([[max_temp, min_temp, rh1, rh2, sunshine, rainfall]])
    input_scaled = X_scaler.transform(input_data)
    
    # Predict and inverse transform
    pred_scaled = model.predict(input_scaled)
    pred = Y_scaler.inverse_transform(pred_scaled)

    st.subheader("Predicted Disease Severity (Scale 0–5):")
    st.write(f"**Algal Rust (ARP)**: {pred[0][0]:.2f}")
    st.write(f"**Slow Wilt (SWP)**: {pred[0][1]:.2f}")
    st.write(f"**Quick Wilt (QWP)**: {pred[0][2]:.2f}")
    st.write(f"**Pollu Disease (PDP)**: {pred[0][3]:.2f}")

    st.success("Prediction complete.")
