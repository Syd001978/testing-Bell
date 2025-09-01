# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------
# Load model & scaler
# --------------------------
model = joblib.load("model_malware.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------
# Define training columns (numeric only, sesuai training)
# --------------------------
training_cols = [
    'Feature1', 'Feature2', 'Feature3',  # ganti dengan nama kolom numeric asli
    # ...
]

# --------------------------
# Streamlit UI
# --------------------------
st.title("Malware Domain Detector")

st.write("Masukkan  domain untuk mendeteksi apakah domain malware atau tidak.")

# Form input user
with st.form("input_form"):
    user_input = {}
    for col in training_cols:
        # semua numeric input
        user_input[col] = st.number_input(f"{col}", value=0.0, step=1.0)
    submitted = st.form_submit_button("Predict")

# --------------------------
# Predict
# --------------------------
if submitted:
    # Buat DataFrame dari input user
    df_input = pd.DataFrame([user_input])

    # Pastikan urutan kolom sesuai training
    df_input = df_input[training_cols]

    # Transformasi menggunakan scaler
    X_scaled = scaler.transform(df_input)

    # Prediksi
    pred = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    # Tampilkan hasil
    st.write(f"**Prediksi:** {'Malware' if pred==1 else 'Non-Malware'}")
    st.write(f"**Probabilitas:** Malware: {pred_proba[1]:.2f}, Non-Malware: {pred_proba[0]:.2f}")
