# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# -----------------------------
# Load model, scaler, dan kolom
# -----------------------------
model = joblib.load("model_malware.pkl")
scaler = joblib.load("scaler.pkl")
training_cols = joblib.load("training_cols.pkl")

# -----------------------------
# Fungsi ekstraksi fitur domain
# -----------------------------
def extract_features(domain):
    """Ekstraksi fitur numerik dari domain (contoh sederhana)."""
    features = {
        "length": len(domain),
        "num_digits": sum(c.isdigit() for c in domain),
        "num_dashes": domain.count("-"),
        "num_dots": domain.count("."),
        "num_underscores": domain.count("_"),
        # bisa ditambah fitur lain sesuai yang dipakai saat training
    }
    df = pd.DataFrame([features])
    
    # pastikan urutan kolom sesuai training
    for col in training_cols:
        if col not in df.columns:
            df[col] = 0  # kolom yang tidak ada di input diisi 0
    df = df[training_cols]
    
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Malware Domain Detection 🔒")
st.write("Masukkan nama domain, dan mesin akan memprediksi apakah malware atau tidak.")

domain_input = st.text_input("Domain:")

if st.button("Prediksi"):
    if not domain_input:
        st.warning("Tolong masukkan domain terlebih dahulu!")
    else:
        # ekstraksi fitur
        df_input = extract_features(domain_input)

        # scaling
        X_scaled = scaler.transform(df_input)

        # prediksi
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0].max()

        st.write(f"**Hasil Prediksi:** {pred}")
        st.write(f"**Probabilitas:** {proba:.2f}")
