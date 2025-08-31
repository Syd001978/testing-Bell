import streamlit as st
import joblib
import pandas as pd

# --- Load model & scaler ---
model = joblib.load("model_malware.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🕵️ Malware Domain Detector")

# Input domain
domain = st.text_input("Masukkan domain:", "example.com")

if st.button("Deteksi"):
    # ---- TODO: ekstraksi fitur dari domain ----
    # misalnya panjang domain
    features = {
        "length": len(domain),
        "digit_count": sum(c.isdigit() for c in domain),
        "dot_count": domain.count("."),
    }
    df_input = pd.DataFrame([features])

    # Scaling
    X_scaled = scaler.transform(df_input)

    # Predict
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    st.write(f"**Hasil Deteksi:** {'⚠️ Malware' if pred==1 else '✅ Aman'}")
    st.write("Probabilitas:", prob)
    st.write(f"- Probabilitas Aman: {prob[0]:.2%}")
    st.write(f"- Probabilitas Malware: {prob[1]:.2%}")
