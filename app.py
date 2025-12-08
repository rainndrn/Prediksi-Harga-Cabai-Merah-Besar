import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

# ===== Load Model =====
@st.cache_resource
def load_sarima():
    with open("model_sarima.zip", "rb") as f:
        model = pickle.load(f)
    return model

model = load_sarima()

# ===== UI =====
st.title("📈 Prediksi Harga Cabai Merah Besar - SARIMA")
st.markdown("Aplikasi ini menggunakan model **SARIMA** untuk memprediksi harga cabai merah besar di Kabupaten Bekasi.")

st.sidebar.header("Pengaturan Prediksi")
n_days = st.sidebar.slider("Prediksi berapa hari ke depan?", 1, 30, 7)

if st.button("Prediksi Sekarang"):
    forecast = model.forecast(steps=n_days)

    st.subheader(f"📅 Hasil Prediksi {n_days} Hari ke Depan")
    df_pred = pd.DataFrame({
        "Tanggal": pd.date_range(start=datetime.today(), periods=n_days),
        "Prediksi Harga": forecast
    })

    st.dataframe(df_pred)

    # Plot
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_pred["Tanggal"], df_pred["Prediksi Harga"], marker="o")
    ax.set_title("Prediksi Harga Cabai")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga")
    st.pyplot(fig)

st.markdown("---")
st.info("Model: SARIMA • Dibuat tanpa TensorFlow supaya stabil untuk deployment Streamlit.")

