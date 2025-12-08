import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ===============================
# CONFIG STREAMLIT
# ===============================
st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Besar",
    layout="wide"
)

st.title("🌶️ Prediksi Harga Cabai Merah Besar di Kabupaten Bekasi")
st.write("Metode: **SARIMA** dan **LSTM** (Dashboard)")

# ===============================
# LOAD SARIMA FROM ZIP
# ===============================
st.sidebar.header("🔧 Load Model")

if not os.path.exists("model_sarima.pkl"):
    if os.path.exists("model_sarima.zip"):
        with zipfile.ZipFile("model_sarima.zip", "r") as z:
            z.extractall()
        st.sidebar.success("Model SARIMA diekstrak!")
    else:
        st.sidebar.error("model_sarima.zip tidak ditemukan!")

# Load SARIMA
try:
    model_sarima = joblib.load("model_sarima.pkl")
    st.sidebar.success("Model SARIMA berhasil dimuat!")
except:
    model_sarima = None
    st.sidebar.error("Gagal load model_sarima.pkl")

# ===============================
# LOAD LSTM
# ===============================
try:
    model_lstm = load_model("model_lstm.h5")
    st.sidebar.success("Model LSTM berhasil dimuat!")
except:
    model_lstm = None
    st.sidebar.error("model_lstm.h5 tidak ditemukan!")

# ===============================
# LOAD SCALER
# ===============================
scaler = None
if os.path.exists("scaler.pkl"):
    scaler = joblib.load("scaler.pkl")
    st.sidebar.info("Scaler berhasil dimuat.")

# ===============================
# UPLOAD DATA USER
# ===============================
st.header("📤 Upload Data Harga Cabai")
uploaded = st.file_uploader("Upload file CSV (harus ada kolom: tanggal_lengkap, cabe_merah_besar)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    
    if "tanggal_lengkap" not in df.columns or "cabe_merah_besar" not in df.columns:
        st.error("CSV wajib memiliki kolom tanggal_lengkap dan cabe_merah_besar!")
        st.stop()

    df["tanggal_lengkap"] = pd.to_datetime(df["tanggal_lengkap"], errors="coerce")
    df = df.sort_values("tanggal_lengkap")

    st.success("Data berhasil dimuat!")
    st.dataframe(df.head())
else:
    df = None
    st.warning("Upload CSV dulu untuk melihat data.")

# ===============================
# INPUT N DAYS
# ===============================
st.header("🎯 Pengaturan Prediksi")
n_days = st.number_input("Prediksi berapa hari ke depan?", min_value=1, max_value=90, value=30)

# ===============================
# RUN PREDIKSI
# ===============================
if st.button("🚀 Jalankan Prediksi"):

    if df is None:
        st.error("Upload data CSV terlebih dahulu.")
        st.stop()

    series = df["cabe_merah_besar"].values
    
    # ---------------------------
    # PREDIKSI SARIMA
    # ---------------------------
    st.subheader("📈 Prediksi SARIMA")
    
    if model_sarima:
        sarima_pred = model_sarima.forecast(steps=n_days)
        st.line_chart(sarima_pred)

        sarima_result = pd.DataFrame({
            "hari_ke": range(1, n_days+1),
            "prediksi": sarima_pred
        })
        st.dataframe(sarima_result)
    else:
        st.error("Model SARIMA tidak tersedia.")

    # ---------------------------
    # PREDIKSI LSTM
    # ---------------------------
    st.subheader("📈 Prediksi LSTM")

    if not model_lstm:
        st.error("Model LSTM tidak tersedia.")
        st.stop()

    # Scaling
    data_raw = series.reshape(-1, 1)
    if scaler:
        data_scaled = scaler.transform(data_raw)
    else:
        data_scaled = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min())

    # Window size = 30
    window = 30
    last_window = data_scaled[-window:].reshape(1, window, 1)

    lstm_pred_list = []
    current_input = last_window

    for _ in range(n_days):
        pred = model_lstm.predict(current_input, verbose=0)
        lstm_pred_list.append(pred[0][0])

        new_window = np.append(current_input[0, 1:, 0], pred[0][0])
        current_input = new_window.reshape(1, window, 1)

    lstm_pred = np.array(lstm_pred_list).reshape(-1, 1)

    # Inverse transform
    if scaler:
        lstm_pred = scaler.inverse_transform(lstm_pred)
    else:
        lstm_pred = lstm_pred * (data_raw.max() - data_raw.min()) + data_raw.min()

    st.line_chart(lstm_pred)

    lstm_result = pd.DataFrame({
        "hari_ke": range(1, n_days+1),
        "prediksi": lstm_pred.flatten()
    })
    st.dataframe(lstm_result)

# ===============================
# END
# ===============================