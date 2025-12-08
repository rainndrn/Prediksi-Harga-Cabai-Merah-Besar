import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ========================================================
# STREAMLIT CONFIG
# ========================================================
st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Besar",
    layout="wide",
    page_icon="🌶️"
)

# ========================================================
# HEADER
# ========================================================
st.markdown("""
# 🌶️ Prediksi Harga Cabai Merah Besar  
### Kabupaten Bekasi — Metode SARIMA & LSTM  
Dashboard prediksi harga harian berbasis machine learning.
""")

st.markdown("---")

# ========================================================
# SIDEBAR - LOAD MODEL
# ========================================================
with st.sidebar:
    st.header("🔧 Load Model")

    # ============= Extract SARIMA From ZIP =============
    if not os.path.exists("model_sarima.pkl") and os.path.exists("model_sarima.zip"):
        with zipfile.ZipFile("model_sarima.zip", "r") as z:
            z.extractall()
        st.success("SARIMA extracted!")

    # ============= Load SARIMA =============
    try:
        model_sarima = joblib.load("model_sarima.pkl")
        st.success("Model SARIMA loaded")
    except:
        model_sarima = None
        st.error("Model SARIMA not found")

    # ============= Load LSTM =============
    try:
        model_lstm = load_model("model_lstm.h5")
        st.success("Model LSTM loaded")
    except:
        model_lstm = None
        st.error("Model LSTM not found")


# ========================================================
# TAB MENU
# ========================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 SARIMA", "🔮 LSTM"])

# ========================================================
# UPLOAD DATA (GLOBAL)
# ========================================================
with st.expander("📤 Upload Dataset di Sini"):
    uploaded = st.file_uploader(
        "Upload file CSV (harus ada kolom: tanggal_lengkap, cabe_merah_besar)", 
        type=["csv"]
    )

if uploaded:
    df = pd.read_csv(uploaded)
    
    if "tanggal_lengkap" not in df.columns or "cabe_merah_besar" not in df.columns:
        st.error("CSV wajib memiliki kolom tanggal_lengkap dan cabe_merah_besar!")
        st.stop()

    # Convert date
    df["tanggal_lengkap"] = pd.to_datetime(df["tanggal_lengkap"], errors="coerce")
    df = df.sort_values("tanggal_lengkap")

    series = df["cabe_merah_besar"].values
else:
    df = None
    series = None


# ========================================================
# TAB 1: DASHBOARD
# ========================================================
with tab1:
    st.subheader("📊 Dashboard Harga Harian")

    if df is None:
        st.info("Upload dataset terlebih dahulu.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📈 Grafik Harga Cabai Merah Besar")
            st.line_chart(df.set_index("tanggal_lengkap")["cabe_merah_besar"])

        with col2:
            st.write("### 📘 Statistik Deskriptif")
            st.dataframe(df["cabe_merah_besar"].describe())

        st.markdown("---")
        st.write("### 📅 Range Data")
        st.write(f"**Tanggal awal:** {df['tanggal_lengkap'].min().date()}")
        st.write(f"**Tanggal akhir:** {df['tanggal_lengkap'].max().date()}")


# ========================================================
# TAB 2: PREDIKSI SARIMA
# ========================================================
with tab2:
    st.subheader("📈 Prediksi SARIMA")

    if model_sarima is None:
        st.error("Model SARIMA tidak tersedia.")
    elif df is None:
        st.warning("Upload dataset terlebih dahulu.")
    else:
        n_days = st.number_input("Prediksi berapa hari ke depan?", 1, 90, 30)

        if st.button("🚀 Jalankan Prediksi SARIMA"):
            sarima_pred = model_sarima.forecast(steps=n_days)

            st.write("### Hasil Prediksi SARIMA")
            st.line_chart(sarima_pred)

            st.dataframe(pd.DataFrame({
                "hari_ke": range(1, n_days+1),
                "prediksi": sarima_pred
            }))


# ========================================================
# TAB 3: PREDIKSI LSTM
# ========================================================
with tab3:
    st.subheader("🔮 Prediksi LSTM")

    if model_lstm is None:
        st.error("Model LSTM tidak tersedia.")
    elif df is None:
        st.warning("Upload dataset terlebih dahulu.")
    else:

        n_days = st.number_input("Prediksi berapa hari ke depan?", 1, 90, 30)

        if st.button("🚀 Jalankan Prediksi LSTM"):
            # =========================================
            # 1. Fit scaler dari data user
            # =========================================
            scaler = MinMaxScaler()
            data_raw = series.reshape(-1, 1)
            data_scaled = scaler.fit_transform(data_raw)

            # =========================================
            # 2. Window size dari training kamu = 30
            # =========================================
            window = 30
            last_window = data_scaled[-window:].reshape(1, window, 1)

            # =========================================
            # 3. Multi-step recursive forecasting
            # =========================================
            lstm_pred_list = []
            current_input = last_window

            for _ in range(n_days):
                pred = model_lstm.predict(current_input, verbose=0)
                lstm_pred_list.append(pred[0][0])

                new_window = np.append(current_input[0, 1:, 0], pred[0][0])
                current_input = new_window.reshape(1, window, 1)

            # =========================================
            # 4. Inverse transform
            # =========================================
            lstm_pred = np.array(lstm_pred_list).reshape(-1, 1)
            lstm_pred = scaler.inverse_transform(lstm_pred)

            # =========================================
            # 5. Display output
            # =========================================
            st.write("### Hasil Prediksi LSTM")
            st.line_chart(lstm_pred)

            st.dataframe(pd.DataFrame({
                "hari_ke": range(1, n_days+1),
                "prediksi": lstm_pred.flatten()
            }))