import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import io

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Prediksi Harga Cabai Merah Besar Bekasi", layout="wide")

# =========================
# KONFIGURASI DIREKTORI
# =========================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SARIMA_PATH = os.path.join(MODEL_DIR, "sarima.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ==========================================
# 1. Fungsi Membaca CSV (auto detect separator)
# ==========================================
def read_csv_auto(uploaded_file):
    content = uploaded_file.getvalue()
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            return df
        except:
            pass
    return pd.read_csv(io.BytesIO(content))

# ==========================================
# 2. Preprocessing otomatis
# ==========================================
def preprocess(df, date_col=None, target_hint="cabe"):
    df = df.copy()

    # detect kolom tanggal
    if date_col is None:
        for c in df.columns:
            if "tanggal" in c.lower() or "date" in c.lower():
                date_col = c
                break

    if date_col is None:
        st.error("Tidak menemukan kolom tanggal.")
        st.stop()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    # detect kolom target (mengandung kata 'cabe')
    target_col = None
    for c in df.columns:
        if target_hint.lower() in c.lower():
            target_col = c
            break

    if target_col is None:
        target_col = st.selectbox("Pilih kolom target harga:", df.columns)

    ts = df[[target_col]].astype(float)

    # resampling harian + interpolasi
    ts = ts.asfreq("D")
    ts[target_col] = ts[target_col].interpolate()

    return ts, target_col

# ==========================================
# 3. SARIMA
# ==========================================
def train_sarima(series, order=(1,1,1), seasonal=(1,1,1,7)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    joblib.dump(fitted, SARIMA_PATH)
    return fitted

def predict_sarima(fitted, steps):
    return fitted.forecast(steps=steps)

# ==========================================
# 4. LSTM
# ==========================================
def create_dataset(dataset, step):
    X, y = [], []
    for i in range(len(dataset) - step):
        X.append(dataset[i:i+step])
        y.append(dataset[i+step])
    return np.array(X), np.array(y)

def train_lstm(series, time_step=30, epochs=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1,1))

    X, y = create_dataset(scaled, time_step)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step,1)),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.1,
              epochs=epochs, batch_size=32, verbose=0, callbacks=[es])

    # simpan
    model.save(LSTM_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

def predict_lstm(model, scaler, series, steps, time_step=30):
    scaled = scaler.transform(series.reshape(-1,1))
    temp = list(scaled[-time_step:].flatten())
    outputs = []

    for i in range(steps):
        x = np.array(temp[-time_step:]).reshape(1, time_step, 1)
        y_hat = model.predict(x, verbose=0)[0][0]
        temp.append(y_hat)
        outputs.append(y_hat)

    return scaler.inverse_transform(np.array(outputs).reshape(-1,1)).flatten()

# ==========================================
# == STREAMLIT UI ==
# ==========================================
st.title("📈 Prediksi Harga Cabai Merah Besar – Kabupaten Bekasi")
st.write("Upload dataset harga cabai rawit/cabai merah besar tahun 2023–2024 atau terbaru.")

uploaded = st.file_uploader("Upload CSV", type=["csv", "txt", "xlsx"])

if uploaded:
    if uploaded.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded)
    else:
        df_raw = read_csv_auto(uploaded)

    st.subheader("Preview Data")
    st.write(df_raw.head())

    ts, target_col = preprocess(df_raw)

    st.subheader("Grafik Data Historis")
    st.line_chart(ts[target_col])

    # ======================
    # PILIHAN MODEL
    # ======================
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox("Pilih model", ["SARIMA", "LSTM", "Keduanya"])
    horizon = st.sidebar.number_input("Horizon prediksi (hari):", 1, 365, 30)

    # ======================
    # RETRAIN SARIMA
    # ======================
    if st.sidebar.button("🔄 Retrain SARIMA"):
        fitted = train_sarima(ts[target_col])
        st.sidebar.success("SARIMA berhasil dilatih ulang & disimpan!")

    # Load SARIMA jika ada
    sarima_model = joblib.load(SARIMA_PATH) if os.path.exists(SARIMA_PATH) else None

    # ======================
    # RETRAIN LSTM
    # ======================
    if st.sidebar.button("🔄 Retrain LSTM"):
        model, scaler = train_lstm(ts[target_col].values)
        st.sidebar.success("LSTM berhasil dilatih ulang & disimpan!")

    # Load LSTM jika ada
    lstm_model = load_model(LSTM_PATH) if os.path.exists(LSTM_PATH) else None
    lstm_scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    # ======================
    # PREDIKSI
    # ======================
    if st.sidebar.button("🚀 Jalankan Prediksi"):
        st.subheader("Hasil Prediksi")

        results = {}

        if model_choice in ["SARIMA", "Keduanya"]:
            if sarima_model is None:
                st.error("Model SARIMA belum tersedia. Klik 'Retrain SARIMA' dulu.")
            else:
                sarima_pred = predict_sarima(sarima_model, horizon)
                results["SARIMA"] = sarima_pred

        if model_choice in ["LSTM", "Keduanya"]:
            if lstm_model is None:
                st.error("Model LSTM belum tersedia. Klik 'Retrain LSTM' dulu.")
            else:
                lstm_pred = predict_lstm(
                    lstm_model, lstm_scaler, ts[target_col].values, horizon
                )
                results["LSTM"] = lstm_pred

        # Tampilkan grafik
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(ts[target_col], label="Data Historis")

        future_index = pd.date_range(
            start=ts.index[-1] + pd.Timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        for name, pred in results.items():
            ax.plot(future_index, pred, label=f"Prediksi {name}")

        ax.legend()
        st.pyplot(fig)

        # Tabel hasil
        df_pred = pd.DataFrame(index=future_index)
        for name, pred in results.items():
            df_pred[name] = pred

        st.write("📄 Hasil Prediksi:")
        st.dataframe(df_pred)

        # Download
        csv = df_pred.to_csv().encode("utf-8")
        st.download_button("📥 Download CSV", csv, "prediksi.csv", "text/csv")
