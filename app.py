import gzip
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from datetime import timedelta

st.set_page_config(page_title="Prediksi Harga Cabai Merah Besar - Kabupaten Bekasi", layout="centered")

# --- Helpers ---
@st.cache_data
def load_model(path="model/model_sarima.pkl.gz"):
    with gzip.open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_sample_data(path="data/cabai merah besar.csv"):
    df = pd.read_csv(path, parse_dates=["tanggal_lengkap"]) if st.sidebar.checkbox("Load bundled sample data", value=False) else None
    return df


# --- UI ---
st.title("Prediksi Harga Cabai Merah Besar - Kabupaten Bekasi")
st.markdown("Model: SARIMA (dideploy dari penelitian)")

col1, col2 = st.columns([3,1])
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Indonesia_Bekasi_locator_map.svg/320px-Indonesia_Bekasi_locator_map.svg.png", use_column_width=True)

st.sidebar.header("Pengaturan")
# Forecast horizon: default 14, allow user choose 1-90
horizon = st.sidebar.slider("Jumlah hari prediksi (days)", min_value=1, max_value=90, value=14)

# Allow user to upload CSV or use bundled data
uploaddata = st.sidebar.file_uploader("Upload CSV data (optional)", type=["csv"])
use_sample = st.sidebar.checkbox("Gunakan data sample yang dibundel (jika ada)")

model = None
try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal load model. Pastikan file model berada di model/model_sarima.pkl.gz. Error: {e}")

# load data
if uploaddata is not None:
    df = pd.read_csv(uploaddata, parse_dates=[0])
    df.columns = ["date", "price"] if len(df.columns) >= 2 else ["date"]
elif use_sample:
    df = load_sample_data()
else:
    df = None

if df is not None:
    st.subheader("Data Historis")
    st.write(df.tail(10))
    fig, ax = plt.subplots()
    ax.plot(df["tanggal_lengkap"], df["cabe_merah_besar"], marker=".")
    ax.set_xlabel("Date")
    ax.set_ylabel("Harga")
    ax.set_title("Harga Historis")
    st.pyplot(fig)

st.subheader("Prediksi")
if model is None:
    st.info("Model belum tersedia — upload model di /model/model_sarima.pkl.gz pada repo.")
else:
    # Use model.get_forecast (SARIMAXResults) or model.predict
    try:
        # If model is SARIMAXResults or contains get_forecast
        if hasattr(model, "get_forecast"):
            forecast_obj = model.get_forecast(steps=horizon)
            mean = forecast_obj.predicted_mean
            conf = forecast_obj.conf_int(alpha=0.05)

            # build dataframe of predictions
            last_date = pd.to_datetime(df["date"].max()) if (df is not None and "date" in df.columns) else pd.to_datetime("today")
            pred_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            pred_df = pd.DataFrame({"date": pred_index, "predicted_mean": mean.values})

            st.write(pred_df)

            fig2, ax2 = plt.subplots()
            if df is not None:
                ax2.plot(df["date"], df["price"], label="historis")
            ax2.plot(pred_df["date"], pred_df["predicted_mean"], marker="o", label="prediksi")
            ax2.fill_between(pred_df["date"], conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
            ax2.set_title("Prediksi SARIMA")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Harga")
            ax2.legend()
            st.pyplot(fig2)

            # Offer download
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV prediksi", data=csv, file_name="prediksi_sarima.csv", mime="text/csv")
        else:
            st.error("Model tidak mendukung metode get_forecast(). Pastikan model yang disimpan berasal dari SARIMAXResults dari statsmodels.")
    except Exception as e:
        st.error(f"Gagal membuat prediksi: {e}")

st.markdown("---")
st.markdown("**Catatan:** Pastikan `model/model_sarima.pkl.gz` ada di repo. Jika model disimpan berbeda, ubah path di `load_model()`.")

