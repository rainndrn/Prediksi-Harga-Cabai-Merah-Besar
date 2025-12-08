import gzip
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import io
import os

st.set_page_config(page_title="Prediksi Harga Cabai Merah Besar - Kabupaten Bekasi", layout="centered")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_model(path="model/model_sarima.pkl.gz"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Upload model to model/model_sarima.pkl.gz")
    with gzip.open(path, "rb") as f:
        model = pickle.load(f)
    return model

def try_split_single_column(df):
    """
    If CSV was read as single column like 'tanggal_lengkap;cabe_merah_besar',
    split that column by ';' and create two columns.
    """
    if df.shape[1] == 1:
        col0 = df.columns[0]
        sample = df.iloc[0,0]
        if isinstance(sample, str) and ';' in sample:
            # split every row
            new_df = df[col0].str.split(';', expand=True)
            # give tentative names
            if new_df.shape[1] >= 2:
                new_df.columns = ['tanggal_lengkap', 'cabe_merah_besar'] + [f'col{i}' for i in range(3, new_df.shape[1]+1)]
                return new_df
    return df

@st.cache_data
def load_csv_from_path(path="data/cabai_merah_besar.csv"):
    # Attempt several separators and robust fixes
    if not os.path.exists(path):
        return None
    for sep in [';', ',', '\t']:
        try:
            df = pd.read_csv(path, sep=sep)
            df = try_split_single_column(df)
            return df
        except Exception:
            continue
    # final fallback
    df = pd.read_csv(path, engine='python')
    df = try_split_single_column(df)
    return df

def load_csv_filebuffer(file_buffer):
    # file_buffer is a BytesIO or UploadedFile
    # try common separators, and try splitting single col
    try:
        content = file_buffer.getvalue().decode('utf-8')
    except Exception:
        # if already text-like
        try:
            content = file_buffer.read().decode('utf-8')
        except Exception:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer)
            df = try_split_single_column(df)
            return df
    for sep in [';', ',', '\t']:
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep)
            df = try_split_single_column(df)
            return df
        except Exception:
            continue
    # fallback
    df = pd.read_csv(io.StringIO(content))
    df = try_split_single_column(df)
    return df

def prepare_df_for_plot(df):
    # Normalize column names
    cols_lower = {c.lower(): c for c in df.columns}
    # possible names
    date_col = None
    price_col = None
    # exact expected names
    if 'tanggal_lengkap' in df.columns:
        date_col = 'tanggal_lengkap'
    elif 'tanggal' in cols_lower:
        date_col = cols_lower['tanggal']
    else:
        # try find a column with 'tanggal' substring
        for c in df.columns:
            if 'tanggal' in c.lower() or 'date' in c.lower():
                date_col = c
                break

    # price column candidates
    if 'cabe_merah_besar' in df.columns:
        price_col = 'cabe_merah_besar'
    else:
        for c in df.columns:
            if 'cabe' in c.lower() and ('besar' in c.lower() or 'merah' in c.lower() or 'harga' in c.lower()):
                price_col = c
                break
        if price_col is None:
            for c in df.columns:
                if 'harga' in c.lower() or 'price' in c.lower():
                    price_col = c
                    break

    if date_col is None or price_col is None:
        raise KeyError(f"Kolom tidak ditemukan. Terdeteksi kolom: {list(df.columns)}. Diharapkan 'tanggal_lengkap' & 'cabe_merah_besar' atau kolom sepadan.")

    # parse date (expected format %Y-%m-%d), fallback to to_datetime
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors='coerce')
    if df[date_col].isna().any():
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    df = df[[date_col, price_col]].rename(columns={date_col: 'tanggal_lengkap', price_col: 'cabe_merah_besar'})
    df = df.dropna(subset=['tanggal_lengkap', 'cabe_merah_besar'])
    # ensure daily frequency (we won't resample automatically; we'll assume harian)
    df = df.sort_values('tanggal_lengkap').reset_index(drop=True)
    return df

# -------------------------
# UI
# -------------------------
st.title("Prediksi Harga Cabai Merah Besar - Kabupaten Bekasi")
st.markdown("Model: SARIMA (hasil penelitian) — hanya SARIMA di-deploy pada versi ini.")

col1, col2 = st.columns([3,1])
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Indonesia_Bekasi_locator_map.svg/320px-Indonesia_Bekasi_locator_map.svg.png", width=250)

st.sidebar.header("Pengaturan")
horizon = st.sidebar.slider("Jumlah hari prediksi (days)", min_value=1, max_value=90, value=14)

uploaddata = st.sidebar.file_uploader("Upload CSV data (opsional)", type=["csv"])
use_sample = st.sidebar.checkbox("Gunakan data sample (data/historical_prices.csv jika ada)", value=False)

# Load model
model = None
try:
    model = load_model()
except Exception as e:
    st.warning(f"Model belum ditemukan / gagal dimuat: {e}")

# Load data
df = None
if uploaddata is not None:
    try:
        df = load_csv_filebuffer(uploaddata)
    except Exception as e:
        st.error(f"Gagal membaca file upload: {e}")
elif use_sample:
    df = load_csv_from_path()

if df is None:
    st.info("Tidak ada data historis dimuat. Kamu bisa upload CSV (kolom: tanggal_lengkap, cabe_merah_besar) atau centang 'Gunakan data sample'.")
else:
    try:
        df = prepare_df_for_plot(df)
    except Exception as e:
        st.error(f"Masalah saat memproses data: {e}")
        st.write("Kolom yang tersedia:", list(df.columns))
        st.stop()

    st.subheader("Data Historis (terakhir 10 baris)")
    st.write(df.tail(10))

    fig, ax = plt.subplots()
    ax.plot(df['tanggal_lengkap'], df['cabe_merah_besar'], marker='.', label='historis')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga")
    ax.set_title("Harga Historis Cabe Merah Besar")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

st.markdown("---")
st.subheader("Prediksi (SARIMA)")

if model is None:
    st.error("Model SARIMA tidak tersedia. Pastikan `model/model_sarima.pkl.gz` ada di repo.")
else:
    # Prepare last_date for index
    if df is not None and not df.empty:
        last_date = pd.to_datetime(df['tanggal_lengkap'].max())
    else:
        last_date = pd.to_datetime("today").normalize()
    try:
        # If SARIMAXResults-like
        if hasattr(model, "get_forecast"):
            forecast_obj = model.get_forecast(steps=horizon)
            mean = forecast_obj.predicted_mean
            conf = forecast_obj.conf_int()
            # Build prediction dates
            pred_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
            pred_df = pd.DataFrame({
                "tanggal_lengkap": pred_index,
                "predicted_mean": np.round(mean.values, 2)
            })
            # add conf columns if available
            if conf is not None and conf.shape[1] >= 2:
                pred_df['ci_lower'] = conf.iloc[:,0].values
                pred_df['ci_upper'] = conf.iloc[:,1].values

            st.write(pred_df)

            # Plot combined
            fig2, ax2 = plt.subplots()
            if df is not None and not df.empty:
                ax2.plot(df['tanggal_lengkap'], df['cabe_merah_besar'], label='historis')
            ax2.plot(pred_df['tanggal_lengkap'], pred_df['predicted_mean'], marker='o', label='prediksi')
            if "ci_lower" in pred_df.columns and "ci_upper" in pred_df.columns:
                ax2.fill_between(
                    pred_df["tanggal_lengkap"],
                    pred_df["ci_lower"],
                    pred_df["ci_upper"],
                    alpha=0.2
                )
ax2.set_title("Prediksi SARIMA")
ax2.set_xlabel("Tanggal")
ax2.set_ylabel("Harga")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

except Exception as e:
    st.error(f"Plot error: {e}")
