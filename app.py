import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# ============================
# CONFIG UI
# ============================
st.set_page_config(page_title="Prediksi Harga Cabai 🌶️",
                   page_icon="🌶️",
                   layout="wide")

st.markdown("<h1 style='color:#b30000'>🌶️ Prediksi Harga Cabai Merah Besar</h1>", unsafe_allow_html=True)
st.write("Menggunakan **XGBoost Regression**")

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # rapikan kolom agar bebas spasi & huruf kecil semua
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # cek apakah kolom wajib ada
    required_cols = ["tanggal_lengkap", "cabe_merah_besar"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"Kolom berikut tidak ditemukan dalam file CSV: {missing}")
        st.stop()

    df["tanggal_lengkap"] = pd.to_datetime(df["tanggal_lengkap"], dayfirst=True, errors="coerce")

    if df["tanggal_lengkap"].isna().any():
        st.warning("⚠ Ada tanggal yang gagal diparsing. Pastikan format tanggal benar, misal: 01-01-2023")

    df = df.sort_values("tanggal_lengkap")
    return df

# ============================
# FEATURE ENGINEERING
# ============================
df["day"] = df['tanggal_lengkap'].dt.day
df["month"] = df['tanggal_lengkap'].dt.month
df["year"] = df['tanggal_lengkap'].dt.year

X = df[["day", "month", "year"]]
y = df["cabe_merah_besar"]

# ============================
# TRAIN MODEL
# ============================
train_ratio = 0.8
train_size = int(len(df) * train_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.07,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("MAPE", f"{mape:.2f}%")

# ============================
# FORECASTING
# ============================
st.subheader("📅 Prediksi Masa Depan")
forecast_horizon = st.slider("Prediksi berapa hari ke depan?", 7, 90, 30)

last_date = df['tanggal_lengkap'].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)

future_df = pd.DataFrame({
    "tanggal_lengkap": future_dates,
    "day": future_dates.day,
    "month": future_dates.month,
    "year": future_dates.year
})

future_pred = model.predict(future_df[["day", "month", "year"]])
future_df["prediksi_cabai"] = future_pred

# Gabungkan untuk satu grafik
combined = pd.concat([
    df[['tanggal_lengkap', 'cabe_merah_besar']],
    future_df[['tanggal_lengkap', 'prediksi_cabai']]
])

# ============================
# VISUALISASI
# ============================
st.subheader("📈 Grafik Historis & Prediksi")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['tanggal_lengkap'], y=df['cabe_merah_besar'],
                         mode='lines', name='Historis', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_df['tanggal_lengkap'], y=future_df['prediksi_cabai'],
                         mode='lines', name='Prediksi', line=dict(color='darkred', dash='dot')))
fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ============================
# DOWNLOAD PREDIKSI
# ============================
st.subheader("📥 Download Hasil Prediksi")
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV Prediksi 🌶️",
                   data=csv,
                   file_name="prediksi_cabai.csv",
                   mime="text/csv")

st.success("Prediksi selesai!")
st.caption("👩‍💻 Model: XGBoost Regression | Dibuat dengan Streamlit")


