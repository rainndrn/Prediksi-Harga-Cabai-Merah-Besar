import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import plotly.express as px

# ============================
# APP CONFIG
# ============================
st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Besar",
    page_icon="🌶️",
    layout="wide"
)

st.title("🌶️ Prediksi Harga Cabai Merah Besar")
st.markdown("Menggunakan **XGBoost Regression**")

# ============================
# Upload Data
# ============================
uploaded_file = st.file_uploader("📂 Upload Dataset CSV", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # rapikan nama kolom
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # cek kolom wajib
    required_cols = ["tanggal_lengkap", "cabe_merah_besar"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Kolom '{col}' tidak ditemukan dalam file CSV ❌")
            st.stop()

    df["tanggal_lengkap"] = pd.to_datetime(df["tanggal_lengkap"], dayfirst=True, errors="coerce")

    if df["tanggal_lengkap"].isna().any():
        st.warning("⚠ Format tanggal ada yang error. Pastikan format contoh: 01-01-2023")

    df = df.sort_values("tanggal_lengkap")
    return df

if uploaded_file is None:
    st.info("👆 Silakan upload file CSV terlebih dahulu.")
    st.stop()

df = load_data(uploaded_file)

st.subheader("📊 Data Historis")
st.dataframe(df.tail())

# ============================
# FEATURE ENGINEERING
# ============================
df["day"] = df['tanggal_lengkap'].dt.day
df["month"] = df['tanggal_lengkap'].dt.month
df["year"] = df['tanggal_lengkap'].dt.year

X = df[["year", "month", "day"]]
y = df["cabe_merah_besar"]

# ============================
# TRAIN MODEL
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

st.write(f"**MAE**: {mae:.2f}")
st.write(f"**RMSE**: {rmse:.2f}")

# ============================
# FORECASTING
# ============================
forecast_horizon = st.slider("🗓️ Prediksi berapa hari ke depan?", 1, 30, 7)

last_date = df['tanggal_lengkap'].max()
future_dates = pd.date_range(last_date, periods=forecast_horizon+1, closed='right')

future_df = pd.DataFrame({
    "tanggal_lengkap": future_dates,
    "day": future_dates.day,
    "month": future_dates.month,
    "year": future_dates.year
})

future_pred = model.predict(future_df[["day", "month", "year"]])
future_df["prediksi_cabai"] = future_pred

# ============================
# PLOTTING
# ============================
st.subheader("📈 Grafik Prediksi")

fig = px.line(df, x="tanggal_lengkap", y="cabe_merah_besar", title="Data Historis", color_discrete_sequence=["red"])
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(future_df, x="tanggal_lengkap", y="prediksi_cabai", title="Prediksi Harga", color_discrete_sequence=["darkred"])
st.plotly_chart(fig2, use_container_width=True)

st.success("Prediksi Selesai! 🎉")
st.write("👩‍💻 Model: XGBoost Regression")

