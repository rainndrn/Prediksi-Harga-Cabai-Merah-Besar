import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import plotly.express as px

# ============================
# APP BASIC CONFIG
# ============================
st.set_page_config(
    page_title="Prediksi Harga Cabai Merah Besar",
    page_icon="🌶️",
    layout="wide"
)

st.title("🌶️ Prediksi Harga Cabai Merah Besar")
st.markdown("Menggunakan **XGBoost Regression**")

# ============================
# LOAD DATA
# ============================
FILE_PATH = "cabai merah besar.csv"  # ubah kalau nama file berbeda

@st.cache_data
def load_data():
    df = pd.read_csv(FILE_PATH)
    df['tanggal_lengkap'] = pd.to_datetime(df['tanggal_lengkap'])
    df = df.sort_values('tanggal_lengkap')
    return df

df = load_data()

st.subheader("📊 Data Historis")
st.dataframe(df.tail())

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

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

future_pred = model.predict(future_df[["day","month","year"]])
future_df["prediksi_cabai"] = future_pred

# ============================
# VISUALISASI
# ============================
st.subheader("📈 Grafik Prediksi")

fig = px.line(df, x="tanggal_lengkap", y="cabe_merah_besar",
              title="Data Historis",
              color_discrete_sequence=["red"])
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(future_df, x="tanggal_lengkap", y="prediksi_cabai",
               title="Prediksi Harga Cabai Merah Besar",
               color_discrete_sequence=["darkred"])
st.plotly_chart(fig2, use_container_width=True)

st.success("Prediksi selesai!")

st.write("📌 Model: XGBoost Regression")
st.write("👩‍💻 Dibuat dengan Streamlit")
