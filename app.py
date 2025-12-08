# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import os

st.set_page_config(page_title="Prediksi Harga Cabai - SARIMA & XGBoost", layout="wide")

# ---------- Helpers ----------
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

def mape(a, b):
    return np.mean(np.abs((a - b) / a)) * 100

def make_lag_features(series, lags=[1,7,14]):
    df = pd.DataFrame({"y": series})
    for l in lags:
        df[f"lag_{l}"] = series.shift(l)
    df = df.dropna()
    return df

def forecast_sarima_train_predict(train_series, test_len, order=(1,1,1), seasonal_order=(1,1,1,7)):
    model = SARIMAX(train_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(method='powell', maxiter=500, disp=False)
    pred = res.get_forecast(steps=test_len).predicted_mean
    return res, pred

def forecast_sarima_multi(sarima_res, h):
    pred = sarima_res.get_forecast(steps=h).predicted_mean
    return pred

def forecast_xgb_train_predict(train_df, test_df, feature_cols):
    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.7, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, pred

def forecast_xgb_multi(xgb_model, history_series, h, lags=[1,7,14]):
    # iterative prediction: update lag features with predicted values
    history = list(history_series.copy())
    preds = []
    for i in range(h):
        # build feature vector: lag_1 is last value, lag_7 is value 7 steps back, etc.
        vec = []
        for l in lags:
            if len(history) >= l:
                vec.append(history[-l])
            else:
                vec.append(np.mean(history))  # pad with mean if not enough history
        vec = np.array(vec).reshape(1, -1)
        pred = xgb_model.predict(vec)[0]
        preds.append(pred)
        history.append(pred)
    return pd.Series(preds)

# ---------- UI ----------
st.markdown("<h1 style='color:#b30000'>Prediksi Harga Cabai Merah Besar — SARIMA & XGBoost</h1>", unsafe_allow_html=True)
st.write("Upload file CSV dengan format: `tanggal` (kolom pertama) dan `cabe_merah_besar` (kolom harga).")

uploaded = st.file_uploader("Upload CSV (tanggal_lengkap, cabe_merah_besar)", type=["csv"])

horizon = st.number_input("Horizon prediksi (hari)", min_value=1, max_value=90, value=30)
run_btn = st.button("Proses & Prediksi")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, parse_dates=[0], index_col=0)
    except Exception as e:
        st.error("Gagal membaca file. Pastikan kolom pertama adalah tanggal. Error: " + str(e))
        st.stop()

    # Basic validation
    if df.shape[1] < 1:
        st.error("CSV harus memiliki kolom harga 'cabe_merah_besar'.")
        st.stop()

    # Standardize column name
    if "cabe_merah_besar" not in df.columns:
        # try to find a column similar
        st.warning("Kolom 'cabe_merah_besar' tidak ditemukan. Menggunakan kolom pertama sebagai harga.")
        df.columns = ["cabe_merah_besar"] + list(df.columns[1:])

    series = df["cabe_merah_besar"].astype(float).dropna()
    series = series.sort_index()

    st.subheader("Preview Data")
    st.dataframe(df.tail(10))

    if run_btn:
        with st.spinner("Melatih model dan membuat prediksi... (sebentar ya)"):
            # split train/test by time
            train_size = int(len(series) * 0.8)
            train_series = series.iloc[:train_size]
            test_series = series.iloc[train_size:]

            # ------- SARIMA -------
            try:
                sarima_res, sarima_pred_test = forecast_sarima_train_predict(train_series, len(test_series))
                sarima_rmse = rmse(test_series, sarima_pred_test)
                sarima_mae = mean_absolute_error(test_series, sarima_pred_test)
                sarima_mape = mape(test_series.values, sarima_pred_test)
            except Exception as e:
                st.error(f"SARIMA gagal fit: {e}")
                sarima_res, sarima_pred_test = None, None
                sarima_rmse, sarima_mae, sarima_mape = [np.nan]*3

            # ------- XGBoost -------
            # create lag features across full series then split to preserve time order
            ml_df = make_lag_features(series, lags=[1,7,14])
            # align indices
            train_ml = ml_df.iloc[:(len(ml_df) * 80)//100]
            test_ml = ml_df.iloc[(len(ml_df) * 80)//100:]
            feature_cols = ["lag_1", "lag_7", "lag_14"]

            try:
                xgb_model, xgb_pred_test = forecast_xgb_train_predict(train_ml, test_ml, feature_cols)
                # align y_test index
                y_test = test_ml["y"]
                xgb_rmse = rmse(y_test, xgb_pred_test)
                xgb_mae = mean_absolute_error(y_test, xgb_pred_test)
                xgb_mape = mape(y_test.values, xgb_pred_test)
            except Exception as e:
                st.error(f"XGBoost gagal train: {e}")
                xgb_model, xgb_pred_test = None, None
                xgb_rmse, xgb_mae, xgb_mape = [np.nan]*3

            # ------- Multi-step forecast (horizon) -------
            # SARIMA multi-step
            if sarima_res is not None:
                sarima_fc = forecast_sarima_multi(sarima_res, horizon)
            else:
                sarima_fc = pd.Series([np.nan]*horizon, index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D'))

            # XGBoost multi-step
            if xgb_model is not None:
                xgb_fc = forecast_xgb_multi(xgb_model, series, horizon, lags=[1,7,14])
                # build index
                idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
                xgb_fc.index = idx
            else:
                xgb_fc = pd.Series([np.nan]*horizon, index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D'))

            # Save models (optional)
            os.makedirs("models", exist_ok=True)
            if sarima_res is not None:
                dump(sarima_res, "models/sarima.joblib", compress=3)
            if xgb_model is not None:
                dump(xgb_model, "models/xgb.joblib", compress=3)

            # ------- Display results -------
            st.success("Selesai. Hasil di bawah:")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("SARIMA - Evaluasi pada test set")
                st.write(f"MAE: {sarima_mae:.2f}")
                st.write(f"RMSE: {sarima_rmse:.2f}")
                st.write(f"MAPE: {sarima_mape:.2f} %")
            with col2:
                st.subheader("XGBoost - Evaluasi pada test set")
                st.write(f"MAE: {xgb_mae:.2f}")
                st.write(f"RMSE: {xgb_rmse:.2f}")
                st.write(f"MAPE: {xgb_mape:.2f} %")

            # Plot historical + forecasts
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(series.index, series.values, label="Historis", color="#8b0000")
            # overlay test true values
            ax.plot(test_series.index, test_series.values, label="Actual (test)", color="#ff7f7f")
            if sarima_fc is not None:
                ax.plot(sarima_fc.index, sarima_fc.values, linestyle="--", label="SARIMA Forecast", color="#d62728")
            if xgb_fc is not None:
                ax.plot(xgb_fc.index, xgb_fc.values, linestyle="--", label="XGBoost Forecast", color="#ff4500")
            ax.set_title("Harga Historis dan Prediksi")
            ax.legend()
            st.pyplot(fig)

            # Show forecast tables
            st.subheader("Hasil Prediksi (per model)")
            df_out = pd.DataFrame({
                "date": sarima_fc.index,
                "sarima_forecast": sarima_fc.values,
                "xgb_forecast": xgb_fc.values
            }).set_index("date")
            st.dataframe(df_out)

            # Download button
            csv_bytes = df_out.to_csv().encode('utf-8')
            st.download_button(label="Download hasil prediksi (CSV)", data=csv_bytes, file_name="forecast_cabe.csv", mime="text/csv")

else:

    st.info("Silakan upload dataset CSV terlebih dahulu.")
