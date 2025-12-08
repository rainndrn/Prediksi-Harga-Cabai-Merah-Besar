# Prediksi Harga Cabai Merah Besar - Deploy Streamlit (SARIMA)


Instruksi singkat:
1. Taruh file `model_sarima.pkl.gz` di folder `model/`.
2. (Optional) taruh sample data csv di `data/historical_prices.csv` dengan dua kolom: `date`, `price`.
3. Buat repo di GitHub, commit semua file.
4. Di Streamlit Community Cloud, hubungkan repo dan deploy.


Notes:
- Jika file model kamu berisi objek `SARIMAXResults` dari statsmodels, `get_forecast(steps)` akan bekerja.
- App menyediakan slider untuk memilih horizon prediksi (1-90 hari) — default 14.
