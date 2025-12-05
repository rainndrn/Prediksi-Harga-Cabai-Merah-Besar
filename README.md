# Prediksi Harga Cabai Merah Besar di Kabupaten Bekasi  
Menggunakan Metode SARIMA & LSTM – Aplikasi Streamlit

Aplikasi ini dibuat untuk memprediksi harga **cabai merah besar** di Kabupaten Bekasi menggunakan dua pendekatan model:

- **SARIMA** (Statistical Time Series Model)
- **LSTM** (Deep Learning Recurrent Neural Network)

Aplikasi dapat dijalankan secara lokal atau dideploy langsung ke **Streamlit Cloud**.

---

## 🚀 Fitur Aplikasi
- Upload dataset CSV (bebas separator: `,`, `;`, `\t`)
- Preprocessing otomatis (interpolasi, resampling harian)
- Visualisasi data historis harga cabai
- Pilihan model:
  - SARIMA
  - LSTM
  - Keduanya
- Prediksi hingga 365 hari ke depan
- Download hasil prediksi dalam format CSV
- Retrain model SARIMA & LSTM kapan saja
- Penyimpanan model otomatis ke folder `models/`

---

## 📦 Struktur Repository
