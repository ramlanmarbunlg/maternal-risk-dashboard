# 🏥 Dashboard Sistem Monitoring Risiko Kesehatan Ibu & Rekomendasi Pelatihan SDM Kesehatan

Aplikasi web berbasis **Streamlit** dan **Machine Learning** yang dirancang sebagai Sistem Pendukung Keputusan Klinis (*Clinical Decision Support System / CDSS*). Aplikasi ini berfungsi untuk mengklasifikasikan tingkat risiko kesehatan ibu hamil (`Low Risk` vs `High Risk`), menyediakan fitur **Simulasi Intervensi Klinis (*What-If Analysis*)** secara *real-time*, serta merekomendasikan prioritas modul pelatihan bagi Tenaga Kesehatan (Bidan/Puskesmas) berbasis bobot *Feature Importance*.

---

## 🌟 Fitur Utama Dashboard

Dashboard ini terbagi menjadi **4 Tab Utama**:

1. **📊 Overview & Exploratory Data Analysis (EDA)**
   - Ringkasan metrik sampel dataset kesehatan ibu hamil.
   - Visualisasi distribusi tingkat risiko (*Risk Level*) dan indikator medis (misal: Gula Darah / BS).
   - Pratinjau tabel dataset interaktif dengan kontrol penyesuaian baris data.

2. **🤖 Performa Model Machine Learning**
   - Perbandingan metrik evaluasi (*Accuracy, Precision, Recall, F1-Score, AUC-ROC*) dari 4 algoritma:
     - *Logistic Regression* (dengan *StandardScaler*)
     - *Decision Tree*
     - *Random Forest*
     - *Gradient Boosting (XGBoost)*
   - Visualisasi *Interactive Confusion Matrix* menggunakan Plotly.
   - Grafik perbandingan Kurva ROC (*Receiver Operating Characteristic*).

3. **🔮 Simulasi Intervensi Klinis (What-If Analysis) & Diagnosis**
   - Form input parameter fisiologis pasien (Usia, Tekanan Darah, Gula Darah, BMI, Riwayat Komplikasi, dll.).
   - Prediksi status risiko pasien (`Low Risk` vs `High Risk`) beserta persentase probabilitasnya.
   - **Simulasi Intervensi Klinis (*What-If Analysis*):** Slider interaktif *real-time* untuk menguji dampak penurunan/kenaikan gula darah dan tekanan darah terhadap estimasi risiko pasien.
   - Fitur **Ekspor Laporan (.txt)** ringkasan diagnosis pasien.

4. **🎯 Rekomendasi Prioritas Pelatihan SDM Kesehatan**
   - Analisis kontribusi variabel klinis (*Feature Importance*) dari model *Random Forest* dan *Gradient Boosting*.
   - Matriks pemetaan modul pelatihan berprioritas tinggi untuk meningkatkan kapasitas Bidan Desa dan Petugas KIA Puskesmas.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python 3.10+
- **Framework Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Processing & Analytics:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost
- **Visualisasi Data:** Plotly (Express & Graph Objects)

---

## 📂 Struktur Repositori

```text
maternal-risk-dashboard/
│
├── .venv/                                # Virtual Environment (ignored)
├── .gitignore                            # Berkas pengecualian Git
├── app.py                                # Kode utama aplikasi Streamlit
├── maternal_health_cleaned_missing_handled.csv  # Dataset kesehatan ibu hamil
├── requirements.txt                      # Daftar pustaka Python/dependencies
└── README.md                             # Dokumentasi proyek