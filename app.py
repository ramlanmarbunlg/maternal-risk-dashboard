import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Monitoring Risk Maternal & SDM Kesehatan",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Subtitle
st.title("🏥 Dashboard Sistem Monitoring Risiko Kesehatan Ibu & Rekomendasi Pelatihan SDM Kesehatan")
st.markdown("""
Aplikasi prototipe ini memanfaatkan **Machine Learning** untuk mengklasifikasikan risiko kesehatan ibu hamil (`Low Risk` vs `High Risk`), 
menyediakan analisis interaktif **What-If Analysis**, serta merekomendasikan fokus materi pelatihan bagi Tenaga Kesehatan berbasis bobot *Feature Importance*.
""")

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('maternal_health_cleaned_missing_handled.csv', sep=';')
    return df

df = load_data()

@st.cache_resource
def train_models_and_evaluate(df):
    df_copy = df.copy()
    df_copy['RiskLevel_Binary'] = df_copy['Risk Level'].map({'Low': 0, 'High': 1})
    
    X = df_copy.drop(columns=['Risk Level', 'RiskLevel_Binary'])
    y = df_copy['RiskLevel_Binary']
    
    # Train Test Split (80:20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling Features for distance/linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting (XGBoost)': GradientBoostingClassifier(random_state=42)
    }
    
    trained_models = {}
    metrics_list = []
    eval_details = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        trained_models[name] = model
        
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        metrics_list.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC-ROC': auc
        })
        
        # Save Details for Confusion Matrix & ROC
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        eval_details[name] = {
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'auc': auc,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
    metrics_df = pd.DataFrame(metrics_list)
    return trained_models, metrics_df, eval_details, feature_cols, scaler

feature_cols = df.drop(columns=['Risk Level']).columns
trained_models, metrics_df, eval_details, feature_cols, scaler = train_models_and_evaluate(df)

# -----------------------------------------------------------------------------
# 3. NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & EDA", 
    "🤖 Performa Model ML", 
    "🔮 Simulasi Intervensi Klinis (What-If Analysis) & Diagnosis", 
    "🎯 Rekomendasi Pelatihan SDM"
])

# =============================================================================
# TAB 1: OVERVIEW & EDA
# =============================================================================
with tab1:
    st.header("Visualisasi & Eksplorasi Data Kesehatan Ibu")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel Data", f"{len(df)} Ibu Hamil")
    col2.metric("Risiko Rendah (Low Risk)", f"{(df['Risk Level']=='Low').sum()} Ibu", delta="60.1%", delta_color="normal")
    col3.metric("Risiko Tinggi (High Risk)", f"{(df['Risk Level']=='High').sum()} Ibu", delta="39.9%", delta_color="inverse")
    
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Distribusi Target (Risk Level)")
        risk_counts = df['Risk Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Jumlah']
        
        fig_eda = px.bar(
            risk_counts, 
            x='Risk Level', 
            y='Jumlah',
            color='Risk Level',
            color_discrete_map={'Low': '#2ecc71', 'High': '#e74c3c'},
            text='Jumlah',
            title="Jumlah Pasien Berdasarkan Tingkat Risiko"
        )
        fig_eda.update_traces(textposition='outside')
        fig_eda.update_layout(showlegend=False, yaxis_title="Jumlah Pasien", xaxis_title="Risk Level")
        st.plotly_chart(fig_eda, use_container_width=True)
        
    with col_chart2:
        st.subheader("Distribusi Gula Darah berdasarkan Risk Level")
        fig_box = px.box(
            df, 
            x='Risk Level', 
            y='BS', 
            color='Risk Level',
            color_discrete_map={'Low': '#2ecc71', 'High': '#e74c3c'},
            title="Saran Distribusi Gula Darah (mmol/L)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Eksplorasi Dataset Maternal Health")

    # Pilihan jumlah baris
    num_rows = st.slider("Tampilkan jumlah baris data:", min_value=5, max_value=len(df), value=20, step=5)

    # Tampilkan data sesuai jumlah pilihan
    st.dataframe(df.head(num_rows), use_container_width=True, height=400)
    st.caption(f"Menampilkan {num_rows} dari total {len(df)} baris data.")

# =============================================================================
# TAB 2: MODEL PERFORMANCE (WITH CONFUSION MATRIX & ROC CURVE)
# =============================================================================
with tab2:
    st.header("Perbandingan Performa 4 Algoritma Machine Learning")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.subheader("Visualisasi Metrik Evaluasi")
        metrics_melted = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        fig_metrics = px.bar(
            metrics_melted,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title="Perbandingan Metrik Evaluasi Per Model",
            range_y=[0.8, 1.05],
            text_auto='.3f'
        )
        fig_metrics.update_layout(yaxis_title="Skor Evaluasi", xaxis_title="Algoritma")
        st.plotly_chart(fig_metrics, use_container_width=True)

    with col_perf2:
        st.subheader("Analisis Interactive Confusion Matrix")
        selected_cm_model = st.selectbox("Pilih Model untuk Confusion Matrix:", list(trained_models.keys()), key="cm_select")
        
        cm = eval_details[selected_cm_model]['cm']
        cm_df = pd.DataFrame(cm, index=['Low Risk (Aktual)', 'High Risk (Aktual)'], columns=['Low Risk (Pred)', 'High Risk (Pred)'])
        
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale='Blues',
            title=f"Confusion Matrix - {selected_cm_model}"
        )
        fig_cm.update_layout(xaxis_title="Prediksi Model", yaxis_title="Kondisi Real Pasien")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.subheader("Kurva ROC (Receiver Operating Characteristic)")
    fig_roc = go.Figure()
    
    for name in trained_models.keys():
        fpr = eval_details[name]['fpr']
        tpr = eval_details[name]['tpr']
        auc_val = eval_details[name]['auc']
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC = {auc_val:.3f})"))
        
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), showlegend=False))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="Perbandingan Kurva ROC Seluruh Model")
    st.plotly_chart(fig_roc, use_container_width=True)

# =============================================================================
# TAB 3: PREDICTION FORM & WHAT-IF ANALYSIS & REPORT EXPORT (FIXED REAL-TIME)
# =============================================================================
with tab3:
    st.header("Form Simulasi Prediksi Risiko & Analysis Interaktif What-If")
    st.markdown("Input parameter medis ibu hamil di bawah ini untuk memperoleh diagnosis risiko secara presisi:")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        age = st.number_input("Usia (Tahun)", 10, 60, 25, key="input_age")
        sys_bp = st.number_input("Systolic BP (mmHg)", 70, 200, 110, key="input_sys")
        dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 140, 70, key="input_dia")
        bs = st.number_input("Gula Darah / BS (mmol/L)", 3.0, 25.0, 7.5, key="input_bs")
        
    with col_b:
        body_temp = st.number_input("Suhu Tubuh (°F)", 95, 105, 98, key="input_temp")
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", 10.0, 50.0, 23.0, key="input_bmi")
        heart_rate = st.number_input("Detak Jantung (Heart Rate)", 50, 140, 74, key="input_hr")
        
    with col_c:
        prev_comp = st.selectbox("Riwayat Komplikasi Sblmnya", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", key="input_prev")
        pre_diab = st.selectbox("Diabetes Bawaan (Preexisting)", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", key="input_pre")
        gest_diab = st.selectbox("Diabetes Kehamilan (Gestational)", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", key="input_gest")
        mental_health = st.selectbox("Gangguan Kesehatan Mental", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak", key="input_mental")

    selected_model_name = st.selectbox("Pilih Model Machine Learning:", list(trained_models.keys()), index=2, key="pred_model_select")
    
    # Inisialisasi state jika tombol belum diklik
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    if st.button("🔮 Analisis Prediksi Risiko Pasien", type="primary"):
        st.session_state.analyzed = True

    # Render Hasil & What-If Analysis hanya jika tombol sudah pernah diklik
    if st.session_state.analyzed:
        input_data = pd.DataFrame([[
            age, sys_bp, dia_bp, bs, body_temp, bmi, 
            prev_comp, pre_diab, gest_diab, mental_health, heart_rate
        ]], columns=feature_cols)

        model = trained_models[selected_model_name]
        
        if selected_model_name == 'Logistic Regression':
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
        else:
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        
        # Display Result Card Utama
        st.subheader("📋 Hasil Diagnosis Baseline")
        if pred == 1:
            st.error(f"⚠️ **STATUS DIAGNOSIS: HIGH RISK (RISIKO TINGGI)** | Estimasi Probabilitas Komplikasi: **{prob*100:.2f}%**")
            st.warning("**Rekomendasi Medis:** Lakukan rujukan segera ke RSUD / Dokter Spesialis Kandungan, evaluasi profil metabolik, serta pantau gula darah intensif.")
        else:
            st.success(f"✅ **STATUS DIAGNOSIS: LOW RISK (RISIKO RENDAH)** | Estimasi Probabilitas Risiko Tinggi: **{prob*100:.2f}%**")
            st.info("**Rekomendasi Medis:** Lanjutkan perawatan antenatal (ANC) rutin di Puskesmas atau Posyandu setempat.")
            
        # WHAT-IF ANALYSIS SECTION (PERSISTENT & REAL-TIME)
        st.markdown("---")
        st.subheader("💡 Simulasi Interaktif What-If (Edukasi & Intervensi Fisiologis)")
        st.caption("Geser slider di bawah untuk melihat perubahan estimasi risiko secara langsung tanpa menghilangkan tampilan:")
        
        col_wi1, col_wi2 = st.columns(2)
        with col_wi1:
            sim_bs = st.slider("Simulasi Penurunan/Kenaikan Gula Darah (BS mmol/L):", 3.0, 25.0, float(bs), key="slider_bs")
            sim_sys_bp = st.slider("Simulasi Penurunan/Kenaikan Tekanan Darah Sistolik (mmHg):", 70, 200, int(sys_bp), key="slider_sys")
            
        with col_wi2:
            sim_input = input_data.copy()
            sim_input['BS'] = sim_bs
            sim_input['Systolic BP'] = sim_sys_bp
            
            if selected_model_name == 'Logistic Regression':
                sim_input_scaled = scaler.transform(sim_input)
                sim_pred = model.predict(sim_input_scaled)[0]
                sim_prob = model.predict_proba(sim_input_scaled)[0][1]
            else:
                sim_pred = model.predict(sim_input)[0]
                sim_prob = model.predict_proba(sim_input)[0][1]
                
            status_text = "HIGH RISK" if sim_pred == 1 else "LOW RISK"
            
            st.markdown("#### Hasil Setelah Intervensi:")
            if sim_pred == 1:
                st.error(f"**{status_text}** (Probabilitas Risiko: **{sim_prob*100:.2f}%**)")
            else:
                st.success(f"**{status_text}** (Probabilitas Risiko: **{sim_prob*100:.2f}%**)")
            
            # Indikator Perubahan Probabilitas
            delta_prob = (sim_prob - prob) * 100
            if delta_prob < 0:
                st.info(f"🟢 **Penurunan Risiko:** Intervensi berhasil menurunkan probabilitas risiko sebesar **{abs(delta_prob):.2f}%**.")
            elif delta_prob > 0:
                st.warning(f"🔴 **Peningkatan Risiko:** Perubahan variabel meningkatkan probabilitas risiko sebesar **{delta_prob:.2f}%**.")
            else:
                st.caption("⚪ Belum ada perubahan pada parameter simulasi.")

        # EXPORT SUMMARY REPORT (TXT/CSV DOWNLOAD)
        st.markdown("---")
        st.subheader("📥 Unduh Ringkasan Hasil Diagnosis Pasien")
        
        report_text = f"""==================================================
LAPORAN SUMMARY DIAGNOSIS MATERNAL HEALTH (CDSS)
==================================================
Usia Pasien             : {age} Tahun
Tekanan Darah (S/D)     : {sys_bp}/{dia_bp} mmHg
Gula Darah (BS)         : {bs} mmol/L
Indeks Massa Tubuh (BMI): {bmi} kg/m2
Riwayat Komplikasi      : {'Ya' if prev_comp==1 else 'Tidak'}
--------------------------------------------------
Model Diagnostik        : {selected_model_name}
Hasil Diagnostik        : {'HIGH RISK' if pred==1 else 'LOW RISK'}
Probabilitas Risiko     : {prob*100:.2f}%
=================================================="""

        st.download_button(
            label="📄 Unduh Ringkasan Diagnosis (.txt)",
            data=report_text,
            file_name=f"Diagnosis_Pasien_Usia_{age}.txt",
            mime="text/plain"
        )

# =============================================================================
# TAB 4: SDM TRAINING RECOMMENDATIONS
# =============================================================================
with tab4:
    st.header("🎯 Rekomendasi Prioritas Pelatihan SDM Kesehatan (Bidan/Puskesmas)")
    st.markdown("""
    Pengalokasian kapasitas pelatihan SDM Kesehatan didasarkan pada **bobot kontribusi variabel klinis (Feature Importance)** 
    yang dihitung secara matematis oleh algoritma Machine Learning.
    """)
    
    rf_importance = trained_models['Random Forest'].feature_importances_
    gb_importance = trained_models['Gradient Boosting (XGBoost)'].feature_importances_
    
    fi_df = pd.DataFrame({
        'Indikator Klinis': feature_cols,
        'Random Forest (%)': rf_importance * 100,
        'Gradient Boosting (%)': gb_importance * 100
    }).sort_values(by='Random Forest (%)', ascending=True)
    
    st.subheader("📊 Visualisasi Kontribusi Variabel Klinis (Feature Importance)")
    selected_fi_model = st.radio(
        "Tampilkan Bobot Variabel Berdasarkan Model:",
        ['Random Forest (%)', 'Gradient Boosting (%)'],
        horizontal=True
    )
    
    fig_fi = px.bar(
        fi_df,
        y='Indikator Klinis',
        x=selected_fi_model,
        orientation='h',
        text_auto='.1f',
        title=f"Tingkat Kepentingan Variabel Medis ({selected_fi_model})",
        color=selected_fi_model,
        color_continuous_scale='Reds'
    )
    fig_fi.update_layout(xaxis_title="Persentase Kontribusi (%)", yaxis_title="Variabel Medis", height=500)
    st.plotly_chart(fig_fi, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📚 Matriks Rekomendasi Modul Pelatihan SDM Kesehatan")
    
    col_mod1, col_mod2, col_mod3 = st.columns(3)
    
    with col_mod1:
        st.info("### Modul 1: Skrining Diabetes & Metabolik\n"
                "**Prioritas Utama (~43%-45% Kontribusi)**\n\n"
                "* **Variabel Kunci:** `Preexisting Diabetes`, `Gestational Diabetes`, `BS` (Gula Darah).\n"
                "* **Target Peserta:** Bidan Desa & Ahli Gizi Puskesmas.\n"
                "* **Materi Inti:** Skrining TTGO, manajemen insulin dasar, serta konseling diet rendah glikemik.")
        
    with col_mod2:
        st.warning("### Modul 2: Manajemen Obesitas & BMI Kehamilan\n"
                 "**Prioritas Kedua (~17%-24% Kontribusi)**\n\n"
                 "* **Variabel Kunci:** `BMI` (Indeks Massa Tubuh).\n"
                 "* **Target Peserta:** Petugas KIA & Kader Kesehatan Posyandu.\n"
                 "* **Materi Inti:** Pemantauan *Weight Gain Chart* selama gestasi, edukasi nutrisi, serta deteksi risiko sindrom metabolik.")
        
    with col_mod3:
        st.success("### Modul 3: Vital Sign & Kesehatan Mental\n"
                   "**Prioritas Ketiga (~12%-20% Kontribusi)**\n\n"
                   "* **Variabel Kunci:** `Heart Rate`, `Mental Health`, `Systolic/Diastolic BP`.\n"
                   "* **Target Peserta:** Dokter Puskesmas & Bidan Koordinator.\n"
                   "* **Materi Inti:** Deteksi dini aritmia maternal, penanganan preeklamsia awal, serta instrumen EPDS (*Edinburgh Postnatal Depression Scale*).")
        