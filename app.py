# Import & Page Config
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Monitoring Risk Maternal & SDM Kesehatan",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== GERBANG PASSWORD ====
def check_password():
    """Mengembalikan True jika pengguna memasukkan password yang benar."""
    
    # Jika password belum diatur di secrets, lewati gerbang (untuk development lokal)
    if "APP_PASSWORD" not in st.secrets:
        return True

    # Jika sudah login, kembalikan True
    if st.session_state.get("authenticated"):
        return True

    # Tampilkan form login
    st.title("🔒 Akses Terbatas")
    st.markdown("Silakan masukkan password untuk mengakses dashboard.")
    
    pwd = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Masuk", type="primary"):
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun() # Muat ulang halaman setelah login berhasil
        else:
            st.error("Password salah. Silakan coba lagi.")
    
    return False

# Jalankan gerbang password. Jika gagal, hentikan aplikasi.
if not check_password():
    st.stop()

st.title("Dashboard Sistem Monitoring Risiko Kesehatan Ibu & Rekomendasi Pelatihan SDM Kesehatan")
st.markdown("""
Aplikasi prototipe ini memanfaatkan **Machine Learning** untuk mengklasifikasikan risiko kesehatan ibu hamil (`Low Risk` vs `High Risk`), 
menyediakan analisis interaktif **What-If Analysis**, serta merekomendasikan fokus materi pelatihan bagi Tenaga Kesehatan berbasis bobot *Feature Importance*.

> ⚠️ **Disclaimer:** Aplikasi ini adalah **prototipe penelitian**, bukan alat diagnostik klinis. Hasil prediksi tidak menggantikan penilaian medis profesional.
""")

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CLEANING
# -----------------------------------------------------------------------------
TARGET = 'Risk Level'
NUM_COLS = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Heart Rate']
BIN_COLS = ['Previous Complications', 'Preexisting Diabetes',
            'Gestational Diabetes', 'Mental Health']
UNITS = {
    'Age': 'tahun', 'Systolic BP': 'mmHg', 'Diastolic': 'mmHg', 'BS': 'mmol/L',
    'Body Temp': '°F', 'BMI': 'kg/m²', 'Heart Rate': 'bpm',
}
BMI_MIN_VALID = 10.0


def clean_data(raw: pd.DataFrame):
    """Membersihkan data & mengembalikan (df_bersih, log_pembersihan)."""
    df = raw.copy()
    df.columns = df.columns.str.strip()
    log = {'n_awal': len(df), 'n_missing': int(df.isna().sum().sum())}

    # 1) Hapus data ganda
    log['n_duplikat'] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 2) BMI tidak valid → imputasi median
    bmi_invalid = df['BMI'] < BMI_MIN_VALID
    log['n_bmi_invalid'] = int(bmi_invalid.sum())
    if bmi_invalid.any():
        median_valid = df.loc[~bmi_invalid, 'BMI'].median()
        df.loc[bmi_invalid, 'BMI'] = median_valid
        log['bmi_median_imputasi'] = round(float(median_valid), 2)

    # 3) Validasi domain
    log['n_sys_le_dia'] = int((df['Systolic BP'] <= df['Diastolic']).sum())

    log['n_akhir'] = len(df)
    return df, log


@st.cache_data
def load_data():
    raw = pd.read_csv('maternal_health_cleaned_missing_handled.csv', sep=';')
    return clean_data(raw)


df, cleaning_log = load_data()

# -----------------------------------------------------------------------------
# 3. FUNGSI BANTU STATISTIK
# -----------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace('.', ',')


def describe_numeric(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUM_COLS:
        s = d[c]
        rows.append({
            'Variabel': c, 'Satuan': UNITS[c],
            'Mean': round(s.mean(), 2), 'SD': round(s.std(), 2),
            'Min': round(s.min(), 2), 'Q1': round(s.quantile(.25), 2),
            'Median': round(s.median(), 2), 'Q3': round(s.quantile(.75), 2),
            'Max': round(s.max(), 2),
        })
    return pd.DataFrame(rows)


def describe_binary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in BIN_COLS:
        n_ya = int(d[c].sum())
        rows.append({'Variabel': c, 'n "Ya"': n_ya,
                     '% "Ya"': round(n_ya / len(d) * 100, 1),
                     'n "Tidak"': len(d) - n_ya})
    return pd.DataFrame(rows)


def describe_by_risk(d: pd.DataFrame) -> pd.DataFrame:
    low, high = d[d[TARGET] == 'Low'], d[d[TARGET] == 'High']
    rows = []
    for c in NUM_COLS:
        _, p = stats.mannwhitneyu(low[c], high[c], alternative='two-sided')
        rows.append({'Variabel': c,
                     'Low Risk': f"{low[c].mean():.2f} ± {low[c].std():.2f}",
                     'High Risk': f"{high[c].mean():.2f} ± {high[c].std():.2f}",
                     'Uji': 'Mann-Whitney U', 'p-value': fmt_p(p)})
    for c in BIN_COLS:
        _, p, _, _ = stats.chi2_contingency(pd.crosstab(d[c], d[TARGET]))
        rows.append({'Variabel': c,
                     'Low Risk': f"{int(low[c].sum())} ({low[c].mean()*100:.1f}%)",
                     'High Risk': f"{int(high[c].sum())} ({high[c].mean()*100:.1f}%)",
                     'Uji': 'Chi-square', 'p-value': fmt_p(p)})
    return pd.DataFrame(rows)


def outlier_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in NUM_COLS:
        q1, q3 = d[c].quantile(.25), d[c].quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((d[c] < lo) | (d[c] > hi)).sum())
        rows.append({'Variabel': c, 'Batas Bawah': round(lo, 2), 'Batas Atas': round(hi, 2),
                     'Jumlah Outlier': n_out, '% Outlier': round(n_out / len(d) * 100, 1)})
    return pd.DataFrame(rows)


def corr_frame(d: pd.DataFrame) -> pd.DataFrame:
    out = d.drop(columns=[TARGET]).copy()
    out['Risk Level (High=1)'] = (d[TARGET] == 'High').astype(int)
    return out


def corr_with_target(d: pd.DataFrame, method: str) -> pd.DataFrame:
    cf = corr_frame(d)
    y = cf['Risk Level (High=1)']
    fn = stats.pearsonr if method == 'pearson' else stats.spearmanr
    rows = []
    for c in cf.columns.drop('Risk Level (High=1)'):
        r, p = fn(cf[c], y)
        rows.append({'Variabel': c, 'r': round(float(r), 3), '|r|': round(abs(float(r)), 3),
                     'p-value': fmt_p(float(p))})
    return pd.DataFrame(rows).sort_values('|r|', ascending=False).reset_index(drop=True)

# -----------------------------------------------------------------------------
# 4. TRAINING MODELS & EVALUATION (dengan 5-Fold CV + Hyperparameter Tuning)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_models_and_evaluate(df):
    df_copy = df.copy()
    df_copy['RiskLevel_Binary'] = df_copy[TARGET].map({'Low': 0, 'High': 1})

    X = df_copy.drop(columns=[TARGET, 'RiskLevel_Binary'])
    y = df_copy['RiskLevel_Binary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definisi model & hyperparameter grid
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs']},
            'scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]},
            'scaled': False
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
            'scaled': False
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
            'scaled': False
        }
    }

    trained_models = {}
    metrics_list = []
    eval_details = {}
    cv_results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, cfg in model_configs.items():
        model = cfg['model']
        params = cfg['params']
        use_scaled = cfg['scaled']

        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # Hyperparameter tuning
        grid = GridSearchCV(model, params, cv=cv, scoring='f1', n_jobs=-1)
        grid.fit(X_tr, y_train)
        best_model = grid.best_estimator_

        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_tr, y_train, cv=cv, scoring='accuracy')

        # Prediksi
        y_pred = best_model.predict(X_te)
        y_prob = best_model.predict_proba(X_te)[:, 1]

        trained_models[name] = best_model

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
            'AUC-ROC': auc,
            'CV Accuracy (mean ± SD)': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        })

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        eval_details[name] = {
            'cm': cm, 'fpr': fpr, 'tpr': tpr, 'auc': auc,
            'y_test': y_test, 'y_pred': y_pred, 'y_prob': y_prob,
            'best_params': grid.best_params_,
            'X_test': X_te
        }
        cv_results[name] = cv_scores

    metrics_df = pd.DataFrame(metrics_list)
    return trained_models, metrics_df, eval_details, cv_results, scaler


feature_cols = df.drop(columns=[TARGET]).columns
trained_models, metrics_df, eval_details, cv_results, scaler = train_models_and_evaluate(df)

# -----------------------------------------------------------------------------
# 5. NAVIGATION TABS (5 tabs)
# -----------------------------------------------------------------------------
tab1, tab_desc, tab2, tab3, tab4 = st.tabs([
    "📊 Overview & EDA",
    "📐 Statistik Deskriptif & Korelasi",
    "🤖 Performa Model ML",
    "🔮 Simulasi Intervensi Klinis (What-If)",
    "🎯 Rekomendasi Pelatihan SDM"
])

# =============================================================================
# TAB 1: OVERVIEW & EDA
# =============================================================================
with tab1:
    st.header("Visualisasi & Eksplorasi Data Kesehatan Ibu")

    n_total = len(df)
    n_low = int((df[TARGET] == 'Low').sum())
    n_high = int((df[TARGET] == 'High').sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sampel Data", f"{n_total} Ibu Hamil")
    col2.metric("Risiko Rendah (Low Risk)", f"{n_low} Ibu")
    col2.caption(f"{n_low/n_total*100:.1f}% dari total sampel")
    col3.metric("Risiko Tinggi (High Risk)", f"{n_high} Ibu")
    col3.caption(f"{n_high/n_total*100:.1f}% dari total sampel")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Distribusi Target (Risk Level)")
        risk_counts = df[TARGET].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Jumlah']

        fig_eda = px.bar(
            risk_counts, x='Risk Level', y='Jumlah', color='Risk Level',
            color_discrete_map={'Low': '#2ecc71', 'High': '#e74c3c'},
            text='Jumlah', title="Jumlah Pasien Berdasarkan Tingkat Risiko"
        )
        fig_eda.update_traces(textposition='outside')
        fig_eda.update_layout(showlegend=False, yaxis_title="Jumlah Pasien", xaxis_title="Risk Level")
        st.plotly_chart(fig_eda, use_container_width=True)

    with col_chart2:
        st.subheader("Distribusi Gula Darah berdasarkan Risk Level")
        fig_box = px.box(
            df, x='Risk Level', y='BS', color='Risk Level',
            color_discrete_map={'Low': '#2ecc71', 'High': '#e74c3c'},
            title="Distribusi Gula Darah (mmol/L)"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Eksplorasi Dataset Maternal Health")
    num_rows = st.slider("Tampilkan jumlah baris data:", min_value=5, max_value=len(df), value=20, step=5)
    st.dataframe(df.head(num_rows), use_container_width=True, height=400)
    st.caption(f"Menampilkan {num_rows} dari total {len(df)} baris data.")
    
    # =============================================================================
# TAB 2: STATISTIK DESKRIPTIF & KORELASI
# =============================================================================
with tab_desc:
    st.header("Statistik Deskriptif & Analisis Korelasi")

    with st.expander("🧹 Ringkasan Pembersihan Data (Data Cleaning)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baris Awal", cleaning_log['n_awal'])
        c2.metric("Data Ganda Dihapus", cleaning_log['n_duplikat'])
        c3.metric("BMI Tidak Valid (diimputasi)", cleaning_log['n_bmi_invalid'])
        c4.metric("Sampel Akhir", cleaning_log['n_akhir'])
        st.caption(
            f"Missing value: {cleaning_log['n_missing']} • "
            f"BMI < {BMI_MIN_VALID:.0f} diganti median BMI valid "
            f"({cleaning_log.get('bmi_median_imputasi', '-')}) • "
            f"Baris dengan Systolic ≤ Diastolic: {cleaning_log['n_sys_le_dia']}"
        )

    st.subheader(f"Tabel Statistik Deskriptif (N = {len(df)})")
    st.markdown("**Variabel numerik**")
    st.dataframe(describe_numeric(df), use_container_width=True, hide_index=True)
    st.markdown("**Variabel kategorik (biner)**")
    st.dataframe(describe_binary(df), use_container_width=True, hide_index=True)

    st.subheader("Perbandingan Karakteristik: Low Risk vs High Risk")
    st.dataframe(describe_by_risk(df), use_container_width=True, hide_index=True)
    st.caption("Numerik: rata-rata ± SD, uji Mann-Whitney U. Kategorik: n (%), uji Chi-square. Signifikan bila p < 0,05.")

    st.markdown("---")
    st.subheader("Distribusi Variabel berdasarkan Risk Level")
    var_sel = st.selectbox("Pilih variabel:", NUM_COLS, key="desc_var")
    fig_hist = px.histogram(
        df, x=var_sel, color=TARGET, barmode='overlay', opacity=0.65, marginal='box',
        color_discrete_map={'Low': '#2ecc71', 'High': '#e74c3c'},
        title=f"Distribusi {var_sel} ({UNITS[var_sel]})"
    )
    fig_hist.update_layout(yaxis_title="Frekuensi", xaxis_title=f"{var_sel} ({UNITS[var_sel]})")
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Deteksi Outlier (metode IQR 1,5×)"):
        st.dataframe(outlier_table(df), use_container_width=True, hide_index=True)
        st.caption("Outlier tidak otomatis dihapus; nilai ekstrem pada tekanan darah dan gula darah dapat mencerminkan kondisi klinis nyata.")

    st.markdown("---")
    st.subheader("Heatmap Korelasi Antarvariabel")
    method_label = st.radio("Metode korelasi:", ["Pearson", "Spearman"], horizontal=True, key="corr_method")
    method = method_label.lower()

    corr_mat = corr_frame(df).corr(method=method)
    fig_corr = px.imshow(
        corr_mat, text_auto='.2f', color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1, aspect='auto',
        title=f"Matriks Korelasi {method_label} (termasuk target: High Risk = 1)"
    )
    fig_corr.update_layout(height=620)
    st.plotly_chart(fig_corr, use_container_width=True)

    feat_corr = corr_mat.drop(index='Risk Level (High=1)', columns='Risk Level (High=1)')
    pairs = [(a, b, feat_corr.loc[a, b]) for i, a in enumerate(feat_corr.columns)
             for b in feat_corr.columns[i + 1:] if abs(feat_corr.loc[a, b]) >= 0.7]
    if pairs:
        st.warning("Pasangan fitur berkorelasi tinggi (|r| ≥ 0,7): " +
                   "; ".join(f"{a} – {b} (r = {r:.2f})" for a, b, r in pairs))
    else:
        st.caption("Tidak ada pasangan fitur dengan |r| ≥ 0,7 (indikasi multikolinearitas rendah).")

    st.subheader("Korelasi Setiap Variabel terhadap Risiko Tinggi (High Risk)")
    ct = corr_with_target(df, method)
    col_l, col_r = st.columns([1, 1.2])
    with col_l:
        st.dataframe(ct[['Variabel', 'r', 'p-value']], use_container_width=True, hide_index=True)
        if method == 'pearson':
            st.caption("Pearson dengan target biner = koefisien point-biserial.")
    with col_r:
        ct_plot = ct.sort_values('r')
        fig_ct = px.bar(
            ct_plot, x='r', y='Variabel', orientation='h', text='r',
            color='r', color_continuous_scale='RdBu_r', range_color=[-1, 1],
            title=f"Korelasi {method_label} terhadap Risk Level (High = 1)"
        )
        fig_ct.update_layout(xaxis_title="Koefisien korelasi (r)", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_ct, use_container_width=True)
        
 
# TAB 3: MODEL PERFORMANCE ML (dengan CV, Odds Ratio, Permutation Importance)
with tab2:
    st.header("Perbandingan Performa 4 Algoritma Machine Learning")

    st.subheader("Tabel Metrik Evaluasi (dengan 5-Fold CV)")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Hyperparameter Terbaik per Model")
    for name, details in eval_details.items():
        st.markdown(f"**{name}**: `{details['best_params']}`")

    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        st.subheader("Visualisasi Metrik Evaluasi")
        metrics_melted = metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                                          var_name='Metric', value_name='Score')
        fig_metrics = px.bar(
            metrics_melted, x='Model', y='Score', color='Metric', barmode='group',
            title="Perbandingan Metrik Evaluasi Per Model", range_y=[0.8, 1.05], text_auto='.3f'
        )
        fig_metrics.update_layout(yaxis_title="Skor Evaluasi", xaxis_title="Algoritma")
        st.plotly_chart(fig_metrics, use_container_width=True)

    with col_perf2:
        st.subheader("Analisis Interactive Confusion Matrix")
        selected_cm_model = st.selectbox("Pilih Model untuk Confusion Matrix:", list(trained_models.keys()), key="cm_select")
        cm = eval_details[selected_cm_model]['cm']
        cm_df = pd.DataFrame(cm, index=['Low Risk (Aktual)', 'High Risk (Aktual)'],
                             columns=['Low Risk (Pred)', 'High Risk (Pred)'])
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                           title=f"Confusion Matrix - {selected_cm_model}")
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

    # Odds Ratio dari Logistic Regression
    st.markdown("---")
    st.subheader("Odds Ratio (Logistic Regression)")
    st.caption("Odds Ratio > 1 berarti variabel meningkatkan risiko; < 1 berarti menurunkan risiko.")
    lr_model = trained_models['Logistic Regression']
    odds_ratios = np.exp(lr_model.coef_[0])
    or_df = pd.DataFrame({
        'Variabel': feature_cols,
        'Odds Ratio': odds_ratios
    }).sort_values('Odds Ratio', ascending=False)
    fig_or = px.bar(
        or_df, x='Odds Ratio', y='Variabel', orientation='h', text_auto='.2f',
        color='Odds Ratio', color_continuous_scale='RdBu_r',
        title="Odds Ratio dari Logistic Regression"
    )
    st.plotly_chart(fig_or, use_container_width=True)

    # Permutation Importance
    st.markdown("---")
    st.subheader("Permutation Importance (pada Data Uji)")
    st.caption("Permutation importance mengukur penurunan performa model saat nilai suatu fitur diacak. Lebih netral daripada impurity-based importance.")
    selected_perm_model = st.selectbox("Pilih Model untuk Permutation Importance:", list(trained_models.keys()), key="perm_select")
    model_perm = trained_models[selected_perm_model]
    X_test_perm = eval_details[selected_perm_model]['X_test']
    y_test_perm = eval_details[selected_perm_model]['y_test']

    perm_result = permutation_importance(model_perm, X_test_perm, y_test_perm, n_repeats=10, random_state=42, scoring='f1')
    perm_df = pd.DataFrame({
        'Variabel': feature_cols,
        'Importance Mean': perm_result.importances_mean,
        'Importance SD': perm_result.importances_std
    }).sort_values('Importance Mean', ascending=True)

    fig_perm = px.bar(
        perm_df, x='Importance Mean', y='Variabel', orientation='h',
        error_x='Importance SD', text_auto='.4f',
        title=f"Permutation Importance - {selected_perm_model}"
    )
    st.plotly_chart(fig_perm, use_container_width=True)
    
# TAB 4: SIMULASI INTERVENSI (WHAT-IF) & SKRINING RISIKO 
# (Dengan BMI & HR Slider + Warning)
with tab3:
    st.header("Form Simulasi Prediksi Risiko & Analysis Interaktif What-If")
    st.markdown("Input parameter medis ibu hamil di bawah ini untuk memperoleh **prediksi risiko**:")
    st.warning("⚠️ **Disclaimer:** Hasil ini adalah **prediksi risiko berbasis model ML**, bukan diagnosis klinis. Konsultasikan dengan tenaga kesehatan profesional.")

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

    # Peringatan bila input di luar rentang data latih
    warnings = []
    if heart_rate < df['Heart Rate'].min() or heart_rate > df['Heart Rate'].max():
        warnings.append(f"Heart Rate {heart_rate} di luar rentang data latih ({df['Heart Rate'].min():.0f}–{df['Heart Rate'].max():.0f} bpm).")
    if bmi < df['BMI'].min() or bmi > df['BMI'].max():
        warnings.append(f"BMI {bmi} di luar rentang data latih ({df['BMI'].min():.1f}–{df['BMI'].max():.1f} kg/m²).")
    if sys_bp < df['Systolic BP'].min() or sys_bp > df['Systolic BP'].max():
        warnings.append(f"Systolic BP {sys_bp} di luar rentang data latih ({df['Systolic BP'].min():.0f}–{df['Systolic BP'].max():.0f} mmHg).")
    if warnings:
        st.warning("⚠️ **Peringatan Input:** " + " ".join(warnings) + " Hasil prediksi mungkin kurang akurat.")

    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    if st.button("Analisis Prediksi Risiko Pasien", type="primary"):
        st.session_state.analyzed = True

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
        st.subheader("📋 Hasil Prediksi Baseline")
        if pred == 1:
            st.error(f"⚠️ **STATUS: HIGH RISK (RISIKO TINGGI)** | Estimasi Probabilitas: **{prob*100:.2f}%**")
            st.warning("**Rekomendasi:** Lakukan rujukan segera ke RSUD / Dokter Spesialis Kandungan, evaluasi profil metabolik, serta pantau gula darah intensif.")
        else:
            st.success(f"✅ **STATUS: LOW RISK (RISIKO RENDAH)** | Estimasi Probabilitas Risiko Tinggi: **{prob*100:.2f}%**")
            st.info("**Rekomendasi:** Lanjutkan perawatan antenatal (ANC) rutin di Puskesmas atau Posyandu setempat.")

        # WHAT-IF ANALYSIS
        st.markdown("---")
        st.subheader("💡 Simulasi Interaktif What-If")
        st.caption("Geser slider di bawah untuk melihat perubahan estimasi risiko secara real-time:")
        st.info("ℹ️ **Catatan:** Simulasi ini bersifat **prediktif, bukan kausal**. Perubahan probabilitas tidak berarti intervensi klinis pasti menurunkan risiko pasien.")

        col_wi1, col_wi2 = st.columns(2)
        with col_wi1:
            sim_bs = st.slider("Simulasi Gula Darah (BS mmol/L):", 3.0, 25.0, float(bs), key="slider_bs")
            sim_sys_bp = st.slider("Simulasi Tekanan Darah Sistolik (mmHg):", 70, 200, int(sys_bp), key="slider_sys")
            sim_bmi = st.slider("Simulasi BMI (kg/m²):", 10.0, 50.0, float(bmi), key="slider_bmi")
            sim_hr = st.slider("Simulasi Heart Rate (bpm):", 50, 140, int(heart_rate), key="slider_hr")

        with col_wi2:
            sim_input = input_data.copy()
            sim_input['BS'] = sim_bs
            sim_input['Systolic BP'] = sim_sys_bp
            sim_input['BMI'] = sim_bmi
            sim_input['Heart Rate'] = sim_hr

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
                st.error(f"**{status_text}** (Probabilitas: **{sim_prob*100:.2f}%**)")
            else:
                st.success(f"**{status_text}** (Probabilitas: **{sim_prob*100:.2f}%**)")

            delta_prob = (sim_prob - prob) * 100
            if delta_prob < 0:
                st.info(f"🟢 **Penurunan Risiko:** Probabilitas turun **{abs(delta_prob):.2f}%**.")
            elif delta_prob > 0:
                st.warning(f"🔴 **Peningkatan Risiko:** Probabilitas naik **{delta_prob:.2f}%**.")
            else:
                st.caption("⚪ Belum ada perubahan.")

        # EXPORT
        st.markdown("---")
        st.subheader("📥 Unduh Ringkasan Prediksi")
        report_text = f"""==================================================
LAPORAN SUMMARY PREDIKSI RISIKO MATERNAL (PROTOTIPE CDSS)
==================================================
Usia Pasien             : {age} Tahun
Tekanan Darah (S/D)     : {sys_bp}/{dia_bp} mmHg
Gula Darah (BS)         : {bs} mmol/L
Indeks Massa Tubuh (BMI): {bmi} kg/m2
Heart Rate              : {heart_rate} bpm
Riwayat Komplikasi      : {'Ya' if prev_comp==1 else 'Tidak'}
--------------------------------------------------
Model Prediksi          : {selected_model_name}
Hasil Prediksi          : {'HIGH RISK' if pred==1 else 'LOW RISK'}
Probabilitas Risiko     : {prob*100:.2f}%
--------------------------------------------------
DISCLAIMER: Ini adalah prediksi berbasis model ML, bukan diagnosis klinis.
=================================================="""

        st.download_button(
            label="📄 Unduh Ringkasan Prediksi (.txt)",
            data=report_text,
            file_name=f"Prediksi_Risiko_Usia_{age}.txt",
            mime="text/plain"
        )

# =============================================================================
# TAB 5: REKOMENDASI PELATIHAN SDM (BERBASIS PERHITUNGAN)
# =============================================================================
with tab4:
    st.header("Rekomendasi Prioritas Pelatihan SDM Kesehatan (Bidan/Puskesmas)")
    st.markdown("""
    Pengalokasian kapasitas pelatihan SDM Kesehatan didasarkan pada **indeks kebutuhan pelatihan** 
    yang dihitung dari **bobot variabel (feature importance) × prevalensi faktor pada kelompok High Risk**.
    """)

    # Hitung feature importance
    rf_importance = trained_models['Random Forest'].feature_importances_
    xgb_importance = trained_models['XGBoost'].feature_importances_

    # Hitung prevalensi faktor pada High Risk (untuk variabel biner)
    high_df = df[df[TARGET] == 'High']
    prevalence = {}
    for col in feature_cols:
        if col in BIN_COLS:
            prevalence[col] = high_df[col].mean()
        else:
            # Untuk variabel kontinu, gunakan proporsi di atas median High Risk
            median_high = high_df[col].median()
            prevalence[col] = (high_df[col] > median_high).mean()

    # Hitung indeks kebutuhan pelatihan
    fi_df = pd.DataFrame({
        'Variabel': feature_cols,
        'RF Importance (%)': rf_importance * 100,
        'XGB Importance (%)': xgb_importance * 100,
        'Prevalensi High Risk': [prevalence[c] for c in feature_cols]
    })

    fi_df['Indeks Kebutuhan (RF)'] = fi_df['RF Importance (%)'] * fi_df['Prevalensi High Risk']
    fi_df['Indeks Kebutuhan (XGB)'] = fi_df['XGB Importance (%)'] * fi_df['Prevalensi High Risk']

    fi_df = fi_df.sort_values('Indeks Kebutuhan (RF)', ascending=True)

    st.subheader("Visualisasi Kontribusi Variabel Klinis (Feature Importance)")
    selected_fi_model = st.radio(
        "Tampilkan Bobot Variabel Berdasarkan Model:",
        ['RF Importance (%)', 'XGB Importance (%)'], horizontal=True
    )
    fig_fi = px.bar(
        fi_df, y='Variabel', x=selected_fi_model, orientation='h', text_auto='.1f',
        title=f"Tingkat Kepentingan Variabel Medis ({selected_fi_model})",
        color=selected_fi_model, color_continuous_scale='Reds'
    )
    fig_fi.update_layout(xaxis_title="Persentase Kontribusi (%)", yaxis_title="Variabel Medis", height=500)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Indeks Kebutuhan Pelatihan
    st.markdown("---")
    st.subheader("Indeks Kebutuhan Pelatihan SDM Kesehatan")
    st.caption("Indeks = Feature Importance × Prevalensi faktor pada kelompok High Risk. Semakin tinggi, semakin prioritas.")
    fig_idx = px.bar(
        fi_df, y='Variabel', x='Indeks Kebutuhan (RF)', orientation='h', text_auto='.2f',
        title="Indeks Kebutuhan Pelatihan (Berbasis Random Forest)",
        color='Indeks Kebutuhan (RF)', color_continuous_scale='OrRd'
    )
    fig_idx.update_layout(xaxis_title="Indeks Kebutuhan", yaxis_title="Variabel Medis", height=500)
    st.plotly_chart(fig_idx, use_container_width=True)

    # Matriks Rekomendasi Modul
    st.markdown("---")
    st.subheader("Matriks Rekomendasi Modul Pelatihan SDM Kesehatan")

    # Hitung total indeks per kelompok modul
    modul_1_vars = ['Preexisting Diabetes', 'Gestational Diabetes', 'BS']
    modul_2_vars = ['BMI']
    modul_3_vars = ['Heart Rate', 'Mental Health', 'Systolic BP', 'Diastolic', 'Previous Complications', 'Age', 'Body Temp']

    idx_1 = fi_df[fi_df['Variabel'].isin(modul_1_vars)]['Indeks Kebutuhan (RF)'].sum()
    idx_2 = fi_df[fi_df['Variabel'].isin(modul_2_vars)]['Indeks Kebutuhan (RF)'].sum()
    idx_3 = fi_df[fi_df['Variabel'].isin(modul_3_vars)]['Indeks Kebutuhan (RF)'].sum()
    total_idx = idx_1 + idx_2 + idx_3

    pct_1 = idx_1 / total_idx * 100
    pct_2 = idx_2 / total_idx * 100
    pct_3 = idx_3 / total_idx * 100

    col_mod1, col_mod2, col_mod3 = st.columns(3)

    with col_mod1:
        st.info(f"### Modul 1: Skrining Diabetes & Metabolik\n"
                f"**Prioritas Utama (~{pct_1:.1f}% Indeks)**\n\n"
                f"* **Variabel Kunci:** `Preexisting Diabetes`, `Gestational Diabetes`, `BS`.\n"
                f"* **Target Peserta:** Bidan Desa & Ahli Gizi Puskesmas.\n"
                f"* **Materi Inti:** Skrining TTGO, manajemen insulin dasar, konseling diet rendah glikemik.")

    with col_mod2:
        st.warning(f"### Modul 2: Manajemen Obesitas & BMI Kehamilan\n"
                   f"**Prioritas Kedua (~{pct_2:.1f}% Indeks)**\n\n"
                   f"* **Variabel Kunci:** `BMI`.\n"
                   f"* **Target Peserta:** Petugas KIA & Kader Kesehatan Posyandu.\n"
                   f"* **Materi Inti:** Pemantauan *Weight Gain Chart*, edukasi nutrisi, deteksi risiko sindrom metabolik.")

    with col_mod3:
        st.success(f"### Modul 3: Vital Sign & Kesehatan Mental\n"
                   f"**Prioritas Ketiga (~{pct_3:.1f}% Indeks)**\n\n"
                   f"* **Variabel Kunci:** `Heart Rate`, `Mental Health`, `Systolic/Diastolic BP`.\n"
                   f"* **Target Peserta:** Dokter Puskesmas & Bidan Koordinator.\n"
                   f"* **Materi Inti:** Deteksi dini aritmia maternal, penanganan preeklamsia awal, instrumen EPDS.")

    # Tabel Indeks Lengkap
    st.markdown("---")
    st.subheader("Tabel Lengkap Indeks Kebutuhan Pelatihan")
    st.dataframe(
        fi_df[['Variabel', 'RF Importance (%)', 'XGB Importance (%)', 'Prevalensi High Risk', 'Indeks Kebutuhan (RF)', 'Indeks Kebutuhan (XGB)']]
        .sort_values('Indeks Kebutuhan (RF)', ascending=False),
        use_container_width=True, hide_index=True
    )
    
    