import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Prediksi Risiko Kehamilan",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0e1117; }

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 14px;
}

.result-card {
    padding: 22px;
    border-radius: 16px;
    border-left: 6px solid;
    background: rgba(255,255,255,0.06);
}

.rendah { border-color: #2ecc71; }
.sedang { border-color: #f1c40f; }
.tinggi { border-color: #e74c3c; }

.footer {
    text-align: center;
    opacity: 0.6;
    font-size: 13px;
    margin-top: 30px;
}

.metric-box{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px 14px;
  min-height: 86px;
}
.metric-label{
  font-size: 12px;
  opacity: .75;
  margin-bottom: 6px;
}
.metric-value{
  font-size: 20px;
  font-weight: 700;
  line-height: 1.15;
  white-space: normal;
  word-break: break-word;
}
.metric-unit{
  font-size: 12px;
  opacity: .75;
  margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

def metric_box(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""
        <div class="metric-box">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}{f'<span class="metric-unit">{unit}</span>' if unit else ''}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    return joblib.load("modelLinearRegression.joblib")

model = load_model()

if "form_id" not in st.session_state:
    st.session_state.form_id = 0
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "temp_is_f_auto" not in st.session_state:
    st.session_state.temp_is_f_auto = False

FEATURES = ["Usia", "GulaDarah", "SuhuTubuh", "DenyutJantung", "MAP"]

def kategori_risiko(skor: float):
    if skor <= 0.33:
        return "🟢 Risiko Rendah", "rendah", "Lanjutkan pemantauan ANC rutin"
    elif skor <= 0.66:
        return "🟡 Risiko Sedang", "sedang", "Perlu pemantauan lebih sering"
    else:
        return "🔴 Risiko Tinggi", "tinggi", "Segera rujuk ke dokter"

def kategori_risiko_text(skor: float) -> str:
    if skor <= 0.33: return "Rendah"
    if skor <= 0.66: return "Sedang"
    return "Tinggi"

def safe_read_csv(uploaded_file) -> pd.DataFrame:
    for sep in [",", ";", "\t", "|"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip() for c in df2.columns]

    rename_map = {
        "umur": "Usia", "usia": "Usia",
        "gula_darah": "GulaDarah", "gula": "GulaDarah", "blood_sugar": "GulaDarah",
        "suhu": "SuhuTubuh", "suhu_tubuh": "SuhuTubuh", "body_temp": "SuhuTubuh",
        "denyut": "DenyutJantung", "nadi": "DenyutJantung", "heart_rate": "DenyutJantung",
        "sbp": "TekananDarahSistolik", "sistolik": "TekananDarahSistolik", "systolic": "TekananDarahSistolik",
        "dbp": "TekananDarahDiastolik", "diastolik": "TekananDarahDiastolik", "diastolic": "TekananDarahDiastolik",
        "map": "MAP", "mean_arterial_pressure": "MAP",
        "tingkatrisiko": "TingkatRisiko", "skorrisiko": "TingkatRisiko", "risk_score": "TingkatRisiko",
    }

    cols_lower = {c.lower(): c for c in df2.columns}
    for k, v in rename_map.items():
        if k in cols_lower:
            df2 = df2.rename(columns={cols_lower[k]: v})

    return df2

def detect_temp_unit_is_f(df: pd.DataFrame) -> bool:
    if "SuhuTubuh" not in df.columns:
        return False
    med_t = to_num(df["SuhuTubuh"]).median()
    return bool(pd.notna(med_t) and 70 <= med_t <= 110)

def apply_temp_conversion(df: pd.DataFrame, temp_is_f: bool) -> pd.DataFrame:
    df2 = df.copy()
    if "SuhuTubuh" in df2.columns:
        t = to_num(df2["SuhuTubuh"])
        df2["SuhuTubuh"] = (t - 32) * 5/9 if temp_is_f else t
    return df2

def ensure_map(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "MAP" not in df2.columns:
        df2["MAP"] = np.nan

    if {"TekananDarahSistolik", "TekananDarahDiastolik"}.issubset(df2.columns):
        sbp = to_num(df2["TekananDarahSistolik"])
        dbp = to_num(df2["TekananDarahDiastolik"])
        map_val = (sbp + 2.0 * dbp) / 3.0
        df2["MAP"] = df2["MAP"].fillna(map_val)

    df2["MAP"] = to_num(df2["MAP"])
    return df2

def make_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = to_num(out[c])

    med = out[FEATURES].median(numeric_only=True)
    out[FEATURES] = out[FEATURES].fillna(med)
    return out[FEATURES]

def predict_score(feature_df: pd.DataFrame) -> np.ndarray:
    raw = model.predict(feature_df)
    return np.clip(raw, 0.0, 1.0)

def validate_screening_inputs(usia, gula, suhu, denyut, sbp, dbp) -> list[str]:
    warns = []
    if usia > 60:
        warns.append("Usia di atas 60 tahun, pastikan input benar.")
    if gula > 300:
        warns.append("Gula darah sangat tinggi (>300). Pastikan satuan mg/dL dan input benar.")
    if suhu > 45:
        warns.append("Suhu >45°C tidak realistis. Jika input dari °F, pastikan sudah dikonversi.")
    if denyut > 200:
        warns.append("Denyut jantung >200 bpm jarang. Pastikan input benar.")
    if sbp > 250 or dbp > 150:
        warns.append("Tekanan darah sangat tinggi. Pastikan input benar.")
    return warns

st.sidebar.markdown("## 📌 Menu")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["🩺 Screening", "📊 Visualisasi CSV"],
    label_visibility="collapsed"
)

if page == "🩺 Screening":
    st.title("🩺 Prediksi Risiko Kesehatan Kehamilan")
    st.caption("Aplikasi pendukung keputusan klinis untuk perawat dan bidan")

    st.markdown(f"🕒 **Waktu Pemeriksaan:** {datetime.now().strftime('%d %B %Y, %H:%M')}")
    st.divider()

    left, right = st.columns([1.15, 1])
    fid = st.session_state.form_id

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📋 Data Klinis Ibu Hamil</div>", unsafe_allow_html=True)

        usia = st.number_input("Usia Ibu (tahun)", step=1, min_value=0, key=f"usia_{fid}")
        gula = st.number_input("Gula Darah (mg/dL)", step=0.1, min_value=0.0, key=f"gula_{fid}")
        suhu = st.number_input("Suhu Tubuh (°C)", step=0.1, min_value=0.0, key=f"suhu_{fid}")
        denyut = st.number_input("Denyut Jantung (bpm)", step=1, min_value=0, key=f"denyut_{fid}")

        st.markdown("**Tekanan Darah**")
        c1, c2 = st.columns(2)
        sbp = c1.number_input("Sistolik (mmHg)", step=1, min_value=0, key=f"sbp_{fid}")
        dbp = c2.number_input("Diastolik (mmHg)", step=1, min_value=0, key=f"dbp_{fid}")

        map_val = (sbp + 2 * dbp) / 3.0 if (sbp > 0 and dbp > 0) else 0.0
        st.caption(f"MAP (Mean Arterial Pressure): **{map_val:.2f}** mmHg")

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Lakukan Screening", use_container_width=True):
            if all(v == 0 for v in [usia, gula, suhu, denyut, sbp, dbp]):
                st.warning("⚠️ Silakan masukkan data pasien terlebih dahulu.")
            else:
                st.session_state.show_result = True

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        try:
            st.image("ibuHamil.jpg", use_container_width=True)
        except Exception:
            st.info("Gambar `ibuHamil.jpg` tidak ditemukan.")

        st.markdown(
            "Sistem ini membantu **screening awal risiko kehamilan** berdasarkan parameter fisiologis ibu hamil. Hasil bersifat pendukung keputusan klinis."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.show_result:
            warns = validate_screening_inputs(usia, gula, suhu, denyut, sbp, dbp)
            for w in warns:
                st.warning(f"⚠️ {w}")

            input_df = pd.DataFrame([{
                "Usia": usia,
                "GulaDarah": gula,
                "SuhuTubuh": suhu,
                "DenyutJantung": denyut,
                "MAP": map_val
            }])

            skor = float(np.clip(model.predict(input_df)[0], 0.0, 1.0))
            skor_persen = round(skor * 100, 1)

            kategori, css, rekom = kategori_risiko(skor)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Hasil Screening")

            st.progress(skor)
            st.metric("Skor Risiko Kehamilan", f"{skor_persen}%")

            st.markdown(
                f"""
                <div class="result-card {css}">
                    <h4>{kategori}</h4>
                    <p><b>Rekomendasi:</b> {rekom}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("🔄 Screening Pasien Baru", use_container_width=True):
                st.session_state.form_id += 1
                st.session_state.show_result = False
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div class='footer'>© 2026 • Prediksi Risiko Kesehatan Kehamilan • MK Praktikum Unggulan (DGX)</div>",
        unsafe_allow_html=True
    )

else:
    st.title("📊 Visualisasi Data")
    st.caption(
    "Upload data pasien dalam format CSV untuk melihat ringkasan kondisi klinis serta melakukan screening risiko kehamilan secara otomatis.")
    st.divider()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        try:
            df_raw = safe_read_csv(uploaded)
            df = normalize_columns(df_raw)
            df = ensure_map(df)

            temp_is_f_auto = detect_temp_unit_is_f(df)
            st.session_state.uploaded_df = df
            st.session_state.temp_is_f_auto = temp_is_f_auto
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.info("Silakan upload file CSV terlebih dahulu.")
        st.stop()

    with st.expander("⚙️ Pengaturan Unit (Auto-detect)", expanded=False):
        temp_is_f = st.checkbox(
            "Suhu pada CSV adalah °F (akan dikonversi ke °C)",
            value=st.session_state.get("temp_is_f_auto", False)
        )

    df_u = apply_temp_conversion(df, temp_is_f=temp_is_f)
    df_u = ensure_map(df_u)

    tab1, tab2, tab3 = st.tabs(["🩺 Ringkasan Klinis", "🧠 Prediksi Risiko", "📄 Data"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        total = df_u.shape[0]
        miss = int(df_u.isna().sum().sum())

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Pasien", f"{total:,}")
        a2.metric("Jumlah Kolom", f"{df_u.shape[1]:,}")
        a3.metric("Nilai Kosong", f"{miss:,}")
        a4.metric("Sumber", "CSV Upload")

        def mean_of(col):
            if col not in df_u.columns:
                return None
            s = to_num(df_u[col]).dropna()
            return float(s.mean()) if not s.empty else None

        mean_sbp = mean_of("TekananDarahSistolik")
        mean_dbp = mean_of("TekananDarahDiastolik")
        mean_map = mean_of("MAP")
        mean_glu = mean_of("GulaDarah")
        mean_temp = mean_of("SuhuTubuh")
        mean_hr = mean_of("DenyutJantung")

        st.markdown("### 🧾 Ringkasan Rata-rata Klinis")
        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            metric_box("Rerata Sistolik", "-" if mean_sbp is None else f"{mean_sbp:.1f}", "mmHg")
        with k2:
            metric_box("Rerata Diastolik", "-" if mean_dbp is None else f"{mean_dbp:.1f}", "mmHg")
        with k3:
            metric_box("Rerata MAP", "-" if mean_map is None else f"{mean_map:.1f}", "mmHg")
        with k4:
            metric_box("Rerata Gula Darah", "-" if mean_glu is None else f"{mean_glu:.1f}", "(skala data)")
        with k5:
            metric_box("Rerata Suhu", "-" if mean_temp is None else f"{mean_temp:.1f}", "°C")

        st.markdown("### 🚩 Peringatan Cepat")
        w1, w2, w3 = st.columns(3)

        if {"TekananDarahSistolik", "TekananDarahDiastolik"}.issubset(df_u.columns):
            sbp_s = to_num(df_u["TekananDarahSistolik"])
            dbp_s = to_num(df_u["TekananDarahDiastolik"])
            hipertensi = int(((sbp_s >= 140) | (dbp_s >= 90)).fillna(False).sum())
            w1.metric("Dugaan Hipertensi (≥140/90)", f"{hipertensi:,} pasien")
        else:
            w1.info("Kolom sistolik/diastolik tidak ditemukan.")

        if "SuhuTubuh" in df_u.columns:
            temp_s = to_num(df_u["SuhuTubuh"])
            demam = int((temp_s >= 38.0).fillna(False).sum())
            w2.metric("Demam (≥38°C)", f"{demam:,} pasien")
        else:
            w2.info("Kolom suhu tidak ditemukan.")

        if "DenyutJantung" in df_u.columns:
            hr_s = to_num(df_u["DenyutJantung"])
            takikardia = int((hr_s >= 100).fillna(False).sum())
            w3.metric("Takikardia (≥100 bpm)", f"{takikardia:,} pasien")
        else:
            w3.info("Kolom denyut tidak ditemukan.")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🧠 Prediksi Risiko untuk Seluruh Data</div>", unsafe_allow_html=True)

        X = make_feature_df(df_u)
        skor_arr = predict_score(X)

        df_pred = df_u.copy()
        df_pred["SkorRisiko"] = skor_arr
        df_pred["KategoriRisiko"] = [kategori_risiko_text(float(x)) for x in skor_arr]

        counts = df_pred["KategoriRisiko"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).fillna(0).astype(int)
        pcts = (counts / max(1, len(df_pred)) * 100).round(1)

        b1, b2, b3 = st.columns(3)
        b1.metric("🟢 Risiko Rendah", f"{counts['Rendah']:,} ({pcts['Rendah']}%)")
        b2.metric("🟡 Risiko Sedang", f"{counts['Sedang']:,} ({pcts['Sedang']}%)")
        b3.metric("🔴 Risiko Tinggi", f"{counts['Tinggi']:,} ({pcts['Tinggi']}%)")

        fig = plt.figure()
        plt.bar(["Rendah", "Sedang", "Tinggi"], [counts["Rendah"], counts["Sedang"], counts["Tinggi"]])
        plt.title("Distribusi Hasil Prediksi Risiko Kehamilan")
        plt.xlabel("Kategori Risiko")
        plt.ylabel("Jumlah Pasien")
        st.pyplot(fig)

        st.markdown("### 🔍 Daftar Prioritas (Risiko Tinggi)")
        topn = st.slider("Tampilkan berapa pasien risiko tinggi teratas?", 5, 50, 15)

        high_df = (
            df_pred[df_pred["KategoriRisiko"] == "Tinggi"]
            .copy()
            .sort_values("SkorRisiko", ascending=False)
            .head(topn)
        )

        if high_df.empty:
            st.success("Tidak ada pasien kategori risiko tinggi berdasarkan prediksi.")
        else:
            show_cols = [c for c in [
                "Usia", "TekananDarahSistolik", "TekananDarahDiastolik", "MAP",
                "GulaDarah", "SuhuTubuh", "DenyutJantung",
                "SkorRisiko", "KategoriRisiko"
            ] if c in high_df.columns]
            st.dataframe(high_df[show_cols], use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 👀 Preview Data")
        st.dataframe(df_u.head(30), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    X = make_feature_df(df_u)
    skor_arr = predict_score(X)

    df_out = df_u.copy()
    df_out["SkorRisiko"] = skor_arr
    df_out["KategoriRisiko"] = [kategori_risiko_text(float(x)) for x in skor_arr]

    out_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV + Prediksi Risiko",
        data=out_bytes,
        file_name="hasil_prediksi_risiko.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()
    st.markdown(
        "<div class='footer'>© 2026 • Prediksi Risiko Kesehatan Kehamilan • MK Praktikum Unggulan (DGX)</div>",
        unsafe_allow_html=True
    )