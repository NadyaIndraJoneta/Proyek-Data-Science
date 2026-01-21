import os
import re
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard Prediksi Kunjungan Wisata Laut Mandeh",
    page_icon="🛥️",
    layout="wide",
)

# ============================================================
# CUSTOM STYLE (BEDAKAN SIDEBAR VS HALAMAN UTAMA)
# ============================================================
st.markdown(
    """
    <style>
    /* Halaman utama (main page) */
    .stApp {
        background-color: #101a2e;
    }

    /* Sidebar beda warna */
    section[data-testid="stSidebar"] {
        background-color: #08101c; /* lebih gelap dari main page */
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Divider di sidebar */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }

    /* Teks sidebar biar tetap jelas */
    section[data-testid="stSidebar"] * {
        color: #e6e6e6;
    }

    /* Supaya label input/select tetap soft */
    section[data-testid="stSidebar"] label {
        color: #cfd8e3 !important;
    }
    .kpi-card {
        background: linear-gradient(180deg, #0f1a2b, #0b1220);
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        text-align: center;
    }
    .kpi-title {
        font-size: 14px;
        color: #9fb4d9;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DEFAULT_RAW_DATA = "Data Harian Fix mandeh.xlsx"
OUTPUTS_DIR = "outputs"
LOGO_PATH = os.path.join("assets", "logo.png")


# ============================================================
# UTILITIES
# ============================================================
def safe_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def list_files(dir_path: str, exts: Tuple[str, ...]) -> List[str]:
    if not safe_exists(dir_path):
        return []
    files = []
    for f in os.listdir(dir_path):
        if f.lower().endswith(exts):
            files.append(f)
    return sorted(files)


def normalize_colname(c: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(c).strip().lower())


def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = normalize_colname(c)
        if lc in ("tanggal", "date", "ds", "datetime"):
            return c

    best = None
    best_score = 0
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            score = parsed.notna().mean()
            if score > best_score and score >= 0.6:
                best_score = score
                best = c
        except Exception:
            continue
    return best


def detect_numeric_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for c in df.columns:
        lc = normalize_colname(c)
        if any(k in lc for k in keywords) and pd.api.types.is_numeric_dtype(df[c]):
            return c

    if numeric_cols:
        variances = {}
        for c in numeric_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            v = s.var()
            if np.isfinite(v):
                variances[c] = v
        if variances:
            return max(variances, key=variances.get)

    return None


def kpi_format(n: float) -> str:
    n = float(n) if n is not None else 0.0
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.0f}"


@st.cache_data(show_spinner=False)
def load_raw_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    rename_map = {}
    for c in df.columns:
        lc = normalize_colname(c)
        if lc == "tanggal":
            rename_map[c] = "tanggal"
        elif lc in ("asal_kapal", "asal_kpl", "asal"):
            rename_map[c] = "asal_kapal"
        elif lc in ("rute", "route"):
            rename_map[c] = "rute"
        elif lc in ("jumlah_penumpang", "penumpang", "jumlah"):
            rename_map[c] = "jumlah_penumpang"

    df = df.rename(columns=rename_map)

    if "tanggal" not in df.columns:
        dcol = detect_date_col(df)
        if dcol:
            df = df.rename(columns={dcol: "tanggal"})
        else:
            raise ValueError("Kolom tanggal tidak ditemukan di data mentah.")

    if "jumlah_penumpang" not in df.columns:
        pcol = detect_numeric_col(df, ["penumpang", "jumlah_penumpang", "total_penumpang"])
        if pcol:
            df = df.rename(columns={pcol: "jumlah_penumpang"})
        else:
            raise ValueError("Kolom jumlah penumpang tidak ditemukan di data mentah.")

    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    df = df.dropna(subset=["tanggal"]).copy()

    if "asal_kapal" in df.columns:
        df["asal_kapal"] = df["asal_kapal"].astype(str).str.strip()
    else:
        df["asal_kapal"] = "(Tidak tersedia)"

    if "rute" in df.columns:
        df["rute"] = df["rute"].astype(str).str.strip()
    else:
        df["rute"] = "(Tidak tersedia)"

    df["jumlah_penumpang"] = pd.to_numeric(df["jumlah_penumpang"], errors="coerce").fillna(0).astype(int)
    return df


def aggregate_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df_raw.groupby("tanggal", as_index=False)
        .agg(
            total_penumpang=("jumlah_penumpang", "sum"),
            trips=("jumlah_penumpang", "size"),
        )
        .sort_values("tanggal")
    )
    daily["dow"] = daily["tanggal"].dt.day_name()
    daily["bulan"] = daily["tanggal"].dt.month_name()
    daily["is_weekend"] = daily["tanggal"].dt.dayofweek.isin([5, 6]).astype(int)
    return daily


@st.cache_data(show_spinner=False)
def load_forecast_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)

    xls = pd.ExcelFile(path)
    sheet = None
    for s in xls.sheet_names:
        if "forecast" in s.lower():
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)


def standardize_forecast_df(fc: pd.DataFrame) -> pd.DataFrame:
    if fc is None or fc.empty:
        return pd.DataFrame(columns=["tanggal", "prediksi_penumpang"])

    date_col = detect_date_col(fc)
    if date_col is None:
        raise ValueError("Kolom tanggal tidak terdeteksi pada file forecast.")

    fc = fc.copy()
    fc[date_col] = pd.to_datetime(fc[date_col], errors="coerce")
    fc = fc.dropna(subset=[date_col])

    pred_col = detect_numeric_col(fc, ["forecast", "pred", "yhat", "penumpang", "total"])
    if pred_col is None:
        raise ValueError("Kolom prediksi tidak terdeteksi pada file forecast.")

    out = fc[[date_col, pred_col]].rename(columns={date_col: "tanggal", pred_col: "prediksi_penumpang"})
    out = out.sort_values("tanggal").reset_index(drop=True)
    out["prediksi_penumpang"] = pd.to_numeric(out["prediksi_penumpang"], errors="coerce").fillna(0.0)
    return out


# ============================================================
# SIDEBAR (judul dihapus, fokus filter)
# ============================================================
with st.sidebar:
    page = st.radio("Main Menu", ["Home", "Prediksi", "Evaluasi", "Tentang"], index=0)
    st.markdown("---")

    if not safe_exists(OUTPUTS_DIR):
        st.warning(f"Folder `{OUTPUTS_DIR}` belum ada. Buat folder `{OUTPUTS_DIR}` dan taruh file output kamu di sana.")

    st.markdown("### 📁 Pilih File Forecast")
    forecast_candidates = list_files(OUTPUTS_DIR, (".xlsx", ".csv"))
    forecast_candidates = [f for f in forecast_candidates if "forecast" in f.lower()]

    if forecast_candidates:
        forecast_file = st.selectbox("Forecast (outputs/)", options=forecast_candidates, index=0)
        forecast_path = os.path.join(OUTPUTS_DIR, forecast_file)
    else:
        forecast_path = None
        st.info("Tidak ada file forecast di outputs/. Pastikan ada file seperti `FORECAST_BULANAN_2026.xlsx`.")

    st.markdown("---")

    if not safe_exists(DEFAULT_RAW_DATA):
        st.error(f"File data mentah tidak ditemukan: `{DEFAULT_RAW_DATA}`. Taruh file Excel di folder yang sama dengan app.py.")
        st.stop()


# ============================================================
# HEADER DI MAIN PAGE (RATA TENGAH)
# ============================================================
header_l, header_c, header_r = st.columns([0.6, 6.8, 0.6])

with header_l:
    if safe_exists(LOGO_PATH):
        st.markdown(
            """
            <div style="margin-left:-20px;">
            """,
            unsafe_allow_html=True
        )
        st.image(LOGO_PATH, width=250)
        st.markdown("</div>", unsafe_allow_html=True)


with header_c:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 16px 0;">
            <h1 style="margin-center: 15px;">Dashboard Prediksi Kunjungan Wisata Mandeh</h1>
            <h3 style="margin-center: 0;">Kabupaten Pesisir Selatan</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD + FILTER RAW
# ============================================================
df_raw = load_raw_excel(DEFAULT_RAW_DATA)
daily = aggregate_daily(df_raw)

min_date = daily["tanggal"].min().date()
max_date = daily["tanggal"].max().date()

with st.sidebar:
    st.markdown("### 🔎 Filter Data Mentah")
    start_date, end_date = st.date_input(
        "Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    asal_options = ["(Semua)"] + sorted(df_raw["asal_kapal"].unique().tolist())
    rute_options = ["(Semua)"] + sorted(df_raw["rute"].unique().tolist())
    asal_selected = st.selectbox("Asal Kapal", asal_options, index=0)
    rute_selected = st.selectbox("Rute", rute_options, index=0)
    st.caption("Catatan: 1 baris data = 1 trip kapal.")

mask = (df_raw["tanggal"].dt.date >= start_date) & (df_raw["tanggal"].dt.date <= end_date)
df_f = df_raw.loc[mask].copy()

if asal_selected != "(Semua)":
    df_f = df_f[df_f["asal_kapal"] == asal_selected]
if rute_selected != "(Semua)":
    df_f = df_f[df_f["rute"] == rute_selected]

daily_f = aggregate_daily(df_f) if len(df_f) else pd.DataFrame(columns=daily.columns)


# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    st.markdown("## 🏠 Page: Home")

    if daily_f.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
        st.stop()

    total_penumpang = daily_f["total_penumpang"].sum()
    total_trips = daily_f["trips"].sum()
    avg_penumpang_harian = daily_f["total_penumpang"].mean()
    avg_trips_harian = daily_f["trips"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Penumpang</div>
                <div class="kpi-value">{kpi_format(total_penumpang)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Trips</div>
                <div class="kpi-value">{kpi_format(total_trips)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Rata-rata Penumpang/Hari</div>
                <div class="kpi-value">{avg_penumpang_harian:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Rata-rata Trips/Hari</div>
                <div class="kpi-value">{avg_trips_harian:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("---")

    col1, col2, col3 = st.columns((1.2, 1, 1))

    with col1:
        st.markdown("### 📈 Total Penumpang Harian")
        fig_line = px.line(daily_f, x="tanggal", y="total_penumpang")
        fig_line.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.markdown("### 🧭 Rute Paling Ramai (Top 10)")
        top_rute = (
            df_f.groupby("rute", as_index=False)["jumlah_penumpang"].sum()
            .sort_values("jumlah_penumpang", ascending=False)
            .head(10)
        )
        fig_bar = px.bar(top_rute, x="jumlah_penumpang", y="rute", orientation="h")
        fig_bar.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col3:
        st.markdown("### 📊 Distribusi Penumpang Berdasarkan Rute (Top 5)")

        pie_rute = (
            df_f.groupby("rute", as_index=False)["jumlah_penumpang"].sum()
            .sort_values("jumlah_penumpang", ascending=False)
        )

        top_n = 5
        if len(pie_rute) > top_n:
            top = pie_rute.head(top_n).copy()
            other_sum = pie_rute.iloc[top_n:]["jumlah_penumpang"].sum()
            pie_show = pd.concat(
                [top, pd.DataFrame([{"rute": "Lainnya", "jumlah_penumpang": other_sum}])],
                ignore_index=True
            )
        else:
            pie_show = pie_rute

        fig_pie = px.pie(
            pie_show,
            names="rute",
            values="jumlah_penumpang",
            hole=0.45
        )

        fig_pie.update_traces(textinfo="percent+label", textposition="inside")
        fig_pie.update_layout(
            height=420,
            showlegend=True,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="v", x=1.02, y=0.5)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Cuplikan Data Mentah (Trip-level)")
    st.dataframe(df_f.head(50), use_container_width=True)

    st.markdown("### 📅 Data Harian (Agregasi)")
    st.dataframe(daily_f.head(30), use_container_width=True)


# ============================================================
# PAGE: PREDIKSI
# ============================================================
elif page == "Prediksi":
    st.markdown("## 🔮 Page: Prediksi")

    if not forecast_path or not safe_exists(forecast_path):
        st.error(
            "File forecast belum ditemukan.\n\n"
            "✅ Pastikan file forecast berada di folder `outputs/` dan namanya mengandung kata `FORECAST`.\n"
            "Contoh: `FORECAST_BULANAN_2026.xlsx`"
        )
        st.stop()

    try:
        fc_raw = load_forecast_any(forecast_path)
        fc = standardize_forecast_df(fc_raw)
    except Exception as e:
        st.error(f"Gagal membaca forecast: {e}")
        if "fc_raw" in locals():
            st.write("Kolom di file forecast:", list(fc_raw.columns))
        st.stop()

    if fc.empty:
        st.warning("Forecast terbaca tapi isinya kosong.")
        st.stop()

    pred_min = fc["prediksi_penumpang"].min()
    pred_max = fc["prediksi_penumpang"].max()
    pred_avg = fc["prediksi_penumpang"].mean()

    a1, a2, a3 = st.columns(3)
    a1.metric("Rata-rata Prediksi", f"{pred_avg:.0f}")
    a2.metric("Prediksi Minimum", f"{pred_min:.0f}")
    a3.metric("Prediksi Maksimum", f"{pred_max:.0f}")

    st.markdown("---")
    st.markdown("### 📊 Grafik Prediksi Penumpang")
    fig_fc = px.line(fc, x="tanggal", y="prediksi_penumpang")
    fig_fc.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("### 🚨 Indikasi Lonjakan")
    default_thr = int(np.percentile(fc["prediksi_penumpang"], 75))
    threshold = st.slider(
        "Ambang penumpang/hari (status 'Tinggi')",
        min_value=0,
        max_value=int(max(100, fc["prediksi_penumpang"].max())),
        value=default_thr
    )
    fc["status"] = np.where(fc["prediksi_penumpang"] >= threshold, "Tinggi", "Normal")

    st.write("Jumlah hari berstatus **Tinggi**:", int((fc["status"] == "Tinggi").sum()))
    st.dataframe(fc, use_container_width=True)
    st.caption(f"Sumber forecast: `{forecast_path}`")


# ============================================================
# PAGE: EVALUASI
# ============================================================
elif page == "Evaluasi":
    st.markdown("## ✅ Page: Evaluasi")

    st.markdown("### 📌 Grafik Aktual (Total Penumpang Harian)")
    if daily_f.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
        st.stop()

    fig_act = px.line(daily_f, x="tanggal", y="total_penumpang")
    fig_act.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📎 Daftar File di outputs/")
    all_outputs = list_files(OUTPUTS_DIR, (".xlsx", ".png", ".pkl", ".csv"))
    if all_outputs:
        st.write(all_outputs)
        st.caption("File PNG seperti `MODEL_01_aktual_vs_prediksi_test.png` bisa kamu jadikan bukti evaluasi.")
    else:
        st.info("Belum ada file di outputs/ atau folder outputs belum ada.")


# ============================================================
# PAGE: TENTANG
# ============================================================
else:
    st.markdown("## ℹ️ Tentang")
    st.markdown(
        """
**Dashboard Prediksi Kunjungan Wisata Laut Mandeh (XGBoost)**

- Data mentah: trip-level (baris = 1 trip)
- Agregasi harian:
  - total_penumpang = total penumpang per hari
  - trips = jumlah trip per hari
- Prediksi: dibaca dari file forecast di folder outputs/ (XLSX/CSV)

Jika kamu ingin prediksi **langsung dihitung dari model** `model_xgboost_tuned_mandeh.pkl`
(tanpa file forecast), aku bisa buatkan versi lanjutannya.
        """
    )
