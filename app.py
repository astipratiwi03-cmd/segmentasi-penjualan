import os
# -*- coding: utf-8 -*-
"""
Dashboard bisnis untuk segmentasi penjualan produk mingguan.
Sumber utama: dashboard_data/sales_clusters.csv (nilai penjualan asli W0-W51
+ label cluster), product_features.csv, summary.json, cross_silhouette.csv,
dan ari.csv. File evaluasi tambahan bersifat opsional.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Product Portfolio | Sales Review",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Visual system ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root {
  --ink:#202923; --muted:#68736b; --line:#e4e7df; --paper:#f6f6f1;
  --white:#ffffff; --green:#315d45; --green2:#547b5e; --lime:#dce8d7;
  --amber:#c18a3b; --red:#a95645; --blue:#647d8c;
}
html, body, [class*="css"] {font-family:'Inter',sans-serif;color:var(--ink);}
.stApp {background:var(--paper);}
.block-container {max-width:1500px;padding-top:1.5rem;padding-bottom:3rem;}
h1,h2,h3 {color:var(--ink)!important;letter-spacing:-.035em;}
h1 {font-size:2.15rem!important;font-weight:700!important;}
h2 {font-size:1.42rem!important;font-weight:700!important;margin-top:.3rem;}
h3 {font-size:1.02rem!important;font-weight:650!important;}
p,li,label {color:var(--ink);}
[data-testid="stSidebar"] {background:#263b30;}
[data-testid="stSidebar"] * {color:#f4f5ef!important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#d5dfd6!important;}
[data-testid="stMetric"] {
  background:#fff;border:1px solid var(--line);border-radius:5px;
  padding:15px 17px;box-shadow:none;
}
[data-testid="stMetricLabel"] {color:#68736b!important;font-size:.78rem!important;}
[data-testid="stMetricValue"] {color:#263b30!important;font-size:1.75rem!important;font-weight:700!important;}
[data-testid="stMetricDelta"] {font-size:.76rem!important;}
div[data-testid="stTabs"] button {font-weight:600;color:#657067!important;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#315d45!important;border-bottom-color:#315d45!important;}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:4px;}
div.stButton>button, div.stDownloadButton>button {border-radius:4px;font-weight:600;}
hr {border-color:var(--line)!important;}
.kicker {font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:#6c7b6e;font-weight:700;margin-bottom:.35rem;}
.deck {font-size:1rem;color:#59645c;line-height:1.65;max-width:980px;margin-bottom:1rem;}
.rule {border-top:1px solid #dfe4db;margin:1rem 0 1.25rem 0;}
.insight {background:#edf2e9;border-left:3px solid #547b5e;padding:13px 16px;margin:.4rem 0 1rem 0;line-height:1.55;}
.warning-note {background:#f5eee3;border-left:3px solid #c18a3b;padding:13px 16px;margin:.4rem 0 1rem 0;line-height:1.55;}
.mono {font-family:'DM Mono',monospace;}
.small {font-size:.82rem;color:#68736b;}
div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label {font-weight:600;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
DATA = BASE / "dashboard_data"
REQUIRED = ["sales_clusters.csv", "product_features.csv", "summary.json"]

@st.cache_data
def load_data():
    missing = [x for x in REQUIRED if not (DATA / x).exists()]
    if missing:
        raise FileNotFoundError("File belum ditemukan: " + ", ".join(missing))
    sales = pd.read_csv(DATA / "sales_clusters.csv")
    features = pd.read_csv(DATA / "product_features.csv")
    meta = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    optional = {}
    for name, filename in {
        "cross": "cross_silhouette.csv", "ari": "ari.csv",
        "projection": "projection_2d.csv", "product_sil": "product_silhouette.csv",
        "medoids": "medoids.csv"
    }.items():
        path = DATA / filename
        if path.exists():
            try:
                optional[name] = pd.read_csv(path, index_col=0) if name in ("cross", "ari") else pd.read_csv(path)
            except Exception:
                optional[name] = None
    return sales, features, meta, optional

try:
    sales, features, meta, optional = load_data()
except Exception as exc:
    st.error(f"Data dashboard belum siap: {exc}")
    st.info("Pastikan app.py berada di /content dan folder dashboard_data berada di /content/dashboard_data.")
    st.stop()

WEEKS = sorted(
    [c for c in sales.columns if c.startswith("W") and c[1:].isdigit()],
    key=lambda x: int(x[1:])
)
if not WEEKS:
    st.error("Kolom penjualan mingguan W0, W1, ... tidak ditemukan pada sales_clusters.csv.")
    st.stop()

METHODS = {
    "K-Medoids + DTW": ("Cluster_DTW", "dtw", "DTW"),
    "K-Medoids + DDTW": ("Cluster_DDTW", "ddtw", "DDTW"),
    "K-Shape + SBD": ("Cluster_KShape", "kshape", "SBD"),
}
METHOD_NAMES = list(METHODS)
PALETTE = ["#315d45", "#c18a3b", "#647d8c", "#a95645", "#8a7799"]

def fmt(x, d=1):
    try:
        if pd.isna(x): return "—"
        return f"{float(x):,.{d}f}"
    except Exception:
        return "—"

def plot_style(fig, height=350, legend=True):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#344139", size=11),
        margin=dict(l=18, r=18, t=48, b=20),
        title=dict(font=dict(size=14, color="#263b30")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(255,255,255,0)"),
        showlegend=legend,
        hoverlabel=dict(bgcolor="#263b30", font_color="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#dfe4db", tickfont=dict(color="#68736b"))
    fig.update_yaxes(gridcolor="#edf0ea", zerolinecolor="#dfe4db", tickfont=dict(color="#68736b"))
    return fig

def feature_for(product_ids):
    if "product_id" not in features.columns:
        return pd.DataFrame()
    return features[features["product_id"].astype(str).isin([str(x) for x in product_ids])].copy()

def get_cluster_frame(method):
    label_col, _, _ = METHODS[method]
    if label_col not in sales.columns:
        return None, label_col
    out = sales[["Product_Code"] + WEEKS + [label_col]].copy()
    out = out.rename(columns={label_col: "Cluster"})
    out["Cluster"] = pd.to_numeric(out["Cluster"], errors="coerce").fillna(-1).astype(int)
    return out, label_col

def calc_product_metrics(frame):
    x = frame[WEEKS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    means = x.mean(axis=1)
    stds = x.std(axis=1, ddof=1)
    return pd.DataFrame({
        "Product_Code": frame["Product_Code"].astype(str).values,
        "Total 52 minggu": x.sum(axis=1),
        "Rata-rata/minggu": means,
        "Maksimum mingguan": x.max(axis=1),
        "Minggu tanpa penjualan (%)": (x == 0).mean(axis=1) * 100,
        "CV": np.divide(stds, means, out=np.full_like(stds, np.nan), where=means != 0),
        "Simpangan baku": stds,
    })

raw_metrics = calc_product_metrics(sales)
raw = sales[WEEKS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
n_products, n_weeks = raw.shape
total_units = float(raw.sum())
weekly_totals = raw.sum(axis=0)
product_metrics = raw_metrics.copy()
top_products = product_metrics.sort_values("Total 52 minggu", ascending=False)
total_by_week = pd.DataFrame({"Minggu": np.arange(n_weeks), "Unit terjual": weekly_totals})
overall_mean = raw.mean()
zero_share = float((raw == 0).mean() * 100)
best_product = top_products.iloc[0]

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("## PRODUCT PORTFOLIO")
    st.markdown("**Sales performance review**")
    st.markdown('<div class="small">52-week product demand · clustering analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGASI",
        ["Ikhtisar bisnis", "Eksplorasi data asli", "Peluang & risiko produk",
         "Segmentasi produk", "Tinjau satu produk", "Evaluasi metode"],
        label_visibility="visible"
    )
    st.markdown("---")
    method = st.selectbox("Metode segmentasi", METHOD_NAMES, index=0)
    st.markdown('<div class="small">Metode hanya mengubah label cluster. Angka penjualan dan EDA selalu memakai unit asli.</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small">SUMBER<br>Data penjualan mingguan dan hasil clustering dari notebook.</div>', unsafe_allow_html=True)

cluster_frame, label_col = get_cluster_frame(method)
if cluster_frame is None:
    st.error(f"Kolom {label_col} tidak tersedia. Periksa file sales_clusters.csv.")
    st.stop()
cluster_count = int(cluster_frame["Cluster"].nunique())
method_key = METHODS[method][1]
method_meta = meta.get(method_key, {})
silhouette = method_meta.get("silhouette", np.nan)

# --------------------------- Page 1 ---------------------------
if page == "Ikhtisar bisnis":
    st.markdown('<div class="kicker">WEEKLY SALES / MANAGEMENT VIEW</div>', unsafe_allow_html=True)
    st.title("Tinjauan kinerja portofolio produk")
    st.markdown('<div class="deck">Ringkasan permintaan selama 52 minggu untuk membantu tim menentukan produk yang perlu dijaga ketersediaannya, ditinjau ulang, atau dipantau lebih dekat.</div>', unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    a.metric("Total unit terjual", f"{total_units:,.0f}")
    b.metric("Produk dalam portofolio", f"{n_products:,}")
    c.metric("Rata-rata unit / produk / minggu", f"{overall_mean:,.1f}")
    d.metric("Minggu-produk tanpa penjualan", f"{zero_share:.1f}%")

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("### Ritme penjualan mingguan")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=total_by_week["Minggu"], y=total_by_week["Unit terjual"],
            mode="lines+markers", line=dict(color="#315d45", width=2.8),
            marker=dict(size=5), fill="tozeroy", fillcolor="rgba(49,93,69,.09)",
            hovertemplate="Minggu %{x}<br>%{y:,.0f} unit<extra></extra>"
        ))
        fig.update_layout(xaxis_title="Minggu observasi", yaxis_title="Total unit terjual")
        st.plotly_chart(plot_style(fig, 350, False), use_container_width=True)
        st.caption("Grafik menggunakan volume penjualan asli, bukan data z-score.")
    with right:
        st.markdown("### Konsentrasi penjualan")
        top10_share = top_products.head(10)["Total 52 minggu"].sum() / total_units * 100 if total_units else 0
        top20_share = top_products.head(20)["Total 52 minggu"].sum() / total_units * 100 if total_units else 0
        st.metric("Kontribusi 10 produk teratas", f"{top10_share:.1f}%")
        st.metric("Kontribusi 20 produk teratas", f"{top20_share:.1f}%")
        st.markdown('<div class="insight"><b>Produk dengan volume tertinggi</b><br>'
                    f'<span class="mono">{best_product["Product_Code"]}</span> mencatat '
                    f'<b>{best_product["Total 52 minggu"]:,.0f} unit</b> selama periode pengamatan. '
                    'Periksa kestabilan dan frekuensi penjualannya sebelum menetapkan prioritas stok.</div>',
                    unsafe_allow_html=True)
        st.markdown("### Metode segmentasi aktif")
        st.write(f"**{method}** · {cluster_count} cluster")
        st.write(f"Silhouette: **{fmt(silhouette, 3)}**")
        st.caption("Silhouette dipakai untuk menilai pemisahan cluster, bukan ukuran kinerja penjualan.")
    st.markdown("### Produk dengan volume terbesar")
    display_top = top_products.head(10).copy()
    st.dataframe(display_top, use_container_width=True, hide_index=True)
    st.markdown('<div class="warning-note"><b>Catatan untuk keputusan:</b> data ini menggambarkan penjualan historis. '
                'Dashboard tidak mengetahui stok tersedia, margin, lead time, atau kejadian promosi. '
                'Karena itu, prioritas persediaan di bawah merupakan sinyal untuk ditinjau, bukan rekomendasi jumlah pemesanan.</div>',
                unsafe_allow_html=True)

# --------------------------- Page 2 ---------------------------
elif page == "Eksplorasi data asli":
    st.markdown('<div class="kicker">RAW DATA / EXPLORATORY REVIEW</div>', unsafe_allow_html=True)
    st.title("Eksplorasi penjualan asli")
    st.markdown('<div class="deck">Bagian ini memeriksa data unit terjual per produk pada W0–W51 sebelum normalisasi dan clustering. Semua angka di halaman ini berada pada skala penjualan asli.</div>', unsafe_allow_html=True)

    # Data quality checks
    missing_count = int(sales[WEEKS].isna().sum().sum())
    negative_count = int((raw < 0).sum())
    duplicate_count = int(sales["Product_Code"].duplicated().sum())
    q1, q3 = product_metrics["Total 52 minggu"].quantile([.25, .75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    outlier_count = int((product_metrics["Total 52 minggu"] > upper).sum())

    a,b,c,d,e = st.columns(5)
    a.metric("Produk unik", f"{sales['Product_Code'].nunique():,}")
    b.metric("Kolom minggu", f"{n_weeks}")
    c.metric("Nilai hilang", f"{missing_count:,}")
    d.metric("Nilai negatif", f"{negative_count:,}")
    e.metric("Outlier total (IQR)", f"{outlier_count:,}")

    st.markdown("### Ringkasan distribusi seluruh observasi")
    desc = pd.Series(raw.ravel(), name="Unit terjual").describe()
    cols = st.columns(6)
    for col, (lab, val) in zip(cols, [
        ("Observasi", desc["count"]), ("Rata-rata", desc["mean"]),
        ("Median", desc["50%"]), ("Simpangan baku", desc["std"]),
        ("Minimum", desc["min"]), ("Maksimum", desc["max"])
    ]):
        col.metric(lab, f"{val:,.2f}" if lab in ("Rata-rata","Simpangan baku") else f"{val:,.0f}")

    st.markdown("### Bagaimana distribusi permintaan?")
    l,r = st.columns(2)
    with l:
        fig = px.histogram(product_metrics, x="Total 52 minggu", nbins=40,
                           title="Sebaran total unit per produk")
        fig.update_traces(marker_color="#547b5e", opacity=.9)
        fig.update_layout(xaxis_title="Total unit selama 52 minggu", yaxis_title="Jumlah produk")
        st.plotly_chart(plot_style(fig, 330, False), use_container_width=True)
    with r:
        zero_dist = product_metrics.copy()
        fig = px.histogram(zero_dist, x="Minggu tanpa penjualan (%)", nbins=25,
                           title="Frekuensi minggu tanpa penjualan")
        fig.update_traces(marker_color="#c18a3b", opacity=.9)
        fig.update_layout(xaxis_title="Minggu tanpa penjualan (%)", yaxis_title="Jumlah produk")
        st.plotly_chart(plot_style(fig, 330, False), use_container_width=True)

    l,r = st.columns(2)
    with l:
        fig = px.box(product_metrics, y="Total 52 minggu", points="outliers",
                     title="Rentang total penjualan antarproduk")
        fig.update_traces(marker_color="#547b5e", line_color="#315d45")
        st.plotly_chart(plot_style(fig, 320, False), use_container_width=True)
    with r:
        fig = px.scatter(
            product_metrics, x="Rata-rata/minggu", y="CV",
            size="Total 52 minggu", color="Minggu tanpa penjualan (%)",
            hover_name="Product_Code", color_continuous_scale=["#dce8d7","#c18a3b","#a95645"],
            title="Volume rata-rata dan ketidakstabilan permintaan",
            labels={"Rata-rata/minggu":"Rata-rata unit / minggu","CV":"CV","Minggu tanpa penjualan (%)":"Minggu nol (%)"}
        )
        st.plotly_chart(plot_style(fig, 340, False), use_container_width=True)

    st.markdown("### Total penjualan dari minggu ke minggu")
    weekly = pd.DataFrame({"Minggu": np.arange(n_weeks), "Total unit": raw.sum(axis=0),
                           "Rata-rata per produk": raw.mean(axis=0)})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weekly["Minggu"], y=weekly["Total unit"], name="Total unit", marker_color="#547b5e"))
    fig.add_trace(go.Scatter(x=weekly["Minggu"], y=weekly["Rata-rata per produk"], name="Rata-rata per produk",
                             yaxis="y2", line=dict(color="#c18a3b", width=2.5)))
    fig.update_layout(
        title="Aktivitas penjualan per minggu", xaxis_title="Minggu observasi",
        yaxis=dict(title="Total unit"), yaxis2=dict(title="Rata-rata unit / produk", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12)
    )
    st.plotly_chart(plot_style(fig, 390), use_container_width=True)

    st.markdown("### Telusuri data penjualan per produk")
    product_choice = st.selectbox("Pilih produk", sales["Product_Code"].astype(str).tolist(), key="raw_eda_product")
    prow = sales[sales["Product_Code"].astype(str) == product_choice].iloc[0]
    values = pd.to_numeric(prow[WEEKS], errors="coerce").fillna(0).to_numpy(float)
    fig = go.Figure(go.Bar(x=np.arange(n_weeks), y=values, marker_color="#547b5e",
                           hovertemplate="Minggu %{x}<br>%{y} unit<extra></extra>"))
    fig.update_layout(title=f"Penjualan mingguan · {product_choice}", xaxis_title="Minggu", yaxis_title="Unit terjual")
    st.plotly_chart(plot_style(fig, 300, False), use_container_width=True)
    st.dataframe(pd.DataFrame({"Minggu": [f"W{i}" for i in range(n_weeks)], "Unit terjual": values}),
                 use_container_width=True, hide_index=True, height=230)

    st.markdown("### Pemeriksaan data")
    quality = pd.DataFrame([
        ["Jumlah produk", len(sales)], ["Duplikasi kode produk", duplicate_count],
        ["Nilai kosong pada kolom minggu", missing_count], ["Nilai penjualan negatif", negative_count],
        ["Persentase observasi bernilai nol", zero_share]
    ], columns=["Pemeriksaan", "Hasil"])
    st.dataframe(quality, use_container_width=True, hide_index=True)

# --------------------------- Page 3 ---------------------------
elif page == "Peluang & risiko produk":
    st.markdown('<div class="kicker">PORTFOLIO ACTION / PRODUCT PRIORITIES</div>', unsafe_allow_html=True)
    st.title("Peluang dan risiko produk")
    st.markdown('<div class="deck">Gunakan kombinasi volume, konsistensi, dan frekuensi minggu nol untuk menyusun daftar produk yang perlu diperiksa lebih lanjut.</div>', unsafe_allow_html=True)

    m = product_metrics.copy()
    p75 = m["Total 52 minggu"].quantile(.75)
    med_cv = m["CV"].median()
    high_volume = m[m["Total 52 minggu"] >= p75].sort_values("Total 52 minggu", ascending=False)
    unstable = m[(m["CV"] >= med_cv) & (m["Rata-rata/minggu"] > 0)].sort_values("CV", ascending=False)
    intermittent = m[m["Minggu tanpa penjualan (%)"] >= 50].sort_values("Total 52 minggu", ascending=False)
    declining_ids = []
    # Linear trend on original units; descriptive only
    slopes = np.polyfit(np.arange(n_weeks), raw.T, 1)[0]
    trend_table = pd.DataFrame({"Product_Code": sales["Product_Code"].astype(str), "Slope": slopes})
    declining = trend_table[trend_table["Slope"] < 0].sort_values("Slope").head(20)

    a,b,c,d = st.columns(4)
    a.metric("Produk volume tinggi", f"{len(high_volume):,}", "≥ kuartil 3 total")
    b.metric("Produk CV di atas median", f"{len(unstable):,}")
    c.metric("Produk ≥50% minggu nol", f"{len(intermittent):,}")
    d.metric("Produk dengan tren linear turun", f"{(trend_table.Slope < 0).sum():,}")

    st.markdown("### Daftar tinjauan operasional")
    tab1,tab2,tab3,tab4 = st.tabs(["Jaga ketersediaan", "Permintaan tidak stabil", "Penjualan intermiten", "Tren menurun"])
    with tab1:
        st.write("Produk bervolume tinggi layak masuk daftar pemantauan ketersediaan. Ini belum membuktikan produk paling menguntungkan karena data margin dan stok tidak tersedia.")
        st.dataframe(high_volume.head(30), use_container_width=True, hide_index=True)
    with tab2:
        st.write("Variabilitas tinggi dapat membuat kebutuhan persediaan berubah tajam. Tinjau kalender promosi, lead time, dan pola lonjakan.")
        st.dataframe(unstable.head(30), use_container_width=True, hide_index=True)
    with tab3:
        st.write("Minggu tanpa penjualan yang sering dapat menunjukkan permintaan jarang, musiman, atau ketersediaan yang tidak konsisten. Perlu verifikasi dengan data stok.")
        st.dataframe(intermittent.head(30), use_container_width=True, hide_index=True)
    with tab4:
        st.write("Kemiringan dihitung dengan regresi linear sederhana pada 52 minggu. Ini sinyal awal, bukan uji signifikansi atau ramalan.")
        st.dataframe(declining.merge(product_metrics, on="Product_Code", how="left"), use_container_width=True, hide_index=True)

    st.markdown('<div class="warning-note"><b>Batasan bisnis:</b> data yang tersedia hanya unit terjual per minggu. '
                'Dashboard tidak memiliki harga, margin, stok, stockout, biaya simpan, promosi, maupun lead time. '
                'Jangan menafsirkan volume tinggi sebagai laba tinggi atau minggu nol sebagai bukti tidak ada permintaan.</div>',
                unsafe_allow_html=True)

# --------------------------- Page 4 ---------------------------
elif page == "Segmentasi produk":
    st.markdown('<div class="kicker">PRODUCT SEGMENTS / CLUSTER PROFILE</div>', unsafe_allow_html=True)
    st.title("Segmentasi pola penjualan")
    st.markdown('<div class="deck">Cluster mengelompokkan produk berdasarkan kemiripan pola mingguan. Gunakan profil penjualan asli untuk memberi arti operasional pada tiap kelompok.</div>', unsafe_allow_html=True)

    sizes = cluster_frame["Cluster"].value_counts().sort_index()
    a,b,c = st.columns(3)
    a.metric("Jumlah cluster", f"{len(sizes)}")
    b.metric("Produk dalam cluster terbesar", f"{sizes.max():,}")
    b.caption(f"Cluster {sizes.idxmax()}")
    c.metric("Silhouette metode", fmt(silhouette, 3))
    if pd.notna(silhouette) and float(silhouette) < .25:
        st.markdown('<div class="warning-note"><b>Pemisahan cluster lemah.</b> Gunakan cluster sebagai ringkasan pola dominan, bukan batas kelompok yang tegas. Validasi hasil dengan kebutuhan bisnis.</div>', unsafe_allow_html=True)

    st.markdown("### Ukuran dan karakter tiap cluster")
    size_df = sizes.rename_axis("Cluster").reset_index(name="Jumlah produk")
    size_df["Porsi portofolio (%)"] = size_df["Jumlah produk"] / n_products * 100
    fig = px.bar(size_df, x="Cluster", y="Jumlah produk", text="Porsi portofolio (%)",
                 color="Cluster", color_discrete_sequence=PALETTE,
                 title="Jumlah produk per cluster")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(plot_style(fig, 320, False), use_container_width=True)

    chosen = st.selectbox("Pilih cluster untuk ditinjau", sorted(sizes.index), key="cluster_profile")
    part = cluster_frame[cluster_frame["Cluster"] == chosen].copy()
    met = calc_product_metrics(part)
    avg_pattern = part[WEEKS].mean(axis=0).to_numpy(float)
    median_pattern = part[WEEKS].median(axis=0).to_numpy(float)
    a,b,c,d = st.columns(4)
    a.metric("Anggota", f"{len(part):,}")
    b.metric("Rata-rata unit / minggu", fmt(met["Rata-rata/minggu"].mean(), 2))
    c.metric("CV rata-rata", fmt(met["CV"].mean(), 2))
    d.metric("Minggu nol rata-rata", f"{met['Minggu tanpa penjualan (%)'].mean():.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n_weeks), y=avg_pattern, mode="lines",
                             line=dict(color="#315d45", width=3.2), name="Rata-rata cluster"))
    fig.add_trace(go.Scatter(x=np.arange(n_weeks), y=median_pattern, mode="lines",
                             line=dict(color="#c18a3b", width=2, dash="dash"), name="Median cluster"))
    fig.update_layout(title=f"Pola penjualan asli · Cluster {chosen}", xaxis_title="Minggu", yaxis_title="Unit terjual")
    st.plotly_chart(plot_style(fig, 350), use_container_width=True)

    # Business-readable profile text
    mean_v = met["Rata-rata/minggu"].mean()
    cv_v = met["CV"].mean()
    zero_v = met["Minggu tanpa penjualan (%)"].mean()
    if mean_v >= product_metrics["Rata-rata/minggu"].quantile(.75):
        level = "volume relatif tinggi"
        action = "periksa ketersediaan, frekuensi replenishment, dan risiko stockout."
    elif mean_v <= product_metrics["Rata-rata/minggu"].quantile(.25):
        level = "volume relatif rendah"
        action = "tinjau kontribusi produk terhadap assortment dan biaya penyimpanannya."
    else:
        level = "volume menengah"
        action = "atur pemantauan berkala dan bandingkan dengan kapasitas penyimpanan."
    if zero_v >= 50:
        pattern = "permintaan intermiten: setidaknya separuh minggu rata-rata tidak mencatat penjualan."
        action += " Hindari mengandalkan rata-rata mingguan saja untuk menentukan pemesanan."
    elif cv_v > 1.5:
        pattern = "permintaan berfluktuasi tinggi."
        action += " Tinjau periode lonjakan dan kemungkinan pengaruh promosi atau pasokan."
    else:
        pattern = "pola relatif lebih rutin dibanding kelompok dengan frekuensi nol atau CV lebih tinggi."
    st.markdown(f'<div class="insight"><b>Ringkasan cluster {chosen}</b><br>Kelompok ini memiliki {level} dan {pattern}<br><b>Hal yang perlu ditinjau:</b> {action}</div>', unsafe_allow_html=True)

    st.markdown("### Anggota cluster")
    members = met.sort_values("Total 52 minggu", ascending=False)
    st.dataframe(members, use_container_width=True, hide_index=True, height=350)
    st.download_button("Unduh daftar anggota cluster (CSV)", members.to_csv(index=False).encode("utf-8"),
                       file_name=f"cluster_{chosen}_{method_key}.csv", mime="text/csv")

# --------------------------- Page 5 ---------------------------
elif page == "Tinjau satu produk":
    st.markdown('<div class="kicker">SKU REVIEW / PRODUCT DETAIL</div>', unsafe_allow_html=True)
    st.title("Tinjau satu produk")
    st.markdown('<div class="deck">Periksa riwayat unit terjual, konsistensi, dan posisi produk pada ketiga pendekatan clustering.</div>', unsafe_allow_html=True)
    product = st.selectbox("Kode produk", sales["Product_Code"].astype(str).tolist())
    row = sales[sales["Product_Code"].astype(str) == product].iloc[0]
    pm = product_metrics[product_metrics["Product_Code"] == product].iloc[0]
    a,b,c,d = st.columns(4)
    a.metric("Total 52 minggu", f"{pm['Total 52 minggu']:,.0f} unit")
    b.metric("Rata-rata per minggu", fmt(pm["Rata-rata/minggu"], 2))
    c.metric("Minggu tanpa penjualan", f"{pm['Minggu tanpa penjualan (%)']:.1f}%")
    d.metric("CV", fmt(pm["CV"], 2))
    vals = pd.to_numeric(row[WEEKS], errors="coerce").fillna(0).to_numpy(float)
    fig = go.Figure(go.Bar(x=np.arange(n_weeks), y=vals, marker_color="#547b5e"))
    fig.update_layout(title=f"Riwayat unit terjual · {product}", xaxis_title="Minggu", yaxis_title="Unit terjual")
    st.plotly_chart(plot_style(fig, 350, False), use_container_width=True)
    st.markdown("### Penempatan cluster menurut metode")
    boxes = st.columns(3)
    for col, (name, (label, key, distance)) in zip(boxes, METHODS.items()):
        value = row[label] if label in row.index else "—"
        col.metric(name, f"Cluster {int(value)}" if pd.notna(value) else "Tidak tersedia")
        if key in meta:
            col.caption(f"Silhouette: {fmt(meta[key].get('silhouette'), 3)}")
    st.markdown("### Ringkasan penjualan")
    st.dataframe(pm.to_frame("Nilai"), use_container_width=True)

# --------------------------- Page 6 ---------------------------
elif page == "Evaluasi metode":
    st.markdown('<div class="kicker">MODEL REVIEW / METHOD COMPARISON</div>', unsafe_allow_html=True)
    st.title("Perbandingan metode clustering")
    st.markdown('<div class="deck">Evaluasi ini membantu memilih representasi pola yang paling berguna. Kualitas pemisahan cluster perlu dibaca terpisah dari nilai penjualan dan manfaat operasional.</div>', unsafe_allow_html=True)
    rows = []
    for name, (_, key, distance) in METHODS.items():
        info = meta.get(key, {})
        rows.append({"Metode":name, "Jarak":distance, "Jumlah cluster (k)":info.get("k"),
                     "Radius (minggu)":info.get("window", "—"), "Silhouette":info.get("silhouette")})
    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    fig = px.bar(comp, x="Metode", y="Silhouette", color="Metode", color_discrete_sequence=PALETTE,
                 text="Silhouette", title="Silhouette masing-masing metode")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(plot_style(fig, 330, False), use_container_width=True)

    if optional.get("cross") is not None:
        st.markdown("### Silhouette lintas matriks jarak")
        fig = px.imshow(optional["cross"], text_auto=".3f", aspect="auto",
                        color_continuous_scale=["#f3f4ef","#b7cbb8","#315d45"],
                        title="Label cluster dibandingkan pada matriks jarak")
        st.plotly_chart(plot_style(fig, 330, True), use_container_width=True)
    if optional.get("ari") is not None:
        st.markdown("### Kesepakatan label antar-metode (ARI)")
        fig = px.imshow(optional["ari"], text_auto=".3f", aspect="auto", zmin=0, zmax=1,
                        color_continuous_scale=["#f3f4ef","#b7cbb8","#315d45"])
        st.plotly_chart(plot_style(fig, 330, True), use_container_width=True)
    st.markdown('<div class="warning-note"><b>Interpretasi:</b> silhouette rendah berarti kelompok belum terpisah kuat berdasarkan jarak yang digunakan. '
                'Metode dengan silhouette terbesar tidak otomatis paling baik untuk bisnis. Periksa juga apakah profil cluster dapat diterjemahkan menjadi tindakan yang masuk akal.</div>',
                unsafe_allow_html=True)
    st.markdown("### Cara membaca metode")
    method_notes = {
        "K-Medoids + DTW": "Membandingkan deret dengan penyelarasan waktu lokal. Relevan ketika bentuk pola serupa muncul pada minggu yang sedikit bergeser.",
        "K-Medoids + DDTW": "Membandingkan pola perubahan/kemiringan antarperiode. Perlu hati-hati karena turunan dapat memperbesar noise.",
        "K-Shape + SBD": "Membandingkan bentuk deret dengan normalized cross-correlation. Skala volume harus diperiksa lagi menggunakan unit asli."
    }
    for name, note in method_notes.items():
        st.markdown(f"**{name}.** {note}")

st.markdown("---")
st.caption("Product Portfolio | Sales Review · Analisis historis 52 minggu. Tidak mencakup margin, harga, stok, promosi, lead time, atau ramalan permintaan.")
