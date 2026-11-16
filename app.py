# -*- coding: utf-8 -*-
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Segmentasi Pola Penjualan", page_icon="📈", layout="wide")
DATA = Path(__file__).parent / "dashboard_data"
C1, C2, INK, MUT = "#1d4ed8", "#e8890c", "#0f1b2d", "#5b6678"
COL = {1: C1, 2: C2}
NM = {"dtw": "K-Medoids + DTW", "ddtw": "K-Medoids + DDTW", "ks": "K-Shape + SBD"}
CL = {"dtw": "Cluster_DTW", "ddtw": "Cluster_DDTW", "ks": "Cluster_KShape"}

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif}
.block-container{max-width:1180px;padding-top:2rem}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e3e7ee}
h1{font-size:2.5rem!important;letter-spacing:-.03em;line-height:1.1!important}
h2,h3{letter-spacing:-.02em}
.sub{color:#5b6678;font-size:1.05rem;max-width:760px;margin:-.4rem 0 1.4rem}
.kpi{background:#fff;border:1px solid #e3e7ee;border-radius:14px;padding:18px 20px}
.kpi b{display:block;font-size:1.9rem;letter-spacing:-.02em;color:#0f1b2d}.kpi span{color:#5b6678;font-size:.85rem}
.note{background:#fff7e8;border:1px solid #f3dcae;border-radius:10px;padding:12px 16px;font-size:.9rem}
.pill{display:inline-block;padding:2px 12px;border-radius:99px;color:#fff;font-size:.8rem;font-weight:600}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    d = {"s": pd.read_csv(DATA / "sales_clusters.csv"), "f": pd.read_csv(DATA / "product_features.csv"),
         "meta": json.load(open(DATA / "summary.json"))}
    for n in ["k_curve_dtw", "k_curve_ddtw", "evaluation_dtw", "evaluation_ddtw", "evaluation_kshape", "medoids"]:
        p = DATA / f"{n}.csv"
        d[n] = pd.read_csv(p) if p.exists() else None
    for n in ["cross_silhouette", "ari"]:
        p = DATA / f"{n}.csv"
        d[n] = pd.read_csv(p, index_col=0) if p.exists() else None
    for n in ["medoid_patterns_dtw", "medoid_patterns_ddtw", "kshape_centroids"]:
        p = DATA / f"{n}.npy"
        d[n] = np.squeeze(np.load(p)) if p.exists() else None
    pj = glob.glob(str(DATA / "*proj*.csv"))
    d["proj"] = pd.read_csv(pj[0]) if pj else None
    return d


D = load()
S, F, META = D["s"], D["f"], D["meta"]
WK = [c for c in S.columns if c.startswith("W") and c[1:].isdigit()]
X = S[WK].to_numpy(float)
N, M = X.shape
fid = "product_id" if "product_id" in F.columns else F.columns[0]
FE = F.merge(S[["Product_Code"] + list(CL.values())], left_on=fid, right_on="Product_Code")


def label(m, c):
    """Cluster dengan rata-rata penjualan lebih tinggi disebut volume tinggi."""
    mu = {k: X[S[CL[m]] == k].mean() for k in (1, 2)}
    return "Volume tinggi" if mu[c] == max(mu.values()) else "Volume rendah / intermiten"


def style(fig, h=340):
    fig.update_layout(template="simple_white", height=h, margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="Inter", color=INK), legend=dict(orientation="h", y=-.2),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(gridcolor="#eef1f6"); fig.update_yaxes(gridcolor="#eef1f6")
    return fig


def kpi(col, v, t):
    col.markdown(f'<div class="kpi"><b>{v}</b><span>{t}</span></div>', unsafe_allow_html=True)


def head(t, s):
    st.markdown(f"# {t}"); st.markdown(f'<p class="sub">{s}</p>', unsafe_allow_html=True)


def best(m):
    b = META[m]
    return b["silhouette"], b["k"], b.get("window")


# ---------------------------------------------------------------- halaman
def page_ringkasan():
    head("Pola penjualan mingguan 811 produk, dikelompokkan tiga cara",
         "K-Medoids dengan DTW dan DDTW, serta K-Shape dengan SBD, dihitung pada 52 minggu penjualan.")
    c = st.columns(4)
    kpi(c[0], f"{N}", "Produk"); kpi(c[1], f"{M}", "Minggu per produk")
    kpi(c[2], f"{(X == 0).mean() * 100:.1f}%", "Sel penjualan bernilai nol")
    kpi(c[3], f"{META['dtw']['silhouette']:.3f}", "Silhouette tertinggi (DTW)")
    st.write("")
    rows = []
    for m, k in [("dtw", "dtw"), ("ddtw", "ddtw"), ("ks", "kshape")]:
        s, kk, w = best(k)
        sz = S[CL[m]].value_counts().sort_index()
        rows.append({"Metode": NM[m], "Parameter": f"radius {w} minggu" if w is not None else "n_init 10",
                     "k": kk, "Silhouette": round(s, 4), "Ukuran cluster": " / ".join(str(v) for v in sz)})
    st.subheader("Perbandingan metode")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown('<div class="note"><b>Kesimpulan.</b> K-Medoids + DTW memberi silhouette tertinggi, tetapi semua '
                'metode berada di bawah 0,25. Produk tidak membentuk kelompok yang terpisah jelas. Perbedaan utama '
                'antar-cluster adalah volume dan intermitensi penjualan.</div>', unsafe_allow_html=True)


def page_data():
    head("Data dan pra-pemrosesan",
         "Setiap produk adalah satu deret waktu berpanjang m = 52 minggu. Jarak dihitung pada deret ter-normalisasi "
         "agar yang dibandingkan adalah bentuk pola, bukan volume.")
    a, b = st.columns(2)
    with a:
        st.subheader("Normalisasi z-score")
        st.latex(r"z_t=\frac{x_t-\mu}{\sigma},\quad \mu=\frac1m\sum_{t=1}^{m}x_t,\quad "
                 r"\sigma=\sqrt{\frac1m\sum_{t=1}^{m}(x_t-\mu)^2}")
        st.write("Hasilnya rata-rata 0 dan simpangan baku 1 untuk tiap produk. Data asli tetap dipakai untuk profil "
                 "cluster agar angka dapat dibaca dalam satuan unit.")
    with b:
        st.subheader("Total penjualan per minggu")
        fig = go.Figure(go.Scatter(y=X.sum(0), x=list(range(1, M + 1)), mode="lines", line=dict(color=INK, width=2.5)))
        st.plotly_chart(style(fig.update_layout(xaxis_title="Minggu", yaxis_title="Unit")), use_container_width=True)
    st.subheader("Sebaran per produk")
    st.dataframe(FE[[fid, "mean", "total_sales", "cv", "zero_proportion"]].describe().round(2), use_container_width=True)


ALGO = {
    "dtw": ("Mengelompokkan produk menurut kemiripan bentuk deret dengan toleransi pergeseran waktu kecil.",
            ["Normalisasi z-score setiap deret.",
             "Hitung jarak DTW semua pasangan produk pada radius Sakoe-Chiba r (matriks 811 × 811).",
             "Grid search r (52 sampai 0 minggu) dan k = 2 sampai 8, masing-masing dengan K-Medoids (PAM).",
             "Pilih silhouette tertinggi. Bila seri, pilih radius lalu k terkecil.",
             "Ekstraksi fitur dari data asli untuk menafsirkan cluster."],
            [(r"D(i,j)=(a_i-b_j)^2+\min\{D(i-1,j),\,D(i,j-1),\,D(i-1,j-1)\}",
              "Rekursi pemrograman dinamis dengan D(0,0)=0. Jarak akhir d_DTW(a,b)=√D(m,m)."),
             (r"|i-j|\le r", "Pita Sakoe-Chiba. r = 0 setara jarak Euclidean, r = 52 adalah DTW penuh."),
             (r"m_c=\arg\min_{x\in C}\sum_{y\in C}d(x,y)",
              "PAM: BUILD memilih medoid awal yang paling menurunkan biaya, SWAP menukar medoid dengan non-medoid "
              "selama total biaya turun.")]),
    "ddtw": ("Sama seperti DTW tetapi pada turunan deret: yang dibandingkan adalah arah dan laju perubahan.",
             ["Normalisasi z-score setiap deret.", "Ubah tiap deret menjadi turunannya (panjang 52 menjadi 50).",
              "Hitung jarak DTW pada turunan dengan radius 0 sampai 50.",
              "Jalankan K-Medoids (PAM) untuk k = 2 sampai 8, pilih silhouette tertinggi.",
              "Ekstraksi fitur dari data asli untuk profil cluster."],
             [(r"a'_i=\frac{(a_i-a_{i-1})+(a_{i+1}-a_{i-1})/2}{2},\quad i=2,\dots,m-1",
               "Turunan menurut Keogh dan Pazzani (2001): rata-rata kemiringan kiri dan kanan titik."),
              (r"d_{DDTW}(a,b)=d_{DTW}(a',b')", "Rekursi DTW sama, inputnya deret turunan."),
              (r"\text{derau}\uparrow", "Pada penjualan intermiten, turunan memperbesar lonjakan dan minggu nol, "
                                         "sehingga pemisahan cenderung lebih lemah dari DTW.")]),
    "ks": ("Mengelompokkan produk menurut bentuk yang tak terpengaruh pergeseran dan skala. Pusatnya berupa centroid "
           "hasil ekstraksi bentuk.",
           ["Normalisasi z-score setiap deret.", "Mulai dari penugasan cluster acak (n_init = 10, ambil inertia terkecil).",
            "Perbarui centroid: selaraskan anggota dengan pergeseran optimal, lalu ekstraksi bentuk lewat vektor eigen.",
            "Tugaskan ulang tiap produk ke centroid dengan SBD terkecil.",
            "Ulangi langkah 3 dan 4 sampai label tetap atau 100 iterasi. Pilih k dengan silhouette SBD tertinggi."],
           [(r"NCC_c(x,y)=\max_w\frac{CC_w(x,y)}{\|x\|\,\|y\|},\quad CC_w=\sum_l x_{l+w-m}\,y_l",
             "Korelasi silang ternormalisasi untuk semua pergeseran w, dihitung dengan FFT."),
            (r"SBD(x,y)=1-NCC_c(x,y)", "Shape-Based Distance, bernilai 0 sampai 2."),
            (r"\mu^*=\arg\max_\mu\frac{\mu^\top Q S Q\,\mu}{\mu^\top\mu},\; S=\sum_{x\in C}xx^\top,\; Q=I-\tfrac1m\mathbf 1\mathbf 1^\top",
             "Centroid adalah vektor eigen dengan nilai eigen terbesar dari QSQ.")]),
}


def page_method(m):
    key = {"dtw": "dtw", "ddtw": "ddtw", "ks": "kshape"}[m]
    s, k, w = best(key)
    head(NM[m], ALGO[m][0])
    sz = S[CL[m]].value_counts().sort_index()
    c = st.columns(3)
    kpi(c[0], f"{s:.4f}", "Silhouette terbaik")
    kpi(c[1], f"{k}" + (f" · radius {w}" if w is not None else ""), "Jumlah cluster")
    kpi(c[2], " / ".join(str(v) for v in sz), "Produk per cluster")
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Algoritma dan rumus", "Pola cluster", "Pemilihan parameter", "Profil cluster"])
    with t1:
        a, b = st.columns(2)
        with a:
            st.subheader("Langkah algoritma")
            for i, t in enumerate(ALGO[m][1], 1):
                st.markdown(f"{i}. {t}")
        with b:
            st.subheader("Rumus")
            for lt, ex in ALGO[m][2]:
                st.latex(lt); st.caption(ex)
    with t2:
        a, b = st.columns(2)
        fig = go.Figure()
        for c_ in (1, 2):
            sel = X[S[CL[m]] == c_]
            fig.add_scatter(x=list(range(1, M + 1)), y=sel.mean(0), name=f"Cluster {c_}: {label(m, c_)} ({len(sel)})",
                            line=dict(color=COL[c_], width=2.5))
        a.subheader("Rata-rata penjualan per cluster")
        a.plotly_chart(style(fig.update_layout(xaxis_title="Minggu", yaxis_title="Unit")), use_container_width=True)
        pat = D["kshape_centroids"] if m == "ks" else D[f"medoid_patterns_{m}"]
        b.subheader("Centroid (z-score)" if m == "ks" else "Medoid tiap cluster (z-score)")
        if pat is not None:
            fig = go.Figure()
            for i in range(min(2, len(pat))):
                nm = f"Centroid {i + 1}"
                if m != "ks" and D["medoids"] is not None:
                    r = D["medoids"][(D["medoids"]["Cluster"] == i + 1) &
                                     (D["medoids"]["Metode"].str.contains(m.upper(), case=False))]
                    nm = f"Medoid {r['Product_Code'].iloc[0]}" if len(r) else f"Medoid {i + 1}"
                fig.add_scatter(x=list(range(1, len(pat[i]) + 1)), y=pat[i], name=nm, line=dict(color=COL[i + 1], width=2.2))
            b.plotly_chart(style(fig.update_layout(xaxis_title="Minggu", yaxis_title="z-score")), use_container_width=True)
        if D["proj"] is not None and f"{ {'dtw':'DTW','ddtw':'DDTW','ks':'KShape'}[m] }_x" in D["proj"].columns:
            p = {"dtw": "DTW", "ddtw": "DDTW", "ks": "KShape"}[m]
            st.subheader("Peta jarak 2D (MDS klasik)")
            fig = go.Figure()
            for c_ in (1, 2):
                sel = S[CL[m]] == c_
                fig.add_scatter(x=D["proj"][f"{p}_x"][sel], y=D["proj"][f"{p}_y"][sel], mode="markers",
                                name=f"Cluster {c_}", marker=dict(color=COL[c_], size=6, opacity=.55),
                                text=S["Product_Code"][sel])
            st.plotly_chart(style(fig, 420), use_container_width=True)
    with t3:
        ev = D["evaluation_" + key]; kc = D.get("k_curve_" + key)
        if ev is not None and "window" in ev.columns:
            st.subheader("Silhouette menurut radius Sakoe-Chiba")
            fig = go.Figure()
            for kk, g in ev.groupby("k"):
                g = g.sort_values("window")
                fig.add_scatter(x=g["window"], y=g["silhouette"], name=f"k={kk}", mode="lines")
            fig.add_vline(x=w, line_dash="dash", line_color="#c0392b", annotation_text=f"terbaik: {w}")
            st.plotly_chart(style(fig.update_layout(xaxis_title="Radius (minggu)", yaxis_title="Silhouette"), 380),
                            use_container_width=True)
        tab = kc if kc is not None else ev
        if tab is not None:
            st.subheader("Pemilihan k")
            st.dataframe(tab.drop(columns=[c for c in ["Metode", "window", "band_percentage"] if c in tab.columns])
                         .round(4), hide_index=True, use_container_width=True)
    with t4:
        cols = [c for c in ["mean", "total_sales", "trend_slope", "cv", "zero_proportion", "acf_1", "acf_13",
                            "skewness", "peak_strength"] if c in FE.columns]
        pr = FE.groupby(CL[m])[cols].mean().T
        pr.columns = [f"Cluster {c_}: {label(m, c_)}" for c_ in pr.columns]
        pr.loc["jumlah produk"] = FE[CL[m]].value_counts().sort_index().values
        st.dataframe(pr.round(3), use_container_width=True)
        st.markdown('<div class="note">Silhouette di bawah 0,25 berarti batas antar-cluster tidak tegas. Hasil lebih '
                    'tepat dibaca sebagai segmentasi pola dominan (volume dan intermitensi), bukan kelompok terpisah.</div>',
                    unsafe_allow_html=True)


def page_banding():
    head("Perbandingan tiga metode",
         "Silhouette tiap metode dihitung pada matriks jaraknya sendiri. Di sini setiap pengelompokan dievaluasi pada "
         "ketiga matriks jarak yang sama, lalu kesepakatan antar-metode diukur dengan Adjusted Rand Index.")
    a, b = st.columns(2)
    for col, key, ttl, sc in [(a, "cross_silhouette", "Silhouette pada matriks yang sama", "Blues"),
                              (b, "ari", "Adjusted Rand Index", "Blues")]:
        df = D[key]
        if df is None:
            continue
        df = df.drop(columns=[c for c in ["k"] if c in df.columns])
        fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale=sc,
                                   text=np.round(df.values, 3), texttemplate="%{text}"))
        col.subheader(ttl); col.plotly_chart(style(fig, 330), use_container_width=True)


def page_produk():
    head("Jelajah produk", "Cari kode produk, saring berdasarkan cluster, lalu pilih produk untuk melihat deretnya.")
    a, b, c = st.columns([2, 2, 2])
    q = a.text_input("Cari kode produk", "")
    m = b.selectbox("Metode", list(NM), format_func=NM.get)
    cl = c.selectbox("Cluster", ["Semua", 1, 2])
    df = S[["Product_Code"] + list(CL.values())].copy()
    if q:
        df = df[df["Product_Code"].str.contains(q, case=False)]
    if cl != "Semua":
        df = df[df[CL[m]] == cl]
    if df.empty:
        st.info("Tidak ada produk yang cocok. Ubah kata kunci atau filter cluster."); return
    pick = st.selectbox("Produk", df["Product_Code"].tolist())
    i = S.index[S["Product_Code"] == pick][0]
    fig = go.Figure(go.Scatter(x=list(range(1, M + 1)), y=X[i], mode="lines+markers", line=dict(color=INK, width=2)))
    st.plotly_chart(style(fig.update_layout(title=pick, xaxis_title="Minggu", yaxis_title="Unit")), use_container_width=True)
    st.caption(f"{pick}: DTW cluster {S.at[i, 'Cluster_DTW']}, DDTW cluster {S.at[i, 'Cluster_DDTW']}, "
               f"K-Shape cluster {S.at[i, 'Cluster_KShape']}. Rata-rata {X[i].mean():.2f} unit per minggu.")
    st.dataframe(df.head(200), hide_index=True, use_container_width=True)


def page_rumus():
    head("Rumus evaluasi dan unduh data", "Ukuran yang dipakai untuk memilih parameter dan membandingkan metode.")
    a, b = st.columns(2)
    with a:
        st.subheader("Silhouette")
        st.latex(r"s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}")
        st.caption("a(i): rata-rata jarak ke anggota cluster sendiri. b(i): rata-rata jarak ke cluster terdekat lain.")
        st.subheader("Davies-Bouldin")
        st.latex(r"DB=\frac1k\sum_i\max_{j\ne i}\frac{S_i+S_j}{d(m_i,m_j)}")
        st.caption("S_i: rata-rata jarak anggota ke medoid. Makin kecil makin baik.")
        st.subheader("Cost (elbow)")
        st.latex(r"TD=\sum_i\min_{m\in M}d(x_i,m)")
    with b:
        st.subheader("Adjusted Rand Index")
        st.latex(r"ARI=\frac{\sum_{ij}\binom{n_{ij}}2-E}{\frac12\left[\sum_i\binom{a_i}2+\sum_j\binom{b_j}2\right]-E}")
        st.latex(r"E=\frac{\sum_i\binom{a_i}2\sum_j\binom{b_j}2}{\binom n2}")
        st.subheader("Fitur profil")
        st.latex(r"CV=\frac{\sigma}{\mu},\qquad \rho_k=\frac{\sum_t(x_t-\mu)(x_{t+k}-\mu)}{\sum_t(x_t-\mu)^2}")
        st.latex(r"x_t=\beta_0+\beta_1 t\ \Rightarrow\ \text{tren}=\beta_1,\qquad \text{peak}=\frac{\max-\text{median}}{\sigma}")
    st.subheader("Unduh data")
    st.download_button("sales_clusters.csv", S.to_csv(index=False), "sales_clusters.csv", "text/csv")
    st.download_button("product_features.csv", F.to_csv(index=False), "product_features.csv", "text/csv")


PAGES = {"Ringkasan": page_ringkasan, "Data": page_data, NM["dtw"]: lambda: page_method("dtw"),
         NM["ddtw"]: lambda: page_method("ddtw"), NM["ks"]: lambda: page_method("ks"),
         "Perbandingan": page_banding, "Jelajah produk": page_produk, "Rumus dan unduh": page_rumus}
st.sidebar.markdown("### Time Series Clustering")
PAGES[st.sidebar.radio("Navigasi", list(PAGES), label_visibility="collapsed")]()
