# Streamlit Web App Lengkap: Klasterisasi Siswa dengan K-Means & K-Medoids

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.impute import SimpleImputer
from sklearn_extra.cluster import KMedoids
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

st.set_page_config(page_title="Klasterisasi Siswa SMP", layout="wide")
st.title("Klasterisasi Siswa SMP Berdasarkan Nilai Rapor")

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV nilai siswa", type=["csv"])

if uploaded_file is not None:
    # Baca dan bersihkan data
    df = pd.read_csv(uploaded_file, delimiter=';')

    # Ganti koma jadi titik, '-' jadi NaN, dan konversi ke float
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = df[col].str.replace('-', 'NaN').str.replace(',', '.').astype(float)
        except:
            pass

    # Imputasi nilai kosong
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    # Hitung Pengetahuan dan Keterampilan
    df["Pengetahuan_Sains"] = df[["IPA", "MTK", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Pengetahuan_Sosial"] = df[["IPS", "BIN", "BING", "SUN", "PAI", "PKN"]].mean(axis=1)
    df["Keterampilan_Tertinggi"] = df[["PRK", "SBDY", "PNJ"]].max(axis=1)

    fitur = ["Pengetahuan_Sains", "Pengetahuan_Sosial", "Keterampilan_Tertinggi"]
    X = df[fitur]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Pilih Jumlah Klaster")
    k = st.slider("Jumlah Klaster", 2, 10, 3)

    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Klaster_KMeans"] = kmeans.fit_predict(X_scaled)

    # K-Medoids
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    df["Klaster_KMedoids"] = kmedoids.fit_predict(X_scaled)

    # DBI
    dbi_kmeans = davies_bouldin_score(X_scaled, df["Klaster_KMeans"])
    dbi_kmedoids = davies_bouldin_score(X_scaled, df["Klaster_KMedoids"])

    # Predikat Akademik
    def klasifikasi_predikat(sains, sosial, keterampilan):
        rata2 = (sains + sosial + keterampilan) / 3
        if rata2 >= 90:
            return "Sangat Baik"
        elif rata2 >= 80:
            return "Baik"
        elif rata2 >= 70:
            return "Cukup"
        else:
            return "Perlu Bimbingan"

    df["Predikat"] = df.apply(lambda row: klasifikasi_predikat(
        row["Pengetahuan_Sains"], row["Pengetahuan_Sosial"], row["Keterampilan_Tertinggi"]), axis=1)

    # Gabungan Pengetahuan + Keterampilan
    df["Gabungan"] = df.apply(
        lambda row: f"{'Sains' if row['Pengetahuan_Sains'] > row['Pengetahuan_Sosial'] else 'Sosial'} - {['PRK', 'SBDY', 'PNJ'][np.argmax([row['PRK'], row['SBDY'], row['PNJ']])]}" if not pd.isna(row['PRK']) and not pd.isna(row['SBDY']) and not pd.isna(row['PNJ']) else "-",
        axis=1
    )

    st.subheader("Tabel dan Evaluasi")
    st.write(f"**Davies-Bouldin Index K-Means:** {dbi_kmeans:.4f}")
    st.write(f"**Davies-Bouldin Index K-Medoids:** {dbi_kmedoids:.4f}")
    st.dataframe(df[fitur + ["Klaster_KMeans", "Klaster_KMedoids", "Predikat", "Gabungan"]])

    st.subheader("Visualisasi 2D")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=X["Pengetahuan_Sains"], y=X["Keterampilan_Tertinggi"], hue=df["Klaster_KMeans"], palette="tab10", ax=ax1)
        ax1.set_title("K-Means Clustering")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=X["Pengetahuan_Sains"], y=X["Keterampilan_Tertinggi"], hue=df["Klaster_KMedoids"], palette="Set2", ax=ax2)
        ax2.set_title("K-Medoids Clustering")
        st.pyplot(fig2)

    st.subheader("Visualisasi 3D")
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X["Pengetahuan_Sains"], X["Pengetahuan_Sosial"], X["Keterampilan_Tertinggi"], c=df["Klaster_KMeans"], cmap="tab10")
    ax.set_title("K-Means 3D")
    ax.set_xlabel("Sains")
    ax.set_ylabel("Sosial")
    ax.set_zlabel("Keterampilan")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X["Pengetahuan_Sains"], X["Pengetahuan_Sosial"], X["Keterampilan_Tertinggi"], c=df["Klaster_KMedoids"], cmap="Set2")
    ax2.set_title("K-Medoids 3D")
    ax2.set_xlabel("Sains")
    ax2.set_ylabel("Sosial")
    ax2.set_zlabel("Keterampilan")
    st.pyplot(fig)

    st.subheader("Distribusi Pie Chart")
    fig3, ax3 = plt.subplots()
    df["Klaster_KMeans"].value_counts().sort_index().plot.pie(autopct="%1.1f%%", ax=ax3)
    ax3.set_ylabel("")
    ax3.set_title("Distribusi KMeans")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    df["Klaster_KMedoids"].value_counts().sort_index().plot.pie(autopct="%1.1f%%", ax=ax4)
    ax4.set_ylabel("")
    ax4.set_title("Distribusi KMedoids")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    df["Gabungan"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax5, colors=plt.cm.Paired.colors)
    ax5.set_ylabel("")
    ax5.set_title("Gabungan Pengetahuan & Keterampilan")
    st.pyplot(fig5)

    st.subheader("Unduh Hasil")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "hasil_klasterisasi.csv", "text/csv")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
