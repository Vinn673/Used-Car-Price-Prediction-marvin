import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import load_processed_data, load_raw_data
from utils.stats import get_dashboard_metrics, get_top_brands, get_price_by_brand

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

with open("style/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
        <h1>📊 Exploratory Data Analysis</h1>
        <p style='font-size: 1rem; color: #718096;'>
            Analisis distribusi, korelasi, dan pola harga mobil bekas
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

df = load_processed_data()
df1_raw, df2_raw = load_raw_data()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Distribusi Harga", "Brand Analysis", "Tren Tahun", "Dataset Mentah"]
)

# --- Tab 1: Distribusi Harga ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='chart-card'>
                <h3>Distribusi Harga</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["price (Rp)"] / 1e6, bins=30, kde=True, ax=ax, color="#1E3A5F")
        ax.set_xlabel("Harga (Juta Rp)")
        ax.set_ylabel("Jumlah")
        ax.set_title("")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(
            """
            <div class='chart-card'>
                <h3>Boxplot Harga (Deteksi Outlier)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df["price (Rp)"] / 1e6, ax=ax, color="#2D5F8A")
        ax.set_xlabel("Harga (Juta Rp)")
        st.pyplot(fig)
        plt.close()

    st.markdown(
        """
        <div class='card'>
            <h3 class='section-header'>Statistik Deskriptif Harga</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    desc = df["price (Rp)"].describe().to_frame()
    desc.columns = ["Value"]
    st.dataframe(desc, use_container_width=True)


# --- Tab 2: Brand Analysis ---
with tab2:
    col1, col2 = st.columns(2)

    top_brands = get_top_brands(df, 10)

    with col1:
        st.markdown(
            """
            <div class='chart-card'>
                <h3>Top 10 Brand (Jumlah Listing)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=top_brands,
            x="count",
            y="brand",
            ax=ax,
            palette="Blues_r",
        )
        ax.set_xlabel("Jumlah Listing")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(
            """
            <div class='chart-card'>
                <h3>Median Harga per Brand (Top 10)</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        price_brand = get_price_by_brand(df).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=price_brand,
            x="median",
            y="brand",
            ax=ax,
            palette="Blues_r",
        )
        ax.set_xlabel("Median Harga (Rp)")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()


# --- Tab 3: Tren Tahun ---
with tab3:
    st.markdown(
        """
        <div class='chart-card'>
            <h3>Rata-rata Harga per Tahun</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    yearly = df.groupby("year")["price (Rp)"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=yearly, x="year", y="price (Rp)", ax=ax, marker="o", color="#1E3A5F")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Rata-rata Harga (Rp)")
    ax.set_title("")
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class='chart-card'>
                <h3>Distribusi Tahun</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["year"], bins=20, ax=ax, color="#2D5F8A")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(
            """
            <div class='chart-card'>
                <h3>Distribusi Mileage</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["mileage (km)"] / 1000, bins=30, ax=ax, color="#2D5F8A")
        ax.set_xlabel("Mileage (Ribu km)")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)
        plt.close()


# --- Tab 4: Dataset Mentah ---
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class='card'>
                <h3 class='section-header'>Dataset 1: used_car.csv</h3>
                <p><strong>{len(df1_raw):,}</strong> baris &times; <strong>{len(df1_raw.columns)}</strong> kolom</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(df1_raw.head(10), use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div class='card'>
                <h3 class='section-header'>Dataset 2: used_car_data_new.csv</h3>
                <p><strong>{len(df2_raw):,}</strong> baris &times; <strong>{len(df2_raw.columns)}</strong> kolom</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(df2_raw.head(10), use_container_width=True)
