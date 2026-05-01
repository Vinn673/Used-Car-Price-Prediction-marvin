import pandas as pd
import streamlit as st

from utils.data_loader import load_raw_data, load_processed_data, load_clean_data
from utils.stats import get_preprocessing_stats

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")

with open("style/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
        <h1>⚙️ Preprocessing Pipeline</h1>
        <p style='font-size: 1rem; color: #718096;'>
            Tahapan pembersihan dan transformasi data sebelum training
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Load data
df1_raw, df2_raw = load_raw_data()
df_processed = load_processed_data()
df_clean = load_clean_data()

stats = get_preprocessing_stats(
    pd.concat([df1_raw, df2_raw], ignore_index=True), df_clean
)

# --- Step-by-step Pipeline ---
st.markdown(
    """
    <div class='section-header'>
        <h3>Pipeline Steps</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

steps = [
    ("Load 2 Dataset", f"used_car.csv ({len(df1_raw):,} rows) + used_car_data_new.csv ({len(df2_raw):,} rows)"),
    ("Merge & Select Columns", "Kolom: brand, model, year, mileage (km), transmission, price (Rp)"),
    ("Map Transmission", "Dataset 2: id_transmission 1→manual, 2→automatic"),
    ("Normalize Text", "brand, model, transmission → lowercase + strip"),
    ("Mileage Conversion", "Dataset 1: mileage dikali 1000 (ribuan → km)"),
    ("Remove Duplicates", f"{stats['raw_duplicates']} duplikat dihapus"),
    ("Drop Missing Values", f"{stats['raw_missing']} missing values dibersihkan"),
    ("Remove Outlier (IQR)", "Outlier harga di luar Q1-1.5*IQR s/d Q3+1.5*IQR dihapus"),
    ("Feature Engineering", "Tambah kolom car_age = 2026 - year"),
]

for i, (step, desc) in enumerate(steps, 1):
    st.markdown(
        f"""
        <div class='pipeline-step'>
            <div class='step-number'>{i}</div>
            <div>
                <strong>{step}</strong><br>
                <span style='color: #718096; font-size: 0.85rem;'>{desc}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Before / After Stats ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{stats['raw_rows']:,}</div>
            <div class='metric-label'>Raw Rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{stats['clean_rows']:,}</div>
            <div class='metric-label'>Clean Rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value' style='color: #E53E3E;'>{stats['removed_rows']:,}</div>
            <div class='metric-label'>Removed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Data Preview ---
tab1, tab2 = st.tabs(["Data Setelah Merge", "Data Setelah Clean"])

with tab1:
    st.dataframe(df_processed.head(20), use_container_width=True)

with tab2:
    st.dataframe(df_clean.head(20), use_container_width=True)
