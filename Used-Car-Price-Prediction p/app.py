import streamlit as st

from utils.data_loader import load_processed_data, load_raw_data
from utils.stats import get_dashboard_metrics, get_top_brands

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
with open("style/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Data ---
df1_raw, df2_raw = load_raw_data()
df = load_processed_data()
metrics = get_dashboard_metrics(df)

# --- Header ---
st.markdown(
    """
    <div style='text-align: center; padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='font-size: 2.5rem; margin-bottom: 0.25rem; color: #1E3A5F;'>
            🚗 Car Price Prediction Indonesia
        </h1>
        <p style='font-size: 1.1rem; color: #718096;'>
            Prediksi harga mobil bekas menggunakan Machine Learning &mdash;
            Random Forest Regressor
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --- Metric Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{metrics['total_listings']:,}</div>
            <div class='metric-label'>Total Listings</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{metrics['total_brands']}</div>
            <div class='metric-label'>Brands</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>{metrics['total_models']}</div>
            <div class='metric-label'>Models</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-value'>Rp {metrics['median_price'] / 1e6:.0f} Jt</div>
            <div class='metric-label'>Median Price</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Quick Overview ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown(
        """
        <div class='card'>
            <h3 class='section-header'>📊 Dataset Overview</h3>
            <p><strong>Dataset 1:</strong> used_car.csv &mdash; data mentah pertama</p>
            <p><strong>Dataset 2:</strong> used_car_data_new.csv &mdash; data mentah kedua</p>
            <p><strong>Setelah merge &amp; cleaning:</strong> {:,} baris, 6 kolom</p>
        </div>
        """.format(len(df)),
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
        <div class='card'>
            <h3 class='section-header'>🤖 Model Info</h3>
            <p><strong>Algoritma:</strong> Random Forest Regressor</p>
            <p><strong>Features:</strong> Brand, Model, Year, Mileage, Transmission</p>
            <p><strong>Preprocessing:</strong> One-Hot, Label Encoding, Log Transform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Top Brands Quick View ---
top_brands = get_top_brands(df, 5)

st.markdown(
    """
    <div class='section-header'>
        <h3>🏆 Top 5 Brands</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(5)
for idx, row in top_brands.iterrows():
    with cols[idx]:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value' style='font-size: 1.1rem;'>{row['brand'].upper()}</div>
                <div class='metric-label'>{row['count']:,} listings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# --- Navigation Hint ---
st.markdown(
    """
    <div class='info-card'>
        <p>👉 <strong>Navigasi:</strong> Gunakan sidebar di kiri untuk mengakses
        <em>Dataset</em>, <em>EDA</em>, <em>Preprocessing</em>,
        <em>Training</em>, dan <em>Prediction</em>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
