import os

import streamlit as st

from utils.data_loader import load_clean_data
from utils.model_loader import load_model
from utils.predictor import find_similar_listings, get_price_stats, predict_price

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

with open("style/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
        <h1>🔮 Prediksi Harga Mobil</h1>
        <p style='font-size: 1rem; color: #718096;'>
            Masukkan spesifikasi mobil untuk mendapatkan estimasi harga
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Check model exists
if not os.path.exists("model/model.pkl"):
    st.markdown(
        """
        <div class='warning-card'>
            <p>⚠️ <strong>Model belum di-train.</strong> Silakan buka halaman <strong>Training</strong> terlebih dahulu.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

model, le_trans, model_columns = load_model()
df = load_clean_data()

# Get valid brands and models
valid_brands = sorted(df["brand"].unique().tolist())

# --- Input Form ---
st.markdown(
    """
    <div class='section-header'>
        <h3>Spesifikasi Mobil</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", valid_brands)

    # Filter models by brand
    brand_models = sorted(df[df["brand"] == brand]["model"].unique().tolist())
    car_model = st.selectbox("Model", brand_models)

with col2:
    transmission = st.selectbox("Transmisi", ["manual", "automatic"])
    year = st.slider("Tahun", 2000, 2026, 2018)
    mileage = st.slider("Mileage (km)", 0, 300000, 50000, step=5000)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Prediksi Harga", use_container_width=True):
    result = predict_price(
        brand, car_model, transmission, year, mileage,
        model, le_trans, model_columns
    )

    if result["success"]:
        price = result["price"]

        st.markdown(
            f"""
            <div class='prediction-result'>
                <div class='price-label'>Estimasi Harga</div>
                <div class='price-value'>Rp {price:,}</div>
                <div class='price-label'>{brand.upper()} {car_model.upper()} {year} — {mileage:,} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Price stats for this brand/model
        stats = get_price_stats(df, brand, car_model)
        if stats:
            st.markdown(
                """
                <div class='section-header'>
                    <h3>Perbandingan Harga Pasar</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value'>Rp {stats['min']/1e6:.0f} Jt</div>
                        <div class='metric-label'>Termurah</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value'>Rp {stats['median']/1e6:.0f} Jt</div>
                        <div class='metric-label'>Median</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value'>Rp {stats['mean']/1e6:.0f} Jt</div>
                        <div class='metric-label'>Rata-rata</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value'>Rp {stats['max']/1e6:.0f} Jt</div>
                        <div class='metric-label'>Termahal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Similar listings
        similar = find_similar_listings(brand, car_model, year, mileage, df)
        if not similar.empty:
            st.markdown(
                """
                <div class='section-header'>
                    <h3>📋 Listing Serupa</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            display_df = similar.copy()
            display_df["price (Rp)"] = display_df["price (Rp)"].apply(
                lambda x: f"Rp {x:,}"
            )
            display_df["mileage (km)"] = display_df["mileage (km)"].apply(
                lambda x: f"{x:,}"
            )
            display_df.columns = [c.title() for c in display_df.columns]

            st.markdown(
                "<div class='similar-table'>",
                unsafe_allow_html=True,
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            f"""
            <div class='warning-card'>
                <p>❌ {result['error']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
