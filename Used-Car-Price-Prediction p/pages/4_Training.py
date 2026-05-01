import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.data_loader import load_clean_data

st.set_page_config(page_title="Training", page_icon="🏋️", layout="wide")

with open("style/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
        <h1>🏋️ Model Training</h1>
        <p style='font-size: 1rem; color: #718096;'>
            Train Random Forest Regressor untuk prediksi harga mobil bekas
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

df = load_clean_data()

# --- Outlier removal & feature engineering ---
Q1 = df["price (Rp)"].quantile(0.25)
Q3 = df["price (Rp)"].quantile(0.75)
IQR = Q3 - Q1
df = df[
    (df["price (Rp)"] >= Q1 - 1.5 * IQR)
    & (df["price (Rp)"] <= Q3 + 1.5 * IQR)
]
df = df.copy()
df["car_age"] = 2026 - df["year"]

# --- Model Config ---
st.markdown(
    """
    <div class='section-header'>
        <h3>Model Configuration</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    n_estimators = st.slider("n_estimators", 50, 300, 150, step=50)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, step=0.05)

with col2:
    st.markdown(
        """
        <div class='card'>
            <p><strong>Algoritma:</strong> Random Forest Regressor</p>
            <p><strong>Preprocessing:</strong></p>
            <ul>
                <li>Label Encoding (transmission)</li>
                <li>One-Hot Encoding (brand, model)</li>
                <li>Log Transform (price)</li>
                <li>Feature: car_age = 2026 - year</li>
            </ul>
            <p><strong>Target:</strong> price (Rp) — log1p transformed</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Train Button ---
if st.button("🚀 Train Model", use_container_width=True):
    with st.spinner("Training model..."):
        # Encoding
        le_trans = LabelEncoder()
        df["transmission"] = le_trans.fit_transform(df["transmission"])
        df_encoded = pd.get_dummies(df, columns=["brand", "model"], drop_first=True)

        X = df_encoded.drop(columns=["price (Rp)", "year"])
        y = np.log1p(df_encoded["price (Rp)"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        y_test_orig = np.expm1(y_test)

        mae = mean_absolute_error(y_test_orig, pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, pred))
        r2 = r2_score(y_test_orig, pred)

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        joblib.dump(le_trans, "model/le_trans.pkl")
        joblib.dump(list(X.columns), "model/model_columns.pkl")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='section-header'>
            <h3>Training Results</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>Rp {mae/1e6:.1f} Jt</div>
                <div class='metric-label'>MAE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>Rp {rmse/1e6:.1f} Jt</div>
                <div class='metric-label'>RMSE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{r2:.4f}</div>
                <div class='metric-label'>R² Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-card'>
            <p>✅ Model berhasil disimpan ke <strong>model/</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature importance
    st.markdown(
        """
        <div class='chart-card'>
            <h3>Top 10 Feature Importance</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    importance = model.feature_importances_
    feat_df = (
        pd.DataFrame({"feature": X.columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#1E3A5F")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close()
