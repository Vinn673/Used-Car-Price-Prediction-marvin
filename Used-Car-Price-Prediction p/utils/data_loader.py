import pandas as pd
import streamlit as st


@st.cache_data
def load_raw_data():
    """Load both raw CSV datasets."""
    df1 = pd.read_csv("dataset/used_car.csv")
    df2 = pd.read_csv("dataset/used_car_data_new.csv")
    return df1, df2


@st.cache_data
def load_processed_data():
    """Load, merge, and preprocess both datasets."""
    df1, df2 = load_raw_data()

    df2["transmission"] = df2["id_transmission"].map({1: "manual", 2: "automatic"})
    df1["mileage (km)"] = df1["mileage (km)"] * 1000

    df = pd.concat([df1, df2], ignore_index=True)
    df = df[["brand", "model", "year", "mileage (km)", "transmission", "price (Rp)"]]

    df["brand"] = df["brand"].astype(str).str.lower().str.strip()
    df["model"] = df["model"].astype(str).str.lower().str.strip()
    df["transmission"] = df["transmission"].astype(str).str.lower().str.strip()

    return df


@st.cache_data
def load_clean_data():
    """Load processed data with duplicates and NaN removed."""
    df = load_processed_data()
    return df.dropna().drop_duplicates().reset_index(drop=True)
