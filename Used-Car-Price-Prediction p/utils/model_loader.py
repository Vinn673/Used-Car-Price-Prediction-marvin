import joblib
import streamlit as st


@st.cache_resource
def load_model():
    """Load trained model, label encoder, and feature columns."""
    model = joblib.load("model/model.pkl")
    le_trans = joblib.load("model/le_trans.pkl")
    model_columns = joblib.load("model/model_columns.pkl")
    return model, le_trans, model_columns
