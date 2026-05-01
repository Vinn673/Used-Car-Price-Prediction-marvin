import difflib

import numpy as np
import pandas as pd


def predict_price(
    brand: str,
    car_model: str,
    transmission: str,
    year: int,
    mileage: int,
    model,
    le_trans,
    model_columns: list[str],
) -> dict:
    """Run price prediction and return result dict with price and metadata.

    Returns:
        dict with keys: success, price, brand, car_model, transmission, year, mileage, error
    """
    car_age = 2026 - year

    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    input_df["mileage (km)"] = mileage
    input_df["car_age"] = car_age
    input_df["transmission"] = le_trans.transform([transmission])[0]

    # Brand one-hot (drop_first=True: first brand alphabetically has no column, all zeros = that brand)
    brand_col = f"brand_{brand}"
    if brand_col in model_columns:
        input_df[brand_col] = 1

    # Model one-hot
    model_col = f"model_{car_model}"
    if model_col in model_columns:
        input_df[model_col] = 1
    else:
        model_candidates = [
            col.replace("model_", "") for col in model_columns if col.startswith("model_")
        ]
        closest = difflib.get_close_matches(car_model, model_candidates, n=1, cutoff=0.3)
        if closest:
            input_df[f"model_{closest[0]}"] = 1
            car_model = closest[0]
        else:
            return {"success": False, "error": "Model tidak ditemukan di training data"}

    pred_log = model.predict(input_df)[0]
    pred_log = np.clip(pred_log, 0, 20)
    price = np.expm1(pred_log)

    if np.isinf(price) or np.isnan(price):
        return {"success": False, "error": "Prediksi gagal (invalid value)"}

    return {
        "success": True,
        "price": int(price),
        "brand": brand,
        "car_model": car_model,
        "transmission": transmission,
        "year": year,
        "mileage": mileage,
    }


def find_similar_listings(
    brand: str,
    car_model: str,
    year: int,
    mileage: int,
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find similar car listings from the dataset."""
    mask = df["brand"] == brand
    filtered = df[mask]

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.copy()
    filtered["_year_diff"] = abs(filtered["year"] - year)
    filtered["_mileage_diff"] = abs(filtered["mileage (km)"] - mileage)

    model_match = filtered["model"] == car_model
    filtered.loc[model_match, "_model_bonus"] = 0
    filtered.loc[~model_match, "_model_bonus"] = 100

    filtered["_score"] = (
        filtered["_year_diff"] * 5
        + filtered["_mileage_diff"] / 1000
        + filtered["_model_bonus"]
    )

    result = filtered.nsmallest(top_n, "_score")
    result = result.drop(columns=["_year_diff", "_mileage_diff", "_model_bonus", "_score"])

    return result.reset_index(drop=True)


def get_price_stats(df: pd.DataFrame, brand: str, car_model: str | None = None) -> dict:
    """Get price statistics for a brand or specific model."""
    mask = df["brand"] == brand
    filtered = df[mask]

    if car_model:
        model_mask = filtered["model"] == car_model
        filtered = filtered[model_mask]

    if filtered.empty:
        return {}

    prices = filtered["price (Rp)"]
    return {
        "min": int(prices.min()),
        "max": int(prices.max()),
        "mean": int(prices.mean()),
        "median": int(prices.median()),
        "count": len(filtered),
    }
