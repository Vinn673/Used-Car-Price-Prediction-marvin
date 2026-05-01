import pandas as pd


def get_dashboard_metrics(df: pd.DataFrame) -> dict:
    """Get key metrics for the dashboard."""
    return {
        "total_listings": len(df),
        "total_brands": df["brand"].nunique(),
        "total_models": df["model"].nunique(),
        "avg_price": int(df["price (Rp)"].mean()),
        "min_price": int(df["price (Rp)"].min()),
        "max_price": int(df["price (Rp)"].max()),
        "median_price": int(df["price (Rp)"].median()),
    }


def get_top_brands(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N brands by listing count."""
    return df["brand"].value_counts().head(n).reset_index()


def get_price_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Get average price per brand."""
    return (
        df.groupby("brand")["price (Rp)"]
        .agg(["mean", "median", "count"])
        .sort_values("count", ascending=False)
        .reset_index()
    )


def get_price_trend(df: pd.DataFrame, brand: str | None = None) -> pd.DataFrame:
    """Get average price by year, optionally filtered by brand."""
    if brand:
        df = df[df["brand"] == brand]
    return df.groupby("year")["price (Rp)"].mean().reset_index()


def get_preprocessing_stats(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Compare raw vs cleaned data stats."""
    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "removed_rows": len(raw_df) - len(clean_df),
        "raw_duplicates": int(raw_df.duplicated().sum()),
        "raw_missing": int(raw_df.isnull().sum().sum()),
    }
