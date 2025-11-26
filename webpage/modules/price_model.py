#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import io
import uuid

# ----------------------------
# File paths for Mode 1
# ----------------------------
# docker file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZILLOW_PATH = os.path.join(BASE_DIR, "zillow_data.csv")
ZILLOW_PATH_2 = os.path.join(BASE_DIR, "zillow_data2.csv")
ZIP_MAPPING_PATH = os.path.join(BASE_DIR, "zip_to_city.csv")
# ZILLOW_PATH = "zillow_data.csv"
# ZILLOW_PATH_2 = "zillow_data2.csv"
# ZIP_MAPPING_PATH = "zip_to_city.csv"


# ----------------------------
# Load Zillow data
# ----------------------------
def load_zillow() -> pd.DataFrame:
    if not os.path.exists(ZILLOW_PATH) and not os.path.exists(ZILLOW_PATH_2):
        raise FileNotFoundError("No Zillow data files found.")
    df_list = []
    if os.path.exists(ZILLOW_PATH):
        df_list.append(pd.read_csv(ZILLOW_PATH, low_memory=False))
    if os.path.exists(ZILLOW_PATH_2):
        df_list.append(pd.read_csv(ZILLOW_PATH_2, low_memory=False))
    combined = pd.concat(df_list).drop_duplicates(subset=["RegionName"], keep="last") if len(df_list) > 1 else df_list[0]
    return combined

def load_zip_mapping(path=ZIP_MAPPING_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ZIP mapping CSV not found at {path}")
    return pd.read_csv(path, dtype=str)

def zip_to_city_state(zip_code, zip_df):
    match = zip_df[zip_df['ZIP'] == str(zip_code)]
    if match.empty:
        return None, None
    return match.iloc[0]['CITY'], match.iloc[0]['STATE']

# ----------------------------
# Holt-Winters Prediction Functions (Mode 1)
# ----------------------------
def get_city_data(df: pd.DataFrame, city_name: str) -> pd.Series:
    city_row = df[df['RegionName'].str.contains(city_name, case=False, na=False)]
    if city_row.empty:
        raise ValueError(f"No data found for {city_name}")
    date_cols = city_row.columns[5:]
    price_data = city_row[date_cols].iloc[0]
    price_data.index = pd.to_datetime(date_cols, errors='coerce')
    price_data = pd.to_numeric(price_data, errors='coerce')
    return pd.Series(price_data.values, index=price_data.index).dropna()

def predict_from_2024(price_data: pd.Series) -> pd.Series:
    historical = price_data[price_data.index < "2024-01-01"]
    if len(historical) < 12:
        raise ValueError("Not enough historical data for prediction")
    model = ExponentialSmoothing(historical, trend='mul', seasonal='mul', seasonal_periods=12)
    hw_fit = model.fit(optimized=True)
    future_index = pd.date_range(start="2024-01-31", end=price_data.index[-1] + pd.offsets.MonthEnd(12), freq='M')
    forecast = pd.Series(hw_fit.forecast(len(future_index)), index=future_index)
    return forecast

def compare_predictions(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    actual_2024 = actual[(actual.index >= "2024-01-01") & (actual.index <= "2024-12-31")]
    predicted_2024 = predicted[(predicted.index >= "2024-01-01") & (predicted.index <= "2024-12-31")]
    comparison = pd.DataFrame({"Predicted": predicted_2024, "Actual": actual_2024})
    comparison["Error"] = comparison["Predicted"] - comparison["Actual"]
    comparison["Percent_Error"] = comparison["Error"] / comparison["Actual"] * 100
    return comparison

def plot_prediction(city_name: str, price_data: pd.Series, predicted: pd.Series):
    img_bytes = io.BytesIO()
    plt.figure(figsize=(7, 3))
    plt.plot(price_data.index, price_data.values, label='Actual (CSV)', color='blue', linewidth=2)
    plt.plot(predicted.index, predicted.values, label='Predicted 2024+', color='red', linestyle='--', linewidth=2)
    plt.title(f"Actual Prices & Prediction — {city_name}")
    plt.xlabel("Date")
    plt.ylabel("Median Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_bytes, format="png", dpi=150)
    plt.close()
    img_bytes.seek(0)
    return img_bytes
