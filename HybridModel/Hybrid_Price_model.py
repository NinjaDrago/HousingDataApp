#!/usr/bin/env python3
"""
Hybrid Housing Tool
- Mode 1: Price trend prediction using Zillow CSV + Holt-Winters
- Mode 2: Check a property offer (self-contained formula-based)
- Mode 3: City Economic Analysis (population forecast via Census CSV)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import os

from offer_checker import check_offer_formula

# Import population_forecast module
from population_forecast import (
    load_census_data,
    get_population_trend,
    forecast_population,
    plot_population_forecast
)

# ----------------------------
# File paths for Mode 1
# ----------------------------
ZILLOW_PATH = "zillow_data.csv"
ZILLOW_PATH_2 = "zillow_data2.csv"
ZIP_MAPPING_PATH = "zip_to_city.csv"

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
    plt.figure(figsize=(14, 6))
    plt.plot(price_data.index, price_data.values, label='Actual (CSV)', color='blue', linewidth=2)
    plt.plot(predicted.index, predicted.values, label='Predicted 2024+', color='red', linestyle='--', linewidth=2)
    plt.title(f"Actual Prices & Prediction — {city_name}")
    plt.xlabel("Date")
    plt.ylabel("Median Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Mode 3: City Economic Analysis with Holt-Winters Forecast
# ----------------------------
def mode3_city_analysis():
    print("\n--- City Economic Analysis (Population Forecast) ---")
    city = input("Enter city/county name (e.g., Mesa): ").strip()
    state = input("Enter state abbreviation (e.g., CO): ").strip().upper()
    
    try:
        census_df = load_census_data()
        pop_trend = get_population_trend(city, state, census_df)
    except Exception as e:
        pop_trend = None
        print("Population fetch error:", e)
    
    if pop_trend:
        # Forecast 2025
        combined_series, forecast_series = forecast_population(pop_trend, forecast_years=1)
        historical_years = list(map(float, pop_trend.keys()))
        plot_population_forecast(combined_series, city, state, historical_years=historical_years)

        # Print forecast 2025 safely
        print(f"\nForecasted Population for {city}, {state}:")
        for year, pop in forecast_series.items():
            if pd.isna(pop):
                print(f"{int(year)}: Forecast not available")
            else:
                print(f"{int(year)}: {int(round(pop)):,}")
    else:
        print(f"No population data available for {city}, {state}.")


# ----------------------------
# User input helper for Mode 2
# ----------------------------
def user_input_offer():
    city = input("Enter city name: ").strip()
    valid_states = { "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
        "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
        "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"}
    while True:
        state = input("Enter state abbreviation (e.g., CO): ").strip().upper()
        if state in valid_states: break
        print("Please enter a valid 2-letter state abbreviation.")
    while True:
        price_offer = float(input("Enter offer price: "))
        if price_offer > 0: break
        print("Price must be greater than 0.")
    beds = int(input("Number of bedrooms (default 3): ") or 3)
    baths = int(input("Number of bathrooms (default 2): ") or 2)
    sqft = int(input("Square footage (default 1500): ") or 1500)
    lot_size_acres = float(input("Lot size in acres (default 0.1): ") or 0.1)
    current_datetime = datetime.now()
    year_built = int(input("Year built (default 2000): ") or 2000)
    property_type = input("Property type (single/duplex, default single): ").strip().lower() or "single"
    has_amenities = input("Has amenities (shop/irrigation)? (y/n, default n): ").strip().lower() == 'y'
    return (city, state, price_offer, beds, baths, sqft, lot_size_acres, year_built, property_type, has_amenities)

# ----------------------------
# Main Interactive Function
# ----------------------------
def main():
    zillow_df = None
    zip_df = None
    try:
        zillow_df = load_zillow()
        zip_df = load_zip_mapping()
        print("Zillow CSV loaded.")
        print("ZIP mapping loaded.")
    except FileNotFoundError:
        print("Zillow or ZIP mapping CSV not found.")

    while True:
        print("\nChoose mode:")
        print("1: Predict price trend")
        print("2: Check a property offer")
        print("3: City Economic Analysis (population forecast)")
        print("q: Quit")
        choice = input("Enter choice: ").strip().lower()
        if choice=='1':
            city_name = input("Enter a city name (e.g., Grand Junction, CO): ").strip()
            try:
                prices = get_city_data(zillow_df, city_name)
                predicted = predict_from_2024(prices)
                comparison = compare_predictions(prices, predicted)
                print(f"\nPrediction vs Actual — {city_name}:")
                print(comparison)
                plot_prediction(city_name, prices, predicted)
            except ValueError as e:
                print(e)
        elif choice=='2':
            args = user_input_offer()
            result = check_offer_formula(*args)
            print(result)
        elif choice=='3':
            mode3_city_analysis()
        elif choice=='q':
            break
        else:
            print("Invalid choice. Try again.")

if __name__=="__main__":
    main()
