#!/usr/bin/env python3
"""
Hybrid Housing Tool
- Mode 1: Price trend prediction using Zillow CSV + Holt-Winters
- Mode 2: Check a property offer (self-contained formula-based)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import os

from offer_checker import check_offer_formula

# ----------------------------
# File paths for Mode 1 (optional)
# ----------------------------
ZILLOW_PATH = "zillow_data.csv"
ZILLOW_PATH_2 = "zillow_data2.csv"
ZIP_MAPPING_PATH = "zip_to_city.csv"  # CSV with ZIP,CITY,STATE

# ----------------------------
# Load Zillow data (Mode 1)
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


def user_input_offer():
    city = input("Enter city name: ").strip()

    valid_states = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
        "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
        "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
        "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
        "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
    }
    while True:
        state = input("Enter state abbreviation (e.g., CO): ").strip().upper()
        if state in valid_states:
            break
        print("Please enter a valid 2-letter state abbreviation.")

    while True:
        price_offer = float(input("Enter offer price: "))
        if price_offer > 0:
            break
        print("Price must be greater than 0. Please enter a valid number.")

    while True:
        beds = int(input("Number of bedrooms (default 3): ") or 3)
        if beds > 0:
            break
        print("Number of bedrooms must be greater than 0. Please enter a valid number")

    while True:
        baths = int(input("Number of bathrooms (default 2): ") or 2)
        if baths > 0:
            break
        print("Number of bathrooms must be greater than 0. Please enter a valid number")

    while True:
        sqft = int(input("Square footage (default 1500): ") or 1500)
        if sqft > 0:
            break
        print("Square footage of home must be greater than 0. Please enter a valid number")

    while True:
        lot_size_acres = float(input("Lot size in acres (default 0.1): ") or 0.1)
        if sqft > 0:
            break
        else:
            print("Square footage of home must be greater than 0. Please enter a valid number")


    # gets current time
    current_datetime = datetime.now()
    while True:
        year_built = int(input("Year built (default 2000): ") or 2000)
        if year_built < 1500:
            print("house too old (before 1500's). Please enter a valid year")
        elif year_built > current_datetime.year:
            print("Future year given. Please enter a valid year")
        else:
            break

    while True:
        property_type = input("Property type (single/duplex, default single): ").strip().lower()

        if property_type == "" or property_type == "single":
            property_type = "single"
            break
        elif property_type == "duplex":
            break
        else:
            print("Invalid input. Please type 'single' or 'duplex'")

    user_input = input("Has amenities (shop/irrigation)? (y/n, default n): ").strip().lower()

    while True:
        if user_input == "" or user_input == 'n':
            has_amenities = False
            break
        elif user_input == 'y':
            has_amenities = True
            break
        else:
            print("Invalid input. Please type 'y' or 'n'")
            user_input = input("Has amenities (shop/irrigation)? (y/n, default n): ").strip().lower()


    return (city, state, price_offer, beds, baths, sqft, lot_size_acres,
            year_built, property_type, has_amenities)


# ----------------------------
# Main Interactive Function
# ----------------------------
def main():
    # Optional: load Zillow for Mode 1
    zillow_df = None
    zip_df = None
    try:
        zillow_df = load_zillow()
        zip_df = load_zip_mapping()
        print("Zillow CSV loaded for Mode 1 trend predictions.")
    except FileNotFoundError:
        print("Zillow CSV not found. Mode 1 will not work, but Mode 2 is fully functional.")

    while True:
        print("\nChoose mode:")
        print("1: Predict price trend (Zillow + Holt-Winters)")
        print("2: Check a property offer (Formula-based)")
        print("q: Quit")
        choice = input("Enter choice: ").strip().lower()
        
        if choice == '1':
            if zillow_df is None:
                print("Zillow data not available. Cannot use Mode 1.")
                continue
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
        
        elif choice == '2':
            city, state, price_offer, beds, baths, sqft, lot_size_acres, year_built, property_type, has_amenities = user_input_offer()

            result = check_offer_formula(
                city, state, price_offer,
                beds, baths, sqft, lot_size_acres,
                year_built, property_type, has_amenities
            )
            print(result)

        elif choice == 'q':
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
