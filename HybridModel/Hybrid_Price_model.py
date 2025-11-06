#!/usr/bin/env python3
"""
Hybrid Housing Tool
- Mode 1: Price trend prediction using Zillow CSV + Holt-Winters
- Mode 2: Offer checker using formula-based estimation
- Accepts city name or ZIP code for offer checking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# ----------------------------
# File paths
# ----------------------------
ZILLOW_PATH = "zillow_data.csv"
ZIP_MAPPING_PATH = "zip_to_city.csv"  # CSV with ZIP,CITY,STATE

# ----------------------------
# Load Zillow data
# ----------------------------
def load_zillow(path=ZILLOW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Zillow file not found at {path}")
    return pd.read_csv(path, low_memory=False)

# ----------------------------
# Load ZIP -> City mapping
# ----------------------------
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
# Holt-Winters Prediction Functions
# ----------------------------
def get_city_data(df: pd.DataFrame, city_name: str) -> pd.Series:
    city_row = df[df['RegionName'].str.contains(city_name, case=False, na=False)]
    if city_row.empty:
        raise ValueError(f"No data found for {city_name}")
    
    # Only date columns from the 6th column onward
    date_cols = city_row.columns[5:]
    price_data = city_row[date_cols].iloc[0]

    # Convert dates and values
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
    
    comparison = pd.DataFrame({
        "Predicted": predicted_2024,
        "Actual": actual_2024
    })
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
# Formula-based Offer Checker
# ----------------------------
def estimate_home_price(base_price, beds=3, baths=2, sqft=1500, basement=False):
    bed_modifier = 0.05
    bath_modifier = 0.03
    sqft_modifier = 0.0005
    basement_modifier = 0.10 if basement else 0

    price = base_price * (
        1 + bed_modifier*(beds-3) + bath_modifier*(baths-2) + sqft_modifier*(sqft-1500) + basement_modifier
    )
    return price

def check_offer_formula(city, state, price_offer, zillow_df, beds=3, baths=2, sqft=1500, basement=False):
    try:
        prices = get_city_data(zillow_df, city)
        base_price = prices[-1]  # use last known price
    except:
        return f"No Zillow data found for {city}, {state}"

    estimated_price = estimate_home_price(base_price, beds, baths, sqft, basement)

    if price_offer < estimated_price*0.95:
        advice = "Potentially a good deal."
    elif price_offer > estimated_price*1.05:
        advice = "May be overpaying."
    else:
        advice = "Within typical market range."

    return f"Estimated price for {beds} beds, {baths} baths, {sqft} sqft in {city}, {state}: ${estimated_price:,.0f}\n{advice}"

# ----------------------------
# Main Interactive Function
# ----------------------------
def main():
    print("Loading Zillow CSV for trend predictions...")
    zillow_df = load_zillow()
    print("Loading ZIP -> City mapping...")
    zip_df = load_zip_mapping()
    
    while True:
        print("\nChoose mode:")
        print("1: Predict price trend (Zillow + Holt-Winters)")
        print("2: Check a property offer (Formula-based)")
        print("q: Quit")
        choice = input("Enter choice: ").strip().lower()
        
        if choice == '1':
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
            user_input = input("Enter city name or ZIP code: ").strip()
            if user_input.isdigit():
                city, state = zip_to_city_state(user_input, zip_df)
                if city is None:
                    print("ZIP code not found in mapping.")
                    continue
            else:
                city = user_input
                state = input("Enter state abbreviation (e.g., CO): ").strip()
            
            price_offer = float(input("Enter offer price: "))
            beds = int(input("Number of bedrooms (default 3): ") or 3)
            baths = int(input("Number of bathrooms (default 2): ") or 2)
            sqft = int(input("Square footage (default 1500): ") or 1500)
            basement_input = input("Has basement? (y/n, default n): ").strip().lower()
            basement = basement_input == 'y'

            result = check_offer_formula(city, state, price_offer, zillow_df, beds, baths, sqft, basement)
            print(result)
        
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
