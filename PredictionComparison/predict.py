import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------
# Load data
# ---------------------------
def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# ---------------------------
# Extract city prices
# ---------------------------
def get_city_data(df: pd.DataFrame, city_name: str) -> pd.Series:
    city_row = df[df['RegionName'].str.contains(city_name, case=False, na=False)]
    if city_row.empty:
        raise ValueError(f"No data found for {city_name}")
    
    price_data = city_row.drop(
        columns=[col for col in df.columns if col in ['RegionID', 'SizeRank', 'RegionType', 'StateName', 'State', 'Metro', 'CountyName', 'RegionName'] or col.startswith('Unnamed')]
    ).iloc[0]

    price_data.index = pd.to_datetime(price_data.index)
    price_data = pd.to_numeric(price_data, errors='coerce')
    return pd.Series(price_data.values, index=pd.to_datetime(price_data.index)).dropna()

# ---------------------------
# Predict 2024 + 1 year past last CSV data
# ---------------------------
def predict_from_2024(price_data: pd.Series) -> pd.Series:
    # Use all historical data before Jan 2024
    historical = price_data[price_data.index < "2024-01-01"]
    if len(historical) < 12:
        raise ValueError("Not enough historical data for prediction")
    
    model = ExponentialSmoothing(historical, trend='mul', seasonal='mul', seasonal_periods=12)
    hw_fit = model.fit(optimized=True)
    
    # Predict from Jan 2024 to 1 year past last CSV date
    future_index = pd.date_range(start="2024-01-31", end=price_data.index[-1] + pd.offsets.MonthEnd(12), freq='M')
    forecast = pd.Series(hw_fit.forecast(len(future_index)), index=future_index)
    return forecast

# ---------------------------
# Compare predicted vs actual for 2024
# ---------------------------
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

# ---------------------------
# Plot
# ---------------------------
def plot_prediction(city_name: str, price_data: pd.Series, predicted: pd.Series):
    plt.figure(figsize=(14, 6))
    
    # Actual data line (CSV)
    plt.plot(price_data.index, price_data.values, label='Actual (CSV)', color='blue', linewidth=2)
    
    # Predicted line (continuous)
    plt.plot(predicted.index, predicted.values, label='Predicted 2024+', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Actual Prices & Prediction — {city_name}")
    plt.xlabel("Date")
    plt.ylabel("Median Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    df = load_data("zillow_data.csv")
    sample_cities = ["Grand Junction, CO", "Denver, CO", "Seattle, WA", "Miami, FL"]

    for city in sample_cities:
        try:
            prices = get_city_data(df, city)
            predicted = predict_from_2024(prices)
            comparison = compare_predictions(prices, predicted)
            print(f"\nPrediction vs Actual — {city}:")
            print(comparison)
            plot_prediction(city, prices, predicted)
        except ValueError as e:
            print(e)

    city_name = input("\nEnter a city name (e.g., Grand Junction, CO): ").strip()
    try:
        prices = get_city_data(df, city_name)
        predicted = predict_from_2024(prices)
        comparison = compare_predictions(prices, predicted)
        print(f"\nPrediction vs Actual — {city_name}:")
        print(comparison)
        plot_prediction(city_name, prices, predicted)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
