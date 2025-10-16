import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load the Zillow CSV data."""
    df = pd.read_csv(file_path)
    return df

def get_city_data(df: pd.DataFrame, city_name: str) -> pd.Series:
    """Extract and return price data for a specific city."""
    city_row = df[df['RegionName'].str.contains(city_name, case=False, na=False)]

    if city_row.empty:
        raise ValueError(f"No data found for {city_name}")

    # Drop metadata columns to leave only date columns
    price_data = city_row.drop(
        columns=[col for col in df.columns if col in [
            'RegionID', 'SizeRank', 'RegionType', 'StateName', 'State',
            'Metro', 'CountyName', 'RegionName'
        ] or col.startswith('Unnamed')]
    ).iloc[0]
    
    # Convert column names (dates) to datetime and values to numeric
    price_data.index = pd.to_datetime(price_data.index)
    price_data = pd.to_numeric(price_data, errors='coerce').dropna()
    return price_data

def predict_future_prices(price_data: pd.Series, years_to_predict: int = 3) -> pd.Series:
    """Use linear regression to project future prices."""
    # Convert dates to numeric for regression
    X = np.array((price_data.index.year * 12 + price_data.index.month)).reshape(-1, 1)
    y = price_data.values

    model = LinearRegression()
    model.fit(X, y)

    # Predict next N years (12 * years)
    last_date = price_data.index[-1]
    future_months = pd.date_range(last_date, periods=12 * years_to_predict + 1, freq='M')[1:]
    future_X = np.array((future_months.year * 12 + future_months.month)).reshape(-1, 1)
    future_y = model.predict(future_X)

    return pd.Series(future_y, index=future_months)

def plot_trend(city_name: str, price_data: pd.Series, future_data: pd.Series):
    """Plot the historical and projected price data."""
    plt.figure(figsize=(10, 5))
    plt.plot(price_data.index, price_data.values, label='Historical Prices', color='blue')
    plt.plot(future_data.index, future_data.values, label='Projected Prices', color='orange', linestyle='--')
    plt.title(f"Median Home Sale Prices — {city_name}")
    plt.xlabel("Year")
    plt.ylabel("Median Sale Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = load_data("zillow_data.csv")

    city_name = input("Enter a city name (e.g., Grand Junction, CO): ").strip()
    try:
        price_data = get_city_data(df, city_name)
        future_data = predict_future_prices(price_data)
        plot_trend(city_name, price_data, future_data)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
