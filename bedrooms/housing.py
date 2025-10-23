import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def filter_data(state, county, bedroom = None):
    """filters the data for the given state or county"""

    file_paths = {
        1:"data/County_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        2:"data/County_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        3:"data/County_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        4:"data/County_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        5:"data/County_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    }

    dataframes = []

    # Fileter bedroom dataset
    if bedroom is None or bedroom <=0:
        raise ValueError("Bedroom must be provided and greater than 0")
    elif bedroom >5:
        df = pd.read_csv(file_paths[5])
    else:
        df = pd.read_csv(file_paths[bedroom])

    # Filter by county and state
    filtered = df[(df['StateName'] == state) & (df['RegionName'] == county)]

    if filtered.empty:
        print(f"No data found for specification: {county}, {state}")

    # Select date columns
    date_cols = [col for col in filtered.columns if re.match(r'\d{4}-\d{2}', col)]

    # Transpose so dates become rows
    ts = filtered[date_cols].T
    ts.index = pd.to_datetime(ts.index)
    dataframes.append(ts)

    # Merge all bedroom categories on the Date column
    if len(dataframes) == 0:
        raise ValueError("No Data found.")

    final_df = pd.concat(dataframes, axis=1).sort_index()

    if final_df.shape[1] == 1:
        return final_df.iloc[:, 0]
    return final_df

def predict_future_prices(price_data: pd.Series, years_to_predict: int = 3) -> pd.Series:
    """Use linear regression to project future prices"""
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

def linear_r_gd(price_data: pd.Series, years_to_predict: int = 3,
                learning_rate: float = 0.01, n_iterations: int = 10000) -> pd.Series:
    """Predicts future prices using simple linear regression trained via gradient descent."""

    # Convert dates to numeric for regression
    X = np.array(price_data.index.year * 12 + price_data.index.month, dtype=float).reshape(-1, 1)
    y = price_data.values.reshape(-1, 1)

    # Normalize X
    X_mean, X_std = X.mean(), X.std()
    X_norm = (X - X_mean) / X_std

    # Initialize parameters
    m = 0.0
    b = 0.0
    n = len(X)

    # Gradient descent loop
    for _ in range(n_iterations):
        y_pred = m * X_norm + b
        error = y_pred - y

        # Compute gradients
        dm = (2 / n) * np.sum(error * X_norm)
        db = (2 / n) * np.sum(error)

        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db

    # Generate future months
    last_date = price_data.index[-1]
    future_months = pd.date_range(last_date, periods=12 * years_to_predict + 1, freq='M')[1:]
    future_X = np.array(future_months.year * 12 + future_months.month, dtype=float).reshape(-1, 1)

    # Normalize using training data stats
    future_X_norm = (future_X - X_mean) / X_std

    # Predict future prices
    future_y = m * future_X_norm + b

    # Return predictions as Series
    return pd.Series(future_y.flatten(), index=future_months, name="Predicted Price")

def poly_r(price_data: pd.Series, years_to_predict: int = 3, degree: int = 3) -> pd.Series:
    """ Predict future prices using polynomial regression."""

    # Convert dates to numeric for regression
    X = np.array(price_data.index.year * 12 + price_data.index.month, dtype=float).reshape(-1, 1)
    y = price_data.values

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict into the future
    last_date = price_data.index[-1]
    future_months = pd.date_range(last_date, periods=12 * years_to_predict + 1, freq='M')[1:]
    future_X = np.array(future_months.year * 12 + future_months.month, dtype=float).reshape(-1, 1)
    future_X_poly = poly.transform(future_X)

    # Make predictions
    future_y = model.predict(future_X_poly)

    return pd.Series(future_y, index=future_months, name=f"Predicted (Degree {degree})")

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

def main() -> None:
    state_initials = "CO"
    county_name = "Mesa County"
    bedroom_num = 1

    df = filter_data(state_initials, county_name, bedroom_num)

    if not df.empty:
        # future = predict_future_prices(df)
        # future = linear_r_gd(df)
        future = poly_r(df)
        plot_trend(county_name +", " + state_initials, df, future)


if __name__ == "__main__":
    main()
