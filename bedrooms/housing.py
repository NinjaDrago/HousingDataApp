import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


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


def predict_future_prices(price_data: pd.Series, years_to_predict: int = 3,
                          predict_future: bool = True, test_x=None) -> pd.Series:
    """Use linear regression to project future prices"""
    # Prepare training data
    X = np.array(price_data.index.year * 12 + price_data.index.month).reshape(-1, 1)
    y = price_data.values

    model = LinearRegression()
    model.fit(X, y)

    if test_x is not None:
        # Predict for test dates
        X_test = np.array(test_x.year * 12 + test_x.month).reshape(-1, 1)
        preds = model.predict(X_test)
        return pd.Series(preds, index=test_x, name="Predicted (Test)")
    elif predict_future:
        # Future prediction
        last_date = price_data.index[-12]
        future_months = pd.date_range(last_date, periods=12 * years_to_predict + 1, freq="ME")[1:]
        X_future = np.array(future_months.year * 12 + future_months.month).reshape(-1, 1)
        preds = model.predict(X_future)
        return pd.Series(preds, index=future_months, name="Predicted (Future)")
    else:
        # Predict within dataset
        preds = model.predict(X)
        return pd.Series(preds, index=price_data.index, name="Predicted (Train)")


def linear_r_gd(price_data: pd.Series, years_to_predict: int = 3,
                learning_rate: float = 0.1, n_iterations: int = 10000, test_x=None) -> pd.Series:
    """Predicts future prices using simple linear regression trained via gradient descent."""
    X = np.array(price_data.index.year * 12 + price_data.index.month, dtype=float).reshape(-1, 1)
    y = price_data.values.reshape(-1, 1)

    # Normalize
    X_mean, X_std = X.mean(), X.std()
    X_norm = (X - X_mean) / X_std

    # Initialize parameters
    m, b = 0.0, 0.0
    n = len(X)

    # Gradient descent
    for _ in range(n_iterations):
        y_pred = m * X_norm + b
        error = y_pred - y
        dm = (2 / n) * np.sum(error * X_norm)
        db = (2 / n) * np.sum(error)
        m -= learning_rate * dm
        b -= learning_rate * db

    if test_x is not None:
        # Predict for test dates
        X_test = np.array(test_x.year * 12 + test_x.month, dtype=float).reshape(-1, 1)
        X_test_norm = (X_test - X_mean) / X_std
        preds = m * X_test_norm + b
        return pd.Series(preds.flatten(), index=test_x, name="Predicted (Test)")

    else:
        # Future prediction
        last_date = price_data.index[-1]
        future_months = pd.date_range(last_date, periods=12 * years_to_predict + 1, freq="ME")[1:]
        X_future = np.array(future_months.year * 12 + future_months.month, dtype=float).reshape(-1, 1)
        X_future_norm = (X_future - X_mean) / X_std
        preds = m * X_future_norm + b
        return pd.Series(preds.flatten(), index=future_months, name="Predicted (Future)")

def poly_r(price_data: pd.Series, years_to_predict: int = 3, degree: int = 3, test_x=None) -> pd.Series:
    """Predict future prices using polynomial regression."""
    X = np.array(price_data.index.year * 12 + price_data.index.month, dtype=float).reshape(-1, 1)
    y = price_data.values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    if test_x is not None:
        # Predict for test dates
        X_test = np.array(test_x.year * 12 + test_x.month, dtype=float).reshape(-1, 1)
        X_test_poly = poly.transform(X_test)
        preds = model.predict(X_test_poly)
        return pd.Series(preds, index=test_x, name=f"Predicted (Test deg={degree})")

    else:
        # Future prediction
        last_date = price_data.index[-12]
        future_months = pd.date_range(last_date, periods=12 * years_to_predict + 1, freq="ME")[1:]
        X_future = np.array(future_months.year * 12 + future_months.month, dtype=float).reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        preds = model.predict(X_future_poly)
        return pd.Series(preds, index=future_months, name=f"Predicted (Future deg={degree})")

def exp_smoothing(price_data: pd.Series, years_to_predict: int = 3,
                  start_forecast: str = "2024-01-01", test_x=None) -> pd.Series:
    """Exponential smoothing"""

    model = ExponentialSmoothing(price_data,  trend='mul', seasonal='mul', seasonal_periods=12)
    fit = model.fit(optimized=True)

    if test_x is not None:
        # Predict exactly for test/train dates
        n_periods = len(test_x)
        forecast = fit.forecast(n_periods)
        return pd.Series(forecast.values, index=test_x, name="ExpSmoothing (Test)")

    else:
        # Predict future months
        forecast_start = pd.Timestamp(start_forecast) + pd.offsets.MonthEnd(0)
        forecast_end = price_data.index[-1] + pd.offsets.MonthEnd(12 * years_to_predict)
        forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='M')

        forecast_values = fit.forecast(len(forecast_index))
        forecast_series = pd.Series(forecast_values, index=forecast_index, name="ExpSmoothing (From 2024)")
        return forecast_series


def plot_trend(city_name: str, price_data: pd.Series, future_data: pd.Series):
    """Plot the historical and projected price data."""
    plt.figure(figsize=(10, 5))
    plt.plot(price_data.index, price_data.values, label='Historical Prices', color='blue')
    plt.plot(future_data.index, future_data.values,
             label='Projected Prices', color='orange', linestyle='--')
    plt.title(f"Median Home Sale Prices — {city_name}")
    plt.xlabel("Year")
    plt.ylabel("Median Sale Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_model(model_name, model_func, train, test, **kwargs):
    """Train and evaluate a model given train/test data."""
    # calls model functions
    preds = model_func(train, test_x=test.index, **kwargs)

    # Align and compute metrics
    preds = preds.loc[test.index.intersection(preds.index)]
    if preds.empty:
        return f"\n--- {model_name} ---\nNo overlapping dates between train and test.\n"

    return evaluate_predictions(test.loc[preds.index].values, preds.values, model_name)


def evaluate_predictions(y_true, y_pred, label):
    """Calculate and return model evaluation metrics including percent error."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percent Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return (
        f"\n--- {label} ---\n"
        f"MSE: {mse:.2f}\n"
        f"MAE: {mae:.2f}\n"
        f"R²: {r2:.4f}\n"
        f"MAPE: {mape:.2f}%\n"
    )


def random_split(price_data: pd.Series, test_ratio=0.2):
    """Return train/test sets for an 80/20 random split."""
    n = len(price_data)
    test_size = int(n * test_ratio)
    shuffled = price_data.sample(frac=1, random_state=42)
    train = shuffled.iloc[:-test_size].sort_index()
    test = shuffled.iloc[-test_size:].sort_index()
    return train, test, f"Random Split (80/20)"


def recent_split(price_data: pd.Series, cutoff_years=3):
    """Return train/test sets using last N years for testing."""
    cutoff_date = price_data.index.max() - pd.DateOffset(years=cutoff_years)
    train = price_data[price_data.index <= cutoff_date]
    test = price_data[price_data.index > cutoff_date]
    label = f"Recent split (Last {cutoff_years} Years) — cutoff {cutoff_date.date()}"
    return train, test, label


def main() -> None:
    state_initials = "CO"
    county_name = "Mesa County"
    bedroom_num = 1

    df = filter_data(state_initials, county_name, bedroom_num)

    if not df.empty:
        # Future projection for visualization
        future = exp_smoothing(df, years_to_predict=3)
        # future = predict_from_2024(df)
        plot_trend(f"{county_name}, {state_initials}", df, future)


        # Clear old file
        open("model_test_results.txt", "w").close()

        # Models to test
        models = [
            ("Linear Regression", predict_future_prices),
            ("Gradient Descent Linear", linear_r_gd),
            ("Polynomial Regression (deg=3)", poly_r),
            ("Exponential Smoothing", exp_smoothing)
        ]

        with open("model_test_results.txt", "a") as f:
            f.write(f"\nTests for {county_name}, {state_initials}\n")

        # Random split
        train, test, split_label = random_split(df)
        with open("model_test_results.txt", "a") as f:
            f.write(f"\n=== {split_label} ===\n")

        for model_name, func in models:
            result = evaluate_model(model_name, func, train, test)
            with open("model_test_results.txt", "a") as f:
                f.write(result)

        # Recent split
        train, test, split_label = recent_split(df)
        with open("model_test_results.txt", "a") as f:
            f.write(f"\n=== {split_label} ===\n")

        for model_name, func in models:
            result = evaluate_model(model_name, func, train, test)
            with open("model_test_results.txt", "a") as f:
                f.write(result)


if __name__ == "__main__":
    main()
