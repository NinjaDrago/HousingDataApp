import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ----------------------------
# File path for the population Excel
# ----------------------------
POPULATION_XLSX = os.path.join(os.path.dirname(__file__), "SUB-IP-EST2024-ANNRNK.xlsx")

# ----------------------------
# Map state abbreviations to full state names
# ----------------------------
STATE_ABBR_TO_NAME = {
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California",
    "CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia",
    "HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas",
    "KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts",
    "MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana",
    "NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico",
    "NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma",
    "OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota",
    "TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington",
    "WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"
}

# ----------------------------
# Load Excel file
# ----------------------------
def load_census_data() -> pd.DataFrame:
    if not os.path.exists(POPULATION_XLSX):
        raise FileNotFoundError(f"Census Excel file not found at {POPULATION_XLSX}")

    # Skip first 4 rows; header is on row 5
    df = pd.read_excel(
        POPULATION_XLSX,
        engine='openpyxl',
        skiprows=4,
        usecols="B,D,E,F,G,H"  # B = Geographic Area, D-H = 2020-2024
    )

    # Rename columns
    df.columns = ['Geographic Area', '2020', '2021', '2022', '2023', '2024']
    df['Geographic Area'] = df['Geographic Area'].str.strip()

    return df

# ----------------------------
# Normalize city & state
# ----------------------------
def normalize_city_state(user_city: str, user_state_abbr: str, df: pd.DataFrame) -> str:
    state_full = STATE_ABBR_TO_NAME.get(user_state_abbr.upper(), user_state_abbr)
    user_city_lower = user_city.strip().lower()
    
    for ga in df['Geographic Area']:
        if ',' not in ga:
            continue
        city_part, state_part = ga.split(',', 1)
        city_part = city_part.strip()
        state_part = state_part.strip()
        if city_part.lower() == user_city_lower and state_part.lower() == state_full.lower():
            return ga
        if user_city_lower in city_part.lower() and state_part.lower() == state_full.lower():
            return ga
    return None

# ----------------------------
# Get population trend
# ----------------------------
def get_population_trend(city: str, state_abbr: str, df: pd.DataFrame) -> dict:
    csv_city = normalize_city_state(city, state_abbr, df)
    if csv_city is None:
        print(f"City '{city}' in state '{state_abbr}' not found in Excel")
        return None

    row = df[df['Geographic Area'] == csv_city].iloc[0]

    pop_dict = {
        '2020': row['2020'],
        '2021': row['2021'],
        '2022': row['2022'],
        '2023': row['2023'],
        '2024': row['2024']
    }
    return pop_dict

# ----------------------------
# Forecast population with linear regression
# ----------------------------
def forecast_population(pop_dict: dict, forecast_years: int = 1):
    # Convert years to numeric
    years = np.array([int(y) for y in pop_dict.keys()]).reshape(-1,1)
    pops = np.array(list(pop_dict.values()))
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(years, pops)
    
    # Historical series
    pop_series = pd.Series(pops, index=years.flatten())
    
    # Forecast next N years
    future_years = np.array([years.max() + i for i in range(1, forecast_years+1)]).reshape(-1,1)
    forecast_pops = model.predict(future_years)
    forecast_series = pd.Series(forecast_pops, index=future_years.flatten())
    
    # Combine
    combined_series = pd.concat([pop_series, forecast_series])
    
    return combined_series, forecast_series

# ----------------------------
# Plot population trend + forecast
# ----------------------------
def plot_population_forecast(combined_series: pd.Series, city: str, state_abbr: str, historical_years: list = None):
    years = combined_series.index
    pops = combined_series.values

    if historical_years is None:
        historical_years = years[:-1]  # all except last year

    historical_mask = [y in historical_years for y in years]
    forecast_mask = [not m for m in historical_mask]

    plt.figure(figsize=(10,5))
    
    # Plot the full line (connects historical + forecast)
    plt.plot(years, pops, linestyle='-', linewidth=2, color='blue', label='Population Trend')
    
    # Overlay historical markers
    plt.plot(years[historical_mask], pops[historical_mask], 'o', color='green', markersize=8, label='Historical')
    
    # Overlay forecast markers
    plt.plot(years[forecast_mask], pops[forecast_mask], 'x', color='red', markersize=10, label='Forecast')
    
    plt.title(f"Population Trend & Forecast — {city}, {state_abbr}")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

