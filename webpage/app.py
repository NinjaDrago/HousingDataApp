"""
Web app for housing app

how to run:
build:
    docker build -t web-app .
run:
    docker run -p 5000:5000 web-app
"""
import pandas as pd

from flask import Flask, render_template, request, send_file
from datetime import datetime
from modules.price_model import(
    load_zillow,
    load_zip_mapping,
    get_city_data,
    predict_from_2024,
    compare_predictions,
    plot_prediction
)
from modules.offer_checker import check_offer_formula
from modules.population_forecast import (
    load_census_data,
    get_population_trend,
    forecast_population,
    plot_population_forecast
)

app = Flask(__name__)

valid_states = { "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
                "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
                "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR",
                "PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"}

@app.route("/")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict")
def predict_pg():
    return render_template("prediction.html", title="Predict")

@app.route("/process_predict", methods=["GET", "POST"])
def process_predict():
    error = None
    result = None

    if request.method == "POST":
        try:
            city = request.form.get("city")
            state = request.form.get("state")

            if state:
                state = state.upper()
                if state not in valid_states:
                    raise ValueError("Not a valid state")

            location = city+", "+state

        except ValueError as ve:
            error = str(ve)
    return render_template("prediction.html", title="Predict", result= location, city=location, error=error)

# Route to show image
@app.route("/plot_predict.png")
def plot_predict():
    zillow_df = None
    zip_df = None
    try:
        zillow_df = load_zillow()
        zip_df = load_zip_mapping()
        print("Zillow CSV loaded.")
        print("ZIP mapping loaded.")
    except FileNotFoundError:
        raise FileNotFoundError("Zillow or ZIP mapping CSV not found.")

    city = request.args.get("city")
    if not city:
        return "No city provided", 400

    try:
        prices = get_city_data(zillow_df, city)
        predicted = predict_from_2024(prices)

        # prices = get_city_data(zillow_df, location)
        # predicted = predict_from_2024(prices)
        compare = compare_predictions(prices, predicted)
        print(f"\nPrediction vs Actual — {city}:")
        print(compare)

        img_bytes = plot_prediction(city, prices, predicted)
        return send_file(img_bytes, mimetype="image/png")

    except Exception as e:
        return f"Error generating plot: {e}", 500

@app.route("/offer")
def check_offer_pg():
    return render_template("offer.html", title="Offer")

@app.route("/process_offer", methods=["GET", "POST"])
def process_offer():
    result = None
    error = None

    if request.method=="POST":
        try:
            city = request.form.get("city")
            state = request.form.get("state")
            price_offer = int(request.form.get("offer"))
            beds = int(request.form.get("beds") or 3)
            baths = int(request.form.get("baths") or 2)
            sqft = int(request.form.get("sqft") or 1500)
            lot_size_acres = float(request.form.get("lot_size") or 0.1)
            year_built = int(request.form.get("year_built") or 2000)
            property_type = request.form.get("Property_type") or "single"
            has_amenities = request.form.get("has_amenities") or "n"

            if state:
                state = state.upper()
                if state not in valid_states:
                    raise ValueError("Not a valid state")
            if price_offer <= 0:
                raise ValueError("Price must be greater than 0.")

            if beds <= 0:
                raise ValueError("Number of bedrooms must be greater than 0.")
            if baths <= 0:
                raise ValueError("Number of bathrooms must be greater than 0.")
            if sqft <= 0:
                raise ValueError("Square footage of home must be greater than 0.")
            if lot_size_acres < 0:
                raise ValueError("Lot size of home must be greater than 0.")

            current_datetime = datetime.now()
            if year_built < 1500:
                raise ValueError("house too old (before 1500's). Please enter a valid year")
            elif year_built > current_datetime.year:
                raise ValueError("Future year given. Please enter a valid year")

            property_type = property_type.lower()
            if property_type not in ("single", "duplex"):
                raise ValueError("Invalid input. Please type 'single' or 'duplex'")
            if has_amenities not in ("y", "n"):
                raise ValueError("Invalid input. Please type 'y' or 'n'")

            # result = f"""Provided: {city}, {state}, {price_offer}, {beds}, {baths}, {sqft}, {lot_size_acres}, {year_built}, {property_type}, {has_amenities}"""
            result = check_offer_formula(city, state, price_offer, beds, baths, sqft, lot_size_acres, year_built, property_type, has_amenities)

        except ValueError as ve:
            error = str(ve)

    return render_template("offer.html", title="Offer", result=result, error=error)

@app.route("/population")
def population_pg():
    return render_template("population.html", title="Population")

@app.route("/process_population", methods=["GET", "POST"])
def process_population():
    error = None
    result = None

    if request.method == "POST":
        try:
            city = request.form.get("city")
            state = request.form.get("state")

            if state:
                state = state.upper()
                if state not in valid_states:
                    raise ValueError("Not a valid state")
            result = f"Provided: {city}, {state}"
        except ValueError as ve:
            error = str(ve)
    return render_template("population.html", title="Population", result=result, city=city, state=state, error=error)

@app.route("/plot_population.png")
def plot_population():
    city = request.args.get("city")
    state = request.args.get("state")
    if not city or not state:
        return "Missing city or state", 400

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
