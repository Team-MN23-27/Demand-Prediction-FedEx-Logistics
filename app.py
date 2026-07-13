"""
FedEx Shipment Demand Prediction - Flask App
==============================================
Date + Hub kudutha, andha day-ku evlo demand (OrderCount) varum nu predict pannum.
Adha vachi evlo workers + delivery vehicles venum nu suggest pannum.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import math
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD MODEL & HISTORICAL DATA (once at startup)
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.txt")
HIST_PATH = os.path.join(BASE_DIR, "data", "historical_orders.csv")

model = lgb.Booster(model_file=MODEL_PATH)

hist_df = pd.read_csv(HIST_PATH)
hist_df["Date"] = pd.to_datetime(hist_df["Date"])
hist_df = hist_df.sort_values(["Hub", "Date"]).reset_index(drop=True)

LAST_DATE = hist_df["Date"].max()

# ==========================================
# 2. FIXED MAPPINGS (from dataset analysis)
# ==========================================
HUB_TO_REGION = {
    "Bengaluru_Hub": "South",
    "Chennai_Hub": "South",
    "Hyderabad_Hub": "South",
    "Delhi_Hub": "North",
    "Kolkata_Hub": "East",
    "Mumbai_Hub": "West",
}
HUBS = sorted(HUB_TO_REGION.keys())

# EXACT category order used during training (alphabetical, from pandas .astype('category'))
HUB_CATEGORIES = ['Bengaluru_Hub', 'Chennai_Hub', 'Delhi_Hub', 'Hyderabad_Hub', 'Kolkata_Hub', 'Mumbai_Hub']
REGION_CATEGORIES = ['East', 'North', 'South', 'West']

# month -> Season_Encoded (derived from dataset: Season followed pure month buckets)
MONTH_TO_SEASON = {
    12: 3, 1: 3, 2: 3,   # Winter
    3: 2, 4: 2, 5: 2,    # Summer
    6: 1, 7: 1, 8: 1, 9: 1,  # Monsoon
    10: 0, 11: 0,        # Post-Monsoon/Autumn
}

# Scaler params reverse-engineered from the training dataset (StandardScaler on lag/rolling features)
SCALER_MEAN = {
    "lag1": 287.13655084, "lag7": 287.41994852, "lag14": 287.74787645,
    "roll_mean7": 287.2576025, "roll_std7": 179.12065258,
}
SCALER_STD = {
    "lag1": 197.38921045, "lag7": 197.61504403, "lag14": 197.75503062,
    "roll_mean7": 90.63609427, "roll_std7": 61.83228815,
}

# ==========================================
# 3. CAPACITY ASSUMPTIONS (workers/vehicles)
# ==========================================
SHIPMENTS_PER_WORKER = 40    # 1 worker handles 40 shipments/day
SHIPMENTS_PER_VEHICLE = 150  # 1 delivery vehicle carries 150 shipments/day


def get_lag_features(hub, target_date):
    """
    Andha hub-oda actual order history-la irundhu lag/rolling features calculate pannrom.
    Target date dataset range-ku appuram irundha (future), last known values-ah proxy-a use pannrom.
    """
    hub_hist = hist_df[hist_df["Hub"] == hub].sort_values("Date")

    if target_date <= LAST_DATE:
        # Date already dataset-la irukku -> exact past values eduthukalam
        past = hub_hist[hub_hist["Date"] < target_date]
    else:
        # Future date -> last available history-ah use pannrom (proxy)
        past = hub_hist

    if len(past) == 0:
        # No history at all -> fallback to overall mean
        raw_lag1 = raw_lag7 = raw_lag14 = SCALER_MEAN["lag1"]
        raw_roll_mean7 = SCALER_MEAN["roll_mean7"]
        raw_roll_std7 = SCALER_MEAN["roll_std7"]
    else:
        series = past.set_index("Date")["OrderCount"]
        raw_lag1 = series.iloc[-1] if len(series) >= 1 else SCALER_MEAN["lag1"]
        raw_lag7 = series.iloc[-7] if len(series) >= 7 else raw_lag1
        raw_lag14 = series.iloc[-14] if len(series) >= 14 else raw_lag7
        last7 = series.iloc[-7:] if len(series) >= 7 else series
        raw_roll_mean7 = last7.mean()
        raw_roll_std7 = last7.std() if len(last7) > 1 else 0.0
        if pd.isna(raw_roll_std7):
            raw_roll_std7 = 0.0

    # scale using the reverse-engineered StandardScaler params
    scaled = {
        "OrderCount_lag_1": (raw_lag1 - SCALER_MEAN["lag1"]) / SCALER_STD["lag1"],
        "OrderCount_lag_7": (raw_lag7 - SCALER_MEAN["lag7"]) / SCALER_STD["lag7"],
        "OrderCount_lag_14": (raw_lag14 - SCALER_MEAN["lag14"]) / SCALER_STD["lag14"],
        "OrderCount_rolling_mean_7": (raw_roll_mean7 - SCALER_MEAN["roll_mean7"]) / SCALER_STD["roll_mean7"],
        "OrderCount_rolling_std_7": (raw_roll_std7 - SCALER_MEAN["roll_std7"]) / SCALER_STD["roll_std7"],
    }
    return scaled


def build_feature_row(hub, target_date):
    """Oru date + hub kudutha, model expect panra ellaa 17 features-um build pannrom."""
    region = HUB_TO_REGION[hub]
    year = target_date.year
    month = target_date.month
    day = target_date.day
    day_of_week = target_date.weekday()          # Monday=0 ... Sunday=6
    week_of_year = target_date.isocalendar()[1]
    is_weekend = 1 if day_of_week in (5, 6) else 0
    quarter = (month - 1) // 3 + 1
    season_encoded = MONTH_TO_SEASON[month]

    # Simplifying assumption: default "no holiday / no festival" unless caller overrides
    holiday_indicator = 0
    festival_encoded = 2  # 2 == "None" (majority class in training data)

    lag_feats = get_lag_features(hub, target_date)

    row = {
        "Holiday_Indicator_Encoded": holiday_indicator,
        "Festival_Name_Encoded": festival_encoded,
        "Season_Encoded": season_encoded,
        "Year": year,
        "Month": month,
        "Day": day,
        "DayOfWeek": day_of_week,
        "WeekOfYear": week_of_year,
        "IsWeekend": is_weekend,
        "Quarter": quarter,
        **lag_feats,
        "Hub": hub,
        "Region": region,
    }
    return row


def predict_demand(hub, target_date):
    row = build_feature_row(hub, target_date)
    X = pd.DataFrame([row])

    # keep the exact same column order used at training time
    col_order = [
        'Holiday_Indicator_Encoded', 'Festival_Name_Encoded', 'Season_Encoded',
        'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 'Quarter',
        'OrderCount_lag_1', 'OrderCount_lag_7', 'OrderCount_lag_14',
        'OrderCount_rolling_mean_7', 'OrderCount_rolling_std_7',
        'Hub', 'Region'
    ]
    X = X[col_order]

    # categorical dtype with EXACT same categories as training (alphabetical order)
    X["Hub"] = pd.Categorical(X["Hub"], categories=HUB_CATEGORIES)
    X["Region"] = pd.Categorical(X["Region"], categories=REGION_CATEGORIES)

    pred = model.predict(X)[0]
    pred = max(0, round(pred))
    return pred


def suggest_resources(predicted_demand):
    workers = math.ceil(predicted_demand / SHIPMENTS_PER_WORKER)
    vehicles = math.ceil(predicted_demand / SHIPMENTS_PER_VEHICLE)
    return workers, vehicles


# ==========================================
# 4. ROUTES
# ==========================================
@app.route("/")
def home():
    return render_template("index.html", hubs=HUBS, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        date_str = request.form.get("date")
        hub = request.form.get("hub")

        if not date_str or not hub:
            return render_template("index.html", hubs=HUBS, result=None,
                                    error="Date and Hub rendume kudukanum machi!")

        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        region = HUB_TO_REGION[hub]

        predicted_demand = predict_demand(hub, target_date)
        workers, vehicles = suggest_resources(predicted_demand)

        result = {
            "date": date_str,
            "hub": hub,
            "region": region,
            "demand": predicted_demand,
            "workers": workers,
            "vehicles": vehicles,
        }
        return render_template("index.html", hubs=HUBS, result=result)

    except Exception as e:
        return render_template("index.html", hubs=HUBS, result=None,
                                error=f"Error: {str(e)}")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API version - Chart.js dashboard or external tools ku."""
    data = request.get_json()
    date_str = data.get("date")
    hub = data.get("hub")

    if not date_str or not hub or hub not in HUB_TO_REGION:
        return jsonify({"error": "Invalid date or hub"}), 400

    target_date = datetime.strptime(date_str, "%Y-%m-%d")
    predicted_demand = predict_demand(hub, target_date)
    workers, vehicles = suggest_resources(predicted_demand)

    return jsonify({
        "date": date_str,
        "hub": hub,
        "region": HUB_TO_REGION[hub],
        "predicted_demand": predicted_demand,
        "workers_needed": workers,
        "vehicles_needed": vehicles,
    })


if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
