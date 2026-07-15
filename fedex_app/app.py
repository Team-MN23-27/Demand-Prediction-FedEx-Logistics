"""
FedEx Shipment Demand Prediction - Flask App
==============================================
Date + Hub kudutha, andha day-ku evlo demand (OrderCount) varum nu predict pannum.
Adha vachi evlo workers + delivery vehicles venum nu suggest pannum.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import math
import os
import sqlite3

app = Flask(__name__)
app.secret_key = "fedex-demand-app-secret-key"  # flash messages ku venum

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD MODEL & HISTORICAL DATA (once at startup)
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.txt")
HIST_PATH = os.path.join(BASE_DIR, "data", "historical_orders.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "fedex_resources.db")

model = lgb.Booster(model_file=MODEL_PATH)

hist_df = pd.read_csv(HIST_PATH)
hist_df["Date"] = pd.to_datetime(hist_df["Date"])
hist_df = hist_df.sort_values(["Hub", "Date"]).reset_index(drop=True)

LAST_DATE = hist_df["Date"].max()

# Average demand per hub - "high demand" threshold-ku use pannrom
HUB_AVG_DEMAND = hist_df.groupby("Hub")["OrderCount"].mean().to_dict()


def get_db():
    """SQLite connection - workers, vehicles, bookings ella-thaiyum idhula than vachirukom."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db_if_needed():
    """DB illama irundha, CSV files-la irundhu fresh-a build pannrom."""
    if os.path.exists(DB_PATH):
        return
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE workers (
        WorkerID TEXT PRIMARY KEY, Name TEXT, Phone TEXT, Hub TEXT,
        ShiftStart TEXT, ShiftEnd TEXT, BaseStatus TEXT)""")
    cur.execute("""CREATE TABLE vehicles (
        VehicleID TEXT PRIMARY KEY, VehicleType TEXT, DriverName TEXT, Phone TEXT,
        Hub TEXT, BaseStatus TEXT)""")
    cur.execute("""CREATE TABLE bookings (
        BookingID INTEGER PRIMARY KEY AUTOINCREMENT, ResourceType TEXT NOT NULL,
        ResourceID TEXT NOT NULL, Hub TEXT NOT NULL, DateFrom TEXT NOT NULL,
        DateTo TEXT NOT NULL, Status TEXT NOT NULL DEFAULT 'active',
        BookedBy TEXT, CreatedAt TEXT NOT NULL)""")
    conn.commit()

    workers_csv = pd.read_csv(os.path.join(BASE_DIR, "data", "workers_data.csv")).rename(columns={"Status": "BaseStatus"})
    workers_csv.to_sql("workers", conn, if_exists="append", index=False)
    vehicles_csv = pd.read_csv(os.path.join(BASE_DIR, "data", "vehicles_data.csv")).rename(columns={"Status": "BaseStatus"})
    vehicles_csv.to_sql("vehicles", conn, if_exists="append", index=False)
    conn.close()


init_db_if_needed()

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


def get_available_staff(hub, target_date=None, needed_workers=None, needed_vehicles=None):
    """
    Andha hub-la 'Available' workers/vehicles list pannrom - SQLite-la irundhu.
    target_date kudutha, andha date-ku already booked resources exclude pannrom.
    """
    conn = get_db()

    workers = conn.execute(
        "SELECT * FROM workers WHERE Hub = ? AND BaseStatus = 'Available'", (hub,)
    ).fetchall()
    vehicles = conn.execute(
        "SELECT * FROM vehicles WHERE Hub = ? AND BaseStatus = 'Available'", (hub,)
    ).fetchall()

    if target_date:
        d = target_date.strftime("%Y-%m-%d") if hasattr(target_date, "strftime") else target_date
        booked_workers = {r["ResourceID"] for r in conn.execute(
            "SELECT ResourceID FROM bookings WHERE ResourceType='worker' AND Status='active' "
            "AND DateFrom <= ? AND DateTo >= ?", (d, d)).fetchall()}
        booked_vehicles = {r["ResourceID"] for r in conn.execute(
            "SELECT ResourceID FROM bookings WHERE ResourceType='vehicle' AND Status='active' "
            "AND DateFrom <= ? AND DateTo >= ?", (d, d)).fetchall()}
        workers = [w for w in workers if w["WorkerID"] not in booked_workers]
        vehicles = [v for v in vehicles if v["VehicleID"] not in booked_vehicles]

    conn.close()

    workers_list = [dict(w) for w in workers]
    vehicles_list = [dict(v) for v in vehicles]

    if needed_workers is not None:
        workers_list = workers_list[:needed_workers]
    if needed_vehicles is not None:
        vehicles_list = vehicles_list[:needed_vehicles]

    return workers_list, vehicles_list


def suggest_resources(predicted_demand):
    workers = math.ceil(predicted_demand / SHIPMENTS_PER_WORKER)
    vehicles = math.ceil(predicted_demand / SHIPMENTS_PER_VEHICLE)
    return workers, vehicles


def compute_gap(needed, available):
    """
    Compares needed vs available resources and returns shortage/surplus info.
    Returns None if 'available' was not provided by the user.
    """
    if available is None:
        return None
    gap = needed - available
    if gap > 0:
        status = "shortage"
        message = f"Short by {gap}. Need to hire/arrange {gap} more."
    elif gap < 0:
        status = "surplus"
        message = f"{abs(gap)} extra available (surplus)."
    else:
        status = "exact"
        message = "Exact match — no extra needed."
    return {"needed": needed, "available": available, "gap": gap,
            "status": status, "message": message}


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
                                    error="Please provide both Date and Hub!")

        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        region = HUB_TO_REGION[hub]

        predicted_demand = predict_demand(hub, target_date)
        workers, vehicles = suggest_resources(predicted_demand)

        # Available workers/vehicles - optional fields
        avail_workers_str = request.form.get("available_workers", "").strip()
        avail_vehicles_str = request.form.get("available_vehicles", "").strip()

        available_workers = int(avail_workers_str) if avail_workers_str else None
        available_vehicles = int(avail_vehicles_str) if avail_vehicles_str else None

        worker_gap = compute_gap(workers, available_workers)
        vehicle_gap = compute_gap(vehicles, available_vehicles)

        # Hub average - "high demand" badge kaamikanuma nu check panna mattum
        hub_avg = HUB_AVG_DEMAND.get(hub, 0)
        is_high_demand = predicted_demand > hub_avg

        # Staff list - EPPOVUM kaamikum (high demand irundhalum illanalum)
        # target_date pass panrom, andha date-ku already booked resources exclude aagum
        avail_workers_list, avail_vehicles_list = get_available_staff(hub, target_date=target_date)
        staff_list = {
            "hub_avg": round(hub_avg, 1),
            "workers": avail_workers_list,
            "vehicles": avail_vehicles_list,
        }

        result = {
            "date": date_str,
            "hub": hub,
            "region": region,
            "demand": predicted_demand,
            "workers": workers,
            "vehicles": vehicles,
            "worker_gap": worker_gap,
            "vehicle_gap": vehicle_gap,
            "is_high_demand": is_high_demand,
            "staff_list": staff_list,
        }
        return render_template("index.html", hubs=HUBS, result=result,
                                prev_avail_workers=avail_workers_str,
                                prev_avail_vehicles=avail_vehicles_str)

    except Exception as e:
        return render_template("index.html", hubs=HUBS, result=None,
                                error=f"Error: {str(e)}")


@app.route("/book", methods=["POST"])
def book_resource():
    """Worker/Vehicle-ah oru date range-ku book pannrom."""
    resource_type = request.form.get("resource_type")   # 'worker' or 'vehicle'
    resource_id = request.form.get("resource_id")
    hub = request.form.get("hub")
    date_from = request.form.get("date_from")
    date_to = request.form.get("date_to") or date_from
    booked_by = request.form.get("booked_by", "").strip() or "Unknown"

    if not all([resource_type, resource_id, hub, date_from]):
        flash("Booking failed: missing required fields.", "danger")
        return redirect(url_for("home"))

    if date_to < date_from:
        flash("Booking failed: 'Date To' cannot be before 'Date From'.", "danger")
        return redirect(url_for("home"))

    conn = get_db()
    # Already booked-a irukka nu check pannrom (overlap check)
    overlap = conn.execute(
        "SELECT * FROM bookings WHERE ResourceType=? AND ResourceID=? AND Status='active' "
        "AND DateFrom <= ? AND DateTo >= ?", (resource_type, resource_id, date_to, date_from)
    ).fetchone()

    if overlap:
        flash(f"This {resource_type} is already booked for an overlapping date range!", "danger")
    else:
        conn.execute(
            "INSERT INTO bookings (ResourceType, ResourceID, Hub, DateFrom, DateTo, Status, BookedBy, CreatedAt) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
            (resource_type, resource_id, hub, date_from, date_to, booked_by, datetime.now().isoformat())
        )
        conn.commit()
        flash(f"Booked successfully from {date_from} to {date_to}!", "success")
    conn.close()

    return redirect(url_for("view_bookings", hub=hub))


@app.route("/bookings")
def view_bookings():
    """Ella active bookings-um kaamikum page - hub filter pannalam."""
    hub_filter = request.args.get("hub", "")
    conn = get_db()

    query = """
        SELECT b.BookingID, b.ResourceType, b.ResourceID, b.Hub, b.DateFrom, b.DateTo,
               b.Status, b.BookedBy, b.CreatedAt,
               COALESCE(w.Name, v.DriverName) as ResourceName,
               COALESCE(w.Phone, v.Phone) as ResourcePhone
        FROM bookings b
        LEFT JOIN workers w ON b.ResourceType='worker' AND b.ResourceID = w.WorkerID
        LEFT JOIN vehicles v ON b.ResourceType='vehicle' AND b.ResourceID = v.VehicleID
        WHERE b.Status = 'active'
    """
    params = []
    if hub_filter:
        query += " AND b.Hub = ?"
        params.append(hub_filter)
    query += " ORDER BY b.DateFrom DESC"

    bookings = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    return render_template("bookings.html", hubs=HUBS, bookings=bookings, hub_filter=hub_filter)


@app.route("/bookings/cancel/<int:booking_id>", methods=["POST"])
def cancel_booking(booking_id):
    """Booking-ah cancel pannrom (undo) - Status 'cancelled' aaki vidrom."""
    conn = get_db()
    conn.execute("UPDATE bookings SET Status='cancelled' WHERE BookingID=?", (booking_id,))
    conn.commit()
    hub = request.form.get("hub", "")
    conn.close()
    flash("Booking cancelled.", "info")
    return redirect(url_for("view_bookings", hub=hub))


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

    hub_avg = HUB_AVG_DEMAND.get(hub, 0)
    is_high_demand = predicted_demand > hub_avg

    response = {
        "date": date_str,
        "hub": hub,
        "region": HUB_TO_REGION[hub],
        "predicted_demand": predicted_demand,
        "workers_needed": workers,
        "vehicles_needed": vehicles,
        "hub_avg_demand": round(hub_avg, 1),
        "is_high_demand": is_high_demand,
    }

    avail_workers_list, avail_vehicles_list = get_available_staff(hub, target_date=target_date)
    response["available_workers"] = avail_workers_list
    response["available_vehicles"] = avail_vehicles_list

    return jsonify(response)


if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
