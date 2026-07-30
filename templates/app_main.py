from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import math
import os
import sqlite3

app = Flask(__name__)
app.secret_key = "fedex-demand-app-secret-key"  # Needed for Flask's "flash" messages (temporary success/error popups) to work securely

# BASE_DIR = the folder where this app.py file lives.
# We use this so file paths (models, data, db) work correctly no matter where the app is run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. LOAD MODEL & HISTORICAL DATA (runs once, when the server starts)
# ============================================================

MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.txt")
HIST_PATH = os.path.join(BASE_DIR, "data", "historical_orders.csv")
DB_PATH = os.path.join(BASE_DIR, "data", "fedex_resources.db")

# Load the pre-trained LightGBM model from disk (trained earlier in Colab)
model = lgb.Booster(model_file=MODEL_PATH)

# Load historical shipment order data (used to build lag/rolling features for predictions)
hist_df = pd.read_csv(HIST_PATH)
hist_df["Date"] = pd.to_datetime(hist_df["Date"])
hist_df = hist_df.sort_values(["Hub", "Date"]).reset_index(drop=True)

# The most recent date present in our historical dataset
LAST_DATE = hist_df["Date"].max()

# Average demand per hub - used later to decide if a day counts as "high demand"
HUB_AVG_DEMAND = hist_df.groupby("Hub")["OrderCount"].mean().to_dict()


def get_db():
    """
    Opens a connection to the SQLite database.
    This single database file stores everything: workers, vehicles, bookings, and login users.
    row_factory = sqlite3.Row lets us access columns by name (like a dictionary) instead of by index number.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db_if_needed():
    """
    Runs only ONCE - the very first time the app starts.
    If the database file doesn't exist yet, this function creates all the required
    tables from scratch and fills them with starter data from the CSV files.
    On every later run, since the .db file already exists, this function does nothing.
    """
    if os.path.exists(DB_PATH):
        return  # DB already built - nothing to do

    conn = get_db()
    cur = conn.cursor()

    # Table: list of delivery workers per hub
    cur.execute("""CREATE TABLE workers (
        WorkerID TEXT PRIMARY KEY, Name TEXT, Phone TEXT, Hub TEXT,
        ShiftStart TEXT, ShiftEnd TEXT, BaseStatus TEXT)""")

    # Table: list of delivery vehicles per hub
    cur.execute("""CREATE TABLE vehicles (
        VehicleID TEXT PRIMARY KEY, VehicleType TEXT, DriverName TEXT, Phone TEXT,
        Hub TEXT, BaseStatus TEXT)""")

    # Table: records every booking made for a worker or vehicle (with date range)
    cur.execute("""CREATE TABLE bookings (
        BookingID INTEGER PRIMARY KEY AUTOINCREMENT, ResourceType TEXT NOT NULL,
        ResourceID TEXT NOT NULL, Hub TEXT NOT NULL, DateFrom TEXT NOT NULL,
        DateTo TEXT NOT NULL, Status TEXT NOT NULL DEFAULT 'active',
        BookedBy TEXT, CreatedAt TEXT NOT NULL)""")

    # Table: login users for the web app
    cur.execute("""CREATE TABLE users (
        UserID INTEGER PRIMARY KEY AUTOINCREMENT, Username TEXT UNIQUE NOT NULL,
        PasswordHash TEXT NOT NULL, FullName TEXT NOT NULL, CreatedAt TEXT NOT NULL)""")
    conn.commit()

    # Create one default demo login account so you can log in immediately
    # Username: admin | Password: admin123
    cur.execute(
        "INSERT INTO users (Username, PasswordHash, FullName, CreatedAt) VALUES (?, ?, ?, ?)",
        ("admin", generate_password_hash("admin123"), "Admin", datetime.now().isoformat())
    )
    conn.commit()

    # Load starter worker/vehicle data from CSV files into the new database tables
    workers_csv = pd.read_csv(os.path.join(BASE_DIR, "data", "workers_data.csv")).rename(columns={"Status": "BaseStatus"})
    workers_csv.to_sql("workers", conn, if_exists="append", index=False)
    vehicles_csv = pd.read_csv(os.path.join(BASE_DIR, "data", "vehicles_data.csv")).rename(columns={"Status": "BaseStatus"})
    vehicles_csv.to_sql("vehicles", conn, if_exists="append", index=False)
    conn.close()


# Call it immediately on startup - it will only actually build the DB the first time
init_db_if_needed()


# ============================================================
# 2. FIXED MAPPINGS (derived from analyzing the training dataset)
# ============================================================

# Which region each hub belongs to
HUB_TO_REGION = {
    "Bengaluru_Hub": "South",
    "Chennai_Hub": "South",
    "Hyderabad_Hub": "South",
    "Delhi_Hub": "North",
    "Kolkata_Hub": "East",
    "Mumbai_Hub": "West",
}
HUBS = sorted(HUB_TO_REGION.keys())

# IMPORTANT: This must be the EXACT same category order pandas used when the model was trained
# (alphabetical order, from .astype('category')). If this order is wrong, predictions will be wrong.
HUB_CATEGORIES = ['Bengaluru_Hub', 'Chennai_Hub', 'Delhi_Hub', 'Hyderabad_Hub', 'Kolkata_Hub', 'Mumbai_Hub']
REGION_CATEGORIES = ['East', 'North', 'South', 'West']

# Maps a calendar month -> Season_Encoded number (matches how "Season" was encoded during training)
MONTH_TO_SEASON = {
    12: 3, 1: 3, 2: 3,   # Winter
    3: 2, 4: 2, 5: 2,    # Summer
    6: 1, 7: 1, 8: 1, 9: 1,  # Monsoon
    10: 0, 11: 0,        # Post-Monsoon/Autumn
}

# These mean/std values were reverse-engineered from the original training data.
# The model was trained on SCALED (StandardScaler) lag/rolling features, so at prediction
# time we must scale new values using the SAME formula: (value - mean) / std
SCALER_MEAN = {
    "lag1": 287.13655084, "lag7": 287.41994852, "lag14": 287.74787645,
    "roll_mean7": 287.2576025, "roll_std7": 179.12065258,
}
SCALER_STD = {
    "lag1": 197.38921045, "lag7": 197.61504403, "lag14": 197.75503062,
    "roll_mean7": 90.63609427, "roll_std7": 61.83228815,
}


# ============================================================
# 3. CAPACITY ASSUMPTIONS (how many workers/vehicles are needed per shipment volume)
# ============================================================

SHIPMENTS_PER_WORKER = 40    # Assumption: 1 worker can handle 40 shipments/day
SHIPMENTS_PER_VEHICLE = 150  # Assumption: 1 delivery vehicle can carry 150 shipments/day


def get_lag_features(hub, target_date):
    """
    Builds the 'lag' and 'rolling average' features the model needs.
    Lag features = past values (e.g. what was demand 1 day ago, 7 days ago, 14 days ago).
    Rolling features = average/std of the last 7 days.
    These help the model understand recent trends, not just the calendar date.
    """
    hub_hist = hist_df[hist_df["Hub"] == hub].sort_values("Date")

    if target_date <= LAST_DATE:
        # This date already exists in our historical dataset -> use the real past values
        past = hub_hist[hub_hist["Date"] < target_date]
    else:
        # This is a FUTURE date (beyond our data) -> use all available history as a proxy/estimate
        past = hub_hist

    if len(past) == 0:
        # No history at all for this hub -> fall back to the overall dataset mean
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

    # Scale each raw value using the same StandardScaler formula used at training time
    scaled = {
        "OrderCount_lag_1": (raw_lag1 - SCALER_MEAN["lag1"]) / SCALER_STD["lag1"],
        "OrderCount_lag_7": (raw_lag7 - SCALER_MEAN["lag7"]) / SCALER_STD["lag7"],
        "OrderCount_lag_14": (raw_lag14 - SCALER_MEAN["lag14"]) / SCALER_STD["lag14"],
        "OrderCount_rolling_mean_7": (raw_roll_mean7 - SCALER_MEAN["roll_mean7"]) / SCALER_STD["roll_mean7"],
        "OrderCount_rolling_std_7": (raw_roll_std7 - SCALER_MEAN["roll_std7"]) / SCALER_STD["roll_std7"],
    }
    return scaled


def get_recent_trend(hub, target_date, days=14):
    """
    Returns the last N days of ACTUAL historical demand for a hub.
    This is only for displaying a trend chart to the user - it's not used by the model.
    """
    hub_hist = hist_df[hist_df["Hub"] == hub].sort_values("Date")

    if target_date <= LAST_DATE:
        past = hub_hist[hub_hist["Date"] < target_date]
    else:
        past = hub_hist

    recent = past.tail(days)
    trend = [{"date": d.strftime("%b %d"), "demand": int(v)}
             for d, v in zip(recent["Date"], recent["OrderCount"])]
    return trend


def build_feature_row(hub, target_date):
    """
    Given a hub and a date, this builds ALL the input features (17 total) that the
    LightGBM model expects, in one row, ready to be fed into model.predict().
    """
    region = HUB_TO_REGION[hub]
    year = target_date.year
    month = target_date.month
    day = target_date.day
    day_of_week = target_date.weekday()          # Monday=0 ... Sunday=6
    week_of_year = target_date.isocalendar()[1]
    is_weekend = 1 if day_of_week in (5, 6) else 0
    quarter = (month - 1) // 3 + 1
    season_encoded = MONTH_TO_SEASON[month]

    # Simplifying assumption: default to "no holiday / no festival" unless the caller overrides this
    holiday_indicator = 0
    festival_encoded = 2  # 2 == "None" (this was the majority/most common class in the training data)

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
    """
    Runs the actual ML prediction: builds the feature row, arranges it in the exact
    column order the model was trained on, and returns the predicted shipment demand
    (rounded to a whole number, never negative).
    """
    row = build_feature_row(hub, target_date)
    X = pd.DataFrame([row])

    # Keep the EXACT same column order used at training time - LightGBM is sensitive to this
    col_order = [
        'Holiday_Indicator_Encoded', 'Festival_Name_Encoded', 'Season_Encoded',
        'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 'Quarter',
        'OrderCount_lag_1', 'OrderCount_lag_7', 'OrderCount_lag_14',
        'OrderCount_rolling_mean_7', 'OrderCount_rolling_std_7',
        'Hub', 'Region'
    ]
    X = X[col_order]

    # Convert Hub/Region to categorical dtype with the SAME category list/order as training
    X["Hub"] = pd.Categorical(X["Hub"], categories=HUB_CATEGORIES)
    X["Region"] = pd.Categorical(X["Region"], categories=REGION_CATEGORIES)

    pred = model.predict(X)[0]
    pred = max(0, round(pred))  # Demand can never be negative
    return pred


def get_available_staff(hub, target_date=None, needed_workers=None, needed_vehicles=None):
    """
    Fetches the list of 'Available' workers and vehicles for a given hub from SQLite.
    If a target_date is given, it also excludes any worker/vehicle that already has
    an active booking overlapping that date (so we don't double-book anyone).
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
        # Find worker/vehicle IDs that are already booked for this date
        booked_workers = {r["ResourceID"] for r in conn.execute(
            "SELECT ResourceID FROM bookings WHERE ResourceType='worker' AND Status='active' "
            "AND DateFrom <= ? AND DateTo >= ?", (d, d)).fetchall()}
        booked_vehicles = {r["ResourceID"] for r in conn.execute(
            "SELECT ResourceID FROM bookings WHERE ResourceType='vehicle' AND Status='active' "
            "AND DateFrom <= ? AND DateTo >= ?", (d, d)).fetchall()}
        # Remove already-booked resources from the available list
        workers = [w for w in workers if w["WorkerID"] not in booked_workers]
        vehicles = [v for v in vehicles if v["VehicleID"] not in booked_vehicles]

    conn.close()

    workers_list = [dict(w) for w in workers]
    vehicles_list = [dict(v) for v in vehicles]

    # Optionally trim the list down to only as many as are actually needed
    if needed_workers is not None:
        workers_list = workers_list[:needed_workers]
    if needed_vehicles is not None:
        vehicles_list = vehicles_list[:needed_vehicles]

    return workers_list, vehicles_list


def suggest_resources(predicted_demand):
    """
    Converts a predicted demand number into how many workers and vehicles are needed,
    using the simple capacity assumptions defined above (ceil = always round UP,
    since you can't have a fraction of a worker/vehicle).
    """
    workers = math.ceil(predicted_demand / SHIPMENTS_PER_WORKER)
    vehicles = math.ceil(predicted_demand / SHIPMENTS_PER_VEHICLE)
    return workers, vehicles


def compute_gap(needed, available):
    """
    Compares how many resources are NEEDED vs how many are ACTUALLY AVAILABLE,
    and returns a friendly shortage/surplus/exact-match message.
    Returns None if the user didn't provide an 'available' count.
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


def login_required(f):
    """
    A decorator (@login_required) you can add above any route function.
    If the user is NOT logged in (no 'user_id' in their session), they get
    redirected to the login page instead of seeing the protected page.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "info")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated



# ============================================================
# 4. AUTH ROUTES (login / register / logout)
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE Username = ?", (username,)).fetchone()
        conn.close()

        # check_password_hash compares the entered password against the securely stored hash
        if user and check_password_hash(user["PasswordHash"], password):
            session["user_id"] = user["UserID"]
            session["username"] = user["Username"]
            session["full_name"] = user["FullName"]
            flash(f"Welcome back, {user['FullName']}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()

        if not username or not password or not full_name:
            flash("Please fill in all fields.", "danger")
            return render_template("register.html")

        conn = get_db()
        existing = conn.execute("SELECT * FROM users WHERE Username = ?", (username,)).fetchone()
        if existing:
            conn.close()
            flash("That username is already taken.", "danger")
            return render_template("register.html")

        # Never store plain-text passwords - generate_password_hash securely hashes it first
        conn.execute(
            "INSERT INTO users (Username, PasswordHash, FullName, CreatedAt) VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), full_name, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()  # Wipes all session data, effectively logging the user out
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))



# ============================================================
# 5. MAIN ROUTES
# ============================================================

@app.route("/")
@login_required
def home():
    # Homepage - shows the prediction form with a dropdown of all hubs
    return render_template("index.html", hubs=HUBS)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Handles the prediction form submission:
    1. Reads the chosen date + hub
    2. Runs the ML model to predict demand
    3. Works out how many workers/vehicles are needed
    4. Compares against what the user says is available (if provided)
    5. Fetches the actual available staff/vehicles from the database
    6. Renders a results page with everything, plus a 14-day trend chart
    """
    try:
        date_str = request.form.get("date")
        hub = request.form.get("hub")

        if not date_str or not hub:
            return render_template("index.html", hubs=HUBS,
                                    error="Please provide both Date and Hub!")

        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        region = HUB_TO_REGION[hub]

        predicted_demand = predict_demand(hub, target_date)
        workers, vehicles = suggest_resources(predicted_demand)

        # Available workers/vehicles are OPTIONAL fields the user may type in manually
        avail_workers_str = request.form.get("available_workers", "").strip()
        avail_vehicles_str = request.form.get("available_vehicles", "").strip()

        available_workers = int(avail_workers_str) if avail_workers_str else None
        available_vehicles = int(avail_vehicles_str) if avail_vehicles_str else None

        worker_gap = compute_gap(workers, available_workers)
        vehicle_gap = compute_gap(vehicles, available_vehicles)

        # Compare predicted demand against the hub's historical average - used to show a "high demand" badge
        hub_avg = HUB_AVG_DEMAND.get(hub, 0)
        is_high_demand = predicted_demand > hub_avg

        # Fetch the real list of available staff/vehicles from the database (ALWAYS shown,
        # regardless of whether demand is high or not), excluding anyone already booked for this date
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

        trend = get_recent_trend(hub, target_date, days=14)

        return render_template("result.html", result=result, trend=trend,
                                prev_avail_workers=avail_workers_str,
                                prev_avail_vehicles=avail_vehicles_str)

    except Exception as e:
        # Catch-all so a bad input never crashes the whole app - shows the error on the homepage instead
        return render_template("index.html", hubs=HUBS,
                                error=f"Error: {str(e)}")


@app.route("/book", methods=["POST"])
@login_required
def book_resource():
    """
    Handles booking one or more workers/vehicles for a date range.
    Checks for overlapping bookings before confirming, so the same
    worker/vehicle can never be double-booked for overlapping dates.
    """
    worker_ids = request.form.getlist("worker_ids")
    vehicle_ids = request.form.getlist("vehicle_ids")
    hub = request.form.get("hub")
    date_from = request.form.get("date_from")
    date_to = request.form.get("date_to") or date_from
    booked_by = session.get("full_name", "Unknown")

    if not hub or not date_from:
        flash("Booking failed: missing hub or date.", "danger")
        return redirect(url_for("home"))

    if not worker_ids and not vehicle_ids:
        flash("Please select at least one worker or vehicle to book.", "info")
        return redirect(url_for("view_bookings", hub=hub))

    if date_to < date_from:
        flash("Booking failed: 'Date To' cannot be before 'Date From'.", "danger")
        return redirect(url_for("home"))

    conn = get_db()
    booked_count = 0
    skipped = []

    # Combine both worker and vehicle selections into one list to process together
    selections = [("worker", rid) for rid in worker_ids] + [("vehicle", rid) for rid in vehicle_ids]

    for resource_type, resource_id in selections:
        # Check if this resource already has an active booking that overlaps the requested date range
        overlap = conn.execute(
            "SELECT * FROM bookings WHERE ResourceType=? AND ResourceID=? AND Status='active' "
            "AND DateFrom <= ? AND DateTo >= ?", (resource_type, resource_id, date_to, date_from)
        ).fetchone()

        if overlap:
            skipped.append(resource_id)  # Already booked - skip this one
        else:
            conn.execute(
                "INSERT INTO bookings (ResourceType, ResourceID, Hub, DateFrom, DateTo, Status, BookedBy, CreatedAt) "
                "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
                (resource_type, resource_id, hub, date_from, date_to, booked_by, datetime.now().isoformat())
            )
            booked_count += 1

    conn.commit()
    conn.close()

    if booked_count:
        msg = f"Booked {booked_count} resource(s) successfully from {date_from} to {date_to}!"
        if skipped:
            msg += f" ({len(skipped)} already booked, skipped.)"
        flash(msg, "success")
    else:
        flash("All selected resources were already booked for that date range.", "danger")

    return redirect(url_for("view_bookings", hub=hub))

    return redirect(url_for("view_bookings", hub=hub))  # (unreachable - kept as-is from original)


@app.route("/bookings")
@login_required
def view_bookings():
    """
    Shows a table of all active bookings, optionally filtered by hub.
    Joins the bookings table with workers/vehicles to show the person's name and phone number.
    """
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
@login_required
def cancel_booking(booking_id):
    """
    Cancels a booking by marking its Status as 'cancelled' instead of deleting it,
    so we keep a history of past bookings.
    """
    conn = get_db()
    conn.execute("UPDATE bookings SET Status='cancelled' WHERE BookingID=?", (booking_id,))
    conn.commit()
    hub = request.form.get("hub", "")
    conn.close()
    flash("Booking cancelled.", "info")
    return redirect(url_for("view_bookings", hub=hub))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    A JSON API endpoint (no login required) - useful for testing with Postman/curl,
    or for other programs/services to call this prediction logic directly.
    Expects JSON body: {"date": "YYYY-MM-DD", "hub": "Chennai_Hub"}
    """
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


# ============================================================
# Entry point: only runs when this file is executed directly
# (e.g. `python app.py`), NOT when imported as a module elsewhere.
# ============================================================
if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
