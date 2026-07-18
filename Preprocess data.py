

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 1. LOAD RAW DATA

RAW_FILE = "raw_shipment_data.csv"   
df = pd.read_csv(RAW_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Hub", "Date"]).reset_index(drop=True)


# 2. HOLIDAY / FESTIVAL / SEASON ENCODING


HOLIDAY_CALENDAR = {
    (1, 14): "Pongal",
    (8, 15): "Independence_Day",
    (10, 2): "Gandhi_Jayanti",
}

def get_festival_name(date):
    key = (date.month, date.day)
    return HOLIDAY_CALENDAR.get(key, "None")

df["Festival_Name"] = df["Date"].apply(get_festival_name)
df["Holiday_Indicator"] = (df["Festival_Name"] != "None").astype(int)

# Season purely month-based (verified against dataset)
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}
df["Season"] = df["Date"].dt.month.map(MONTH_TO_SEASON)


# Holiday_Indicator: 0/1 already numeric
# Festival_Name: Gandhi_Jayanti=0, Independence_Day=1, None=2, Pongal=3
# Season: Autumn=0, Monsoon=1, Summer=2, Winter=3
FESTIVAL_MAP = {"Gandhi_Jayanti": 0, "Independence_Day": 1, "None": 2, "Pongal": 3}
SEASON_MAP = {"Autumn": 0, "Monsoon": 1, "Summer": 2, "Winter": 3}

df["Holiday_Indicator_Encoded"] = df["Holiday_Indicator"]
df["Festival_Name_Encoded"] = df["Festival_Name"].map(FESTIVAL_MAP)
df["Season_Encoded"] = df["Season"].map(SEASON_MAP)


# 3. DATE-BASED FEATURES

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.weekday          # Monday=0 ... Sunday=6
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
df["Quarter"] = df["Date"].dt.quarter


# 4. LAG & ROLLING FEATURES (per Hub, chronological)

df = df.sort_values(["Hub", "Date"]).reset_index(drop=True)

df["OrderCount_lag_1"] = df.groupby("Hub")["OrderCount"].shift(1)
df["OrderCount_lag_7"] = df.groupby("Hub")["OrderCount"].shift(7)
df["OrderCount_lag_14"] = df.groupby("Hub")["OrderCount"].shift(14)

df["OrderCount_rolling_mean_7"] = df.groupby("Hub")["OrderCount"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
df["OrderCount_rolling_std_7"] = df.groupby("Hub")["OrderCount"].transform(
    lambda x: x.shift(1).rolling(7).std()
)


lag_cols = [
    "OrderCount_lag_1", "OrderCount_lag_7", "OrderCount_lag_14",
    "OrderCount_rolling_mean_7", "OrderCount_rolling_std_7",
]
df = df.dropna(subset=lag_cols).reset_index(drop=True)


# 5. SCALE LAG/ROLLING FEATURES (StandardScaler)

scaler = StandardScaler()
df[lag_cols] = scaler.fit_transform(df[lag_cols])

import joblib
joblib.dump(scaler, "lag_feature_scaler.pkl")
print("Scaler saved as lag_feature_scaler.pkl (prediction-ku idhே use pannanum)")


# 6. FINAL COLUMN ORDER & SAVE

final_cols = [
    "Date", "Hub", "Region", "OrderCount", "NumberOfPieces", "TotalRevenue",
    "Holiday_Indicator_Encoded", "Festival_Name_Encoded", "Season_Encoded",
    "Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "IsWeekend", "Quarter",
    "OrderCount_lag_1", "OrderCount_lag_7", "OrderCount_lag_14",
    "OrderCount_rolling_mean_7", "OrderCount_rolling_std_7",
]
df_final = df[final_cols]
df_final.to_csv("preprocessed_demand_data.csv", index=False)

print(f"Preprocessing done! Shape: {df_final.shape}")
print(df_final.head())
