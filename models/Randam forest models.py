import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. DATA LOAD & SORTING

file_path = r'/content/preprocessed_demand_data.csv'
df = pd.read_csv(file_path)

# Date column-ah datetime-ah maathi, chronological order-la sort pannrom
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)


# 2. FEATURE SELECTION & ONE-HOT ENCODING

target = 'OrderCount'

# Data Leakage thaduka 'NumberOfPieces' and 'TotalRevenue'-ah thookiyachu!
features = [
    'Holiday_Indicator_Encoded', 'Festival_Name_Encoded', 'Season_Encoded',
    'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 'Quarter',
    'OrderCount_lag_1', 'OrderCount_lag_7', 'OrderCount_lag_14',
    'OrderCount_rolling_mean_7', 'OrderCount_rolling_std_7'
]

# CRITICAL FOR RANDOM FOREST: Scikit-learn direct-ah text/category string dtypes-ah edukathu.
# Adhnala 'Hub' and 'Region'-ah pd.get_dummies() vachi One-Hot Encode pannrom.
df_encoded = pd.get_dummies(df, columns=['Hub', 'Region'], drop_first=True)

# Update all features list after encoding
all_features = [col for col in df_encoded.columns if col != target and col != 'Date' and col not in ['NumberOfPieces', 'TotalRevenue']]

X = df_encoded[all_features]
y = df_encoded[target]


# 3. TIME-SERIES TRAIN & TEST SPLIT

split_idx = int(len(df_encoded) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# 4. RANDOM FOREST MODEL TRAINING

print("Training Random Forest model... (This might take a few seconds)")

# n_estimators=100 text trees build pannum parallel-ah
# n_jobs=-1 kudutha unga system-oda ella CPU cores-ayum use panni fast-ah train aagum
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,           # Tree depth-ah limit panrom overfitting thaduka
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. PREDICTION & EVALUATION

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Model Evaluation ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# 6. FEATURE IMPORTANCE

importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Important Features ---")
print(importance.head(5))
