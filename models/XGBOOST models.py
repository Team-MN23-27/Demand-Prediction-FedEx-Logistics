import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. DATA LOAD & SORTING
# ==========================================
file_path = r'/content/preprocessed_demand_data.csv'
df = pd.read_csv(file_path)

# Date column-ah datetime-ah maathi, chronological order-la sort pannrom
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ==========================================
# 2. FEATURE SELECTION
# ==========================================
target = 'OrderCount'

# Data Leakage thaduka 'NumberOfPieces' and 'TotalRevenue'-ah thookiyachu!
features = [
    'Holiday_Indicator_Encoded', 'Festival_Name_Encoded', 'Season_Encoded',
    'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 'Quarter',
    'OrderCount_lag_1', 'OrderCount_lag_7', 'OrderCount_lag_14',
    'OrderCount_rolling_mean_7', 'OrderCount_rolling_std_7'
]

categorical_features = ['Hub', 'Region']

# XGBoost categorical encoding features use panna 'category' dtypes mukkiyam
for col in categorical_features:
    df[col] = df[col].astype('category')

all_features = features + categorical_features

X = df[all_features]
y = df[target]

# ==========================================
# 3. TIME-SERIES TRAIN & TEST SPLIT
# ==========================================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ==========================================
# 4. XGBoost MODEL TRAINING
# ==========================================
print("Training XGBoost model...")

# Regressor object initialisation
# enable_categorical=True kudutha text strings encoding separate-ah panna thevai illai
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,    # CRITICAL FOR XGBOOST CATEGORIES
    random_state=42,
    early_stopping_rounds=50
)

# Model fitting with validation tracking
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

# ==========================================
# 5. PREDICTION & EVALUATION
# ==========================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- XGBoost Model Evaluation ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================
importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Important Features ---")
print(importance.head(5))
