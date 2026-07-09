import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load Dataset
df = pd.read_csv("cleaned_shipment_demand_dataset.csv")

# -----------------------------
# Convert Date Column
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month
df["Day_Num"] = df["Date"].dt.day

# Remove original Date column
df.drop("Date", axis=1, inplace=True)

# -----------------------------
# Convert categorical columns
# -----------------------------
df = pd.get_dummies(df, columns=["Day_of_Week", "Holiday"], drop_first=True)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Shipment_Volume", axis=1)
y = df["Shipment_Volume"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE :", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2 Score :", r2_score(y_test, y_pred))

# Accuracy (using R²)
r2 = r2_score(y_test, y_pred)
accuracy = r2 * 100
print("Accuracy :", round(accuracy, 2), "%")

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "random_forest_model.pkl")

print("Model Saved Successfully!")