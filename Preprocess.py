# ==========================================
# Demand Prediction - Data Cleaning
# ==========================================

import pandas as pd
import numpy as np

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_excel("shipment_demand_dataset.xlsx")

print("Dataset Loaded Successfully!\n")

# -----------------------------
# 2. Display Dataset Information
# -----------------------------
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

# -----------------------------
# 3. Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 4. Remove Duplicate Rows
# -----------------------------
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# -----------------------------
# 5. Fill Missing Values
# -----------------------------

# Numeric columns → Mean
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns → Mode
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# 6. Remove Extra Spaces
# -----------------------------
for col in categorical_cols:
    df[col] = df[col].str.strip()

# -----------------------------
# 7. Convert Date Columns
# -----------------------------
date_columns = [
    "Shipment_Date",
    "Order_Date",
    "Delivery_Date"
]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -----------------------------
# 8. Check Missing Values Again
# -----------------------------
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# 9. Save Cleaned Dataset
# -----------------------------
df.to_csv("cleaned_shipment_demand_dataset.csv", index=False)

print("\nCleaning Completed Successfully!")
print("Cleaned dataset saved as:")
print("cleaned_shipment_demand_dataset.csv")