## Team Members : Mathan
##                Mohammed Fazil
##                A.Muthulakshmi
##                S.Nesha
##                Rajasarathi

## B.Tech Artificial Intelligence and Data Science
## EGS Pillay Engineering College

## Title : Demand Prediction in FedEx Logistics Using Historical Data

## Project Overview
This project predicts future shipment demand using historical logistics data. The system helps logistics companies like FedEx estimate shipment volumes in advance, enabling better fleet allocation, warehouse planning, and staffing decisions.

## Problem Statement
Logistics companies experience fluctuating shipment volumes due to seasonal trends, holidays, promotional sales, and regional demand changes. Manual forecasting methods often fail to accurately predict these variations, leading to inefficient resource utilization and increased operational costs.

## Objective
- Predict future shipment demand using historical data.
- Improve warehouse resource planning.
- Optimize fleet allocation.
- Reduce delivery delays.
- Support better staffing decisions.

## Dataset
The dataset contains historical shipment records with features such as:
- Date
- Region
- Hub
- Shipment Volume / Demand
- Vehicle Type
- Holiday Indicator
- Weather (if available)
- Day of Week
- Month

## Data Preprocessing
The following preprocessing steps were performed:
- Removed duplicate records.
- Handled missing values.
- Converted date columns into datetime format.
- Encoded categorical variables.
- Cleaned unnecessary spaces.
- Saved the cleaned dataset.

## Machine Learning Model
Algorithm Used : XGBoost Regressor

## Features Used
Date
Year
Week
Day
Day_of_Week
Quarter
Shipment_Volume
Holiday
Previous_Day_Demand
Previous_Week_Demand
Rolling_7Day_Avg
Rolling_30Day_Avg
Total_Weight_lbs
Total_Revenue
Load_Count

## Target Variable
  Shipment_Volume

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook / VS Code

## Project Workflow
1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Demand Prediction
9. Visualization of Results

## Project Structure


Demand-Prediction/
│
├── dataset/
│   └── shipment_demand_dataset.xlsx
│
├── notebooks/
│   └── demand_prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── prediction.py
│
├── models/
│   └── xgboost_model.pkl
│
├── results/
│   ├── prediction.csv
│   └── graphs.png
│
├── README.md
└── requirements.txt


## Future Enhancements
- Real-time demand prediction
- Weather API integration
- Interactive dashboard using Power BI
- Route optimization integration
- Deployment using FastAPI
