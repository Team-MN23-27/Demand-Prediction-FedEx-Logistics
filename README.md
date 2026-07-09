## Team Members : 
##                K.Mathan
##                M.Mohammed Fazil
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
Algorithm Used : LightGBM

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

Demand-Prediction-Project/
│
├── data/
│   ├── shipment_demand_dataset.xlsx          
│   └── cleaned_shipment_demand_dataset.csv   
│
├── models/
│   ├── lightgbm_model.pkl                    
│   └── features.pkl                          
│
├── src/
│   ├── preprocess.py                         # Data Cleaning
│   ├── train_lightgbm.py                     # Model Training
│   ├── predict.py                            # Prediction Script (Optional)
│   └── app.py                                # Flask API
│
├── templates/
│   └── index.html                            # Flask Frontend (Optional)
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── fedex_logo.png
│
├── results/
│   ├── evaluation_metrics.txt
│   ├── prediction_results.csv
│   └── feature_importance.png
│
├── notebooks/
│   └── Demand_Prediction.ipynb
│
├── requirements.txt                          
├── README.md                                 
├── .gitignore                                

                                 
## Future Enhancements
- Real-time demand prediction
- Integrate real-time shipment data for live demand forecasting.
- Add weather and holiday information to improve predictions.
- Develop a dashboard to visualize demand trends.
- Deploy the model as a web application using FastAPI.
- Support hub-wise and region-wise demand prediction.
- Enable automatic fleet and warehouse resource planning.
  
