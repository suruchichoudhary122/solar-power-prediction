# ☀️ Solar Power Output Prediction

An end-to-end machine learning project to predict solar plant AC power output 
and detect underperforming inverters using real plant generation and weather data.

## 🔍 Project Overview
- **Domain:** Renewable Energy / Solar O&M
- **ML Task:** Regression (Power Output Prediction)
- **Dataset:** [Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

## ⚙️ Tech Stack
Python · XGBoost · Scikit-learn · SHAP · Streamlit · Pandas · Matplotlib

## 📊 Results
| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.9929 | 642.58 kW | 438.17 kW |
| Random Forest | 0.9934 | 620.21 kW | 407.36 kW |
| XGBoost | 0.9934 | 619.20 kW | 395.12 kW |

## 🔧 Features
- Exploratory Data Analysis (12 plots)
- Feature engineering (time, physics-based, lag features)
- 3 ML models trained and compared
- SHAP explainability (beeswarm, waterfall, bar plots)
- Anomaly detection — flags inverters below 80% performance ratio
- Interactive Streamlit dashboard

## 🚀 Run the Dashboard
pip install -r requirements.txt
streamlit run app.py

## 📁 Project Structure
notebooks/        → EDA, preprocessing, modeling, evaluation
app.py            → Streamlit dashboard
model_*.pkl       → Saved trained models
anomaly_report.csv → Flagged inverter anomaly log
