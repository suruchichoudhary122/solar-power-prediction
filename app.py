# ============================================================
# Solar Power Prediction — Streamlit Dashboard
# app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from xgboost import XGBRegressor

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Solar Power Prediction Dashboard",
    page_icon="☀️",
    layout="wide"
)

# ============================================================
# LOAD DATA & MODELS
# ============================================================
@st.cache_resource
def load_models():
    xgb    = joblib.load("model_xgboost.pkl")
    rf     = joblib.load("model_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    return xgb, rf, scaler

@st.cache_data
def load_data():
    df     = pd.read_csv("plant1_processed.csv",   parse_dates=["DATE_TIME"])
    gen    = pd.read_csv("Plant_1_Generation_Data.csv")
    anomalies = pd.read_csv("anomaly_report.csv")
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True)
    return df, gen, anomalies

xgb, rf, scaler = load_models()
df, gen, anomalies = load_data()

FEATURES = [
    "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
    "IRR_TEMP_RATIO", "TEMP_DELTA", "HOUR_SIN", "HOUR_COS",
    "DAY_OF_YEAR", "DC_POWER_LAG1", "DC_POWER_LAG2", "IRR_LAG1"
]

TARGET    = "AC_POWER"
split_idx = int(len(df) * 0.80)
X_test    = df[FEATURES].iloc[split_idx:]
y_test    = df[TARGET].iloc[split_idx:]
y_pred    = xgb.predict(X_test)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=80)
st.sidebar.title("Solar Dashboard")
st.sidebar.markdown("**Plant 1 — Power Prediction & O&M**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔮 Predict Power Output",
    "📈 Model Performance",
    "⚠️ Anomaly Detection",
    "🧠 SHAP Explainability"
])

st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Suruchi** | ECE Internship Project")
st.sidebar.markdown("Data: [Kaggle Solar Dataset](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.title(" Solar Power Generation — Plant 1")
    st.markdown("End-to-end ML project for solar power prediction and inverter health monitoring.")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inverters",      f"{gen['SOURCE_KEY'].nunique()}")
    col2.metric("Avg Daily Yield",      f"{df['DAILY_YIELD'].mean():.0f} kWh")
    col3.metric("Peak AC Power",        f"{df['AC_POWER'].max():.0f} kW")
    col4.metric("Anomalous Inverters",  f"{anomalies['SOURCE_KEY'].nunique()}")

    st.markdown("---")

    # Daily AC Power Time Series
    st.subheader("Daily Total AC Power Output")
    daily = df.groupby(df["DATE_TIME"].dt.date)["AC_POWER"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, color="darkorange", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("AC Power (kW)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Irradiation vs DC Power
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Irradiation vs DC Power")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["IRRADIATION"], df["AC_POWER"],
                   alpha=0.2, s=8, color="steelblue")
        ax.set_xlabel("Irradiation (W/m²)")
        ax.set_ylabel("AC Power (kW)")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Hourly Average AC Power")
        hourly = df.groupby("HOUR")["AC_POWER"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hourly.index, hourly.values, color="coral", edgecolor="white")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Avg AC Power (kW)")
        st.pyplot(fig)
        plt.close()

# ============================================================
# PAGE 2 — PREDICT POWER OUTPUT
# ============================================================
elif page == "🔮 Predict Power Output":
    st.title("🔮 Predict Solar AC Power Output")
    st.markdown("Adjust the input parameters to get a real-time power prediction.")

    col1, col2 = st.columns(2)

    with col1:
        irradiation  = st.slider("Irradiation (W/m²)",       0.0,   1.0,   0.5,  0.01)
        ambient_temp = st.slider("Ambient Temperature (°C)",  20.0,  45.0,  30.0, 0.5)
        module_temp  = st.slider("Module Temperature (°C)",   20.0,  60.0,  40.0, 0.5)
        hour         = st.slider("Hour of Day",               6,     18,    12)

    with col2:
        day_of_year  = st.slider("Day of Year",               1,     365,   180)
        dc_lag1      = st.slider("DC Power 15 min ago (kW)",  0.0,   10000.0, 5000.0, 100.0)
        dc_lag2      = st.slider("DC Power 30 min ago (kW)",  0.0,   10000.0, 4800.0, 100.0)
        irr_lag1     = st.slider("Irradiation 15 min ago",    0.0,   1.0,   0.48,  0.01)

    # Derived features
    irr_temp_ratio = irradiation / (module_temp + 1)
    temp_delta     = module_temp - ambient_temp
    hour_sin       = np.sin(2 * np.pi * hour / 24)
    hour_cos       = np.cos(2 * np.pi * hour / 24)

    input_df = pd.DataFrame([{
        "IRRADIATION"         : irradiation,
        "AMBIENT_TEMPERATURE" : ambient_temp,
        "MODULE_TEMPERATURE"  : module_temp,
        "IRR_TEMP_RATIO"      : irr_temp_ratio,
        "TEMP_DELTA"          : temp_delta,
        "HOUR_SIN"            : hour_sin,
        "HOUR_COS"            : hour_cos,
        "DAY_OF_YEAR"         : day_of_year,
        "DC_POWER_LAG1"       : dc_lag1,
        "DC_POWER_LAG2"       : dc_lag2,
        "IRR_LAG1"            : irr_lag1
    }])

    prediction = xgb.predict(input_df)[0]

    st.markdown("---")
    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)
    col1.metric(" Predicted AC Power", f"{prediction:,.0f} kW")
    col2.metric(" Module Temp",        f"{module_temp:.1f} °C")
    col3.metric(" Irradiation",        f"{irradiation:.2f} W/m²")

    # Gauge-style bar
    max_power = df["AC_POWER"].max()
    pct       = min(prediction / max_power, 1.0)
    st.markdown(f"**Output vs Plant Peak Capacity:** {pct*100:.1f}%")
    st.progress(float(pct))

# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("RMSE",     f"{rmse:.2f} kW")
    col3.metric("MAE",      f"{mae:.2f} kW")

    st.markdown("---")

    # Model comparison table
    st.subheader("Model Comparison")
    comparison = pd.read_csv("model_comparison.csv")
    st.dataframe(comparison, use_container_width=True)

    st.markdown("---")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted AC Power (XGBoost)")
    test_dates = df["DATE_TIME"].iloc[split_idx:].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(test_dates, y_test.values, label="Actual",    color="steelblue",  linewidth=1)
    ax.plot(test_dates, y_pred,        label="Predicted", color="darkorange", linewidth=1, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("AC Power (kW)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Residuals
    st.subheader("Residual Distribution")
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=50, color="slategray", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Error (kW)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close()

# ============================================================
# PAGE 4 — ANOMALY DETECTION
# ============================================================
elif page == "⚠️ Anomaly Detection":
    st.title("⚠️ Inverter Anomaly Detection")
    st.markdown("Inverters flagged when daily yield drops **below 80%** of plant average.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Inverters",     gen["SOURCE_KEY"].nunique())
    col2.metric("Anomalous Inverters", anomalies["SOURCE_KEY"].nunique())
    col3.metric("Total Anomaly Days",  len(anomalies))

    st.markdown("---")

    st.subheader("Anomaly Log")
    st.dataframe(anomalies.sort_values("PERF_RATIO").reset_index(drop=True),
                 use_container_width=True)

    st.markdown("---")

    @st.cache_data
    def compute_inv_daily(_gen):
        inv_daily = (_gen.groupby(["SOURCE_KEY", _gen["DATE_TIME"].dt.date])["DAILY_YIELD"]
                        .max().reset_index())
        inv_daily.columns = ["SOURCE_KEY", "DATE", "DAILY_YIELD"]
        overall_avg = inv_daily.groupby("DATE")["DAILY_YIELD"].transform("mean")
        inv_daily["PERF_RATIO"] = inv_daily["DAILY_YIELD"] / overall_avg
        return inv_daily

    inv_daily = compute_inv_daily(gen)

    st.subheader("Performance Ratio — Inverter Deep Dive")
    selected = st.selectbox("Select Inverter", inv_daily["SOURCE_KEY"].unique())
    inv_data = inv_daily[inv_daily["SOURCE_KEY"] == selected].sort_values("DATE")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(inv_data["DATE"], inv_data["PERF_RATIO"],
            color="steelblue", linewidth=1.5)
    ax.axhline(0.80, color="crimson", linestyle="--", label="Threshold (80%)")
    ax.fill_between(inv_data["DATE"], inv_data["PERF_RATIO"], 0.80,
                    where=inv_data["PERF_RATIO"] < 0.80,
                    color="crimson", alpha=0.3, label="Underperformance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Performance Ratio")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)
    plt.close()
# ============================================================
# PAGE 5 — SHAP EXPLAINABILITY
# ============================================================
# Sample for SHAP speed
# ============================================================
# PAGE 5 — SHAP EXPLAINABILITY
# ============================================================
elif page == "🧠 SHAP Explainability":
    st.title("🧠 SHAP Feature Explainability")
    st.markdown("Understanding **why** the model makes each prediction.")

    @st.cache_data
    def compute_shap(_model, _X_train, _X_test):
        explainer   = shap.Explainer(_model, _X_train)
        shap_values = explainer(_X_test)
        return shap_values

    # Sample 300 rows for speed
    X_test_shap = X_test.sample(300, random_state=42)

    with st.spinner("Computing SHAP values — only done once..."):
        shap_values = compute_shap(xgb,
                                   df[FEATURES].iloc[:split_idx],
                                   X_test_shap)

    # Summary plot
    st.subheader("Feature Impact Summary")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_shap,
                      feature_names=FEATURES, show=False)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Bar plot
    st.subheader("Mean Absolute SHAP Value per Feature")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test_shap,
                      feature_names=FEATURES,
                      plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Waterfall
    st.subheader("Single Prediction Explained (Waterfall)")
    sample = st.slider("Select test sample index", 0, len(X_test_shap)-1, 10)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_values[sample], show=False)
    st.pyplot(fig)
    plt.close()
