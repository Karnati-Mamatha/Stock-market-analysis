# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")

# Load CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"

# -------------------------------
# Landing Page (Mint Green)
# -------------------------------
if st.session_state.page == "landing":
    st.markdown('<div class="landing">', unsafe_allow_html=True)
    st.title("📊 Stock Forecasting Dashboard")
    st.write("Welcome! This is your mint green title page.")
    if st.button("Go to Data Overview"):
        st.session_state.page = "overview"
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Data Overview Page
# -------------------------------
elif st.session_state.page == "overview":
    st.markdown('<div class="overview">', unsafe_allow_html=True)
    st.title("📄 Data Overview")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)
        st.session_state.df = df
        st.write(df.head())
        st.write(df.describe())
        if st.button("Exploratory Plots"):
            st.session_state.page = "explore"
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Exploratory Plots Page
# -------------------------------
elif st.session_state.page == "explore":
    st.markdown('<div class="explore">', unsafe_allow_html=True)
    st.title("📈 Exploratory Plots")
    df = st.session_state.df
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Date'], df['Close'])
    ax.set_title("Close Price Trend")
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Date'], df['Volume'])
    ax.set_title("Volume Traded Over Time")
    st.pyplot(fig)
    if st.button("Forecasting"):
        st.session_state.page = "forecast"
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Forecasting Page
# -------------------------------
elif st.session_state.page == "forecast":
    st.markdown('<div class="forecast">', unsafe_allow_html=True)
    st.title("🤖 Model Forecasting")
    df = st.session_state.df
    model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA", "Random Forest", "XGBoost"])
    days_ahead = st.slider("Days to Predict", 1, 30, 7)

    train_size = int(len(df) * 0.8)
    train_ts = df["Close"][:train_size]
    test_ts = df["Close"][train_size:]

    # Feature engineering
    for lag in range(1, 8):
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df.dropna(inplace=True)

    feature_columns = ['Open','High','Low','Volume','Adj Close',
                       'lag_close_1','lag_close_2','lag_close_3','lag_close_4',
                       'lag_close_5','lag_close_6','lag_close_7',
                       'MA7','MA30']
    X = df[feature_columns]
    y = df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_choice == "ARIMA":
        model = ARIMA(train_ts, order=(5,1,0)).fit()
        forecast = model.forecast(len(test_ts)+days_ahead)
        pred = forecast[-days_ahead:]
    elif model_choice == "SARIMA":
        model = SARIMAX(train_ts, order=(2,1,2), seasonal_order=(1,1,1,12)).fit()
        forecast = model.forecast(len(test_ts)+days_ahead)
        pred = forecast[-days_ahead:]
    elif model_choice == "Random Forest":
        rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train_scaled, y_train)
        pred = rf.predict(X_test_scaled[:days_ahead])
    else:
        xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8).fit(X_train_scaled, y_train)
        pred = xgb.predict(X_test_scaled[:days_ahead])

    future_dates = pd.date_range(df['Date'].iloc[-1], periods=days_ahead+1, freq='D')[1:]
    forecast_table = pd.DataFrame({"Date": future_dates, "Predicted Close": pred})
    st.write("📋 Forecast Table")
    st.write(forecast_table)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Date'], df['Close'], label="Historical")
    ax.plot(future_dates, pred, label=f"{model_choice} Forecast", linestyle="--", marker="o")
    ax.legend()
    st.pyplot(fig)

    if st.button("Comparison"):
        st.session_state.page = "compare"
    st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------
# Comparison Page
# -------------------------------
elif st.session_state.page == "compare":
    st.markdown('<div class="compare">', unsafe_allow_html=True)
    st.title("📊 Model Comparison")

    df = st.session_state.df
    train_size = int(len(df) * 0.8)
    train_ts = df["Close"][:train_size]
    test_ts = df["Close"][train_size:]

    # Feature engineering
    for lag in range(1, 8):
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df.dropna(inplace=True)

    feature_columns = [
        'Open','High','Low','Volume','Adj Close',
        'lag_close_1','lag_close_2','lag_close_3','lag_close_4',
        'lag_close_5','lag_close_6','lag_close_7',
        'MA7','MA30'
    ]
    X = df[feature_columns]
    y = df["Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Collect results
    results = {}

    # ARIMA
    arima_model = ARIMA(train_ts, order=(5,1,0)).fit()
    arima_pred = arima_model.forecast(len(test_ts))
    results["ARIMA"] = (
        mean_absolute_error(test_ts, arima_pred),
        np.sqrt(mean_squared_error(test_ts, arima_pred))
    )

    # SARIMA
    sarima_model = SARIMAX(train_ts, order=(2,1,2), seasonal_order=(1,1,1,12)).fit()
    sarima_pred = sarima_model.forecast(len(test_ts))
    results["SARIMA"] = (
        mean_absolute_error(test_ts, sarima_pred),
        np.sqrt(mean_squared_error(test_ts, sarima_pred))
    )

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    results["Random Forest"] = (
        mean_absolute_error(y_test, rf_pred),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    )

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8
    ).fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    results["XGBoost"] = (
        mean_absolute_error(y_test, xgb_pred),
        np.sqrt(mean_squared_error(y_test, xgb_pred))
    )

    # Build comparison table
    comparison = pd.DataFrame(results, index=["MAE", "RMSE"]).T
    st.write("### 📋 Model Performance Table")
    st.write(comparison)

    # Identify best model (lowest RMSE)
    best_model = comparison["RMSE"].idxmin()
    best_rmse = comparison.loc[best_model, "RMSE"]
    best_mae = comparison.loc[best_model, "MAE"]
    st.success(f"🏆 Best Model: {best_model} — RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}")

    # Bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(comparison.index, comparison["MAE"],
                color=["#6baed6" if m != best_model else "#31a354" for m in comparison.index])
    axes[0].set_title("MAE Comparison")
    axes[0].set_ylabel("MAE")

    axes[1].bar(comparison.index, comparison["RMSE"],
                color=["#9ecae1" if m != best_model else "#31a354" for m in comparison.index])
    axes[1].set_title("RMSE Comparison")
    axes[1].set_ylabel("RMSE")

    plt.tight_layout()
    st.pyplot(fig)

    # Navigation button
    if st.button("Back to Landing"):
        st.session_state.page = "landing"

    st.markdown('</div>', unsafe_allow_html=True)


