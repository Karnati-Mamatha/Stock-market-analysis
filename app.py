# dashboard_app.py

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
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# STREAMLIT DASHBOARD
# -------------------------------
st.set_page_config(page_title="Stock Market Price Forecasting Dashboard", layout="wide")

# Custom CSS for background colors
st.markdown("""
    <style>
    .landing {background-color: #f0f8ff; padding: 30px; border-radius: 10px;}
    .overview {background-color: #fff0f5; padding: 20px; border-radius: 10px;}
    .explore {background-color: #f5fffa; padding: 20px; border-radius: 10px;}
    .forecast {background-color: #ffffe0; padding: 20px; border-radius: 10px;}
    .compare {background-color: #f0fff0; padding: 20px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Landing Page
# -------------------------------
st.markdown("<div class='landing'>", unsafe_allow_html=True)
st.title("📊 Stock Market Price Forecasting Dashboard")
st.write("Welcome! Upload your dataset and navigate through the tabs below to explore insights, forecasts, and model comparisons.")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "SARIMA", "Random Forest", "XGBoost"])
days_ahead = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Overview", "📈 Exploratory Plots", "🤖 Model Forecasting", "📊 Comparison"])

    # -------------------------------
    # Tab 1: Data Overview
    # -------------------------------
    with tab1:
        st.markdown("<div class='overview'>", unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        st.write(df.head())
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        st.write("Summary Statistics")
        st.write(df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Tab 2: Exploratory Plots
    # -------------------------------
    with tab2:
        st.markdown("<div class='explore'>", unsafe_allow_html=True)
        st.subheader("Close Price Trend")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Date'], df['Close'])
        ax.set_title("Close Price Trend")
        st.pyplot(fig)

        st.subheader("Volume Trend")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Date'], df['Volume'])
        ax.set_title("Volume Traded Over Time")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Tab 3: Model Forecasting
    # -------------------------------
    with tab3:
        st.markdown("<div class='forecast'>", unsafe_allow_html=True)
        st.subheader(f"{model_choice} Forecast for Next {days_ahead} Days")

        # Train-Test Split
        train_size = int(len(df) * 0.8)
        train_ts = df["Close"][:train_size]
        test_ts = df["Close"][train_size:]

        # Feature Engineering for ML
        for lag in range(1, 8):
            df[f"lag_close_{lag}"] = df["Close"].shift(lag)
        df["MA7"] = df["Close"].rolling(7).mean()
        df["MA30"] = df["Close"].rolling(30).mean()
        df["Volatility_7"] = df["Close"].rolling(7).std()
        df["Volatility_30"] = df["Close"].rolling(30).std()
        df["Return_7"] = df["Close"].pct_change(7)
        df["Return_30"] = df["Close"].pct_change(30)
        df['day'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['weekday'] = df['Date'].dt.dayofweek
        df.dropna(inplace=True)

        feature_columns = [
            'Open','High','Low','Volume','Adj Close',
            'lag_close_1','lag_close_2','lag_close_3','lag_close_4',
            'lag_close_5','lag_close_6','lag_close_7',
            'MA7','MA30','Volatility_7','Volatility_30',
            'Return_7','Return_30','day','month','year','weekday'
        ]
        X = df[feature_columns]
        y = df["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model selection
        if model_choice == "ARIMA":
            model = ARIMA(train_ts, order=(5,1,0))
            fit = model.fit()
            forecast = fit.forecast(len(test_ts)+days_ahead)
            pred = forecast[-days_ahead:]
        elif model_choice == "SARIMA":
            model = SARIMAX(train_ts, order=(2,1,2), seasonal_order=(1,1,1,12))
            fit = model.fit()
            forecast = fit.forecast(len(test_ts)+days_ahead)
            pred = forecast[-days_ahead:]
        elif model_choice == "Random Forest":
            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            rf.fit(X_train_scaled, y_train)
            future_X = X_test_scaled[:days_ahead]
            pred = rf.predict(future_X)
        elif model_choice == "XGBoost":
            xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8)
            xgb.fit(X_train_scaled, y_train)
            future_X = X_test_scaled[:days_ahead]
            pred = xgb.predict(future_X)

        # Forecast Table
        future_dates = pd.date_range(df['Date'].iloc[-1], periods=days_ahead+1, freq='D')[1:]
        forecast_table = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": pred
        })
        st.write("📋 Forecast Table")
        st.write(forecast_table)

        # Visualization
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['Date'], df['Close'], label="Historical Close")
        ax.plot(future_dates, pred, label=f"{model_choice} Forecast", linestyle="--", marker="o")
        ax.set_title(f"{model_choice} Closing Price Forecast")
        ax.legend()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Tab 4: Comparison
    # -------------------------------
    with tab4:
        st.markdown("<div class='compare'>", unsafe_allow_html=True)
        st.subheader("Model Comparison (MAE & RMSE)")
        # Quick evaluation on test set
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
    st.write(comparison)

    # Identify best model (lowest RMSE)
    best_model = comparison["RMSE"].idxmin()
    best_rmse = comparison.loc[best_model, "RMSE"]
    best_mae = comparison.loc[best_model, "MAE"]
    st.success(f"🏆 Best Model: {best_model}  —  RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}")

    # Bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(comparison.index, comparison["MAE"], color=[
        "#6baed6" if m != best_model else "#31a354" for m in comparison.index
    ])
    axes[0].set_title("MAE Comparison")
    axes[0].set_ylabel("MAE")

    axes[1].bar(comparison.index, comparison["RMSE"], color=[
        "#9ecae1" if m != best_model else "#31a354" for m in comparison.index
    ])
    axes[1].set_title("RMSE Comparison")
    axes[1].set_ylabel("RMSE")

    plt.tight_layout()
    st.pyplot(fig)




