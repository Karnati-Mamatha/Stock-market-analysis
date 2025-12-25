# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# STREAMLIT APP
# -------------------------------
st.title("📈 Stock Forecasting Model Comparison")
st.write("This app compares ARIMA, SARIMA, Random Forest, and XGBoost models to find the best performer.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Daily_Return'] = df['Close'].pct_change()

    # Feature Engineering
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
    df.reset_index(drop=True, inplace=True)

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    train_size = int(len(df) * 0.8)
    train_ts = df["Close"][:train_size]
    test_ts = df["Close"][train_size:]

    # -------------------------------
    # ARIMA
    # -------------------------------
    arima_model = ARIMA(train_ts, order=(5,1,0))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(len(test_ts))
    arima_mae = mean_absolute_error(test_ts, arima_pred)
    arima_rmse = np.sqrt(mean_squared_error(test_ts, arima_pred))

    # -------------------------------
    # SARIMA
    # -------------------------------
    sarima_model = SARIMAX(train_ts, order=(2,1,2), seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit()
    sarima_pred = sarima_fit.forecast(len(test_ts))
    sarima_mae = mean_absolute_error(test_ts, sarima_pred)
    sarima_rmse = np.sqrt(mean_squared_error(test_ts, sarima_pred))

    # -------------------------------
    # Machine Learning Models
    # -------------------------------
    feature_columns = [
        'Open','High','Low','Volume','Adj Close',
        'lag_close_1','lag_close_2','lag_close_3','lag_close_4',
        'lag_close_5','lag_close_6','lag_close_7',
        'MA7','MA30','Volatility_7','Volatility_30',
        'Daily_Return','Return_7','Return_30',
        'day','month','year','weekday'
    ]
    X = df[feature_columns]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # XGBoost
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8)
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    # -------------------------------
    # Model Comparison
    # -------------------------------
    comparison = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA", "Random Forest", "XGBoost"],
        "MAE": [arima_mae, sarima_mae, rf_mae, xgb_mae],
        "RMSE": [arima_rmse, sarima_rmse, rf_rmse, xgb_rmse]
    })

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(comparison)

    # Best Model
    best_model = comparison.loc[comparison["RMSE"].idxmin()]
    st.success(f"🏆 Best Model: **{best_model['Model']}** with RMSE = {best_model['RMSE']:.4f}")

    # -------------------------------
    # Plot Actual vs Predicted for Best Model
    # -------------------------------
    st.subheader("📉 Actual vs Predicted (Best Model)")
    fig, ax = plt.subplots(figsize=(12,5))

    if best_model["Model"] == "ARIMA":
        ax.plot(test_ts.values, label="Actual")
        ax.plot(arima_pred.values, label="ARIMA Prediction", linestyle="--")
    elif best_model["Model"] == "SARIMA":
        ax.plot(test_ts.values, label="Actual")
        ax.plot(sarima_pred.values, label="SARIMA Prediction", linestyle="--")
    elif best_model["Model"] == "Random Forest":
        ax.plot(y_test.values, label="Actual")
        ax.plot(rf_pred, label="Random Forest Prediction", linestyle="--")
    else:
        ax.plot(y_test.values, label="Actual")
        ax.plot(xgb_pred, label="XGBoost Prediction", linestyle="--")

    ax.set_title(f"{best_model['Model']} – Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)
