import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# TensorFlow/Keras for LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"

# Define background colors for each page
page_styles = {
    "landing": "background-color:#f0f8ff;",
    "overview": "background-color:#fff0f5;",
    "explore": "background-color:#f5fffa;",
    "forecast": "background-color:#ffffe0;"
}

def set_background(style):
    st.markdown(f"""
        <style>
        body {{
            {style}
        }}
        </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Landing Page
# -------------------------------
if st.session_state.page == "landing":
    set_background(page_styles["landing"])
    st.title("📊 Stock Forecasting Dashboard")
    st.write("Welcome! Click below to begin.")
    if st.button("Go to Data Overview"):
        st.session_state.page = "overview"

# -------------------------------
# Data Overview Page
# -------------------------------
elif st.session_state.page == "overview":
    set_background(page_styles["overview"])
    st.title("📄 Data Overview")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)
        st.session_state.df = df
        st.write(df.head())
        st.write(df.describe())
        if st.button("Exploratory Plots"):
            st.session_state.page = "explore"

# -------------------------------
# Exploratory Plots Page
# -------------------------------
elif st.session_state.page == "explore":
    set_background(page_styles["explore"])
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

# -------------------------------
# Forecasting Page (LSTM only)
# -------------------------------
elif st.session_state.page == "forecast":
    set_background(page_styles["forecast"])
    st.title("🤖 LSTM Forecasting")
    df = st.session_state.df
    days_ahead = st.slider("Days to Predict", 1, 30, 7)

    train_size = int(len(df) * 0.8)
    series = df["Close"].values

    # LSTM model
    n_input = 10
    generator = TimeseriesGenerator(series[:train_size], series[:train_size],
                                    length=n_input, batch_size=32)
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_input, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(generator, epochs=20, verbose=0)

    # Forecast next days
    pred = []
    batch = series[-n_input:].reshape((1, n_input, 1))
    for i in range(days_ahead):
        yhat = lstm_model.predict(batch, verbose=0)[0][0]
        pred.append(yhat)
        batch = np.append(batch[:,1:,:], [[[yhat]]], axis=1)

    future_dates = pd.date_range(df['Date'].iloc[-1], periods=days_ahead+1, freq='D')[1:]
    forecast_table = pd.DataFrame({"Date": future_dates, "Predicted Close": pred})
    st.write("📋 Forecast Table")
    st.write(forecast_table)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Date'], df['Close'], label="Historical")
    ax.plot(future_dates, pred, label="LSTM Forecast", linestyle="--", marker="o")
    ax.legend()
    st.pyplot(fig)

    # Navigation
    if st.button("Back to Overview"):
        st.session_state.page = "overview"
    if st.button("Back to Landing"):
        st.session_state.page = "landing"
