# streamlit_forecast.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("https://drive.google.com/uc?id=1PgfZ2EAaQF-4wIPs_hQgVgDCJnEd7Bxi&export=download")
    items = pd.read_csv("lookup_item.csv")
    premises = pd.read_csv("lookup_premise.csv")

    df = df.merge(items, on="item_code", how="left")
    df = df.merge(premises, on="premise_code", how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# --- Sidebar ---
st.sidebar.header("Filter Options")
state_choice = st.sidebar.selectbox("Select State", df["state"].dropna().unique())
item_choice = st.sidebar.selectbox("Select Item", df["item"].dropna().unique())
method_choice = st.sidebar.radio("Forecasting Method", ["Holt", "ARIMA", "Prophet"])

# --- Filter Data ---
filtered = df[(df["state"] == state_choice) & (df["item"] == item_choice)].copy()
daily = filtered.groupby("date")["price"].mean().reset_index()
daily.set_index("date", inplace=True)

st.title("Malaysia Item Price Forecasting")
st.subheader(f"{item_choice} - {state_choice}")

if len(daily) > 10:
    if method_choice == "Holt":
        # Holt’s Method
        model = ExponentialSmoothing(daily, trend="add", seasonal=None)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(30)

    elif method_choice == "ARIMA":
        # ARIMA (basic order, can be auto-tuned with pmdarima)
        model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,7))  # weekly seasonality
        fit = model.fit(disp=False)
        forecast = fit.forecast(30)

    elif method_choice == "Prophet":
        # Prophet expects columns 'ds' and 'y'
        df_prophet = daily.reset_index().rename(columns={"date": "ds", "price": "y"})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=30, freq="D")
        forecast_df = model.predict(future)
        forecast = forecast_df.set_index("ds")["yhat"].tail(30)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(daily.index, daily["price"], label="Historical")
    ax.plot(forecast.index, forecast, "rx-", label="Forecast")
    ax.set_title(f"{method_choice} Forecast for {item_choice} in {state_choice}")
    ax.set_ylabel("Average Price (RM)")
    ax.legend()
    st.pyplot(fig)

    # --- Show forecast table ---
    st.subheader("Forecasted Prices (Next 30 Days)")
    st.dataframe(forecast.round(2).to_frame("Forecast Price (RM)"))

else:
    st.warning("Not enough data for this selection.")

