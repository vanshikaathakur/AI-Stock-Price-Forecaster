import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib  # Corrected import
import plotly.graph_objects as go
from datetime import timedelta

# --- Model Loading ---
@st.cache_resource 
def load_prediction_model():
    # Make sure these files are in the same folder as app.py
    model = load_model('stock_lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_prediction_model()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}. Ensure 'stock_lstm_model.h5' and 'scaler.pkl' are present.")

# --- Prediction Logic ---
def get_forecast(ticker, days):
    df = yf.download(ticker, period='1y', interval='1d')
    if df.empty: return None, None, None
    
    # Preprocessing
    data = df['Close'].values.reshape(-1, 1)
    # Applying Winsorization logic to input data to match training
    lower, upper = pd.Series(data.flatten()).quantile([0.05, 0.95])
    data = np.clip(data, lower, upper)
    
    scaled_data = scaler.transform(data)
    
    # Recursive Multi-Step Forecasting
    last_sequence = scaled_data[-60:]
    future_preds = []
    current_batch = last_sequence.reshape(1, 60, 1)

    for _ in range(days):
        pred = model.predict(current_batch, verbose=0)[0]
        future_preds.append(pred)
        # Shift sequence: Remove first, add predicted to end
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    
    forecast_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    forecast_dates = [df.index[-1] + timedelta(days=i+1) for i in range(days)]
    
    return df, forecast_dates, forecast_prices

# --- Dashboard UI ---
st.set_page_config(page_title="Stock AI Predictor", layout="wide")
st.title("📈 AI Stock Price Forecaster")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL")
    forecast_days = st.slider("Days to Forecast", 1, 30, 7)
    predict_btn = st.button("Generate Forecast")

if predict_btn:
    with st.spinner('Calculating...'):
        hist_df, dates, prices = get_forecast(ticker, forecast_days)
        
        if hist_df is not None:
            # Plotly Visuals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df.index[-90:], y=hist_df['Close'][-90:], name="Past Price"))
            fig.add_trace(go.Scatter(x=dates, y=prices, name="AI Prediction", line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Table
            res_df = pd.DataFrame({'Date': dates, 'Price': prices}).set_index('Date')
            st.table(res_df)
        else:
            st.error("Ticker not found.")