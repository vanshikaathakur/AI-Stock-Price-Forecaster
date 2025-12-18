# 📈 AI Stock Price Forecaster

An end-to-end **Time Series Forecasting web application** built using **LSTM (Long Short-Term Memory)** neural networks. This project allows users to input a stock ticker (e.g., `AAPL`) and forecast future stock **closing prices for 7 to 30 days** using historical market data. The application is deployed as an **interactive Streamlit dashboard**.

---

## 🚀 Project Overview

Stock price prediction is a classic and challenging **time series forecasting** problem. Traditional statistical models often struggle with non-linear patterns and long-term dependencies present in financial data. This project leverages **deep learning (LSTM)** to capture temporal patterns and trends in historical stock prices and generate future forecasts.

The application is designed to be **user-friendly**, **interactive**, and **portfolio-ready**, making it suitable for data science and data analyst interviews.

---

## 🎯 Key Features

* 🔍 User input for **stock ticker symbol** (e.g., AAPL, TSLA, MSFT)
* 📆 Selectable **forecast horizon** (7–30 days)
* 📊 Interactive visualization of:

  * Historical stock prices
  * AI-predicted future prices
* 📋 Tabular view of forecasted values with dates
* 🌐 Web-based interface built using **Streamlit**

---

## 🧠 Machine Learning Approach

### Model Used

* **LSTM (Long Short-Term Memory)** neural network

### Why LSTM?

* Captures **long-term dependencies** in sequential data
* Handles **non-linear patterns** better than classical models
* Widely used in financial time series forecasting

---

## 🗂️ Project Workflow

1. **Data Collection**

   * Historical stock data fetched using Yahoo Finance API
   * Uses daily closing prices

2. **Data Preprocessing**

   * Handling missing values
   * Scaling data using MinMaxScaler
   * Creating time-series sequences (sliding window approach)

3. **Train-Test Split**

   * Time-based split (no shuffling)

4. **Model Training**

   * LSTM layers with dropout for regularization
   * Optimizer: Adam
   * Loss Function: Mean Squared Error (MSE)

5. **Forecasting**

   * Recursive multi-step forecasting
   * Generates predictions for future dates

6. **Visualization & Deployment**

   * Interactive charts using Plotly
   * Streamlit dashboard for user interaction

---

## 🖥️ Web Application Interface

The Streamlit dashboard includes:

* Sidebar for user inputs (stock ticker & forecast days)
* Line chart comparing historical prices and AI predictions
* Forecast results displayed in a table

---

## 🛠️ Tech Stack

| Category             | Tools & Libraries        |
| -------------------- | ------------------------ |
| Programming Language | Python                   |
| Data Handling        | Pandas, NumPy            |
| Visualization        | Plotly, Matplotlib       |
| Machine Learning     | TensorFlow, Keras        |
| Data Source          | Yahoo Finance (yfinance) |
| Web Framework        | Streamlit                |
| Version Control      | Git & GitHub             |

---

## 📁 Project Structure

```
AI-Stock-Price-Forecaster/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── lstm_model.h5
│
├── app.py              # Streamlit application
├── utils.py            # Helper functions
├── requirements.txt    # Project dependencies
└── README.md
```

---

## ▶️ How to Run the Project Locally

1. **Clone the repository**

```bash
git clone https://github.com/your-username/AI-Stock-Price-Forecaster.git
cd AI-Stock-Price-Forecaster
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

4. Open your browser and go to:

```
http://localhost:8501
```

---

## 📊 Sample Use Case

* Input Stock Ticker: `AAPL`
* Forecast Days: `7`
* Output:

  * Visual forecast curve
  * Predicted closing prices for the next 7 days

---

## ⚠️ Limitations

* Stock markets are influenced by external factors (news, events) not included in the model
* Predictions are **not financial advice**
* Model performance may vary across different stocks

---

## 📌 Future Enhancements

* Add confidence intervals for predictions
* Support multiple stocks comparison
* Integrate energy consumption forecasting
* Deploy on Streamlit Cloud / Render

---

## 📄 Resume-Ready Description

> Developed an LSTM-based time series forecasting web application to predict stock closing prices for 7–30 days, featuring real-time data fetching, interactive visualizations, and deployment using Streamlit.

---

## 🙌 Acknowledgements

* Yahoo Finance for stock market data
* Streamlit for rapid web app development
* TensorFlow/Keras for deep learning support

---

⭐ If you like this project, feel free to star the repository and share feedback!
