# 🌸 Task 2: Predict Future Stock Prices

## 📌 Objective
# The goal of this task is to predict the next day’s closing price of a stock using historical data.
# A Linear Regression model is applied to learn patterns from previous stock prices.


# Import required libraries
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# --------------------------
# CLI Stock Symbol Selection
# --------------------------
# Use command line argument or default to Apple

if len(sys.argv) > 1:
    stock_symbol = sys.argv[1].upper()  # e.g., TSLA, GOOGL
else:
    stock_symbol = "AAPL"  # default

print(f"Selected Stock: {stock_symbol}")


# --------------------------
# Load Historical Stock Data
# --------------------------
df = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")
print(df.head())

# The dataset contains historical stock data including Open, High, Low, Close, Adj Close, and Volume.
# This will be used to predict the next day’s closing price.



# -----------------------------
# # Exploratory Data Analysis (EDA)
# -----------------------------
# Plot Closing Price Trend

plt.figure(figsize=(10,4))
plt.plot(df["Close"], label="Closing Price")
plt.title(f"{stock_symbol} Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.savefig(f"outputs/{stock_symbol}_close_trend.png")
plt.show()

# Closing price shows an overall trend with fluctuations.
# Visual inspection helps understand stock behavior over time.


# -----------------------------
# Plot Volume Trend
# -----------------------------

plt.figure(figsize=(10,4))
plt.plot(df["Volume"], label="Trading Volume", color="orange")
plt.title(f"{stock_symbol} Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.savefig(f"outputs/{stock_symbol}_volume_trend.png")
plt.show()


# Volume spikes often correspond to large price movements.
# Volume is an important feature for predicting price changes.


# -----------------------------------------
# Prepare Target Variable (Next Day Close)
# -----------------------------------------
# Predict next day's closing price

df["Target_Close"] = df["Close"].shift(-1)
df.dropna(inplace=True)


# --------------------------
# Feature Selection
# --------------------------
X = df[["Open", "High", "Low", "Volume"]]
y = df["Target_Close"]

# shift(-1) creates the "next day Close" target. Dataset is now ready for ML model.


# --------------------------
# Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Important for time series
)

# Time series data should not be shuffled to preserve temporal order.


# --------------------------
# Train Linear Regression Model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Linear Regression models relationship between features and next-day close price.


# --------------------------
# Prediction & Evaluation
# --------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# MAE indicates average error of prediction. Smaller MAE = better accuracy.


# --------------------------
# Plot Visualize Actual vs Predicted
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Closing Price")
plt.plot(predictions, label="Predicted Closing Price")
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

# Save output plot
plt.savefig(f"outputs/{stock_symbol}_stock_prediction.png")
plt.show()

# Model captures general trend; minor deviations during high volatility.
# Linear Regression provides baseline prediction.
