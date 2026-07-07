# ================================
# Financial Data Analysis & Forecasting
# Dataset: Porsche Stock Data
# ================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv("porsche_stock_data.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nShape of Dataset")
print(df.shape)

# ---------------------------------
# Data Cleaning
# ---------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values("date")

# Check data types
print("\nData Types")
print(df.dtypes)

# Save cleaned dataset
df.to_csv("porsche_stock_data_cleaned.csv", index=False)

print("\nData Cleaning Completed Successfully!")

# ---------------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------------

print("\nStatistical Summary")
print(df.describe())

# Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(
    df[['open','high','low','close','volume']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------
# Closing Price Trend
# ---------------------------------

plt.figure(figsize=(12,6))
plt.plot(df['date'], df['close'], color='blue')
plt.title("Closing Price Trend")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid(True)
plt.show()

# ---------------------------------
# Opening Price Trend
# ---------------------------------

plt.figure(figsize=(12,6))
plt.plot(df['date'], df['open'], color='green')
plt.title("Opening Price Trend")
plt.xlabel("Date")
plt.ylabel("Opening Price")
plt.grid(True)
plt.show()

# ---------------------------------
# Trading Volume
# ---------------------------------

plt.figure(figsize=(12,6))
plt.bar(df['date'], df['volume'])
plt.title("Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.show()

# ---------------------------------
# Distribution of Closing Price
# ---------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['close'], bins=30, kde=True)
plt.title("Distribution of Closing Price")
plt.show()

# ---------------------------------
# Boxplot
# ---------------------------------

plt.figure(figsize=(8,5))
sns.boxplot(y=df['close'])
plt.title("Boxplot of Closing Price")
plt.show()

# ---------------------------------
# Moving Average
# ---------------------------------

df['MovingAverage30'] = df['close'].rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(df['date'], df['close'], label='Close Price')
plt.plot(df['date'], df['MovingAverage30'], label='30-Day Moving Average')
plt.title("Closing Price with Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ---------------------------------
# Forecasting
# ---------------------------------

print("\nForecasting Next 30 Days")

model = ExponentialSmoothing(
    df['close'],
    trend='add',
    seasonal=None
)

fit = model.fit()

forecast = fit.forecast(30)

print("\nForecast Values")
print(forecast)

# Forecast Plot

plt.figure(figsize=(12,6))

plt.plot(df['close'], label='Actual Closing Price')

future_index = range(len(df), len(df)+30)

plt.plot(
    future_index,
    forecast,
    label='Forecast',
    linewidth=3
)

plt.title("30-Day Stock Price Forecast")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()

# Save Forecast

forecast_df = pd.DataFrame({
    "Forecast_Close_Price": forecast
})

forecast_df.to_csv("forecast_30_days.csv", index=False)

print("\nForecast Saved Successfully!")

print("\nProject Completed Successfully!")