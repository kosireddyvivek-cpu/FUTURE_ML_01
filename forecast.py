import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("train.csv")

# Convert date column
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Sort data
df = df.sort_values("Order Date")

# ==========================================
# 2. DATA PREPARATION
# ==========================================

# Create Year-Month
df["YearMonth"] = df["Order Date"].dt.to_period("M")

# Aggregate monthly sales
monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()

# Convert back to timestamp
monthly_sales["Order Date"] = monthly_sales["YearMonth"].dt.to_timestamp()

# Create time-based features
monthly_sales["Month_Number"] = range(len(monthly_sales))  # Trend
monthly_sales["Month"] = monthly_sales["Order Date"].dt.month  # Seasonality

# ==========================================
# 3. MODEL TRAINING
# ==========================================

X = monthly_sales[["Month_Number", "Month"]]
y = monthly_sales["Sales"]

model = LinearRegression()
model.fit(X, y)

# ==========================================
# 4. PREDICTIONS
# ==========================================

# Training predictions
train_predictions = model.predict(X)

# Future 6 months
future_index = np.arange(len(monthly_sales), len(monthly_sales) + 6)

future_df = pd.DataFrame({
    "Month_Number": future_index,
    "Month": [(m % 12) + 1 for m in future_index]
})

future_predictions = model.predict(future_df)

# Create future dates
last_date = monthly_sales["Order Date"].max()
future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(),
    periods=6,
    freq="MS"
)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y, train_predictions)
rmse = np.sqrt(mean_squared_error(y, train_predictions))
r2 = r2_score(y, train_predictions)

print("\n===== MODEL EVALUATION =====")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 4))

# ==========================================
# 6. EXPORT DATA FOR POWER BI
# ==========================================

historical_df = pd.DataFrame({
    "Date": monthly_sales["Order Date"],
    "Sales": y,
    "Type": "Actual"
})

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Sales": future_predictions,
    "Type": "Forecast"
})

final_output = pd.concat([historical_df, forecast_df])

final_output.to_csv("sales_forecast_output.csv", index=False)

print("CSV file created: sales_forecast_output.csv")

# ==========================================
# 7. VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(monthly_sales["Order Date"], y,
         marker='o', label="Actual Sales")

plt.plot(monthly_sales["Order Date"], train_predictions,
         label="Trend + Seasonality Fit")

plt.plot(future_dates, future_predictions,
         linestyle="--", marker='o',
         label="Future 6-Month Forecast")

plt.grid(True)
plt.legend()
plt.title("Sales Forecast with Trend and Seasonality")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.show()