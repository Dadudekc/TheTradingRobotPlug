import pandas as pd
import pandas_ta as ta

# Mock data
data = {
    "Date": pd.date_range(start="2025-01-01", periods=50, freq="D"),
    "Close": [100 + i * 0.5 + (i % 5 - 2) for i in range(50)]  # Simulated price data
}
mock_df = pd.DataFrame(data)

# Add debug: print dataset before MACD calculation
print("Initial Data:\n", mock_df.head())

# MACD calculation
fast_period = 12
slow_period = 21
signal_period = 5

mock_df["MACD"], mock_df["Signal"], mock_df["Histogram"] = ta.macd(
    close=mock_df["Close"],
    fast=fast_period,
    slow=slow_period,
    signal=signal_period
)

# Debug: Check intermediate values
print("Data after MACD Calculation:\n", mock_df[["Date", "Close", "MACD", "Signal", "Histogram"]].head())
