# evaluation/metrics.py
import pandas as pd

def calculate_performance(df: pd.DataFrame) -> dict:
    # E.g., a simple total return calculation
    first = df["Close"].iloc[0]
    last = df["Close"].iloc[-1]
    total_return = (last - first) / first * 100
    return {"Total Return (%)": total_return}
