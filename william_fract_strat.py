import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def flatten_yf_multilevel(df):
    """
    Flattens a 3-level yfinance DataFrame with structure:
    Level 0: Price
    Level 1: OHLCV
    Level 2: Ticker(s)
    """
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
        # Drop the first level ('Price') and last level (ticker symbol)
        df.columns = df.columns.droplevel([1, 2])
    elif isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        # Drop ticker level if present
        df.columns = df.columns.droplevel(0)
    return df

def detect_fractals(df):
    df = df.copy()
    df['fractal_up'] = False
    df['fractal_down'] = False

    for i in range(2, len(df) - 2):
        high = float(df['High'].iloc[i])
        low = float(df['Low'].iloc[i])

        if (
            high > float(df['High'].iloc[i-1]) and
            high > float(df['High'].iloc[i-2]) and
            high > float(df['High'].iloc[i+1]) and
            high > float(df['High'].iloc[i+2])
        ):
            df.at[df.index[i], 'fractal_up'] = True

        if (
            low < float(df['Low'].iloc[i-1]) and
            low < float(df['Low'].iloc[i-2]) and
            low < float(df['Low'].iloc[i+1]) and
            low < float(df['Low'].iloc[i+2])
        ):
            df.at[df.index[i], 'fractal_down'] = True

    return df

def backtest_fractals(df, initial_capital=10000):
    df = df.copy()
    df['Position'] = 0
    position = 0

    for i in range(1, len(df)):
        if position == 0 and df['fractal_up'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i-1]:
            position = 1
        elif position == 0 and df['fractal_down'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i-1]:
            position = -1
        elif position == 1 and df['fractal_down'].iloc[i-1]:
            position = 0
        elif position == -1 and df['fractal_up'].iloc[i-1]:
            position = 0

        df.at[df.index[i], 'Position'] = position

    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Position'].shift(1)
    df['Equity'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
    return df

# Download data
df = yf.download("AAPL", start="2023-01-01", end="2025-01-01", group_by='ticker')
print("Original columns:", df.columns.tolist())

df = flatten_yf_multilevel(df)
print("Flattened columns:", df.columns.tolist())

df = detect_fractals(df)
print(df.head())
print(df[df['fractal_up'] | df['fractal_down']].head())
df = backtest_fractals(df)

print(df[df['fractal_up']].head(10))
print(df[df['fractal_down']].head(10))
print("Total fractal_ups:", df['fractal_up'].sum())
print("Total fractal_downs:", df['fractal_down'].sum())

# Plot equity
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Equity'], label="Fractal Strategy", color='blue')
plt.title("Bill Williams Fractal Strategy Backtest (AAPL)")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(True)
plt.show()
