# ...existing code...
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from utils.synthetic_data_service import SyntheticOHLCVGenerator
import yfinance as yf

def find_pivots(data, smooth_window=10):
    """
    Finds local swing highs and lows in the data and returns a DataFrame
    with the pivot rows (sorted by index).
    """
    df = data.copy()
    df = df.reset_index()
    # ensure Close column exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    df['smooth'] = df['Close'].rolling(window=smooth_window, min_periods=1, center=False).mean()
    local_max = argrelextrema(df['smooth'].values, np.greater)[0]
    local_min = argrelextrema(df['smooth'].values, np.less)[0]

    idx = np.concatenate([local_max, local_min])
    if len(idx) == 0:
        return pd.DataFrame(columns=df.columns)

    pivots = df.iloc[np.sort(idx)].copy()
    pivots = pivots.loc[~pivots.index.duplicated(keep='first')]
    return pivots

def is_bullish_butterfly(pivots):
    """
    Checks if a sequence of 5 pivot rows (X, A, B, C, D) is a bullish butterfly.
    Expects pivots to be a DataFrame with exactly 5 rows.
    """
    if not isinstance(pivots, pd.DataFrame) or len(pivots) != 5:
        return False

    # extract close prices as floats
    try:
        closes = [float(pivots.iloc[i]['Close']) for i in range(5)]
    except Exception:
        return False

    Xc, Ac, Bc, Cc, Dc = closes

    xa_swing = abs(Xc - Ac)
    if xa_swing == 0:
        return False

    ab_retracement = abs(Bc - Ac) / xa_swing
    bc_swing = abs(Cc - Bc)
    if bc_swing == 0:
        return False
    cd_extension = abs(Dc - Cc) / bc_swing
    xd_extension = abs(Dc - Xc) / xa_swing

    is_ab = (0.736 <= ab_retracement <= 0.836)  # ~0.786 ± 0.05
    is_cd = (1.568 <= cd_extension <= 2.29)     # ~1.618 - 2.24 (relaxed a bit)
    is_xd = (1.222 <= xd_extension <= 1.668)    # ~1.272 - 1.618 (relaxed)

    return is_ab and is_cd and is_xd

def find_butterfly_patterns(ticker, period="1y", smooth_window=10):
    """
    Downloads data for `ticker` and finds all bullish butterfly patterns.
    Returns the full data and a list of DataFrames (each with 5 rows) representing patterns.
    """
    data = yf.download(ticker, period=period)
    if data.empty:
        return data, []

    pivots = find_pivots(data, smooth_window=smooth_window)
    patterns = []
    # iterate over contiguous groups of 5 pivot rows
    for i in range(len(pivots) - 4):
        segment = pivots.iloc[i:i+5]
        if is_bullish_butterfly(segment):
            patterns.append(segment)
    return data, patterns

def plot_pattern(data, pattern, ticker="TICKER"):
    """
    Plots the stock data and overlays the detected pattern.
    `pattern` is a DataFrame with 5 rows (X,A,B,C,D).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='C0')

    if isinstance(pattern, pd.DataFrame) and len(pattern) == 5:
        labels = ['X', 'A', 'B', 'C', 'D']
        for i, (idx, row) in enumerate(pattern.iterrows()):
            ax.plot(row['Date'] if 'Date' in row.index else data.index[idx], row['Close'],
                    marker='o', linestyle='None', color='red')
            ax.annotate(labels[i], (data.index[idx], row['Close']),
                        textcoords="offset points", xytext=(0,8), ha='center', color='red')

    ax.set_title(f'Butterfly Pattern for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    ticker = 'SPX'
    data, patterns = find_butterfly_patterns(ticker, period="1y", smooth_window=10)

    if patterns:
        print(f"Found {len(patterns)} butterfly patterns for {ticker}.")
        # plot first pattern as example
        plot_pattern(data, patterns[0], ticker=ticker)
    else:
        print(f"No butterfly patterns found for {ticker}.")