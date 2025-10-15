import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from utils.synthetic_data_service import SyntheticOHLCVGenerator
import yfinance as yf
def find_pivots(data):
    """
    Finds local swing highs and lows in the data.
    """
    # Use a rolling window to smooth the data for clearer pivot detection
    data['smooth'] = data['Close'].rolling(window=10).mean()
    local_max = argrelextrema(data['smooth'].values, np.greater)[0]
    local_min = argrelextrema(data['smooth'].values, np.less)[0]
    
    pivots = pd.concat([data.iloc[local_max], data.iloc[local_min]]).sort_index()
    pivots = pivots[~pivots.index.duplicated(keep='first')]
    return pivots

def is_bullish_butterfly(pivots):
    """
    Checks if a sequence of 5 pivots (X, A, B, C, D) is a bullish butterfly pattern.
    """
    if len(pivots) != 5:
        return False

    X, A, B, C, D = pivots.iloc[0], pivots.iloc[1], pivots.iloc[2], pivots.iloc[3], pivots.iloc[4]

    # Validate the Fibonacci ratios
    xa_swing = abs(X['Close'] - A['Close'])
    ab_retracement = abs(B['Close'] - A['Close']) / xa_swing
    bc_swing = abs(C['Close'] - B['Close'])
    cd_extension = abs(D['Close'] - C['Close']) / bc_swing
    xd_extension = abs(D['Close'] - X['Close']) / xa_swing

    is_ab_retracement_valid = 0.786 - 0.05 <= ab_retracement <= 0.786 + 0.05
    is_cd_extension_valid = 1.618 - 0.05 <= cd_extension <= 2.24 + 0.05
    is_xd_extension_valid = 1.272 - 0.05 <= xd_extension <= 1.618 + 0.05
    
    return all([is_ab_retracement_valid, is_cd_extension_valid, is_xd_extension_valid])

def find_butterfly_patterns(ticker, period="1y"):
    """
    Downloads data and finds all butterfly patterns.
    """
    data = yf.download(ticker, period=period)
    pivots = find_pivots(data)
    
    patterns = []
    for i in range(len(pivots) - 4):
        segment = pivots.iloc[i:i+5]
        if is_bullish_butterfly(segment):
            patterns.append(segment)
            
    return data, patterns

def plot_pattern(data, pattern):
    """
    Plots the stock data and the detected pattern.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    
    if pattern:
        for p in pattern:
            ax.plot(p.index, p['Close'], marker='o', linestyle='--', color='red')
            ax.text(p.index[0], p['Close'][0], 'X')
            ax.text(p.index[1], p['Close'][1], 'A')
            ax.text(p.index[2], p['Close'][2], 'B')
            ax.text(p.index[3], p['Close'][3], 'C')
            ax.text(p.index[4], p['Close'][4], 'D')
        
    ax.set_title(f'Butterfly Pattern for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()

# --- Example Usage ---
ticker = 'AAPL'
data, patterns = find_butterfly_patterns(ticker)

if patterns:
    print(f"Found {len(patterns)} butterfly patterns for {ticker}.")
    plot_pattern(data, patterns)
else:
    print(f"No butterfly patterns found for {ticker}.")

