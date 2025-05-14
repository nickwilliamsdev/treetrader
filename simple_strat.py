import os
import pandas as pd
from feature_functions import apply_zscore_signal
from utils.backtester import Backtester

# Directory containing asset data
asset_dir = './hist_data/stocks/'

# Initialize EV dictionary
asset_ev_dict = {}

# Iterate over all assets
for asset_file in os.listdir(asset_dir):
    if asset_file.endswith('.txt'):
        asset_path = os.path.join(asset_dir, asset_file)
        df = pd.read_csv(asset_path, sep=',')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Apply feature engineering
        df = apply_zscore_signal(df)
        
        # Ensure the Signal column exists
        if 'Signal' not in df.columns:
            df['Signal'] = 0  # Default to no signals
        
        # Calculate expected return
        df['ExpectedReturn'] = df['Signal'] * df['Close'].pct_change().shift(-1)
        ev = df['ExpectedReturn'].mean()
        
        # Store EV
        asset_ev_dict[asset_file] = ev

# Select the highest EV asset
best_asset = max(asset_ev_dict, key=asset_ev_dict.get)
print(f"The highest EV asset is: {best_asset}")

# Backtest the strategy on the best asset
df = pd.read_csv(os.path.join(asset_dir, best_asset), sep=',')
df['Date'] = pd.to_datetime(df['Date'])
df = apply_zscore_signal(df)

# Ensure the Signal column exists
if 'Signal' not in df.columns:
    df['Signal'] = 0  # Default to no signals

backtester = Backtester(initial_balance=10000)
result = backtester.simple_backtest(df, signal_column='Signal')

# Plot the performance of the backtest
backtester.plot_performance(result)