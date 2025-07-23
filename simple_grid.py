import os
import pandas as pd
from feature_functions import apply_zscore_signal
from utils.backtester import Backtester

# Directory containing asset data
asset_dir = './hist_data/crypto/kraken_4hr/'

# Grid search ranges
window_range = range(5, 21, 2)      # e.g., 5, 7, 9, ..., 19
thresh_range = [0.5, 0.75, 1.0, 1.25, 1.5]

results = []

for reg_window in window_range:
    for reg_thresh in thresh_range:
        asset_ev_dict = {}
        for asset_file in os.listdir(asset_dir):
            if asset_file.endswith('.txt'):
                asset_path = os.path.join(asset_dir, asset_file)
                df = pd.read_csv(asset_path, sep=',')
                df['Date'] = pd.to_datetime(df['date'], unit='s')
                df = apply_zscore_signal(df, reg_window, reg_thresh)
                if 'Signal' not in df.columns:
                    df['Signal'] = 0
                df['ExpectedReturn'] = df['Signal'] * df['close'].pct_change().shift(-1)
                ev = df['ExpectedReturn'].mean()
                asset_ev_dict[asset_file] = ev

        best_asset = max(asset_ev_dict, key=asset_ev_dict.get)
        df = pd.read_csv(os.path.join(asset_dir, best_asset), sep=',')
        df['Date'] = pd.to_datetime(df['date'], unit='s')
        df = apply_zscore_signal(df, reg_window, reg_thresh)
        if 'Signal' not in df.columns:
            df['Signal'] = 0
        backtester = Backtester(initial_balance=10000)
        result = backtester.simple_backtest(df, signal_column='Signal')
        perf = backtester.calculate_performance(result)
        results.append({
            'window': reg_window,
            'thresh': reg_thresh,
            'asset': best_asset,
            'final_balance': perf['Final Balance'],
            'total_return': perf['Total Return (%)'],
            'sharpe': perf['Sharpe Ratio']
        })
        print(f"window={reg_window}, thresh={reg_thresh}, asset={best_asset}, final_balance={perf['Final Balance']:.2f}, sharpe={perf['Sharpe Ratio']:.2f}")

# Find the best parameter set
best = max(results, key=lambda x: x['final_balance'])
print("\nBest parameters:")
print(best)