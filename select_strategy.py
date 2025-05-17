import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing asset data
asset_dir = './hist_data/crypto/kraken_4hr/'

# Parameters
window = 55  # lookback window for momentum/volatility
transaction_cost = 0.001  # 0.1%
threshold = 0.0  # minimum score to consider switching

# Load all assets and calculate momentum/volatility score
asset_dfs = {}
for asset_file in os.listdir(asset_dir):
    if asset_file.endswith('.txt'):
        asset_path = os.path.join(asset_dir, asset_file)
        df = pd.read_csv(asset_path, sep=',')
        df['Date'] = pd.to_datetime(df['date'], unit='s')
        df = df.set_index('Date')
        returns = df['close'].pct_change()
        momentum = returns.rolling(window).mean()
        volatility = returns.rolling(window).std()
        score = momentum / volatility
        df['Score'] = score
        asset_dfs[asset_file] = df[['Score', 'close']]

# Combine all scores into a single DataFrame
score_df = pd.DataFrame(index=sorted(set().union(*[df.index for df in asset_dfs.values()])))
for asset, df in asset_dfs.items():
    score_df[asset] = df['Score']

# Shift to avoid lookahead bias
score_df_shifted = score_df.shift(1)
best_scores = score_df_shifted.max(axis=1)
best_picks = score_df_shifted.idxmax(axis=1)
filtered_picks = best_picks.where(best_scores > threshold, None)

def backtest_best_pick(filtered_picks, price_dfs, initial_balance=10000, transaction_cost=0.001):
    portfolio = []
    balance = initial_balance
    prev_asset = None
    prev_price = None

    for date, asset in filtered_picks.items():
        if pd.isna(asset):
            portfolio.append((date, balance, None))
            continue
        try:
            price = price_dfs[asset].loc[date, 'close']
        except KeyError:
            portfolio.append((date, balance, asset))
            continue

        # Transaction cost if switching assets
        if prev_asset is not None and prev_asset != asset:
            balance *= (1 - transaction_cost)

        if prev_asset is not None and prev_price is not None:
            try:
                prev_close = price_dfs[prev_asset].loc[date, 'close']
                balance *= prev_close / prev_price
            except KeyError:
                pass

        prev_asset = asset
        prev_price = price
        portfolio.append((date, balance, asset))

    return pd.DataFrame(portfolio, columns=['Date', 'Portfolio', 'Asset']).set_index('Date')

# Prepare price DataFrames for backtest
price_dfs = {}
for asset_file in os.listdir(asset_dir):
    if asset_file.endswith('.txt'):
        asset_path = os.path.join(asset_dir, asset_file)
        df = pd.read_csv(asset_path, sep=',')
        df['Date'] = pd.to_datetime(df['date'], unit='s')
        df = df.set_index('Date')
        price_dfs[asset_file] = df

# Run backtest
portfolio_df = backtest_best_pick(filtered_picks, price_dfs, initial_balance=10000, transaction_cost=transaction_cost)

# Plot performance
plt.figure(figsize=(12, 6))
plt.plot(portfolio_df.index, portfolio_df['Portfolio'], label='Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Momentum/Volatility Filter Strategy Performance')
plt.legend()
plt.tight_layout()
plt.show()