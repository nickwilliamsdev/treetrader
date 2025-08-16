from utils.synthetic_data_service import SyntheticOHLCVGenerator
from utils.backtester import Backtester
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def add_fibonacci_levels(df, lookback=20):
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 1.0]
    for ratio in fib_ratios:
        df[f'fib_{int(ratio*1000)}'] = np.nan

    for i in range(lookback, len(df)):
        high = df['high'].iloc[i-lookback:i].max()
        low = df['low'].iloc[i-lookback:i].min()
        for ratio in fib_ratios:
            level = high - (high - low) * ratio
            df.at[df.index[i], f'fib_{int(ratio*1000)}'] = level

    return df

# Example usage
df = SyntheticOHLCVGenerator(n_steps=1000, mu=0.001, sigma=0.1, dt=1, seed=42).generate(start=100.0)
df = add_fibonacci_levels(df)
print(df.tail())
plt.plot(df['time'], df['close'], label='Synthetic Close Price')
for ratio in [0.0, 0.236, 0.382, 0.5, 0.618, 1.0]:
    plt.axhline(y=df[f'fib_{int(ratio*1000)}'].iloc[-1], color='r', linestyle='--', label=f'Fib {ratio}')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['close'], label='Synthetic Close Price')
for ratio in [0.0, 0.236, 0.382, 0.5, 0.618, 1.0]:
    plt.plot(df['time'], df[f'fib_{int(ratio*1000)}'], linestyle='--', label=f'Fib {ratio}')
plt.legend()
plt.show()