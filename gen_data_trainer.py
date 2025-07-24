from utils.synthetic_data_service import SyntheticOHLCVGenerator
from utils.backtester import Backtester
from matplotlib import pyplot as plt
import numpy as np

# Generate synthetic OHLCV data
ohlc_gen_service = SyntheticOHLCVGenerator(n_steps=100, mu=0.005, sigma=0.1, dt=1, seed=42)
test_df = ohlc_gen_service.generate(start=100.0)
plt.plot(test_df['time'], test_df['close'], label='Synthetic Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Synthetic OHLCV Data')
plt.legend()
plt.show()
# Generate random buy/sell signals: 1 (buy), -1 (sell), 0 (hold)
np.random.seed(42)
test_df['Signal'] = np.random.choice([1, -1, 0], size=len(test_df))

# Run backtest
backtester = Backtester(initial_balance=10000)
result_df = backtester.simple_backtest(test_df, signal_column='Signal')

# Calculate and print performance
perf = backtester.calculate_performance(result_df)
print(perf)

# Plot performance
backtester.plot_performance(result_df)