from ..backtester import Backtester

# Initialize the backtester
backtester = Backtester(initial_balance=10000, transaction_cost=0.001)

# Perform a simple backtest
df = backtester.simple_backtest(df, signal_column='Signal')
performance = backtester.calculate_performance(df)
print("Simple Backtest Performance:", performance)

# Perform a martingale backtest
df = backtester.martingale_backtest(df, signal_column='Signal', base_bet_fraction=0.1)
performance = backtester.calculate_performance(df)
print("Martingale Backtest Performance:", performance)