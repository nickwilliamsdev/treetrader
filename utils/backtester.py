import pandas as pd
import numpy as np
class Backtester:
    def __init__(self, initial_balance=10000, transaction_cost=0.001):
        """
        Initialize the backtester with starting balance and transaction cost.
        """
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

    def simple_backtest(self, df, signal_column):
        """
        Perform a simple backtest with full portfolio entries and exits.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data and signals.
            signal_column (str): Column name for buy/sell signals (1 for buy, -1 for sell).
        
        Returns:
            pd.DataFrame: DataFrame with portfolio value over time.
        """
        balance = 10000  # Starting balance
        position = 0  # Current position (number of shares)
        transaction_cost = 0.001  # 0.1% per trade
        df['Portfolio'] = 0  # Track portfolio value

        for i in range(len(df)):
            if df.loc[i, signal_column] == 1 and balance > 0:  # Buy signal
                position = (balance * (1 - transaction_cost)) / df.loc[i, 'close']
                balance = 0
            elif df.loc[i, signal_column] == -1 and position > 0:  # Sell signal
                balance = position * df.loc[i, 'close'] * (1 - transaction_cost)
                position = 0
            # Update portfolio value using .loc to avoid chained assignment
            df.loc[i, 'Portfolio'] = balance + (position * df.loc[i, 'close'])

        return df

    def martingale_backtest(self, df, signal_column, base_bet_fraction=0.1):
        """
        Perform a backtest using a martingale betting strategy.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data and signals.
            signal_column (str): Column name for buy/sell signals (1 for buy, -1 for sell).
            base_bet_fraction (float): Fraction of balance to bet initially (e.g., 0.1 for 10%).
        
        Returns:
            pd.DataFrame: DataFrame with portfolio value over time.
        """
        balance = self.initial_balance
        position = 0  # Number of shares held
        bet_fraction = base_bet_fraction
        df['Portfolio'] = 0  # Track portfolio value

        for i in range(len(df)):
            if df[signal_column].iloc[i] == 1 and balance > 0:  # Buy signal
                bet_amount = balance * bet_fraction
                position += (bet_amount * (1 - self.transaction_cost)) / df['close'].iloc[i]
                balance -= bet_amount
                bet_fraction *= 2  # Double the bet for martingale
            elif df[signal_column].iloc[i] == -1 and position > 0:  # Sell signal
                balance += position * df['close'].iloc[i] * (1 - self.transaction_cost)
                position = 0
                bet_fraction = base_bet_fraction  # Reset bet fraction after a win
            df['Portfolio'].iloc[i] = balance + (position * df['close'].iloc[i])

        return df

    def calculate_performance(self, df, risk_free_rate=0.0):
        """
        Calculate performance metrics for the backtest.
        
        Args:
            df (pd.DataFrame): DataFrame containing portfolio values.
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation (annualized, e.g., 0.02 for 2%).
        
        Returns:
            dict: Dictionary with performance metrics (final balance, total return, etc.).
        """
        final_balance = df['Portfolio'].iloc[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100

        # Calculate daily returns
        df['Daily Return'] = df['Portfolio'].pct_change().fillna(0)

        # Max Drawdown
        cumulative_max = df['Portfolio'].cummax()
        drawdown = (df['Portfolio'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100  # as percentage

        # Sharpe Ratio (assume 252 trading days)
        excess_daily_return = df['Daily Return'] - (risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_daily_return) / np.std(excess_daily_return, ddof=1) * np.sqrt(252) if np.std(excess_daily_return, ddof=1) > 0 else np.nan

        return {
            'Initial Balance': self.initial_balance,
            'Final Balance': final_balance,
            'Total Return (%)': total_return,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }

    def plot_performance(self, df):
        """
        Plot the portfolio value over time.

        Args:
            df (pd.DataFrame): DataFrame containing portfolio values with a 'Portfolio' column.
        """
        import matplotlib.pyplot as plt

        if 'Portfolio' not in df.columns:
            raise ValueError("The DataFrame must contain a 'Portfolio' column to plot performance.")

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Portfolio'], label='Portfolio Value', color='blue')
        plt.title("Portfolio Performance Over Time")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid()
        plt.show()