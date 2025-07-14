import numpy as np
import pandas as pd

class SyntheticWienerGenerator:
    def __init__(self, n_steps=1000, mu=0.0, sigma=1.0, dt=1.0, seed=None):
        """
        n_steps: Number of time steps
        mu: Drift coefficient
        sigma: Diffusion coefficient (volatility)
        dt: Time increment
        seed: Random seed for reproducibility
        """
        self.n_steps = n_steps
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.seed = seed

    def generate(self, start=0.0):
        """
        Generate a synthetic Wiener process.
        Returns a pandas DataFrame with columns ['time', 'value'].
        """
        np.random.seed(self.seed)
        time = np.arange(self.n_steps + 1) * self.dt
        increments = np.random.normal(self.mu * self.dt, self.sigma * np.sqrt(self.dt), self.n_steps)
        values = np.concatenate([[start], start + np.cumsum(increments)])
        return pd.DataFrame({'time': time, 'value': values})

# Example usage:
# generator = SyntheticWienerGenerator(n_steps=500, mu=0.01, sigma=0.1, dt=1, seed=42)
# df = generator.generate(start=100)
# print(df.head())


class SyntheticOHLCVGenerator:
    def __init__(self, n_steps=1000, mu=0.0, sigma=0.01, dt=1.0, seed=None):
        """
        n_steps: Number of time steps
        mu: Drift coefficient
        sigma: Diffusion coefficient (volatility)
        dt: Time increment
        seed: Random seed for reproducibility
        """
        self.n_steps = n_steps
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.seed = seed

    def generate(self, start=100.0):
        """
        Generate synthetic OHLCV data using geometric Brownian motion for price.
        Returns a pandas DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume'].
        """
        np.random.seed(self.seed)
        time = np.arange(self.n_steps) * self.dt

        # Generate log returns
        log_returns = np.random.normal(
            (self.mu - 0.5 * self.sigma ** 2) * self.dt,
            self.sigma * np.sqrt(self.dt),
            self.n_steps
        )
        # Generate price series
        price = start * np.exp(np.cumsum(log_returns))
        close = price
        open_ = np.concatenate([[start], close[:-1]])
        # Simulate high/low as deviations from open/close
        spread = np.abs(np.random.normal(0, self.sigma * start * 0.5, self.n_steps))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        # Simulate volume as random positive values
        volume = np.random.randint(100, 10000, self.n_steps)

        df = pd.DataFrame({
            'time': time,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        return df