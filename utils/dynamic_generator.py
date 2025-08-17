import numpy as np
import pandas as pd

class DynamicSyntheticOHLCVGenerator:
    def __init__(self, n_steps=1000, base_mu=0.0, base_sigma=0.01, dt=1.0, seed=None):
        """
        n_steps: Number of time steps
        base_mu: Base drift coefficient
        base_sigma: Base volatility coefficient
        dt: Time increment
        seed: Random seed for reproducibility
        """
        self.n_steps = n_steps
        self.base_mu = base_mu
        self.base_sigma = base_sigma
        self.dt = dt
        self.seed = seed

    def generate(self, start=100.0):
        """
        Generate synthetic OHLCV data with fluctuating log returns.
        Returns a pandas DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume'].
        """
        np.random.seed(self.seed)
        time = np.arange(self.n_steps) * self.dt

        # Simulate dynamic drift and volatility
        mu = self.base_mu + np.sin(np.linspace(0, 10 * np.pi, self.n_steps)) * 0.001  # Sinusoidal drift
        sigma = self.base_sigma + np.random.normal(0, 0.001, self.n_steps)  # Random volatility

        # Generate log returns dynamically
        log_returns = np.random.normal(mu * self.dt, sigma * np.sqrt(self.dt), self.n_steps)

        # Generate price series
        price = start * np.exp(np.cumsum(log_returns))
        close = price
        open_ = np.concatenate([[start], close[:-1]])
        spread = np.abs(np.random.normal(0, self.base_sigma * start * 0.5, self.n_steps))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
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