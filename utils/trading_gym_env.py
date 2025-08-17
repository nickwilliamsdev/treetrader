import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import random

class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for a single-asset trading agent.

    The environment simulates a simplified trading scenario with actions to Buy, Sell, or Hold.
    The state is a rolling window of historical market data.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=60, render_mode=None):
        super().__init__()
        
        # DataFrame containing historical market data (e.g., OHLCV)
        self.df = df
        self.window_size = window_size
        
        # The number of features in our state space
        # Here we assume a simple state of OHLCV
        # In a real scenario, you'd add more indicators (e.g., moving averages)
        self.features = df.shape[1]

        # The observation space is a window of historical market data.
        # It's a Box space with shape (window_size, num_features).
        # We define a low and high value to bound the space.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.features), dtype=np.float32
        )

        # The action space is discrete: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Initialize environment state
        self.current_step = self.window_size
        self.done = False
        self.render_mode = render_mode
        
        # Portfolio attributes
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        
        # For rendering/tracking
        self.history = []

    def _get_state(self):
        """
        Retrieves the current state from the DataFrame.
        The state is the last `window_size` number of rows.
        """
        # Ensure we don't go out of bounds
        start_index = max(0, self.current_step - self.window_size)
        end_index = self.current_step
        
        # Get the window and convert to a numpy array
        state = self.df.iloc[start_index:end_index].values
        
        # If the window is smaller than window_size, pad with zeros
        if state.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - state.shape[0], self.features))
            state = np.vstack((padding, state))

        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        
        Returns:
            tuple: The initial observation and a dictionary of info.
        """
        super().reset(seed=seed)
        
        # Reset position to a random starting point to prevent memorization
        self.current_step = random.randint(self.window_size, len(self.df) - 2)
        
        # Reset portfolio
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.done = False
        self.history = []

        # Get the initial observation
        observation = self._get_state()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """
        Executes a single step in the environment based on the given action.
        
        Args:
            action (int): The action to take (0=Hold, 1=Buy, 2=Sell).
        
        Returns:
            tuple: A tuple containing the new observation, reward, done flag, truncated flag, and a dictionary of info.
        """
        self.current_step += 1
        
        # Get the current price (we'll use the 'close' price for simplicity)
        current_price = self.df.iloc[self.current_step]['close']
        
        # Get the next price for calculating reward
        next_price = self.df.iloc[self.current_step + 1]['close']

        # Determine the action and update the portfolio
        if action == 1:  # Buy
            if self.balance > 0:
                shares_bought = self.balance / current_price
                self.shares += shares_bought
                self.balance = 0
        elif action == 2:  # Sell
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0

        # Calculate the net worth for this step
        # This is the sum of cash balance and the value of any held shares
        current_net_worth = self.balance + self.shares * current_price
        
        # Calculate the reward
        # The reward is the change in net worth
        reward = current_net_worth - self.net_worth
        
        # Update net worth for the next step's reward calculation
        self.net_worth = current_net_worth
        
        # Update history for rendering
        self.history.append({'net_worth': self.net_worth})
        
        # Check for termination conditions
        # The episode ends if we run out of steps or our net worth drops too low.
        truncated = self.current_step >= len(self.df) - 2
        
        # You can also add a condition to end the episode if the agent is performing poorly
        # self.done = self.net_worth < self.initial_balance / 2
        
        self.done = truncated
        
        # Get the new observation
        observation = self._get_state()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, self.done, truncated, info

    def render(self, mode="human"):
        """
        A simple rendering function to print the current state.
        """
        if mode == "human":
            print(f"Step: {self.current_step} | "
                  f"Balance: {self.balance:.2f} | "
                  f"Shares: {self.shares:.2f} | "
                  f"Net Worth: {self.net_worth:.2f}")


