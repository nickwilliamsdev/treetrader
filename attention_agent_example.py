from networks.attention_agent import CausalAttentionAgent
import torch
from utils.trading_gym_env import TradingEnv
from utils.synthetic_data_service import SyntheticOHLCVGenerator
import pandas as pd
import matplotlib.pyplot as plt
from diffevo import DDIMScheduler, BayesianGenerator


if __name__ == '__main__':
    dummy_df = SyntheticOHLCVGenerator(n_steps=500, mu=0.001, sigma=0.1, dt=1, seed=42).generate(start=100.0)
    dummy_df = dummy_df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # --- Example Usage ---
    # Define hyperparameters for the model
    STATE_DIM = len(dummy_df.columns)  # e.g., Open, High, Low, Close, Volume, and other indicators
    ACTION_DIM = 3  # e.g., Buy, Sell, Hold
    SEQ_LEN = 60    # Look-back window of 60 timesteps

    # Instantiate the model
    agent_model = CausalAttentionAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, seq_len=SEQ_LEN)
    
    
    # The `window_size` here should match the `seq_len` of your PyTorch model
    env = TradingEnv(df=dummy_df, window_size=SEQ_LEN)

    # Reset the environment
    observation, info = env.reset()
    
    print("Initial observation shape:", observation.shape)
    
    # Store rewards for visualization
    rewards = []

    # Example of a random walk in the environment
    for _ in range(100):
        # Convert observation to a PyTorch tensor and add batch dimension
        batched_observation = torch.unsqueeze(torch.tensor(observation, dtype=torch.float32), dim=0)

        # Get action from the model
        action = torch.argmax(agent_model(batched_observation), dim=-1)
        observation, reward, done, truncated, info = env.step(action)
        
        rewards.append(reward)  # Store the reward
        
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}")
        
        if done or truncated:
            print("Episode finished.")
            break
    plt.plot(env.history)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.title("Trading Agent Performance")
    plt.show()