from utils.trading_gym_env import TradingEnv
from utils.synthetic_data_service import SyntheticOHLCVGenerator
import pandas as pd
if __name__ == '__main__':
    dummy_df = SyntheticOHLCVGenerator(n_steps=500, mu=0.001, sigma=0.1, dt=1, seed=42).generate(start=100.0)
    dummy_df = dummy_df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    # The `window_size` here should match the `seq_len` of your PyTorch model
    env = TradingEnv(df=dummy_df, window_size=60)
    
    # Reset the environment
    observation, info = env.reset()
    
    print("Initial observation shape:", observation.shape)
    
    # Store rewards for visualization
    rewards = []

    # Example of a random walk in the environment
    for _ in range(100):
        # A random action (0=Hold, 1=Buy, 2=Sell)
        action = env.action_space.sample()
        
        observation, reward, done, truncated, info = env.step(action)
        
        rewards.append(reward)  # Store the reward
        
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}, Truncated: {truncated}")
        
        if done or truncated:
            print("Episode finished.")
            break

    # Plot the rewards
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(env.history, marker='o', label='Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.show()

