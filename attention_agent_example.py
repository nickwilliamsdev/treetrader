from networks.attention_agent import CausalAttentionAgent
import torch
from utils.trading_gym_env import TradingEnv
from utils.synthetic_data_service import SyntheticOHLCVGenerator
import pandas as pd
import matplotlib.pyplot as plt
from diffevo import DDIMScheduler, BayesianGenerator
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils.fitness_funcs import batched_fitness_function
from api_wrappers.kraken_wrapper import KrakenWrapper
kw = KrakenWrapper()
dfs = kw.load_hist_files()
print(dfs)
def add_features(df):
    # Ensure columns are numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # Add log returns
    df['log_return'] = np.log(df['close']).diff()

    # Add moving averages
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()

    # Add Fibonacci levels
    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 1.0]
    for ratio in fib_ratios:
        df[f'fib_{int(ratio * 1000)}'] = np.nan
    for i in range(20, len(df)):
        high = df['high'].iloc[i-20:i].max()
        low = df['low'].iloc[i-20:i].min()
        for ratio in fib_ratios:
            level = high - (high - low) * ratio
            df.at[i, f'fib_{int(ratio * 1000)}'] = level

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    return df

# Add features to all dataframes
dfs = {asset: add_features(df) for asset, df in dfs.items()}
def split_train_test(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df

# Split all dataframes into train/test sets
train_test_data = {asset: split_train_test(df) for asset, df in dfs.items()}
def prepare_data(df, seq_len):
    states = []
    price_changes = []

    for i in range(len(df) - seq_len):
        state = df.iloc[i:i+seq_len][['log_return', 'ma_10', 'ma_50']].values
        price_change = df.iloc[i+seq_len]['log_return']
        states.append(state)
        price_changes.append(price_change)

    states = torch.tensor(states, dtype=torch.float32)  # Shape: (num_samples, seq_len, num_features)
    price_changes = torch.tensor(price_changes, dtype=torch.float32)  # Shape: (num_samples,)
    return states, price_changes

# Prepare data for all assets
seq_len = 50
train_data = {asset: prepare_data(train, seq_len) for asset, (train, _) in train_test_data.items()}
test_data = {asset: prepare_data(test, seq_len) for asset, (_, test) in train_test_data.items()}

def run(x_array, population, agent):
    rewards = []
    # Example of a random walk in the environment
    for xp in population:
        vector_to_parameters(torch.tensor(xp, dtype=torch.float32), agent.parameters())
        rewards.append(batched_fitness_function(agent, x_array))
    return rewards

# Define hyperparameters
POP_SIZE = 100
SCALING = 0.1

# Instantiate the model
STATE_DIM = train_data[list(train_data.keys())[0]][0].shape[-1]  # Number of features
ACTION_DIM = 2  # Buy, Sell
agent_model = CausalAttentionAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, seq_len=seq_len)

# Initialize population
dim = parameters_to_vector(agent_model.parameters()).shape[0]
population = torch.randn(POP_SIZE, dim) * SCALING

# Fitness function
def fitness_function(population, agent, train_data):
    rewards = []
    for params in population:
        vector_to_parameters(params, agent.parameters())
        batch_states = torch.cat([data[0] for data in train_data.values()], dim=0)  # Combine all assets
        batch_price_changes = torch.cat([data[1] for data in train_data.values()], dim=0)
        reward = batched_fitness_function(agent, batch_states, batch_price_changes)
        rewards.append(reward.sum().item())  # Sum rewards across all assets
    return rewards

# Train with diffusion evolution
scheduler = DDIMScheduler(num_step=seq_len)
for step in range(100):  # Number of training steps
    rewards = fitness_function(population, agent_model, train_data)
    print(f"Step {step}, Best Reward: {max(rewards)}")
    population = scheduler.step(population, rewards)