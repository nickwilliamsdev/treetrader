from networks.attention_agent import CausalAttentionAgent
import torch
from utils.trading_gym_env import TradingEnv
from utils.synthetic_data_service import SyntheticOHLCVGenerator
import pandas as pd
import matplotlib.pyplot as plt
from diffevo import DDIMScheduler, BayesianGenerator
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils.fitess_funcs import batched_fitness_function
from api_wrappers.kraken_wrapper import KrakenWrapper

dfs = KrakenWrapper.get_usdt_assets()


def run(x_array, population, agent):
    rewards = []
    # Example of a random walk in the environment
    for xp in population:
        vector_to_parameters(torch.tensor(xp, dtype=torch.float32), agent.parameters())
        rewards.append(batched_fitness_function(agent, x_array))
    return rewards

if __name__ == '__main__':
    dummy_df = SyntheticOHLCVGenerator(n_steps=500, mu=0.001, sigma=0.1, dt=1, seed=42).generate(start=100.0)
    dummy_df = dummy_df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # --- Example Usage ---
    # Define hyperparameters for the model
    STATE_DIM = len(dummy_df.columns)  # e.g., Open, High, Low, Close, Volume, and other indicators
    ACTION_DIM = 3  # e.g., Buy, Sell, Hold
    SEQ_LEN = 400   # Look-back window of 400 timesteps
    POP_SIZE = 100  # Population size for the evolutionary algorithm
    SCALING = .1 # scaling for fitness

    scheduler = DDIMScheduler(num_step=SEQ_LEN)

    # Instantiate the model
    agent_model = CausalAttentionAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, seq_len=SEQ_LEN)
    dim = parameters_to_vector(agent_model.parameters()).shape[0]
    x = torch.randn(POP_SIZE, dim)
    
    for
    # list for training rewards and history
    reward_history = []
    population_history = [x * SCALING]
    x0_population = [x * SCALING]
    observations = []