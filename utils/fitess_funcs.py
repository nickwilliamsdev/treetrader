import torch
import numpy as np

def batched_fitness_function(agent, states, price_changes):
    """
    Evaluate the agent on multiple assets simultaneously using a batched tensor.

    Args:
        agent: The trading agent (PyTorch model).
        states: A tensor of shape (num_steps, batch_dim, state_dim) representing the state of each asset over time.
        price_changes: A tensor of shape (num_steps, batch_dim) representing percentage price changes for each asset.

    Returns:
        torch.Tensor: A tensor of shape (batch_dim, 1) containing the overall return for each asset.
    """
    num_steps, batch_dim, state_dim = states.shape
    rewards = torch.zeros(batch_dim, 1)  # Initialize rewards for each asset

    for t in range(num_steps - 1):  # Iterate over time steps
        # Get the current state for all assets
        current_states = states[t]  # Shape: (batch_dim, state_dim)

        # Get actions for all assets in the batch
        actions = agent(current_states)  # Shape: (batch_dim, action_dim)
        actions = torch.argmax(actions, dim=-1)  # Convert to discrete actions (0=Hold, 1=Buy, 2=Sell)

        # Convert actions to -1 (Sell), 0 (Hold), 1 (Buy)
        actions = actions.float() - 1  # Map (0, 1, 2) -> (-1, 0, 1)

        # Calculate rewards as the product of actions and percentage changes
        rewards += (actions * price_changes[t + 1]).unsqueeze(-1)  # Shape: (batch_dim, 1)

    return rewards