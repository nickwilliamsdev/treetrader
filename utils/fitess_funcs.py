import torch

def batched_fitness_function(agent, states, price_changes):
    """
    Evaluate the agent on multiple assets simultaneously using a batched tensor.

    Args:
        agent: The trading agent (PyTorch model).
        states: A tensor of shape (batch, ep_len, seq_len, features) representing the state of each asset over time.
        price_changes: A tensor of shape (batch, ep_len) representing percentage price changes for each asset.

    Returns:
        torch.Tensor: A tensor of shape (batch, 1) containing the overall return for each asset.
    """
    batch, ep_len, seq_len, features = states.shape
    rewards = torch.zeros(batch, 1, device=states.device)  # Initialize rewards for each asset

    for t in range(ep_len - 1):  # Iterate over episode timesteps
        # Get the current state for all assets at time t
        current_states = states[:, t, :, :]  # Shape: (batch, seq_len, features)

        # Get actions for all assets in the batch
        actions = agent(current_states)  # Shape: (batch, 2) -> 2 actions: Buy, Sell
        actions = torch.argmax(actions, dim=-1)  # Convert to discrete actions (0=Sell, 1=Buy)

        # Convert actions to -1 (Sell) and 1 (Buy)
        actions = actions.float() * 2 - 1  # Map (0, 1) -> (-1, 1)

        # Calculate rewards as the product of actions and percentage changes
        rewards += (actions * price_changes[:, t + 1]).unsqueeze(-1)  # Shape: (batch, 1)

    return rewards