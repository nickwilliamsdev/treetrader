# --- Example Usage ---
# Define hyperparameters for the model
STATE_DIM = 10  # e.g., Open, High, Low, Close, Volume, and other indicators
ACTION_DIM = 3  # e.g., Buy, Sell, Hold
SEQ_LEN = 60    # Look-back window of 60 timesteps

# Instantiate the model
agent_model = CausalAttentionAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, seq_len=SEQ_LEN)

# Create a dummy batch of historical data
batch_size = 4
dummy_input = torch.randn(batch_size, SEQ_LEN, STATE_DIM)

# Get the action logits from the model
action_logits = agent_model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", action_logits.shape)