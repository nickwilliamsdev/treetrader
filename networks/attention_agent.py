import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# A CausalSelfAttention module that applies a causal mask to the attention scores.
# This prevents the model from looking at future data points in a sequence.
class CausalSelfAttention(nn.Module):
    """
    Implements a single-head or multi-head causal self-attention mechanism.
    
    The causal mask is applied to the attention scores to ensure that a token
    at position `t` can only attend to tokens at positions `0` to `t`. This is
    essential for autoregressive tasks like time-series prediction or RL trading,
    where future information is not available.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # Ensure that the embedding dimension is divisible by the number of heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass for the CausalSelfAttention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Project input to query, key, and value vectors
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Calculate attention scores (dot product of queries and keys)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 3. Apply causal mask
        # The mask is a lower triangular matrix with -inf on the upper triangle.
        # This prevents attention to future tokens.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        scores = scores + causal_mask
        
        # 4. Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 5. Calculate the weighted sum of values and concatenate heads
        # context shape: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, v)
        # Reshape back to (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 6. Apply final linear projection
        output = self.out_proj(context)
        return output

# The main RL agent model that uses the causal attention block.
class CausalAttentionAgent(nn.Module):
    """
    A full agent model for RL trading using a causal attention mechanism.
    
    This model takes a history of market states and outputs action logits.
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, num_heads=4, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        
        # Project the input features to the embedding dimension
        self.input_projection = nn.Linear(state_dim, embed_dim)
        
        # A stack of causal attention blocks (a single block for simplicity)
        self.attention_block = CausalSelfAttention(embed_dim, num_heads)
        
        # A position-wise feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), # GELU activation is common in transformers
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # The policy head outputs logits for each possible action
        self.policy_head = nn.Linear(embed_dim, action_dim)

    def forward(self, state_history):
        """
        Forward pass for the CausalAttentionAgent.
        
        Args:
            state_history (torch.Tensor): A tensor of historical states
                                          of shape (batch_size, seq_len, state_dim).
        
        Returns:
            torch.Tensor: Logits for the actions, of shape (batch_size, action_dim).
        """
        # Ensure the input shape is correct
        assert state_history.shape[1] == self.seq_len and state_history.shape[2] == self.state_dim, \
            "Input tensor dimensions do not match the model's sequence length or state dimension."
            
        # Project the input features
        projected_input = self.input_projection(state_history)
        
        # Pass through the attention block
        attention_output = self.attention_block(projected_input)
        
        # Apply a residual connection and layer normalization
        attention_output = F.layer_norm(attention_output + projected_input, attention_output.size()[1:])

        # Pass through the feed-forward network
        ff_output = self.feed_forward(attention_output)
        
        # Apply another residual connection and layer normalization
        ff_output = F.layer_norm(ff_output + attention_output, ff_output.size()[1:])
        
        # Extract the final representation for the most recent timestep.
        # This is the representation from which we will derive our action.
        current_representation = ff_output[:, -1, :]
        
        # Policy output
        action_logits = self.policy_head(current_representation)
        
        return action_logits


