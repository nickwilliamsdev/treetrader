import torch
import torch.nn as nn
import torch.nn.functional as F

class ListNetRanker(nn.Module):
    def __init__(self, n_features, hidden=9044):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, X):
        # X: (batch, list_size, n_features)
        scores = self.net(X)  # (batch, list_size, 1)
        return scores.squeeze(-1)  # (batch, list_size)

class AttentionListNetRanker(nn.Module):
    def __init__(self, n_features, hidden=128, n_heads=4, dropout=0.1):
        """
        Attention-based ListNet Ranker.

        Args:
            n_features (int): Number of input features.
            hidden (int): Number of hidden units in the feedforward layers.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.n_heads = n_heads

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=n_features, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Feedforward layers
        self.ffn = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)  # Output a single score per item
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        Forward pass of the AttentionListNetRanker.

        Args:
            X (torch.Tensor): Input tensor of shape (batch, list_size, n_features).

        Returns:
            torch.Tensor: Output scores of shape (batch, list_size).
        """
        # Apply multi-head attention
        attn_output, _ = self.attention(X, X, X)  # Self-attention
        attn_output = self.dropout(attn_output)

        # Pass through feedforward layers
        scores = self.ffn(attn_output)  # (batch, list_size, 1)
        return scores.squeeze(-1)  # (batch, list_size)