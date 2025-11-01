import torch
import torch.nn as nn
import torch.nn.functional as F

class ListNetRanker(nn.Module):
    def __init__(self, n_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, X):
        # X: (batch, list_size, n_features)
        print(X.shape)
        scores = self.net(X)  # (batch, list_size, 1)
        return scores.squeeze(-1)  # (batch, list_size)

