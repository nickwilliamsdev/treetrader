import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. Define the VAR Model Class ---
class VARModel(nn.Module):
    """
    A PyTorch implementation of a Vector Autoregression (VAR) model.

    The model predicts the next step of multiple time series based on
    their own past values and the past values of other series.
    """
    def __init__(self, num_features: int, lag: int):
        """
        Initializes the VARModel.

        Args:
            num_features (int): The number of time series (variables) in the VAR model.
            lag (int): The order of the VAR model, i.e., how many past time steps
                       are used to predict the current step.
        """
        super(VARModel, self).__init__()
        self.num_features = num_features
        self.lag = lag

        # The input to the linear layer will be the concatenated lagged values
        # for all features. For a lag 'p' and 'k' features, the input size
        # will be k * p. The output size is k (predicting the next step for all features).
        self.linear = nn.Linear(num_features * lag, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the VAR model.

        Args:
            x (torch.Tensor): The input tensor containing lagged values.
                              Shape: (batch_size, num_features * lag)

        Returns:
            torch.Tensor: The predicted next values for all features.
                          Shape: (batch_size, num_features)
        """
        return self.linear(x)