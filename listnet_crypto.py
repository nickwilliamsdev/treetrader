import torch
import torch.nn.functional as F
from .networks.list_net import ListNetRanker
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from api_wrappers.kraken_wrapper import KrakenWrapper

four_hour_wrapper = KrakenWrapper(interval='4hr')

def listnet_loss(scores, true_returns, temperature=0.01):
    """
    Computes the ListNet loss between predicted scores and true returns.

    Args:
        scores (torch.Tensor): Predicted scores, shape (batch, list_size).
        true_returns (torch.Tensor): True returns, shape (batch, list_size).
        temperature (float): Temperature for softmax normalization.

    Returns:
        torch.Tensor: The computed loss.
    """
    P_true = F.softmax(true_returns / temperature, dim=1)
    P_pred = F.softmax(scores, dim=1)
    loss = -(P_true * torch.log(P_pred + 1e-8)).sum(dim=1).mean()
    return loss


def train_listnet(model, dataloader, n_epochs=20, lr=1e-3):
    """
    Trains the ListNet model using the provided dataloader.

    Args:
        model (torch.nn.Module): The ListNet model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        None
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        losses = []
        for X, y in dataloader:  # X: (batch, list_size, n_features)
            opt.zero_grad()
            scores = model(X)
            loss = listnet_loss(scores, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch}: mean loss = {np.mean(losses):.4f}")


def rank_assets(model, features_df, date, feature_cols):
    """
    Ranks assets based on their predicted scores for a given date.

    Args:
        model (torch.nn.Module): The trained ListNet model.
        features_df (pd.DataFrame): DataFrame containing asset features.
        date (str): The date for which to rank assets.
        feature_cols (list): List of feature column names.

    Returns:
        pd.DataFrame: DataFrame of assets ranked by score.
    """
    if 'date' not in features_df.columns:
        raise ValueError("The features_df must contain a 'date' column.")

    subset = features_df[features_df.date == date]
    if subset.empty:
        raise ValueError(f"No data found for the specified date: {date}")

    X = torch.tensor(subset[feature_cols].values, dtype=torch.float32).unsqueeze(0)
    scores = model(X).detach().numpy()[0]
    subset['score'] = scores
    return subset.sort_values('score', ascending=False)


def tournament_rank(model, features_df, date, group_size=10, top_k=2, feature_cols=None):
    """
    Performs tournament-style ranking of assets.

    Args:
        model (torch.nn.Module): The trained ListNet model.
        features_df (pd.DataFrame): DataFrame containing asset features.
        date (str): The date for which to rank assets.
        group_size (int): Number of assets per group in the tournament.
        top_k (int): Number of top assets to select from each group.
        feature_cols (list): List of feature column names.

    Returns:
        pd.DataFrame: DataFrame of assets ranked by score.
    """
    if 'date' not in features_df.columns:
        raise ValueError("The features_df must contain a 'date' column.")

    current = features_df[features_df.date == date]
    if current.empty:
        raise ValueError(f"No data found for the specified date: {date}")

    while len(current) > group_size:
        groups = [current.iloc[i:i + group_size] for i in range(0, len(current), group_size)]
        winners = []
        for g in groups:
            ranked = rank_assets(model, g, date, feature_cols)
            winners.append(ranked.head(top_k))
        current = pd.concat(winners)
    final_rank = rank_assets(model, current, date, feature_cols)
    return final_rank


def validate_listnet(model, dataloader, feature_cols):
    """
    Validates the ListNet model on a validation dataset.

    Args:
        model (torch.nn.Module): The trained ListNet model.
        dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        feature_cols (list): List of feature column names.

    Returns:
        float: The mean validation loss.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            scores = model(X)
            loss = listnet_loss(scores, y)
            losses.append(loss.item())
    mean_loss = np.mean(losses)
    print(f"Validation mean loss: {mean_loss:.4f}")
    model.train()
    return mean_loss


def apply_features(df):
    """
    Applies feature engineering to the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with features added.
    """
    # Stub implementation: Replace this with actual feature engineering logic
    df['close_pct_change'] = df['close'].pct_change().fillna(0)
    df['volume_pct_change'] = df['vol'].pct_change().fillna(0)
    df['high_low_diff'] = df['high'] - df['low']
    df['open_close_diff'] = df['open'] - df['close']
    return df

def apply_target(df):
    """
    Applies target variable engineering to the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with target variable added.
    """
    df['target'] = df['close'].pct_change(5).shift(-5).fillna(0)  # Example: 5-day future return
    return df

class CryptoDataset(Dataset):
    """
    Custom PyTorch Dataset for cryptocurrency data.
    """
    def __init__(self, data, feature_cols, target_col):
        """
        Args:
            data (pd.DataFrame): The input DataFrame.
            feature_cols (list): List of feature column names.
            target_col (str): The target column name.
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        X = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(row[self.target_col], dtype=torch.float32)
        return X, y


def prepare_data(dfs, feature_cols, target_col, train_split=0.8):
    """
    Prepares training and validation datasets from the given DataFrames.

    Args:
        dfs (dict): Dictionary of DataFrames for each cryptocurrency.
        feature_cols (list): List of feature column names.
        target_col (str): The target column name.
        train_split (float): Proportion of data to use for training.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    all_data = pd.concat(dfs.values(), ignore_index=True)
    train_size = int(len(all_data) * train_split)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    train_dataset = CryptoDataset(train_data, feature_cols, target_col)
    val_dataset = CryptoDataset(val_data, feature_cols, target_col)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

def get_dfs():
    return four_hour_wrapper.load_hist_files()

def main():
    """
    Main function to train and validate the ListNet model.
    """
    # Step 1: Fetch and preprocess data
    dfs = get_dfs()
    for crypto, df in dfs.items():
        dfs[crypto] = apply_features(df)

    # Define feature columns and target column
    feature_cols = ['close_pct_change', 'volume_pct_change', 'high_low_diff', 'open_close_diff']
    target_col = 'close_pct_change'  # Example target column

    # Step 2: Prepare training and validation data
    train_loader, val_loader = prepare_data(dfs, feature_cols, target_col)

    # Step 3: Initialize the ListNet model
    input_dim = len(feature_cols)
    model = ListNetRanker(input_dim=input_dim)

    # Step 4: Train the model
    print("Starting training...")
    train_listnet(model, train_loader, n_epochs=20, lr=1e-3)

    # Step 5: Validate the model
    print("Validating the model...")
    validate_listnet(model, val_loader, feature_cols)

    # Step 6: Save the trained model
    torch.save(model.state_dict(), "listnet_model.pth")
    print("Model saved to 'listnet_model.pth'.")


if __name__ == "__main__":
    main()