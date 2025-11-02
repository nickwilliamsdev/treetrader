import torch
import torch.nn.functional as F
from networks.list_net import ListNetRanker
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from api_wrappers.kraken_wrapper import KrakenWrapper
import os

four_hour_wrapper = KrakenWrapper(lb_interval='4hr')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define feature columns and target column
feature_cols = ['close_pct_change', 'volume_pct_change', 'high_low_diff', 'open_close_diff']
target_col = 'close_pct_change'  # Example target column

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
    if true_returns.dim() == 1:  # Ensure true_returns has the correct shape
        true_returns = true_returns.unsqueeze(1)

    P_true = F.softmax(true_returns / temperature, dim=1)
    P_pred = F.softmax(scores, dim=1)
    loss = -(P_true * torch.log(P_pred + 1e-8)).sum(dim=1).mean()
    return loss

def validate_data(X, y):
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError("Input data contains NaN or Inf values.")
    if torch.isnan(y).any() or torch.isinf(y).any():
        raise ValueError("Target data contains NaN or Inf values.")

def train_listnet(model, dataloader, n_epochs=20, lr=1e-3, save_dir="model_archive/listnet"):
    """
    Trains the ListNet model using the provided dataloader and saves the model every 10 epochs.

    Args:
        model (torch.nn.Module): The ListNet model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        save_dir (str): Directory to save the model checkpoints.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        losses = []
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)  # Move data to the device
            validate_data(X, y)
            opt.zero_grad()
            scores = model(X)
            loss = listnet_loss(scores, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"listnet_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
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

def backtest_tournament(model, dfs, feature_cols, target_col, start_date, end_date, group_size=10, top_k=2, initial_cash=10000):
    """
    Backtests the model using tournament ranking.

    Args:
        model (torch.nn.Module): The trained ListNet model.
        dfs (dict): Dictionary of DataFrames for each cryptocurrency.
        feature_cols (list): List of feature column names.
        target_col (str): The target column name.
        start_date (str): Start date for the backtest (e.g., "2023-01-01").
        end_date (str): End date for the backtest (e.g., "2023-12-31").
        group_size (int): Number of assets per group in the tournament.
        top_k (int): Number of top assets to select from each group.
        initial_cash (float): Initial cash for the portfolio.

    Returns:
        pd.DataFrame: DataFrame containing portfolio value over time.
    """
    model.eval()

    # Combine all data into a single DataFrame
    all_data = pd.concat(dfs.values(), keys=dfs.keys(), names=["asset", "index"]).reset_index()
    all_data = all_data[(all_data['date'] >= start_date) & (all_data['date'] <= end_date)]

    # Initialize portfolio
    portfolio_value = initial_cash
    portfolio = {}  # {asset: number of shares}
    portfolio_history = []

    # Iterate over each date in the backtest period
    for date in sorted(all_data['date'].unique()):
        # Filter data for the current date
        daily_data = all_data[all_data['date'] == date]

        # Rank assets using tournament ranking
        ranked_assets = tournament_rank(model, daily_data, date, group_size, top_k, feature_cols)

        # Sell all current holdings
        for asset, shares in portfolio.items():
            if asset in daily_data['asset'].values:
                price = daily_data[daily_data['asset'] == asset]['close'].values[0]
                portfolio_value += shares * price
        portfolio = {}

        # Buy top-ranked assets
        cash_per_asset = portfolio_value / top_k
        for asset in ranked_assets['asset'].head(top_k):
            price = daily_data[daily_data['asset'] == asset]['close'].values[0]
            shares = cash_per_asset // price
            portfolio[asset] = shares
            portfolio_value -= shares * price

        # Record portfolio value
        portfolio_history.append({'date': date, 'portfolio_value': portfolio_value})

    # Convert portfolio history to DataFrame
    portfolio_history = pd.DataFrame(portfolio_history)

    # Add the value of current holdings to the portfolio value
    for asset, shares in portfolio.items():
        if asset in daily_data['asset'].values:
            price = daily_data[daily_data['asset'] == asset]['close'].values[0]
            portfolio_history.loc[portfolio_history.index[-1], 'portfolio_value'] += shares * price

    return portfolio_history

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
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)  # Move data to the device
            scores = model(X)
            loss = listnet_loss(scores, y)
            losses.append(loss.item())
            print(f"Validation Batch {batch_idx}: Loss = {loss.item():.4f}")
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
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
    def __init__(self, data, feature_cols, target_col, list_size=10):
        """
        Args:
            data (pd.DataFrame): The input DataFrame.
            feature_cols (list): List of feature column names.
            target_col (str): The target column name.
            list_size (int): Number of items in each list (group).
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.list_size = list_size

        # Group data into lists
        self.groups = [
            data.iloc[i:i + list_size]
            for i in range(0, len(data) - list_size + 1, list_size)
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        X = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)  # (list_size, n_features)
        y = torch.tensor(group[self.target_col].values, dtype=torch.float32)  # (list_size,)
        return X, y

def prepare_data(dfs, feature_cols, target_col, train_split=0.8, list_size=10):
    """
    Prepares training and validation datasets from the given DataFrames.

    Args:
        dfs (dict): Dictionary of DataFrames for each cryptocurrency.
        feature_cols (list): List of feature column names.
        target_col (str): The target column name.
        train_split (float): Proportion of data to use for training.
        list_size (int): Number of items in each list (group).

    Returns:
        tuple: Training and validation DataLoaders.
    """
    all_data = pd.concat(dfs.values(), ignore_index=True)
    train_size = int(len(all_data) * train_split)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    train_dataset = CryptoDataset(train_data, feature_cols, target_col, list_size=list_size)
    val_dataset = CryptoDataset(val_data, feature_cols, target_col, list_size=list_size)

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

    # Step 2: Prepare training and validation data
    train_loader, val_loader = prepare_data(dfs, feature_cols, target_col)

    # Step 3: Initialize the ListNet model
    input_dim = len(feature_cols)
    model = ListNetRanker(n_features=input_dim).to(device)

    # Step 4: Train the model
    print("Starting training...")
    train_listnet(model, train_loader, n_epochs=2000, lr=1e-3)

    # Step 5: Validate the model
    print("Validating the model...")
    validate_listnet(model, val_loader, feature_cols)

    # Step 6: Save the trained model
    torch.save(model.state_dict(), "listnet_model.pth")
    print("Model saved to 'listnet_model.pth'.")

def test():
    model_path = "listnet_model.pth"
    input_dim = len(feature_cols)
    model = ListNetRanker(n_features=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Step 2: Preprocess the data\
    dfs = get_dfs()
    for crypto, df in dfs.items():
        df['date'] = pd.to_datetime(df['date'])
        dfs[crypto] = apply_features(df)

    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-12-31")
    
    print("Starting backtest...")
    portfolio_history = backtest_tournament(
        model, dfs, feature_cols, target_col,
        start_date=start_date, end_date=end_date,
        group_size=10, top_k=2, initial_cash=10000
    )
    print(portfolio_history)

    # Plot portfolio value over time
    portfolio_history.set_index('date')['portfolio_value'].plot(title="Portfolio Value Over Time")


if __name__ == "__main__":
    test()
    #main()