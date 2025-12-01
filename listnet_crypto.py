import torch
import torch.nn.functional as F
from networks.list_net import ListNetRanker
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from api_wrappers.kraken_wrapper import KrakenWrapper
import os
import matplotlib.pyplot as plt
from scipy.stats import mode
from utils.feature_functions import bar_change_features, apply_slope_features, apply_green_red

four_hour_wrapper = KrakenWrapper(lb_interval='1week')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define feature columns and target column
target_col = 'close_pct_change'  # Example target column

def listnet_loss(scores, true_returns, temperature=0.1):
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

def join_dataframes_on_date(dfs):
    """
    Joins all DataFrames in the dictionary on the 'date' column.

    Args:
        dfs (dict): Dictionary of DataFrames for each cryptocurrency.

    Returns:
        pd.DataFrame: A single DataFrame with all assets joined on the 'date' column.
    """
    joined_df = None
    for asset, df in dfs.items():
        df_join = df.copy()
        df_join['asset'] = asset  # Add an 'asset' column to identify the asset
        if joined_df is None:
            joined_df = df_join
        else:
            #joined_df = pd.merge(joined_df, df_join, on='date', how='outer', suffixes=('', f'_{asset}'))
            joined_df = pd.concat([joined_df, df_join], ignore_index=True)
    print(joined_df.head())
    return joined_df

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
    checkpoint_counter = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            if checkpoint_counter == 10:
                checkpoint_counter = 0
            checkpoint_path = os.path.join(save_dir, f"listnet.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
            checkpoint_counter += 1
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

    X = torch.tensor(subset[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(device)  # (1, list_size, n_features)
    scores = model(X).detach().cpu().numpy().squeeze(0)  # (list_size,)
    subset['score'] = scores
    return subset.sort_values('score', ascending=False)


def tournament_rank(model, features_df, date, group_size=10, feature_cols=None):
    """
    Simplified tournament-style ranking of assets.

    Args:
        model (torch.nn.Module): The trained ListNet model.
        features_df (pd.DataFrame): DataFrame containing asset features.
        date (str): The date for which to rank assets.
        group_size (int): Number of assets per group in the tournament.
        feature_cols (list): List of feature column names.

    Returns:
        pd.DataFrame: DataFrame containing the final ranking of assets.
    """
    if 'date' not in features_df.columns:
        raise ValueError("The features_df must contain a 'date' column.")

    # Filter data for the given date
    current = features_df[features_df.date == date]
    if current.empty:
        raise ValueError(f"No data found for the specified date: {date}")

    # Debugging: Print initial assets
    print(f"Tournament ranking for date: {date}")
    print("Initial assets:")
    print(current[['asset', 'score']])

    # Tournament ranking logic
    while len(current) > 1:
        # Split into groups of `group_size`
        groups = [current.iloc[i:i + group_size] for i in range(0, len(current), group_size)]
        winners = []

        # Rank each group and take the top-ranked asset
        for group in groups:
            ranked = rank_assets(model, group, date, feature_cols)
            winners.append(ranked.iloc[0])  # Take the top-ranked asset

        # Create a new DataFrame with the winners
        current = pd.DataFrame(winners)

        # Debugging: Print winners after each round
        print("Winners after this round:")
        print(current[['asset', 'score']])

    # Return the final winner
    return current

def backtest_tournament_fixed_steps(model, joined_df, feature_cols, target_col, steps=30, group_size=10, top_k=2, initial_cash=10000):
    """
    Backtests the model using tournament ranking over a fixed number of steps.

    Args:
        model (torch.nn.Module): The trained ListNet model.
        joined_df (pd.DataFrame): The joined DataFrame containing all assets.
        feature_cols (list): List of feature column names.
        target_col (str): The target column name.
        steps (int): Number of steps to backtest.
        group_size (int): Number of assets per group in the tournament.
        top_k (int): Number of top assets to select from each group.
        initial_cash (float): Initial cash for the portfolio.

    Returns:
        pd.DataFrame: DataFrame containing portfolio value over time.
    """
    model.eval()

    # Ensure the DataFrame is sorted and aligned
    joined_df = joined_df.sort_values('date').dropna()
    print(joined_df.head())
    # Get the first `steps` unique dates
    unique_dates = joined_df['date'].unique()[:steps]

    # Initialize portfolio
    portfolio_value = initial_cash
    portfolio = {}  # {asset: number of shares}
    portfolio_history = []
    ranked_assets = []

    # Iterate over the selected dates
    for ix, date in enumerate(unique_dates):
        if ix == 0:
            # Filter data for the current date
            daily_data = joined_df[joined_df['date'] == date]
            # Rank assets using tournament ranking
            ranked_assets = rank_assets(model, daily_data, date, feature_cols)
            continue
        # Sell all current holdings
        for asset, shares in portfolio.items():
            matching_rows = daily_data[daily_data['asset'] == asset]
            if not matching_rows.empty:
                price = matching_rows['close'].iloc[0]
                portfolio_value += shares * price
        portfolio = {}

        # Buy top-ranked assets
        cash_per_asset = portfolio_value / min(top_k, len(ranked_assets))
        for asset in ranked_assets['asset'].head(top_k):
            print(f"Buying asset: {asset} on date: {date}")
            matching_rows = daily_data[daily_data['asset'] == asset]
            if not matching_rows.empty:
                price = matching_rows['close'].iloc[0]
                shares = cash_per_asset / price
                portfolio[asset] = shares
        portfolio_history.append({'date': date, 'portfolio_value': portfolio_value})
        portfolio_value = 0
        print(f"Backtesting for date: {date}")
        # Filter data for the current date
        daily_data = joined_df[joined_df['date'] == date]
        # Rank assets using tournament ranking
        ranked_assets = rank_assets(model, daily_data, date, feature_cols)

    # Convert portfolio history to DataFrame
    portfolio_history = pd.DataFrame(portfolio_history)

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
    #df = apply_slope_features(df, columns=['close', 'vol'], lookbacks=[5, 8, 13, 21, 34, 55, 89], dropna=False)
    #df = apply_green_red(df, dropna=False)
    df = bar_change_features(df, lookbacks=[2, 3, 5, 8, 13, 21, 34, 55, 89])
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
    df['target'] = df['close'].pct_change().shift(-1).fillna(0)
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

def prepare_data(dfs, feature_cols, target_col, train_split=0.5, list_size=21):
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
    print(all_data.head())
    train_size = int(len(all_data) * train_split)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    train_dataset = CryptoDataset(train_data, feature_cols, target_col, list_size=list_size)
    val_dataset = CryptoDataset(val_data, feature_cols, target_col, list_size=list_size)

    train_loader = DataLoader(train_dataset, batch_size=89, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=89, shuffle=False)

    return train_loader, val_loader

def get_dfs():
    """
    Fetches and filters DataFrames to ensure all have the same length.

    Returns:
        dict: Filtered dictionary of DataFrames with lengths equal to the mode length.
    """
    dfs = four_hour_wrapper.load_hist_files()

    # Step 1: Calculate the length of each DataFrame
    df_lengths = {crypto: len(df) for crypto, df in dfs.items()}
    print("DataFrame lengths:", df_lengths)

    # Step 2: Find the mode of the lengths
    mode_length = pd.Series(list(df_lengths.values())).mode()[0]
    print("Mode length:", mode_length)

    # Step 3: Filter DataFrames by length
    filtered_dfs = {crypto: df for crypto, df in dfs.items() if len(df) == mode_length}
    print(f"Filtered {len(dfs) - len(filtered_dfs)} DataFrames that do not match the mode length.")
    non_feature_cols = list(dfs.items())[0][1].columns
    return filtered_dfs, non_feature_cols.tolist() 

def main():
    """
    Main function to train and validate the ListNet model.
    """
    # Step 1: Fetch and preprocess data
    feature_cols = []
    dfs, non_feature_cols = get_dfs()
    for crypto, df in dfs.items():
        df['vol'] = df['volume']
        del df['volume']
        #df['date'] = pd.to_datetime(df['date'] / 1000, unit='s')  # Convert timestamps to datetime
        dfs[crypto] = apply_features(df)
        if not feature_cols:
            feature_cols = [col for col in dfs[crypto].columns if col not in non_feature_cols + ['date']]

    # Step 2: Prepare training and validation data
    train_loader, val_loader = prepare_data(dfs, feature_cols, target_col, list_size=5)

    # Step 3: Initialize the ListNet model
    input_dim = len(feature_cols)
    model = ListNetRanker(n_features=input_dim, hidden=1024).to(device)

    # Step 4: Train the model
    print("Starting training...")
    train_listnet(model, train_loader, n_epochs=9044, lr=.0001)

    # Step 5: Validate the model
    print("Validating the model...")
    validate_listnet(model, val_loader, feature_cols)

    # Step 6: Save the trained model
    torch.save(model.state_dict(), "listnet_model.pth")
    print("Model saved to 'listnet_model.pth'.")

def test():
    # Step 2: Preprocess the data
    feature_cols = []
    dfs, non_feature_cols = get_dfs()
    for crypto, df in dfs.items():
        df['vol'] = df['volume']
        del df['volume']
        df['date'] = pd.to_datetime(df['date'] / 1000, unit='s')  # Convert timestamps to datetime
        dfs[crypto] = apply_features(df)
        if not feature_cols:
            feature_cols = [col for col in dfs[crypto].columns if col not in non_feature_cols + ['date']]
    model_path = "./model_archive/listnet/listnet.pth"
    input_dim = len(feature_cols)
    model = ListNetRanker(n_features=input_dim, hidden=1024).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    # Join all DataFrames on the 'date' column
    joined_df = join_dataframes_on_date(dfs)
    print(joined_df.columns)
    print("Starting backtest...")
    portfolio_history = backtest_tournament_fixed_steps(
        model, joined_df, feature_cols, target_col,
        steps=300, group_size=50, top_k=1, initial_cash=10000
    )
    print(portfolio_history)

    # Plot portfolio value over time
    portfolio_history.set_index('date')['portfolio_value'].plot(title="Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.show()


if __name__ == "__main__":
    main()
    test()