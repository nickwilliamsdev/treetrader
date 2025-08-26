from torch.utils.data import Dataset, DataLoader


# Custom PyTorch Dataset for time-series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, target_idx, future_days=5):
        self.data = data
        self.seq_len = seq_len
        self.target_idx = target_idx # Index of the close price feature
        self.future_days = future_days

    def __len__(self):
        # The length is now reduced by the sequence length and the prediction horizon.
        return len(self.data) - self.seq_len - self.future_days + 1

    def __getitem__(self, idx):
        # Get the input sequence of data of length `seq_len`
        # `x` is the input sequence (all features for `seq_len` days)
        x = self.data.iloc[idx:idx + self.seq_len].values

        # `y` is now the percentage change in the close price over the next `future_days`
        current_close = self.data.iloc[idx + self.seq_len - 1, self.target_idx]
        future_close = self.data.iloc[idx + self.seq_len + self.future_days - 1, self.target_idx]

        # Calculate percentage change: ((future - current) / current) * 100
        y = ((future_close - current_close) / current_close) * 100

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)