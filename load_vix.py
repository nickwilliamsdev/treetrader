# file: load_vix.py
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path

CSV_PATH = Path("hist_data/kaggle/vix_daily.csv")
TICKER = "^VIX"   # Yahoo ticker for VIX

def load_existing(csv_path: Path):
    if not csv_path.exists():
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols).set_index(pd.DatetimeIndex([]))
    df = pd.read_csv(csv_path, parse_dates=["index"])
    df = df.set_index("index").sort_index()
    return df

def fetch_yahoo(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    end_exclusive = end_date + timedelta(days=1)
    data = yf.download(ticker, start=start_date.isoformat(), end=end_exclusive.isoformat(), progress=False)
    if data.empty:
        return pd.DataFrame()

    # Flatten MultiIndex -> single-level names (take price level)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [lev0 if isinstance(lev0, str) and lev0.strip() != "" else lev1
                        for lev0, lev1 in data.columns]

    # Lowercase column names
    data.columns = [c.lower() for c in data.columns]

    # Ensure datetime index
    data.index = pd.to_datetime(data.index)

    # Keep only standard columns in canonical order (if present)
    cols = [c for c in ["open", "high", "low", "close"] if c in data.columns]
    data = data[cols]
    print(data.head())
    return data.sort_index()

def append_and_save(csv_path: Path, new_df: pd.DataFrame):
    if new_df.empty:
        print("No new rows to add.")
        return
    existing = load_existing(csv_path)
    combined = pd.concat([existing, new_df])
    # dedupe by index (date), keep the newest (last)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.reset_index().to_csv(csv_path, index=False)
    print(f"Appended {len(new_df)} rows; saved {csv_path} (total rows: {len(combined)})")

def main():
    existing = load_existing(CSV_PATH)
    if existing.empty:
        # If CSV exists but had no rows, set last_date to a reasonable default (e.g., 2004-01-01)
        last_date = existing.index.max()
        if pd.isna(last_date):
            last_date = date(2004, 1, 1)
        else:
            last_date = last_date.date()
    else:
        last_date = existing.index.max().date()

    start_date = last_date + timedelta(days=1)
    today = date.today()
    if start_date > today:
        print("File is already up to date.")
        return

    print(f"Fetching {start_date} -> {today} from Yahoo for {TICKER} ...")
    new_data = fetch_yahoo(TICKER, start_date, today)
    if new_data.empty:
        print("No data returned from Yahoo.")
        return

    # Ensure rows are >= start_date (safety)
    new_data = new_data[new_data.index.date >= start_date]
    if new_data.empty:
        print("No new rows after filtering by start date.")
        return

    append_and_save(CSV_PATH, new_data)

if __name__ == "__main__":
    main()