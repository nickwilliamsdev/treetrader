from massive import RESTClient
import os
api_key = os.environ.get("POLYGON_API_KEY")
if not api_key:
    raise ValueError("POLYGON_API_KEY environment variable not set")

hist_data_dir = "../hist_data/stocks/single/"

client = RESTClient(api_key=api_key)


syms = client.list_tickers(limit=100)
tickers = [sym.ticker for sym in syms]

test_ticker = tickers[0]

ticker_hist = client.get_ticker_details(test_ticker)

