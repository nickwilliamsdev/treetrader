import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
import os
from statistics import mode

class KrakenWrapper(object):
    endpoints = {
        "trade_assets": "https://api.kraken.com/0/public/AssetPairs",
        "ohlc_bars": "https://api.kraken.com/0/public/OHLC?pair={0}&since={1}&interval={2}",
        "ohlc_no_since": "https://api.kraken.com/0/public/OHLC?pair={0}&interval={1}",
        "files_path": "./hist_data/crypto/kraken/"
    }

    bar_data_names = ["date","open","high","low","close","vwap","volume","drop"]

    lookback_intervals = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1hr": 60,
        "4hr": 240,
        "1day": 1440,
        "1week": 10080,
        "2week": 21600
    }

    def __init__(self, key="", secret="", default_lookback = 1000, lb_interval = "1day"):
        self.key = key
        self.secret = secret
        self.look_back = default_lookback
        self.lb_interval = lb_interval
        self.majors = ['BTC', 'ETH', "LTC"]
        return
    
    def update_or_create_hist_files(self, data_dir="./hist_data/crypto/kraken_1day/"):
        """
        Update existing historical files with new data or create new files if they don't exist.
        
        Args:
            data_dir (str): Directory where historical files are stored.
        """
        sym_list = self.get_usdt_assets()
        lb_int = self.lookback_intervals[self.lb_interval]
        lookback_ts = int((datetime.today() - timedelta(self.look_back)).timestamp())

        for sym in sym_list:
            file_path = os.path.join(data_dir, f"{sym}.txt")
            try:
                if os.path.exists(file_path):
                    # Update existing file
                    dataframe = pd.read_csv(file_path)
                    last_ts = dataframe['date'].iloc[-1]
                    hist_req = requests.get(self.endpoints["ohlc_bars"].format(sym, last_ts, lb_int))
                    new_data = hist_req.json()["result"][sym]
                    if new_data:
                        new_df = pd.DataFrame(new_data, columns=self.bar_data_names)
                        new_df["date"] = new_df["date"] * 1000  # Convert timestamp to milliseconds
                        new_df.drop(columns=["drop", "vwap"], inplace=True)
                        dataframe = pd.concat([dataframe, new_df], ignore_index=True)
                        dataframe.to_csv(file_path, index=False)
                        print(f"Updated {sym} with new data.")
                    else:
                        print(f"No new data for {sym}.")
                else:
                    # Create new file
                    hist_req = requests.get(self.endpoints["ohlc_bars"].format(sym, lookback_ts, lb_int))
                    new_data = hist_req.json()["result"][sym]
                    if new_data:
                        new_df = pd.DataFrame(new_data, columns=self.bar_data_names)
                        new_df["date"] = new_df["date"] * 1000  # Convert timestamp to milliseconds
                        new_df.drop(columns=["drop", "vwap"], inplace=True)
                        new_df.to_csv(file_path, index=False)
                        print(f"Created new file for {sym}.")
            except Exception as e:
                print(f"Error processing {sym}: {e}")

    # gets all assets or only assets for a given base pair
    def get_assets(self, base_pair=""):
        tradeable_assets = requests.get(self.endpoints["trade_assets"])
        asset_list = []
        base_len = len(base_pair)
        #df = pd.DataFrame(tradeable_assets.json())
        results = tradeable_assets.json()["result"]
        #print(results)
        for i in results:
            if (base_pair == "" or i[-base_len:] == base_pair):
                asset_list.append(i)
        return asset_list
    
    def get_usdt_assets(self):
        tradeable_assets = requests.get(self.endpoints["trade_assets"])
        asset_list = []
        #df = pd.DataFrame(tradeable_assets.json())
        for i in tradeable_assets.json()["result"]:
            if (i[-4:] == "USDT"):
                asset_list.append(i)
        return asset_list

    def get_file_symbol(self, sym_full):
        stripped = sym_full.split(".")[0][:-3]
        return stripped
    
    def test_get_file_symbol(self):
        df_dict = self.load_hist_files()
        for x in df_dict:
            print(self.get_file_symbol(x))

    def pull_hist(self, syms=None):
        if syms == None:
            syms = self.majors
        for s in syms:
            return

    def pull_kraken_hist_usd_simple(self):
        sym_list = self.get_assets("USD")
        lookback_ts = datetime.today() - timedelta(self.look_back)
        hist_dict = {}
        for i in sym_list:
            hist_req = requests.get(self.endpoints["ohlc_bars"].format(i, lookback_ts, self.lookback_intervals[self.lb_interval]))
            hist_dict[i] = hist_req.json()["result"][i]
        return hist_dict

    def pull_kraken_hist_usd(self):
        sym_list = self.get_usdt_assets()
        lookback_ts = datetime.today() - timedelta(self.look_back)
        hist_dict = {}
        longest = 0
        for i in sym_list:
            hist_req = requests.get(self.endpoints["ohlc_bars"].format(i, lookback_ts, self.lookback_intervals[self.lb_interval]))
            hist_dict[i] = hist_req.json()["result"][i]
            if len(hist_dict[i]) > longest:
                longest = len(hist_dict[i])
        for i in hist_dict:
            if len(hist_dict[i]) == longest:
                with open(f"./hist_data/crypto/kraken_{self.lb_interval}/"+ i + ".txt", "w") as f:
                    f.write("date,open,high,low,close,vwap,vol\n")
                    for l in range(len(hist_dict[i])):
                        next_line = hist_dict[i][l]
                        write_len = len(next_line)-1
                        for x in range(write_len):
                            if(x != write_len-1):
                                f.write(str(next_line[x]) + ",")
                            else:
                                f.write(str(next_line[x]))
                        f.write("\n")

    def pull_kraken_majors_usd(self, lb = "1day"):
        sym_list = [s + "/USD" for s in self.majors]
        lookback_ts = datetime.today() - timedelta(self.look_back * 2)
        lb_int = self.lookback_intervals[lb]
        for i in sym_list:
            hist_req = requests.get(self.endpoints["ohlc_no_since"].format(i, lb_int))
            results = hist_req.json()["result"][i]
            next_since_ts = int(results[0][0]) - (lb_int * self.look_back * 60)
            print(next_since_ts)
            next_hist_req = requests.get(self.endpoints["ohlc_bars"].format(i, next_since_ts, lb_int))
            first_results = next_hist_req.json()["result"][i]
            print(first_results[0][0])
            results = first_results + results
            with open(f"./hist_data/kraken_{lb}/"+ i.split("/")[0] + "USD.txt", "w") as f:
                f.write("time,open,high,low,close,vwap,vol\n")
                for l in range(len(results)):
                    next_line = results[l]
                    for x in range(len(next_line)):
                        if(x != len(next_line)-1):
                            f.write(str(next_line[x]) + ",")
                        else:
                            f.write(str(next_line[x]))
                    f.write("\n")
            print(i, " written")

    def pull_single_sym_hist(self, sym, lb=240):
        lookback_ts = datetime.today() - timedelta(self.look_back) * 500
        hist_req = requests.get(self.endpoints["ohlc_bars"].format(sym, lookback_ts, lb))
        results = hist_req.json()["result"][sym]
        df = pd.DataFrame(results, columns=self.bar_data_names)
        df["date"] = df["date"] * 1000
        df.drop(columns=["drop", "vwap"], inplace=True)
        return df

    def append_new_data(self, data_dir="binance_us", lb=240):
        new_data = False
        for symbol in self.majors:
            kraken_sym = symbol + "/USD"
            symbol = symbol + "USD"
            try:
                dataframe = pd.read_csv(f"./hist_data/{data_dir}/" + symbol + ".txt")
                last_ts = dataframe['date'].iloc[-1]
                df = self.pull_single_sym_hist(kraken_sym, lb)
                print(len(df))
                df = df[df["date"] > last_ts]
                if len(df) > 0:
                    dataframe = dataframe.append(df, ignore_index=True)
                    print("storing new data")
                    dataframe = dataframe.infer_objects()
                    dataframe = dataframe.fillna(method='ffill')
                    dataframe.to_csv(f"./hist_data/{data_dir}/" + symbol + ".txt", index=False)
                    new_data = True
                else:
                    print("no new data for " + symbol)
            except BaseException as err:
                print(f"Unexpected {err}, {type(err)}")
                #raise
        return new_data