import pandas as pd
import numpy as np
import ta
#useful for creating categorical features
return_ranges = [
    (-1.0, -.5),
    (-.5, -.2),
    (-.2,-.1),
    (-.1,-.05),
    (-.05, -.02),
    (-.02, 0.0),
    (0.0, .02)
    ]

# useful for lookbacks, timeframes of any sort
fib_intervals = [2, 3, 5, 8, 13, 21, 34, 55, 89]

def apply_zscore_signal(df, window=13, z_threshold=1.0):
    """
    Apply feature engineering and generate trading signals.
    """
    # Calculate rolling mean and Z-Sco21
    df['RollingMean'] = df['close'].rolling(window=window).mean()
    df['ZScore'] = (df['close'] - df['RollingMean']) / df['close'].rolling(window=window).std()

    # Generate buy/sell signals
    df['Signal'] = 0
    df.loc[df['ZScore'] < -z_threshold, 'Signal'] = 1  # Buy
    df.loc[df['ZScore'] > z_threshold, 'Signal'] = -1  # Sell

    return df

def ensemble_features(df):
    df['ma_short'] = df['close'].rolling(8).mean()
    df['ma_long'] = df['close'].rolling(34).mean()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=13).rsi()
    df['Signal'] = (
        (df['close'].pct_change(5) > 0).astype(int) +
        (df['ma_short'] > df['ma_long']).astype(int) +
        (df['rsi'] < 30).astype(int)
    )
    # Buy if 2 or more signals agree
    df['Signal'] = df['Signal'].apply(lambda x: 1 if x >= 2 else -1 if x <= -2 else 0)
    return df

def cum_norm(df, column='close', drop_min_max=True):
    df['maxes'] = df[column].cummax()
    df['mins'] = df[column].cummin()
    df[f'norm_{column}'] = (df[column] - df['mins']) / (df['maxes'] - df['mins'])
    if drop_min_max:
        df = df.drop(columns=['maxes', 'mins'])
    return df

def percent_change_features(df:pd.DataFrame, columns = None, lookbacks=[1], dropna=True):
    if columns == None:
        columns = df.columns
    for c in columns:
        for l in lookbacks:
            if c != "Date":
                df[f"change_{c}_{l}"] = df[c].pct_change().rolling(l).mean()
    if dropna:
        df.dropna(inplace=True)
    return df

def apply_slope_features(df, columns, lookbacks=fib_intervals, dropna=True):
    # adds the arctan2 values for slopes over fractal lookbacks
    for l in lookbacks:
        for c in columns:
            if c != "Date":
                lookback = df[c].diff(l)
                df[c +"_angle_" + str(l)] = np.arctan2(lookback, l)
    if dropna:
        df.dropna(inplace=True)
    return df

def apply_candlestick_features(df):
	df['high_low'] = (df['high'] - df['low']) / df['low']
	df['open_high'] = (df['high'] - df['open']) / df['open']
	df['close_high'] = (df['high'] - df['close']) / df['close']
	df['open_low'] = (df['open'] - df['low']) / df['open']
	df['close_low'] = (df['close'] - df['low']) / df['close']
	df['open_close'] = ((df['close'] - df['open']) / df['open']) + 1 #make sure all features are positive
	return df

# applies a new column with 1 for green (positive change) or 0 for if its red (negative change)
# we will dropna no matter what
def apply_green_red(df, column="close", pct_change_interval=1, dropna=True):
    df["close_change"] = df[column].pct_change(pct_change_interval)
    if dropna:
        df.dropna(inplace=True) 
    df["green"] = [1 if x > 0.0 else 0 for x in df["close_change"]]
    return df

def rolling_column_change_direction_ratio(df, lookbacks, columns=["close"], pct_change_intervals=[1], keep_change_col=True, keep_green_col=True):
    for p in pct_change_intervals:
        for c in columns:
            df[f"{c}_change_{p}"] = df[c].pct_change(p)
            df[f"{c}_up_{p}"] = [1 if x > 0.0 else 0 for x in df[f"{c}_up_change_{p}"]]
            for lb in lookbacks:
                df[f"{c}_up_ratio_{p}_{lb}"] = df[f"{c}_up_{p}"].rolling(lb).mean()
    df.dropna(inplace=True)
    return df

def rolling_oc_up_down_ratio(df, lookbacks=[], keep_direction_col=True):
    change = df["open"] - df["close"]
    df["direction"] = [1 if x > 0.0 else 0 for x in change]
    for lb in lookbacks:
        df[f"up_v_down_{lb}"] = df["direction"].rolling(lb).mean()
    df.dropna(inplace=True)
    if not keep_direction_col:
        del df["direction"]
    return df

def range_diff(df, lookbacks):
	for l in lookbacks:
		min = df['open'].rolling(l).min()
		max = df['open'].rolling(l).max()
		df[f"diff_min_{l}"] = (df['open'] - min) / df['open']
		df[f"diff_max{l}"] = (max - df['open']) / df['open']
	df.dropna(inplace=True)
	return df		
'''
def apply_change_features(df, lookbacks):
	columns = df.columns
	for c in columns:
		for l in lookbacks:
			if c != "date":
				df[f"change_{c}_{l}"] = df[c].pct_change().rolling(l).mean()
	df.dropna(inplace=True)
	return df
'''     
def apply_features_newest(df):
    print(df.head())
    df['close_change'] = df['Close'].pct_change()
    df['change_shifted'] = df['close_change'].shift(-1)
    df['close_cor'] = df['close_change'].rolling(21).corr(df['change_shifted'])
    df['high_close'] = (df['High'] - df['Close']) / df['Close']
    df['hc_cor'] = df['high_close'].rolling(21).corr(df['change_shifted'])
    df['low_close'] = (df['Close'] - df['Low']) / df['Close']
    df['lc_cor'] = df['low_close'].rolling(21).corr(df['change_shifted'])
    df['high_open'] = (df['High'] - df['Open']) / df['Open']
    df['ho_cor'] = df['high_open'].rolling(21).corr(df['change_shifted'])
    df['low_open'] = (df['Open'] - df['Low']) / df['Open']
    df['lo_cor'] = df['low_open'].rolling(21).corr(df['change_shifted'])
    df['std_volume'] = (df['Volume'] - df['Volume'].rolling(8).mean()) / df['Volume']
    df['vol_cor'] = df['std_volume'].rolling(21).corr(df['change_shifted'])
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

def bar_change_features(feature_df, lookbacks):
    #feature_df = pd.DataFrame()
    feature_df['open_close'] = (feature_df['Close'] - feature_df['Open']) / feature_df['Open']
    feature_df['high_low'] = (feature_df['High'] - feature_df['Low']) / feature_df['Low']
    feature_df['vol_change'] = feature_df["Volume"].pct_change()
    feature_df["close_change"] = feature_df["Close"].pct_change()
    feature_df["high_change"] = feature_df["High"].pct_change()
    feature_df["low_change"] = feature_df["Low"].pct_change()
    feature_df["open_change"] = feature_df["Open"].pct_change()
    for x in lookbacks:
        feature_df[f"oc_{x}"] = feature_df["open_close"].rolling(x).mean()
        feature_df[f"hl_{x}"] = feature_df["high_low"].rolling(x).mean()
        feature_df[f"vc_{x}"] = feature_df["vol_change"].rolling(x).mean()
        feature_df[f"cc_{x}"] = feature_df["close_change"].rolling(x).mean()
        feature_df[f"lc_{x}"] = feature_df["low_change"].rolling(x).mean()
        feature_df[f"hc_{x}"] = feature_df["high_change"].rolling(x).mean()
        feature_df[f"opc_{x}"] = feature_df["open_change"].rolling(x).mean()
    feature_columns = [c for c in feature_df.columns if c not in self.bardata_points and c not in ['level_0', 'index', 'Unnamed: 0']]
    #feature_df[feature_columns] = (feature_df[feature_columns] - feature_df[feature_columns].rolling(89).mean()) / feature_df[feature_columns].rolling(89).std()
    feature_df.dropna(inplace=True)
    feature_df.reset_index(inplace=True)
    print(len(feature_df.columns))
    return feature_df

def apply_roc_features(df):
    columns = df.columns
    for c in columns:
        if c not in ['Unnamed: 0', 'Date', 'Volume']:
            df["roc_" + c] = df[c].pct_change()
            df["roc_" + c] = (df["roc_" + c] - df["roc_" + c].rolling(55).mean()) / df["roc_" + c].rolling(55).std()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

def apply_features_simple(df, lookbacks):
    print(df.head())
    #df['high_close'] = (df['High'] - df['Close']) / df['Close']
    #df['low_close'] = (df['Close'] - df['Low']) / df['Close']
    #df['high_open'] = (df['High'] - df['Open']) / df['Open']
    #df['low_open'] = (df['Open'] - df['Low']) / df['Open']
    df['high_low'] = (df['High'] - df['Low']) / df['Low']
    df['open_close'] = (df['Close'] - df['Open']) / df['Open']
    df['std_volume'] = (df['Volume'] - df['Volume'].rolling(55).mean()) / df['Volume'].rolling(55).mean()
    for l in lookbacks:
        df['hl_' + str(l)] = df['high_low'].rolling(l).mean()
        df['oc_' + str(l)] = df['open_close'].rolling(1).mean()
        df['hl_std_' + str(l)] = df['high_low'].rolling(l).std()
        df['oc_std_' + str(l)] = df['open_close'].rolling(l).std()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

def apply_open_normd(df):
    df['open_normed'] = (df['Open'] - df['Open'].rolling(89).min()) / df['Open'].rolling(89).max()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df
'''    
def apply_slope_features(df, lookbacks):
    # adds the arctan2 values for slopes over fractal lookbacks
    columns = df.columns
    for l in lookbacks:
        for c in columns:
            if c not in ['Unnamed: 0', 'Date', 'Volume']:
                print(c)
                lookback = df[c].diff(l)
                df[c + "_angle_" + str(l)] = np.arctan2(lookback, l)
                df[c + "_angle_" + str(l)] = (df[c +"_angle_" + str(l)] - df[c +"_angle_" + str(l)].rolling(55).min())/(df[c +"_angle_" + str(l)].rolling(55).max() - df[c +"_angle_" + str(l)].rolling(55).min())
                #df[c+"_angle_cor_" + str(l)] = df[c+"_angle_" + str(l)].rolling(8).corr(df["Open"].pct_change())
    #feature_columns = [c for c in df.columns if c not in self.bardata_points and c not in ['level_0', 'index', 'Unnamed: 0']]
    #df[feature_columns] = (df[feature_columns] - df[feature_columns].rolling(89).mean()) / df[feature_columns].rolling(89).std()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    print(df.columns)
    return df
'''
def add_rsi(df, n=14):
    delta = df.Close.diff()
    delta.dropna(inplace=True)
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = dUp.rolling(n).mean()
    RolDown = dDown.rolling(n).mean().abs()
    df["rsi"] = (RolUp / RolDown) / 100
    df.dropna(inplace=True)
    return df


def add_willr(df, n=14):
    top = df["High"].rolling(n).max() - df["Close"]
    bottom = df["High"].rolling(n).max() - df["Low"].rolling(n).max()
    df["willr"] = (top/bottom) / 100
    df.dropna(inplace=True)
    return df


def apply_features(df):
    df['std_close'] = (df['High'] - df['Close'])/df['High']
    df['std_open'] = (df['High'] - df['Open'])/df['High']
    df['std_low'] = (df['High'] - df['Low'])/df['High']
    df['avg_vol_mid'] = pd.Series(np.where(df.Volume.rolling(8).mean() > df.Volume, 1, -1), df.index)
    df['vol_cross'] =  (df.Volume.rolling(3).mean() - df.Volume.rolling(8).mean()) / df.Volume.rolling(8).mean()
    #df = self.add_rsi(df)
    #df = self.add_willr(df)
    df['vol_roc'] =  df.Volume.pct_change(8)
    df["ma_short"] = df.Close.rolling(3).mean()
    df["ma_mid"] = df.Close.rolling(13).mean()
    df["ma_long"] = df.Close.rolling(55).mean()
    df["close_short"] = (df.Close - df.ma_short)  / df.ma_short
    df["short_mid"] = (df.Close - df.ma_mid) / df.ma_mid
    df["mid_long"] = (df.Close - df.ma_long) / df.ma_long
    df["roc"] = df["Close"].pct_change(periods=5)
    #df["roc_long"] = df["Close"].pct_change(periods=55)
    df["roc_mid"] = df["Close"].pct_change(periods=21)
    df["roc_short"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    self.start_idx = df.index[0]
    return df