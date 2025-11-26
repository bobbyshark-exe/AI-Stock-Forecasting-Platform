import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from data.data_preprocessing import merge_and_preprocessing_data

def add_technical_indicators(df: pd.DataFrame, price_col='Close', windows=(5, 10, 20, 50)) -> pd.DataFrame:
    #adds technical indicators (returns, rolling stats, MACD, RSI) to the DataFrame.
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    df['pct_return'] = df[price_col].pct_change()
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))

    for windows in windows:
        df[f'rolling_mean_{window}'] = df[price_col].rolling(window = window).mean()
        df[f'rolling_std_{window}'] = df[price_col].rolling(window = window).stf()

    macd_indicator = MACD(close=df[price_col], window_fast=12, window_slow=26, window_sign =9)
    df['MACD'] = macd_indicator.rsi()

    df = df.dropna()
    return df

def create_lstm_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, seq_len: int = 60) -> tuple:
    #converts a DF into time-step sequences (X,y) for LSTM training
    data = df[feature_cols + [target_col]].values

    X,y = [],[]
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len, :-1])
        y.append(data[i + seq_len, -1])
    
    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.float32)

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values doound in final sequences. Check preprocessing.")
    
    return X, y

def train_val_test_split_sequences(X: np.darray, y: np.ndarray, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
    n_samples = len(X)
    test_split_index = int(n_samples * (1 - test_ratio))
    val_split_index = int(test_split_index * (1 - val_ratio))

    X_train, y_train = X[:val_split_index], y[:val_split_index]
    X_val, y_val = X[val_split_index:test_split_index], y[val_split_index:test_split_index]
    X_test, y_test = X[test_split_index:], y[test_split_index:]

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    print("--- Testing Feature Engineering ---")
    merged_df = merge_and_preproces_data(ticker="MSFT", fred_serie_id = "DGS10", start_date="2018-01-01")
    final_features_df = add_technical_indicators(merged_df.copy())
    print(f"Features created: {list(final_features_df.columns)}")

    

