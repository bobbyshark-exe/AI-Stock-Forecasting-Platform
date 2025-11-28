"""Feature engineering utilities.

Summary of recent feature changes:
- Added RSI indicator calculation and ensured MACD uses the correct method.
- Fixed rolling standard deviation typo and strengthened rolling-statistics generation.
- Added explicit scaling function `scale_data` and sequence creation for LSTM.
- Added defensive checks (price column validation, NaN checks) and clearer
    target handling in the execution block.
"""

import numpy as np
import pandas as pd
# Using the ta library for indicators
from ta.momentum import RSIIndicator 
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from src.data.data_preprocessing import merge_and_preprocess_data

# --- FEATURE CREATION FUNCTIONS ---

def add_technical_indicators(df: pd.DataFrame, price_col='Close', windows=(5, 10, 20, 50)) -> pd.DataFrame:
    """Adds technical indicators (returns, rolling stats, MACD, RSI) to the DataFrame."""
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # 1. Returns
    df['pct_return'] = df[price_col].pct_change()
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))

    # 2. Rolling Statistics
    for window in windows:
        # **FIXED ERROR: Changed 'stf()' to 'std()'**
        df[f'rolling_mean_{window}'] = df[price_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[price_col].rolling(window=window).std()

    # 3. MACD
    macd_indicator = MACD(close=df[price_col], window_fast=12, window_slow=26, window_sign=9)
    # **FIXED ERROR: Changed macd_indicator.rsi() to macd_indicator.macd()**
    df['MACD'] = macd_indicator.macd() 

    # 4. RSI (Missing from original, now added)
    rsi_indicator = RSIIndicator(close=df[price_col], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # Drop NaNs created by rolling windows and indicator calculation (e.g., first 50 days)
    df = df.dropna()
    return df

# --- SCALING AND SEQUENCE CREATION FUNCTIONS ---

def scale_data(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Scales features using MinMaxScaler and returns the scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def create_lstm_sequences(df: pd.DataFrame, feature_cols: list, target_col: str, seq_len: int = 60) -> tuple:
    """Converts a SCALED DF into time-step sequences (X, y) for LSTM training."""
    # Data should be scaled before this function is called
    data = df[feature_cols + [target_col]].values

    X, y = [], []
    for i in range(len(data) - seq_len):
        # Features are columns 0 to -1 (all except the last one, which is the target)
        X.append(data[i:i + seq_len, :-1]) 
        # Target is the value at the step *after* the sequence
        y.append(data[i + seq_len, -1])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("NaN values found in final sequences. Check preprocessing.")
    
    # Keras/TensorFlow LSTM input shape: (samples, timesteps, features)
    return X, y

def train_val_test_split_sequences(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1, test_ratio: float = 0.1) -> tuple:
    """Splits sequential data into non-overlapping training, validation, and testing sets."""
    n_samples = len(X)
    # Determine split indices based on total sample count
    test_split_index = int(n_samples * (1 - test_ratio))
    val_split_index = int(test_split_index * (1 - val_ratio))

    # Split the data chronologically
    X_train, y_train = X[:val_split_index], y[:val_split_index]
    X_val, y_val = X[val_split_index:test_split_index], y[val_split_index:test_split_index]
    X_test, y_test = X[test_split_index:], y[test_split_index:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    print("--- Testing Feature Engineering ---")
    
    # 1. Get and Preprocess Data
    merged_df = merge_and_preprocess_data(ticker="MSFT", fred_series_id="DGS10", start_date="2018-01-01")
    print(f"Preprocessed DF shape: {merged_df.shape}")

    # 2. Add Technical Indicators
    final_features_df = add_technical_indicators(merged_df.copy())
    
    # 3. Define Features and Target
    # NOTE: The target column is set to the price one day ahead (future)
    TARGET_COL = 'Close_Future' 
    final_features_df[TARGET_COL] = final_features_df['Close'].shift(-1)
    final_features_df.dropna(inplace=True) # Drop the last row with the NaN target
    
    # Features for LSTM (only raw prices/volume for pure temporal analysis)
    LSTM_FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"Features created: {list(final_features_df.columns)}")

    # 4. Scale Data (CRITICAL STEP FOR LSTMS)
    scaled_df, scaler = scale_data(final_features_df.copy(), feature_cols=LSTM_FEATURE_COLS + [TARGET_COL])
    
    # 5. Create LSTM Sequences
    X, y = create_lstm_sequences(
        df=scaled_df, 
        feature_cols=LSTM_FEATURE_COLS, 
        target_col=TARGET_COL, 
        seq_len=60
    )
    
    print(f"\nScaled DF columns used: {LSTM_FEATURE_COLS + [TARGET_COL]}")
    print(f"LSTM Sequence X shape (samples, timesteps, features): {X.shape}")
    print(f"LSTM Sequence y shape (samples, target): {y.shape}")
    
    # 6. Split Data
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(X, y)

    print(f"\nTrain set shape (X, y): {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape (X, y): {X_val.shape}, {y_val.shape}")
    print(f"Test set shape (X, y): {X_test.shape}, {y_test.shape}")