# Handles the merging and cleaning the data sources
import pandas as pd
from .data_loader import get_stock_data, get_fred_data
from datetime import datetime

# FIX 1: Renamed function to match src/main.py (removed 'ing')
def merge_and_preprocess_data(ticker="AAPL", fred_series_id="DGS10", start_date="2010-01-01"):
    # Fetches, merges, and cleans stock (yfinance) and economic (FRED) data.
    end_date = datetime.now()

    stock_df = get_stock_data(ticker, period="max")
    fred_df = get_fred_data(fred_series_id, start_date, end_date)

    if stock_df.empty or fred_df.empty:
        print("Error: One or both dataframes are empty. Cannot merge.")
        return pd.DataFrame()
    
    # Select specific columns - keep full OHLCV so downstream pipelines have expected features
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in expected_cols if c not in stock_df.columns]
    if missing:
        print(f"Error: stock data missing expected columns: {missing}")
        return pd.DataFrame()
    stock_df = stock_df[expected_cols]
    
    # FIX 2: Fixed typo 'colums' -> 'columns'
    fred_df.columns = ['FRED_Value']

    # Merge: keep all stock days (left join) and fill FRED data
    merged_df = stock_df.merge(fred_df, left_index=True, right_index=True, how='left')
    merged_df['FRED_Value'] = merged_df['FRED_Value'].ffill() # forward fill
    merged_df = merged_df.dropna() # drop initial NaNs <-- missing data

    print(f"Preprocessing successful. Merged data shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    print("--- Testing Data Preprocessing and Merging ---")
    # Updated function call here too
    final_df = merge_and_preprocess_data(ticker="AAPL", fred_series_id="DGS10", start_date="2018-01-01")
    
    if not final_df.empty:
        print(final_df.head())