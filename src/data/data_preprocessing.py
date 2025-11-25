#Handles the merging and cleaning the data sources
import pandas as pd
from .data_loader import get_stock_data, get_fred_data
from datatime import datetime

def merge_and_preprocessing_data(ticker="AAPL", fred_series_id="DGS10",start_date="2010-01-01"):
    #fetches, merges, and cleans stock (yfinance) and economic (FRED) data.
    end_date = datetime.now()

    stock_df = get_stock_data(ticker, period = "max")
    fred_df = get_fred_data(fred_series_id, start_date, end_date)

    if stock_df.empty or fred_df.empty:
        print("Error: One or both dataframes are empty. Cannot merge.")
        return pd.DataFrame()
    
    stock_df = stock_df[['Close','Volume']]
    fred_df.colums = ['FRED_Value']

    #Merge: keep all stock dayss (left join) and fill FRED data
    merged_df = stock_df.merge(fred_df, left_index=True,right_index=True, how='left')
    merged_df['FRED_Value'] = merged_df['FRED_Value'].ffill() # forward fill
    merged_df = merged_df.dropna() # drop intials NaNs <--missing data

    print(f"Preprocessing sucessful. Merged data shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    print("--- Testing Data Preprocessing and Merging ---")
    final_df = merge_and_preprocessing_data(ticker="AAPL", fred_series_id="DGS10", start_date="2018-01-01")
    
    if not final_df.empty:
        print(final_df.head())
