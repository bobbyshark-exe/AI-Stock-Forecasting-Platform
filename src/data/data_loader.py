import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime  


#goes through and fetches yf data and then sees if the data for the ticker exist and returns (if exists - Data & if not returns error)
def get_stock_data(ticker, period = "5y"):
    #fetches historical data for a given ticker from Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        data = stockhistory(period=period)

        if data.empty:
            print(f"No data found for {ticker} for the period {period}.")
            return pd.DataFrame()
        
        data.index = pd.to_datetime(data.index).tz_locallize(None)  #Ensure timezone-naive
        print(f"Sucessfully fetched yfinance data for {ticker}.")
        return data
    
    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {e}")
        return pd.DataFrame()

#goes through and fetches FRED data and then sees if the data for the time period exist and returns (if exists - Data & if not returns error)
def get_fred_data(series_id, start_date, end_date):
    #fetches FRED data (Fed Reserve Economic Data)
    try:
        data = web.DataReader(series_id, "fred", start_date, end_date)
        print(f"Sucessfully fetched FRED data for {series_id}.")
        return data
    
    except Exception as e:
        print(f"Error fetching FRED data for {series_id}: {e}")
        return pd.DataFrame()


#test run using Apple Stock!
if __name__ == "__main__":
    print("---Testing yfinance (AAPL)---")
    aapl_data = get_stock_data("AAPL", period="1y")
    if not aapl_data.empty:
        print(aapl_data.tail())
    
