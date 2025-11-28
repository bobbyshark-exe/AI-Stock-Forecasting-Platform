import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime


def get_stock_data(ticker, period="5y"):
    """Fetch historical stock data from yfinance for `ticker`.

    Returns a DataFrame indexed by date containing at least the `Close` and `Volume`
    columns. Returns an empty DataFrame on error.
    """
    try:
        stock = yf.Ticker(ticker)
        # use the history() method exposed by yfinance
        data = stock.history(period=period)

        if data is None or data.empty:
            print(f"No data found for {ticker} for the period {period}.")
            return pd.DataFrame()

        # Normalize index to timezone-naive datetimes
        try:
            data.index = pd.to_datetime(data.index).tz_localize(None)
        except Exception:
            # If tz_localize fails (already naive), fall back to plain conversion
            data.index = pd.to_datetime(data.index)

        # Ensure required columns exist
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            print(f"Fetched data for {ticker} missing expected columns: {list(data.columns)}")
            return pd.DataFrame()

        print(f"Successfully fetched yfinance data for {ticker}.")
        return data

    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {e}")
        return pd.DataFrame()

#goes through and fetches FRED data and then sees if the data for the time period exist and returns (if exists - Data & if not returns error)
def get_fred_data(series_id, start_date, end_date):
    #fetches FRED data (Fed Reserve Economic Data)
    try:
        # pandas_datareader accepts strings or datetime objects
        data = web.DataReader(series_id, "fred", start_date, end_date)
        print(f"Successfully fetched FRED data for {series_id}.")
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
    
