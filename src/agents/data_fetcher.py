import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_commodity_data(ticker: str, days: int = 120) -> pd.DataFrame:
    """
    Fetches historical data for a given commodity ticker from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol for the commodity (e.g., 'HG=F' for Copper).
        days (int): The number of days of historical data to fetch.

    Returns:
        pd.DataFrame: A pandas DataFrame with the historical data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for ticker {ticker}. It might be delisted or an invalid ticker.")
            return pd.DataFrame()
        
        # Flatten multi-level column headers if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            # Clean up potential trailing underscores if a level is empty
            data.columns = [col.rstrip('_') for col in data.columns]

        print(f"Successfully fetched {len(data)} data points for {ticker} from {start_date.date()} to {end_date.date()}.")
        return data
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage: Fetch data for Copper
    copper_ticker = 'HG=F' 
    copper_data = fetch_commodity_data(copper_ticker, days=120)
    if not copper_data.empty:
        print("\nCopper Data Head:")
        print(copper_data.head())

    # Example usage: Fetch data for Steel (using a sample steel company ticker as an example, e.g., 'X' for U.S. Steel)
    steel_ticker = 'X'
    steel_data = fetch_commodity_data(steel_ticker, days=120)
    if not steel_data.empty:
        print("\nU.S. Steel Data Head:")
        print(steel_data.head())
