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
    try:
        print(f"Fetching data for {ticker}...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Debug the raw data
        print(f"Raw data shape: {data.shape}")
        print(f"Raw data columns: {data.columns}")
        print(f"Raw data index: {data.index.name}")
        print(f"First few rows of raw data:\n{data.head()}")
        
        if data.empty:
            print(f"No data found for ticker {ticker}.")
            return pd.DataFrame()
        
        # Handle MultiIndex columns (common in Yahoo Finance data)
        if isinstance(data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns, flattening...")
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            # Clean up potential trailing underscores if a level is empty
            data.columns = [col.rstrip('_') for col in data.columns]
            print(f"Flattened columns: {data.columns.tolist()}")

        # Reset index to make 'Date' a column instead of the index
        data.reset_index(inplace=True)
        print(f"After reset_index, columns: {data.columns.tolist()}")
        
        # Ensure we have a 'Close' column (it might be named differently)
        if 'Close' not in data.columns:
            print(f"'Close' column not found. Available columns: {data.columns.tolist()}")
            if 'Adj Close' in data.columns:
                print("Using 'Adj Close' as 'Close'")
                data['Close'] = data['Adj Close']
            elif 'Adj_Close' in data.columns:
                print("Using 'Adj_Close' as 'Close'")
                data['Close'] = data['Adj_Close']
            else:
                # Try to find any column with 'close' in its name (case insensitive)
                close_cols = [col for col in data.columns if 'close' in col.lower()]
                if close_cols:
                    print(f"Using '{close_cols[0]}' as 'Close'")
                    data['Close'] = data[close_cols[0]]
                else:
                    print("WARNING: No suitable price column found!")
        
        print(f"Final columns: {data.columns.tolist()}")
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
