import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

def forecast_prices(data: pd.DataFrame, days_to_predict: int = 60) -> pd.DataFrame:
    """
    Simple forecasting function that uses linear regression for prediction.
    This is a simplified version that prioritizes reliability over accuracy.
    """
    print("\n==== STARTING FORECASTING PROCESS ====\n")
    
    try:
        # Create an empty DataFrame if the input is invalid
        if data is None or data.empty:
            print("Error: No data provided for forecasting.")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI'])
        
        # Print detailed information about the input data
        print(f"Input data shape: {data.shape}")
        print(f"Input data columns: {data.columns.tolist()}")
        print(f"Input data types:\n{data.dtypes}")
        print(f"First few rows of input data:\n{data.head()}")
        
        # STEP 1: Ensure we have the necessary columns
        # Try multiple possible column names for price data
        price_column = None
        possible_price_columns = ['Close', 'Adj Close', 'Adj_Close', 'close', 'adj_close', 'Price', 'price']
        
        for col in possible_price_columns:
            if col in data.columns:
                price_column = col
                print(f"Using '{price_column}' as the price column")
                break
        
        if price_column is None:
            # Try case-insensitive search
            for col in data.columns:
                if any(price_name.lower() in col.lower() for price_name in ['close', 'price']):
                    price_column = col
                    print(f"Using '{price_column}' as the price column")
                    break
        
        if price_column is None:
            print("Error: Could not find a suitable price column")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI'])
        
        # Ensure we have a Date column
        date_column = None
        if 'Date' in data.columns:
            date_column = 'Date'
        elif data.index.name == 'Date':
            print("Date is the index, resetting index")
            data = data.reset_index()
            date_column = 'Date'
        else:
            # Try to find a date column
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]) or 'date' in col.lower():
                    date_column = col
                    print(f"Using '{date_column}' as the date column")
                    break
        
        if date_column is None:
            print("Error: Could not find a date column")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI'])
        
        # STEP 2: Clean and prepare the data
        clean_data = data.copy()
        
        # Convert date column to datetime if it's not already
        clean_data[date_column] = pd.to_datetime(clean_data[date_column], errors='coerce')
        
        # Convert price column to numeric
        clean_data[price_column] = pd.to_numeric(clean_data[price_column], errors='coerce')
        
        # Drop rows with missing values
        clean_data = clean_data.dropna(subset=[date_column, price_column])
        
        # Sort by date
        clean_data = clean_data.sort_values(by=date_column)
        
        # Use only the last 60 days of data for forecasting
        clean_data = clean_data.tail(60).copy()
        
        print(f"Clean data shape: {clean_data.shape}")
        print(f"Clean data head:\n{clean_data.head()}")
        
        if len(clean_data) < 10:
            print("Error: Not enough valid price data points after cleaning.")
            return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI'])
        
        # STEP 3: Create the forecast
        # Get the last date in the data
        last_date = clean_data[date_column].iloc[-1]
        print(f"Last date in data: {last_date}")
        
        # Create a simple trend by fitting a line to the recent data
        y = clean_data[price_column].values
        x = np.arange(len(y)).reshape(-1, 1)
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        
        # Generate future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        # Predict future values
        future_x = np.arange(len(y), len(y) + days_to_predict).reshape(-1, 1)
        predictions = model.predict(future_x)
        
        # Calculate a simple confidence interval
        std_dev = np.std(y) * 1.5
        lower_ci = predictions - std_dev
        upper_ci = predictions + std_dev
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates_str,
            'Forecast': predictions,
            'Lower_CI': lower_ci,
            'Upper_CI': upper_ci
        })
        
        print(f"Successfully created forecast for {days_to_predict} days.")
        print(f"Forecast data head:\n{forecast_df.head()}")
        
        return forecast_df
        
    except Exception as e:
        print(f"Error during forecasting: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return pd.DataFrame(columns=['Date', 'Forecast', 'Lower_CI', 'Upper_CI'])

if __name__ == '__main__':
    from data_fetcher import fetch_commodity_data

    # Fetch sample data for Copper
    copper_ticker = 'HG=F'
    historical_data = fetch_commodity_data(copper_ticker, days=365)

    if not historical_data.empty:
        # Generate the forecast
        price_forecast = forecast_prices(historical_data, days_to_predict=60)
        
        if not price_forecast.empty:
            print("\nForecasted Prices (Next 60 Days):")
            print(price_forecast.to_string())
