import requests
import pandas as pd
from datetime import datetime
import time

def get_unix_timestamp(date_str):
    """
    Converts a date string in the format 'dd-mm-yyyy' to a Unix timestamp.

    Args:
    date_str (str): Date string in the format 'dd-mm-yyyy'.

    Returns:
    int: Unix timestamp corresponding to the input date.
    """
    dt = datetime.strptime(date_str, "%d-%m-%Y")
    return int(time.mktime(dt.timetuple()))

def get_historical_data(symbol, market, resolution, from_timestamp, to_timestamp, is_sepolia=False, retries=3):
    """
    Fetches historical market data for a given symbol and market within a specified time range.

    Args:
    symbol (str): The symbol for the market data.
    market (str): The market identifier.
    resolution (str): The resolution of the data ('5', '15', '30', '60', '240', '720', '1D', '3D', '1W', '1M').
    from_timestamp (int): The start timestamp for the data.
    to_timestamp (int): The end timestamp for the data.
    is_sepolia (bool, optional): Whether to use the Sepolia endpoint. Defaults to False.
    retries (int, optional): The number of retry attempts in case of request failure. Defaults to 3.

    Returns:
    pd.DataFrame: DataFrame containing the historical market data.
    """
    bin_size, bin_unit = get_bin_size_and_unit(resolution)
    base_url = "https://api.overlay.market/sepolia-charts/v1/charts" if is_sepolia else "https://api.overlay.market/charts/v1/charts"
    params = {
        "market": market,
        "binSize": bin_size,
        "binUnit": bin_unit,
        "from": from_timestamp * 1000,
        "to": to_timestamp * 1000,
        "limit": 2000
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                raise ValueError("No data available for the requested period.")
            
            bars = [{
                "datetime": datetime.fromisoformat(bar["_id"]["time"].replace('Z', '+00:00')),
                "low": bar["low"],
                "high": bar["high"],
                "open": bar["open"],
                "close": bar["close"]
            } for bar in data]
            
            df = pd.DataFrame(bars)
            df.set_index('datetime', inplace=True)
            return df
        
        except requests.exceptions.RequestException as e:
            time.sleep(2 ** attempt)
        
    raise ValueError("Failed to fetch data after multiple retries.")

def get_bin_size_and_unit(resolution):
    """
    Maps resolution to bin size and unit.

    Args:
    resolution (str): The resolution of the data.

    Returns:
    tuple: Bin size and unit corresponding to the input resolution.
    """
    bin_size_and_unit_map = {
        "5": (5, "minute"),
        "15": (15, "minute"),
        "30": (30, "minute"),
        "60": (1, "hour"),
        "240": (4, "hour"),
        "720": (12, "hour"),
        "1D": (1, "day"),
        "3D": (3, "day"),
        "1W": (1, "week"),
        "1M": (1, "month")
    }
    return bin_size_and_unit_map.get(resolution, (1, "hour"))

def main(): #To test later
    # Parameters for data fetching
    symbol = "Electric Vehicle Commodity Index"
    market = "0x770e3a8afc5c01855b5bd8eb5b96b23bd7af1e43"
    resolution = "1"
    from_date = "27-05-2024"
    to_date = "03-08-2024"
    from_timestamp = get_unix_timestamp(from_date)
    to_timestamp = get_unix_timestamp(to_date)
    is_sepolia = True

    try:
        # Fetch historical market data
        df = get_historical_data(symbol, market, resolution, from_timestamp, to_timestamp, is_sepolia)
        print("Data fetched successfully.")
        print(df)

        # Save DataFrame to CSV file
        csv_filename = "historical_data1.csv"
        df.to_csv(csv_filename)
        print(f"Data saved to {csv_filename}")

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
