import requests
import pandas as pd

def fetch_binance_klines(symbol="BTCUSDT", interval="1h", total_bars=1500):
    url = "https://data-api.binance.vision/api/v3/klines"
    all_data = []
    end_time = None
    
    while len(all_data) < total_bars:
        # Binance max limit is 1000
        limit = min(1000, total_bars - len(all_data))
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
            
        all_data = data + all_data
        # Update end_time to fetch older data in next loop iteration
        end_time = data[0][0] - 1
        
        if len(data) < limit:
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df.tail(total_bars)
