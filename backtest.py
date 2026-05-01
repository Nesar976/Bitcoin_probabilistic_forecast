import json
import os
import pandas as pd
from data import fetch_binance_klines
from model import predict_next_hour_price_interval
from metrics import calculate_coverage, calculate_average_width, winkler_score

def run_backtest(test_size=720, window_size=500, symbol="BTCUSDT", interval="1h", alpha=0.05):
    total_bars = test_size + window_size
    
    print(f"Fetching {total_bars} bars for backtest ({symbol} {interval})...")
    df = fetch_binance_klines(symbol=symbol, interval=interval, total_bars=total_bars)
    
    results = []
    
    print(f"Running backtest over last {test_size} periods...")
    start_idx = len(df) - test_size - 1
    end_idx = len(df) - 1
    
    for i in range(start_idx, end_idx):
        train_data = df.iloc[i - window_size + 1 : i + 1]
        
        actual_price = df.iloc[i + 1]["close"]
        timestamp = str(df.iloc[i + 1]["open_time"])
        
        # Unpack the 3 values returned
        lower, upper, volatility = predict_next_hour_price_interval(train_data["close"], alpha=alpha)
        
        results.append({
            "timestamp": timestamp,
            "actual_price": actual_price,
            "lower_bound": lower,
            "upper_bound": upper,
            "volatility": volatility
        })
        
        if len(results) % 100 == 0:
            print(f"Completed {len(results)} / {test_size} steps")
            
    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    actuals = [r["actual_price"] for r in results]
    lowers = [r["lower_bound"] for r in results]
    uppers = [r["upper_bound"] for r in results]
    
    cov = calculate_coverage(actuals, lowers, uppers)
    width = calculate_average_width(lowers, uppers)
    winkler = winkler_score(actuals, lowers, uppers, alpha=alpha)
    
    print("\n--- Backtest Results ---")
    print(f"Coverage: {cov:.4f} (Target: ~{1-alpha})")
    print(f"Average Width: {width:.2f}")
    print(f"Winkler Score: {winkler:.2f}")
    print("Results saved to backtest_results.json")
    
    return results

if __name__ == "__main__":
    run_backtest()
