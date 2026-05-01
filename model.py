import numpy as np
from scipy.stats import t

def predict_next_hour_price_interval(prices, num_simulations=10000, alpha=0.05):
    """
    Predicts the next period's confidence interval using Geometric Brownian Motion
    and a Student-t distribution for returns.
    
    Returns:
        lower_bound (float): The lower bound of the interval.
        upper_bound (float): The upper bound of the interval.
        scale (float): The estimated volatility (scale of the t-distribution).
    """
    # Compute log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    # Fit student-t distribution to log returns
    df, loc, scale = t.fit(log_returns)
    
    # Simulate future returns
    simulated_returns = t.rvs(df, loc=loc, scale=scale, size=num_simulations)
    
    # Calculate future prices
    last_price = prices.iloc[-1]
    simulated_prices = last_price * np.exp(simulated_returns)
    
    # Get the quantiles for the prediction interval
    lower_bound = np.percentile(simulated_prices, (alpha / 2) * 100)
    upper_bound = np.percentile(simulated_prices, (1 - alpha / 2) * 100)
    
    return float(lower_bound), float(upper_bound), float(scale)
