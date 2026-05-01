# AlphaI x Polaris Bitcoin Next-Hour Prediction Challenge

<img width="3024" height="1732" alt="image" src="https://github.com/user-attachments/assets/0867f3ee-4c36-4b23-9748-3fdaed01ff31" />


This repository contains a full-stack solution for predicting a 95% confidence interval for the Bitcoin (BTCUSDT) price in the next hour using the Binance API.

## Core Model
The model uses **Geometric Brownian Motion (GBM)**. To account for fat tails and volatility clustering observed in financial markets:
- Log returns are computed from historical data.
- A **Student-t distribution** is fitted to the rolling window of log returns. This handles the fat tails (extreme moves).
- The volatility and drift are implicitly estimated by fitting the distribution to a recent rolling window (default: 500 hours), allowing the model to adapt interval width based on recent volatility.
- We simulate 10,000 future paths and take the 2.5th and 97.5th percentiles to form the 95% confidence interval.

## Structure
- `data.py`: Fetches hourly klines from Binance API.
- `model.py`: Implements the GBM model with a Student-t distribution.
- `metrics.py`: Calculates Coverage, Average Width, and the Winkler Score.
- `backtest.py`: Runs a loop over the last 720 hours to evaluate the model without data leakage.
- `utils.py`: Helper functions for persistence and IO.
- `app.py`: Streamlit dashboard for real-time predictions and monitoring.

## Local Execution Instructions

### 1. Install dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Backtest
To generate the `backtest_results.json` file and see the evaluation metrics, run:
```bash
python backtest.py
```
This fetches the last 1220 bars and backtests the latest 720 bars strictly using historical data.

### 3. Run the Dashboard
To start the live dashboard with real-time predictions, run:
```bash
streamlit run app.py
```
The app will open in your browser (default: `http://localhost:8501`).

## Deployment to Streamlit Cloud
1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in.
3. Click **New app**.
4. Select your GitHub repository and set the **Main file path** to `app.py`.
5. Click **Deploy**.
6. The app will automatically install dependencies from `requirements.txt` and launch.

## Constraints Respected
- **No Data Leakage**: Backtest strictly predicts `t+1` using `[t-window_size, t]`.
- **Volatility Clustering**: Rolling window ensures recent high/low volatility adapts the interval width.
- **Fat Tails**: Use of scipy `t.fit()` successfully handles extremes better than standard normal distributions.
