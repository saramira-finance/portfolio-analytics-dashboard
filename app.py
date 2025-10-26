# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

st.set_page_config(page_title="Portfolio Analytics Dashboard", layout="wide")

st.title("💼 Portfolio Analytics Dashboard")
st.markdown("Analyze, visualize, and optimize your investment portfolio.")

# --- Inputs ---
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.text_input("Enter stock tickers (comma-separated):", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
tickers = [t.strip().upper() for t in tickers]

weights_input = st.sidebar.text_input("Enter weights (comma-separated, must sum to 1):", "0.2,0.2,0.2,0.2,0.2")
weights = np.array([float(x) for x in weights_input.split(",")])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
benchmark_ticker = st.sidebar.text_input("Benchmark (e.g., ^GSPC for S&P 500):", "^GSPC")

# --- Data Download ---
data = yf.download(tickers, start=start_date, end=end_date)["Close"]
benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date)["Close"]

returns = data.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()
portfolio_returns = (returns * weights).sum(axis=1)

# --- Portfolio Performance ---
cumulative_portfolio = (1 + portfolio_returns).cumprod()
cumulative_benchmark = (1 + benchmark_returns).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_portfolio.index, y=cumulative_portfolio, name="Portfolio"))
fig.add_trace(go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark, name="Benchmark"))
fig.update_layout(title="Portfolio vs Benchmark Performance", yaxis_title="Cumulative Return", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)

# --- Key Metrics ---
sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
volatility = portfolio_returns.std() * np.sqrt(252)
total_return = cumulative_portfolio[-1] - 1

st.subheader("📊 Portfolio Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Return", f"{total_return:.2%}")
col2.metric("Annualized Volatility", f"{volatility:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

# --- Risk Section ---
st.subheader("📉 Risk Analysis")

corr_matrix = returns.corr()
st.write("Correlation Matrix:")
st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))

window = 60
rolling_vol = returns.rolling(window).std() * np.sqrt(252)
portfolio_rolling_vol = (rolling_vol * weights).sum(axis=1)
st.line_chart(portfolio_rolling_vol, use_container_width=True)

# --- Optimization ---
st.subheader("🚀 Portfolio Optimization")

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(tickers)
num_portfolios = 5000

results = np.zeros((3, num_portfolios))
weight_array = []

for i in range(num_portfolios):
    w = np.random.random(num_assets)
    w /= np.sum(w)
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = ret / vol
    results[0, i] = vol
    results[1, i] = ret
    results[2, i] = sharpe
    weight_array.append(w)

results_df = pd.DataFrame(results.T, columns=["Volatility", "Return", "Sharpe"])
weights_df = pd.DataFrame(weight_array, columns=tickers)
max_sharpe_idx = results_df["Sharpe"].idxmax()
optimal_portfolio = results_df.iloc[max_sharpe_idx]
optimal_weights = weights_df.iloc[max_sharpe_idx]

fig2 = px.scatter(results_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis",
                  title="Efficient Frontier", hover_data={"Sharpe": ':.2f'})
fig2.add_trace(go.Scatter(x=[optimal_portfolio["Volatility"]], y=[optimal_portfolio["Return"]],
                          mode="markers", marker=dict(color="red", size=10),
                          name="Max Sharpe Portfolio"))
st.plotly_chart(fig2, use_container_width=True)

st.write("**Optimal Portfolio Weights:**")
st.dataframe(optimal_weights)
