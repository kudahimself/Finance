import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



spy = yf.download(tickers='SPY',
                  auto_adjust=False,
                    start='2000-02-01',
                    end='2025-10-31')

spy_ret = np.log(spy['Adj Close']).diff().dropna().rename({'SPY': 'SPY Buy and Hold'}, axis=1)


cumulative_returns = np.exp(np.log1p(spy_ret).cumsum()) - 1

returns = cumulative_returns['SPY Buy and Hold'].reset_index()['SPY Buy and Hold']

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12,8))
returns.plot(ax=ax)
plt.title("Cumulative Returns: RSI and Technical Indicators (BB, Volatility, MACD, ATR) vs SPY")
plt.ylabel("Cumulative Return")
plt.xlabel("Date")
plt.show()