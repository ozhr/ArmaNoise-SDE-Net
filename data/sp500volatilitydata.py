import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ取得
ticker = "^GSPC"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# 日次リターンの計算
data['Return'] = data['Close'].pct_change()

# 実現ボラティリティ（20日移動標準偏差 × √252）
data['RealizedVol'] = data['Return'].rolling(window=20).std() * np.sqrt(252)

# ログボラティリティの計算
data['LogVol'] = np.log(data['RealizedVol'])

# NaNを削除し、t（整数）とxの形式で整形
logvol_df = data[['LogVol']].dropna().reset_index(drop=True)
logvol_df['t'] = np.arange(0, len(logvol_df))
logvol_df = logvol_df[['t', 'LogVol']]
logvol_df.columns = ['t', 'x']

# CSVとして保存
logvol_df.to_csv("log_volatility_sp500.csv", index=False)

# プロット
plt.figure(figsize=(12, 6))
plt.plot(logvol_df['t'], logvol_df['x'])
plt.title('S&P 500 Log Realized Volatility')
plt.xlabel('Date')
plt.ylabel('Log Volatility')
plt.grid(True)
plt.tight_layout()
plt.show()
