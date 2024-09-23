from config import api_key, secret_key
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import numpy as np 

# no keys required for crypto data
client = StockHistoricalDataClient(api_key, secret_key)

request_params = StockBarsRequest(
                        symbol_or_symbols=["SPY"],
                        timeframe=TimeFrame.Day,
                        start=datetime(2022, 7, 1),
                        end=datetime(2022, 8, 1)
                 )

# Fetch the stock bars
bars = client.get_stock_bars(request_params)

# Work directly with the DataFrame
df = bars.df

# Assuming the DataFrame has a 'close' column, calculate Moving Averages
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()

# Make sure 'Signal' column exists before assigning values
df['Signal'] = 0

# Generate Buy/Sell signals using df.loc[] to avoid chained assignment and ensure lengths match
df.loc[df.index[50:], 'Signal'] = np.where(df['SMA_50'][50:] > df['SMA_200'][50:], 1, 0)

# Create a 'Position' column for buy/sell decisions
df['Position'] = df['Signal'].diff()

# Print relevant signals
print(df[['close', 'SMA_50', 'SMA_200', 'Signal', 'Position']])