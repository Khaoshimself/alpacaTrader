from config import api_key, secret_key
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import numpy as np 
from trading_bot import TradingBot
from backtesting import Backtesting


# No keys required for crypto data
client = StockHistoricalDataClient(api_key, secret_key)

request_params = StockBarsRequest(
                        symbol_or_symbols=["AAPL"],
                        timeframe=TimeFrame.Day,
                        start=datetime(2020, 7, 1),
                        end=datetime(2024, 8, 1)
                 )

# Fetch the stock bars
bars = client.get_stock_bars(request_params)
df = bars.df

# Initialize the trading bot
bot = TradingBot(df)

# Run the backtest
backtest = Backtesting(bot)
backtest.run()

# Print out results
bot.print_signals()
