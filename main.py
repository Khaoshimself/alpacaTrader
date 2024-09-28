from config import api_key, secret_key, db_config
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import numpy as np
from trading_bot import TradingBot
import tensorflow as tf
from tensorflow.keras.models import load_model


# Step 1: Fetch Historical Data
def fetch_historical_data(symbol, start_date, end_date):
    client = StockHistoricalDataClient(api_key, secret_key)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = client.get_stock_bars(request_params)
    return bars.df

# Step 2: Main execution function
def main():
    # Define parameters for fetching data
    symbol = "SPY"
    start_date = datetime(2000, 7, 1)
    end_date = datetime(2024, 8, 1)
    
    # Fetch historical data
    df = fetch_historical_data(symbol, start_date, end_date)
    
    # Load the saved LSTM model
    lstm_model = load_model('lstm_trading_model.keras')
    
    # Initialize the TradingBot with data and the loaded LSTM model
    bot = TradingBot(df, lstm_model, db_config=db_config)
    
    # Execute trades
    bot.execute_trades()
    
    # Print trading summary
    bot.summary()

# Ensure the script runs only if executed directly
if __name__ == "__main__":
    main()