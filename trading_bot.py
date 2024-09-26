import numpy as np
from rsi import calculate_rsi
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class TradingBot:
    def __init__(self, df, lstm_model, stop_loss_percentage=0.10):
        self.df = df
        self.lstm_model = lstm_model  # Load your trained LSTM model here
        self.time_steps = 5
        self._prepare_data()  # Prepare data and calculate indicators
        self.trades = []
        self.total_profit = 0.0
        self.profitable_trades = 0
        self.total_trades = 0
        self.current_position = None
        self.stop_loss_percentage = stop_loss_percentage
        

    def _prepare_data(self):   
        # Calculate technical indicators
        self.df['SMA_20'] = self.df['close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['close'].rolling(window=50).mean()
        self.df['RSI'] = calculate_rsi(self.df)  # Assuming you have an RSI function
        self.df['EMA_12'] = self.df['close'].ewm(span=12).mean()
        self.df['EMA_26'] = self.df['close'].ewm(span=26).mean()
        self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9).mean()

        self.df['std_20'] = self.df['close'].rolling(window=20).std()
        self.df['Bollinger_upper'] = self.df['SMA_20'] + (self.df['std_20'] * 2)
        self.df['Bollinger_lower'] = self.df['SMA_20'] - (self.df['std_20'] * 2)

        self.df['high_low'] = self.df['high'] - self.df['low']
        self.df['high_close'] = np.abs(self.df['high'] - self.df['close'].shift())
        self.df['low_close'] = np.abs(self.df['low'] - self.df['close'].shift())
        self.df['true_range'] = self.df[['high_low', 'high_close', 'low_close']].max(axis=1)
        self.df['ATR'] = self.df['true_range'].rolling(window=14).mean()

        self.df['14_low'] = self.df['low'].rolling(window=14).min()
        self.df['14_high'] = self.df['high'].rolling(window=14).max()
        self.df['Stochastic_K'] = (self.df['close'] - self.df['14_low']) / (self.df['14_high'] - self.df['14_low']) * 100

        self.df['momentum_10'] = self.df['close'] - self.df['close'].shift(10)

        # Drop NaN values
        self.df = self.df.dropna()

        # Normalize features
        X = self.df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'Bollinger_upper', 'Bollinger_lower', 'ATR', 'Stochastic_K', 'momentum_10']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)


         # Reshape the data for LSTM (time_steps)
        X_lstm = np.array([X_scaled[i-self.time_steps:i] for i in range(self.time_steps, len(X_scaled))])

        # Use the LSTM model to make predictions
        lstm_predictions = self.lstm_model.predict(X_lstm)
        lstm_predictions = [1 if p > 0.5 else 0 for p in lstm_predictions]  # Convert predictions to binary (buy/sell)

        # Ensure that the number of predictions matches the number of rows after slicing
        if len(lstm_predictions) != len(self.df) - self.time_steps:
            raise ValueError(f"Mismatch between prediction length {len(lstm_predictions)} and DataFrame length {len(self.df) - self.time_steps}")

        # Assign predictions to a new column 'LSTM_signal'
        self.df['LSTM_signal'] = 0  # Initialize column
        self.df.iloc[self.time_steps:, self.df.columns.get_loc('LSTM_signal')] = lstm_predictions

    def execute_trades(self):
        for i, row in self.df.iterrows():
            if row['LSTM_signal'] == 1 and self.current_position is None:
                # Buy signal from LSTM
                self.buy(row['close'], i)
            elif row['LSTM_signal'] == 0 and self.current_position is not None:
                # Sell signal from LSTM
                self.sell(row['close'], i)
            elif self.current_position is not None:
                # Stop-loss check
                if row['close'] < self.current_position['stop_loss_price']:
                    self.sell(row['close'], i)

        self.summary()

    def buy(self, price, timestamp):
        stop_loss_price = price * (1 - self.stop_loss_percentage)
        self.current_position = {'price': price, 'stop_loss_price': stop_loss_price, 'timestamp': timestamp}
        self.trades.append({'action': 'buy', 'price': price, 'timestamp': timestamp})
        print(f"Bought at {price} on {timestamp}, stop-loss set at {stop_loss_price}")

    def sell(self, price, timestamp):
        if self.current_position:
            buy_price = self.current_position['price']
            profit = price - buy_price
            self.total_profit += profit
            self.total_trades += 1
            if profit > 0:
                self.profitable_trades += 1
            self.trades.append({'action': 'sell', 'price': price, 'timestamp': timestamp})
            print(f"Sold at {price} on {timestamp}, profit: {profit:.2f}")
            self.current_position = None

    def summary(self):
        accuracy = (self.profitable_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        print(f"Total profit: {self.total_profit:.2f}, accuracy: {accuracy:.2f}%")
