import numpy as np
from rsi import calculate_rsi

class TradingBot:
    def __init__(self, df):
        self.df = df
        self._prepare_data()
        self.trades = []  # Track each buy and sell trade
        self.total_profit = 0.0  # Track total profit
        self.profitable_trades = 0  # Count of profitable trades
        self.total_trades = 0  # Count of total trades
        self.current_position = None  # Track if holding a position
    
    def _prepare_data(self):
        # Calculate moving averages
        self.df['SMA_50'] = self.df['close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['close'].rolling(window=200).mean()

        # Calculate RSI
        self.df['RSI'] = calculate_rsi(self.df)

        # Initialize Signal column
        self.df['Signal'] = 0

        # Generate buy (1) and sell (-1) signals based on SMA crossover
        valid_sma = self.df['SMA_50'].notna() & self.df['SMA_200'].notna()

        self.df.loc[valid_sma & (self.df['SMA_50'] > self.df['SMA_200']), 'Signal'] = 1  # Buy signal
        self.df.loc[valid_sma & (self.df['SMA_50'] < self.df['SMA_200']), 'Signal'] = -1  # Sell signal

        # Track positions explicitly using buy (1) and sell (-1) signals
        self.df['Position'] = 0
        holding_position = False

        for i in range(1, len(self.df)):
            if self.df.iloc[i]['Signal'] == 1 and not holding_position:
                # Buy signal: open a position
                self.df.at[self.df.index[i], 'Position'] = 1
                holding_position = True
            elif self.df.iloc[i]['Signal'] == -1 and holding_position:
                # Sell signal: close the position
                self.df.at[self.df.index[i], 'Position'] = -1
                holding_position = False
            else:
                # Maintain the current position state
                self.df.at[self.df.index[i], 'Position'] = self.df.iloc[i - 1]['Position']

    def execute_trades(self):
        # Execute trades and track profits
        for i, row in self.df.iterrows():
            if row['Position'] == 1 and self.current_position is None:
                # Buy action
                self.buy(row['close'], i)
            elif row['Position'] == -1 and self.current_position is not None:
                # Sell action
                self.sell(row['close'], i)

        # At the end of execution, show the summary
        self.summary()

    def buy(self, price, timestamp):
        # Record the buy trade
        self.current_position = {'price': price, 'timestamp': timestamp}
        self.trades.append({'action': 'buy', 'price': price, 'timestamp': timestamp})
        print(f"Buy at {price} on {timestamp}")

    def sell(self, price, timestamp):
        # Sell action, calculate profit based on the previous buy price
        if self.current_position:
            buy_price = self.current_position['price']
            profit = price - buy_price
            self.total_profit += profit
            self.total_trades += 1

            # Determine if the trade was profitable
            if profit > 0:
                self.profitable_trades += 1

            # Log the sell trade and remove the current position
            self.trades.append({'action': 'sell', 'price': price, 'timestamp': timestamp})
            print(f"Sell at {price} on {timestamp}, Profit: {profit:.2f}")
            self.current_position = None

    def summary(self):
        # Calculate accuracy
        accuracy = (self.profitable_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        print(f"Total Profit: {self.total_profit:.2f}")
        print(f"Accuracy: {accuracy:.2f}%")

    def print_signals(self):
        print(self.df[['close', 'SMA_50', 'SMA_200', 'RSI', 'Signal', 'Position']])

