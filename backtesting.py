class Backtesting:
    def __init__(self, bot):
        self.bot = bot

    def run(self):
        print("Running backtest on historical data...")
        self.bot.execute_trades()
        print("Backtest complete.")