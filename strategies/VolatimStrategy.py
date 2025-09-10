from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas_ta as ta
import numpy as np

class VolatimStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}  # 5% immediate, 3% after 30min, 1% after 60min
    stoploss = -0.1  # Fixed 10% stoploss
    startup_candle_count = 50  # Wait 50 candles (~250min) for indicator initialization
    use_custom_stoploss = False
    trailing_stop = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=14)

        # Rolling max/min (lookback 20 periods)
        dataframe['rolling_max'] = dataframe['close'].rolling(window=20).max()
        dataframe['rolling_min'] = dataframe['close'].rolling(window=20).min()

        # Local max/min
        dataframe['local_max'] = dataframe['close'] >= dataframe['rolling_max'].shift(1)
        dataframe['local_min'] = dataframe['close'] <= dataframe['rolling_min'].shift(1)

        # Returns for the pair
        dataframe['returns'] = dataframe['close'].pct_change()

        # Rolling mean and cumulative sum of returns
        dataframe['returns_roll_mean'] = dataframe['returns'].rolling(window=20).mean()
        dataframe['returns_roll_mean_cumsum'] = dataframe['returns_roll_mean'].cumsum()

        # Volatility bands (std-based)
        std_multiplier = 2.0
        dataframe['returns_roll_mean_cumsum_upper'] = (
            dataframe['returns_roll_mean_cumsum'] + std_multiplier * dataframe['returns'].rolling(window=20).std()
        )
        dataframe['returns_roll_mean_cumsum_lower'] = (
            dataframe['returns_roll_mean_cumsum'] - std_multiplier * dataframe['returns'].rolling(window=20).std()
        )

        # BTC returns (fetch BTC/USDT data)
        btc_pair = 'BTC/USDT'  # Adjust if needed
        btc_dataframe = self.dp.get_pair_dataframe(pair=btc_pair, timeframe=self.timeframe)
        btc_dataframe['btc_returns'] = btc_dataframe['close'].pct_change()
        btc_dataframe['btc_returns_roll_mean'] = btc_dataframe['btc_returns'].rolling(window=20).mean()
        btc_dataframe['btc_returns_roll_mean_cumsum'] = btc_dataframe['btc_returns_roll_mean'].cumsum()

        # Merge BTC data
        dataframe = dataframe.join(btc_dataframe[['btc_returns', 'btc_returns_roll_mean', 'btc_returns_roll_mean_cumsum']])

        # Volume
        dataframe['volume'] = dataframe['volume']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &  # Oversold (adjustable based on 2.035% observation)
                (dataframe['close'] < dataframe['returns_roll_mean_cumsum_lower']) &  # Below lower band
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean()) &  # Volume above average
                (dataframe['local_min'] == True) &  # Local minimum
                (dataframe['btc_returns_roll_mean_cumsum'] > dataframe['btc_returns_roll_mean_cumsum'].shift(1)) &  # BTC stabilizing
                (dataframe['returns_roll_mean'] > dataframe['returns_roll_mean'].shift(1))  # Pair returns improving
            ),
            ['buy', 'buy_tag']] = (1, 'volatility_breakout')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) |  # Overbought
                (dataframe['close'] > dataframe['returns_roll_mean_cumsum_upper']) |  # Above upper band
                (dataframe['local_max'] == True) |  # Local maximum
                (dataframe['returns_roll_mean_cumsum'] < -0.95)  # Extreme bearish trend (e.g., -95%)
            ),
            ['sell', 'exit_tag']] = (1, 'volatility_exit')
        return dataframe

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,  # ~24 hours
                "trade_limit": 10,
                "stop_duration_candles": 72,  # ~6 hours
                "max_allowed_drawdown": 0.2  # 20% drawdown
            }
        ]