# freqtrade strategy
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class FibonacciGoldenPocketRSIStrategy(IStrategy):
    # Define minimal ROI
    minimal_roi = {
        "0": 0.1,  # Take profit at 10%
    }

    # Define stop loss
    stoploss = -0.05  # 5% stoploss

    # Trailing stop is disabled
    trailing_stop = False

    # Timeframe for analysis
    timeframe = '1h'

    # Custom hyperparameters
    fib_low_factor = DecimalParameter(0.618, 0.65, default=0.618, decimals=3, space='buy', optimize=True)

    def informative_pairs(self):
        # Only trading the pair being analyzed
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI Calculation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Identify the highest high and lowest low within a period
        dataframe['high_rolling'] = dataframe['high'].rolling(window=100).max()
        dataframe['low_rolling'] = dataframe['low'].rolling(window=100).min()

        # Calculate Fibonacci retracement levels
        dataframe['fib_618'] = dataframe['low_rolling'] + (dataframe['high_rolling'] - dataframe['low_rolling']) * 0.618
        dataframe['fib_650'] = dataframe['low_rolling'] + (dataframe['high_rolling'] - dataframe['low_rolling']) * 0.65

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['fib_618']) &  # Price is below Fib 0.618
                (dataframe['close'] > dataframe['fib_650']) &  # Price is above Fib 0.65
                (dataframe['rsi'] < 30)  # RSI is below 30
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['fib_618']) &  # Price is above Fib 0.618
                (dataframe['close'] < dataframe['fib_650']) &  # Price is below Fib 0.65
                (dataframe['rsi'] > 60)  # RSI is above 60
            ),
            'sell'] = 1
        return dataframe