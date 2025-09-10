from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class CustomMACDStrategy(IStrategy):
    """
    Custom strategy based on MACD values and EMA3.
    """
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.1,  # Example ROI (can be adjusted as needed)
    }

    # Stoploss
    stoploss = -0.1  # Example stoploss (adjust as needed)

    # Trailing stoploss
    trailing_stop = False

    # Startup candle count, to ensure indicators have enough data
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several technical indicators to the given DataFrame.
        """
        # Add MACD and related indicators
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Add EMA3
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on the provided parameters, define the buy signal logic.
        """
        dataframe.loc[
            (
                (dataframe['macd'] <= -9.997) &
                (dataframe['macdsignal'] <= -100.000) &
                (dataframe['macdhist'] >= 66569.522) &
                (dataframe['ema3'] <= -0.012)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on the provided parameters, define the sell signal logic.
        """
        dataframe.loc[
            (
                (dataframe['macd'] >= -2.623) &
                (dataframe['macdsignal'] >= -2.623) &
                (dataframe['macdhist'] <= 66569.522)
            ),
            'sell'] = 1

        return dataframe
