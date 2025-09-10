# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AverageStrategy(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        buys and sells on crossovers - doesn't really perfom that well and its just a proof of concept
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "0": 10000,
        
        
        
    }

    # Stoploss:
    stoploss = -0.25

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.3073
    trailing_stop_positive_offset = 0.33535
    trailing_only_offset_is_reached = False
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['maShort'] = ta.EMA(dataframe, timeperiod=360)
        dataframe['maMedium'] = ta.EMA(dataframe, timeperiod=648)
        
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                
                qtpylib.crossed_above(dataframe['maShort'],dataframe['maMedium'])
                
                
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['maMedium'],dataframe['maShort'])
                
                
                
            ),
            'sell'] = 1
        return dataframe
