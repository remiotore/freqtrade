from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SimpleMAStrategy(IStrategy):
    """
    Simple Moving Average Crossover Strategy
    
    This strategy buys when the short-term SMA crosses above the long-term SMA
    and sells when the short-term SMA crosses below the long-term SMA.
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    # Stoploss
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False

    # Timeframe for the strategy
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different moving averages to the dataframe.
        """
        # SMA indicators
        dataframe['sma7'] = ta.SMA(dataframe, timeperiod=7)
        dataframe['sma25'] = ta.SMA(dataframe, timeperiod=25)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['sma7'], dataframe['sma25'])) &  # Signal: short SMA crosses above long SMA
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['sma7'], dataframe['sma25'])) &  # Signal: short SMA crosses below long SMA
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_long'] = 1
        return dataframe
