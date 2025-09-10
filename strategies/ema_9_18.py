from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ema_9_18(IStrategy):
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Enable shorting for futures trading
    can_short = True
    
    # Timeframe for the strategy
    timeframe = '4h'
    
    # Risk management parameters
    stoploss = -0.035  # Initial stop loss at 3.5%
    trailing_stop = True
    trailing_stop_positive = 0.035  # Trail price by 3.5%
    trailing_only_offset_is_reached = False  # Trailing stop activates immediately
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define the technical indicators used in the strategy.
        """
        dataframe['ema9'] = ta.EMA(dataframe['close'], timeperiod=9)
        dataframe['ema18'] = ta.EMA(dataframe['close'], timeperiod=18)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals for long and short positions.
        """
        # Long entry: EMA9 crosses above EMA18
        dataframe.loc[
            (dataframe['ema9'] > dataframe['ema18']) &
            (dataframe['ema9'].shift(1) <= dataframe['ema18'].shift(1)),
            'enter_long'] = 1

        # Short entry: EMA9 crosses below EMA18
        dataframe.loc[
            (dataframe['ema9'] < dataframe['ema18']) &
            (dataframe['ema9'].shift(1) >= dataframe['ema18'].shift(1)),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals for long and short positions.
        """
        # Exit long: EMA9 crosses below EMA18
        dataframe.loc[
            (dataframe['ema9'] < dataframe['ema18']) &
            (dataframe['ema9'].shift(1) >= dataframe['ema18'].shift(1)),
            'exit_long'] = 1

        # Exit short: EMA9 crosses above EMA18
        dataframe.loc[
            (dataframe['ema9'] > dataframe['ema18']) &
            (dataframe['ema9'].shift(1) <= dataframe['ema18'].shift(1)),
            'exit_short'] = 1

        return dataframe
