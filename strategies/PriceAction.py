# https://medium.com/@a.m.saghiri2008/price-action-theory-a-comprehensive-overview-and-a-practical-freqtrade-strategy-2535f49c88bb
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
class PriceActionEngulfingStrategy(IStrategy):
    """
    A basic price action strategy that detects bullish/bearish engulfing candles
    to enter and exit trades.
    """
    # ROI targets
    minimal_roi = {
        "0": 0.10,   # 10% ROI from the start
        "60": 0.05,  # After 60 minutes, reduce target to 5%
        "120": 0.02, # After 120 minutes, reduce target to 2%
        "240": 0     # After 240 minutes, exit whenever profitable
    }
    # Stop-loss at 5%
    stoploss = -0.05
    # Disable trailing stop for simplicity
    trailing_stop = False
    # Strategy timeframe
    timeframe = '1h'
    # Process only new candles
    process_only_new_candles = True
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Reference to previous candle open/close
        dataframe['prev_open'] = dataframe['open'].shift(1)
        dataframe['prev_close'] = dataframe['close'].shift(1)
        # Define a bullish engulfing pattern
        dataframe['bullish_engulfing'] = (
            (dataframe['prev_close'] < dataframe['prev_open']) &  # previous candle bearish
            (dataframe['close'] > dataframe['open']) &            # current candle bullish
            (dataframe['open'] < dataframe['prev_close']) &       # engulf condition
            (dataframe['close'] > dataframe['prev_open'])
        ).astype(int)
        # Define a bearish engulfing pattern
        dataframe['bearish_engulfing'] = (
            (dataframe['prev_close'] > dataframe['prev_open']) &  # previous candle bullish
            (dataframe['close'] < dataframe['open']) &            # current candle bearish
            (dataframe['open'] > dataframe['prev_close']) &       # engulf condition
            (dataframe['close'] < dataframe['prev_open'])
        ).astype(int)
        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter a long position upon detecting a bullish engulfing candle
        dataframe.loc[
            (dataframe['bullish_engulfing'] == 1),
            ['enter_long', 'enter_tag']
        ] = (1, 'bullish_engulfing')
        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit the long position upon detecting a bearish engulfing candle
        dataframe.loc[
            (dataframe['bearish_engulfing'] == 1),
            ['exit_long', 'exit_tag']
        ] = (1, 'bearish_engulfing_exit')
        return dataframe