from datetime import datetime
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as ta
from typing import Optional


class CDCActionZone(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = '1h'

    minimal_roi = {
        "0": 0.05,
        "60": 0.03,
        "120": 0.02,
        "240": 0.01,
    }

    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    can_short = True

    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['FastMA'] = ta.ema(dataframe['close'], 12)
        dataframe['SlowMA'] = ta.ema(dataframe['close'], 26)
        dataframe['Bull'] = (dataframe['FastMA'] > dataframe['SlowMA']).astype(int)
        dataframe['Bear'] = (dataframe['FastMA'] < dataframe['SlowMA']).astype(int)
        dataframe['Green'] = ((dataframe['Bull'] == 1) & (dataframe['close'] > dataframe['FastMA'])).astype(int)
        dataframe['Blue'] = ((dataframe['Bear'] == 1) & (dataframe['close'] > dataframe['FastMA']) & (dataframe['close'] > dataframe['SlowMA'])).astype(int)
        dataframe['LBlue'] = ((dataframe['Bear'] == 1) & (dataframe['close'] > dataframe['FastMA']) & (dataframe['close'] < dataframe['SlowMA'])).astype(int)
        dataframe['Red'] = ((dataframe['Bear'] == 1) & (dataframe['close'] < dataframe['FastMA'])).astype(int)
        dataframe['Orange'] = ((dataframe['Bull'] == 1) & (dataframe['close'] < dataframe['FastMA']) & (dataframe['close'] < dataframe['SlowMA'])).astype(int)
        dataframe['Yellow'] = ((dataframe['Bull'] == 1) & (dataframe['close'] < dataframe['FastMA']) & (dataframe['close'] > dataframe['SlowMA'])).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['Green'] == 1) |
                (dataframe['LBlue'] == 1) |
                (dataframe['Orange'] == 1)
            ),
            'entry_trend'
        ] = 'enter_long'

        dataframe.loc[
            (
                (dataframe['Blue'] == 1) |
                (dataframe['Red'] == 1) |
                (dataframe['Yellow'] == 1)
            ),
            'entry_trend'
        ] = 'enter_short'

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['Green'] == 1) |
                (dataframe['LBlue'] == 1) |
                (dataframe['Orange'] == 1)
            ),
            'exit_trend'
        ] = 'exit_long'

        dataframe.loc[
            (
                (dataframe['Blue'] == 1) |
                (dataframe['Red'] == 1) |
                (dataframe['Yellow'] == 1)
            ),
            'exit_trend'
        ] = 'exit_short'

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                  proposed_leverage: float, max_leverage: float, side: str,
                  **kwargs) -> float:
         """
         Customize leverage for each new trade. This method is only called in futures mode.

         :param pair: Pair that's currently analyzed
         :param current_time: datetime object, containing the current datetime
         :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
         :param proposed_leverage: A leverage proposed by the bot.
         :param max_leverage: Max leverage allowed on this pair
         :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
         :param side: 'long' or 'short' - indicating the direction of the proposed trade
         :return: A leverage amount, which is between 1.0 and max_leverage.
         """
         return 10.0