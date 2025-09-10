from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy

class trend(IStrategy):
    INTERFACE_VERSION = 3
    INTERFACE_VERSION: int = 3

    minimal_roi = {'0': 0.166, '10': 0.024, '36': 0.011, '93': 0}


    stoploss = -0.299
    can_short = True

    trailing_stop = True
    trailing_stop_positive = 0.052
    trailing_stop_positive_offset = 0.147
    trailing_only_offset_is_reached = False
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(dataframe['close'] > dataframe['trend']) & (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) & (dataframe['obv'] > dataframe['obv'].shift(1)), 'enter_long'] = 1

        dataframe.loc[(dataframe['close'] < dataframe['trend']) & (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) & (dataframe['obv'] < dataframe['obv'].shift(1)), 'enter_short'] = -1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[(dataframe['close'] < dataframe['trend']) & (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) & (dataframe['obv'] > dataframe['obv'].shift(1)), 'exit_long'] = 1

        dataframe.loc[(dataframe['close'] > dataframe['trend']) & (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) & (dataframe['obv'] < dataframe['obv'].shift(1)), 'exit_short'] = 1
        return dataframe