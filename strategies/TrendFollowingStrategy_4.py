from functools import reduce

import talib.abstract as ta
from freqtrade.strategy import IStrategy
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class TrendFollowingStrategy_4(IStrategy):

    INTERFACE_VERSION: int = 3

    can_short: bool = True

    minimal_roi = {"0": 0.15, "30": 0.1, "60": 0.05}


    stoploss = -0.2

    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    timeframe = "5m"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()

        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) & 
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)) &
            (dataframe['rsi'] < 30), 
            'enter_long'] = 1

        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) & 
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)) &
            (dataframe['rsi'] > 70), 
            'enter_short'] = -1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) & 
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)), 
            'exit_long'] = 1

        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) & 
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)), 
            'exit_short'] = 1
        
        return dataframe

