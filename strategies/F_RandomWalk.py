
from random import choice, seed
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IntParameter, IStrategy

class FRandomWalk(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True

    minimal_roi = {"0": 0.025, "30": 0.01, "90": 0.005, "180": 0.002, "360": 0.001}
    stoploss = -0.03

    timeframe = '15m'
    startup_candle_count = 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['random_walk'] = choice([1, 1, 0, 0, 0, 0, -1, -1, -1])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['random_walk'] == 1),
            'enter_long'] = 1
        dataframe.loc[
            (dataframe['random_walk'] == -1),
            'enter_short'] = 1
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float,
        proposed_leverage: float, max_leverage: float, entry_tag, side: str,
        **kwargs) -> float:
        return 1.0