



import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)


from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Pivot_tuned(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = '5m'
    inf_timeframe = '1d'

    can_short: bool = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    minimal_roi = {
        "0": 0.33
    }

    stoploss = -0.1

    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True

    process_only_new_candles = False
    use_exit_signal = True

    startup_candle_count: int = 100

    ma_period = IntParameter(4, 32, default=15, space='buy', optimize=True, load=True)
    rsi_long = IntParameter(10, 90, default=79, space='buy', optimize=True, load=True)
    rsi_short = IntParameter(10, 90, default=34, space='buy', optimize=True, load=True)
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]

        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for _ma_period in self.ma_period.range:
            dataframe[f'ema_{_ma_period}'] = ta.EMA(dataframe, timeperiod=_ma_period)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        informative = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.inf_timeframe)
        informative["pivot"] = ((informative["close"] + informative["high"] + informative["low"]) / 3)
        informative["r1"] = 2 * informative["pivot"] - informative["low"]
        informative["s1"] = 2 * informative["pivot"] - informative["high"]
        informative["r2"] = informative["pivot"] + informative["r1"] - informative["s1"]
        informative["s2"] = informative["pivot"] - informative["r1"] + informative["s1"]

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True
        )

        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (qtpylib.crossed_above(dataframe[f'ema_{self.ma_period.value}'], dataframe[f"r2_{self.inf_timeframe}"])) &
            (dataframe['rsi'] < self.rsi_long.value) &
            (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']] = (1, 'pivot_cross_2')





        
        dataframe.loc[
            (qtpylib.crossed_below(dataframe[f'ema_{self.ma_period.value}'], dataframe[f"s2_{self.inf_timeframe}"])) &
            (dataframe['rsi'] > self.rsi_short.value) &
            (dataframe['volume'] > 0),
            ['enter_short', 'enter_tag']] = (1, 'pivot_cross_2')





            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'exit')











        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:

        return 1
    
    @property
    def plot_config(self):
        return {
            'main_plot': {
                f"pivot_{self.inf_timeframe}": {},
                f"r1_{self.inf_timeframe}": {},
                f"s1_{self.inf_timeframe}": {},
                f"r2_{self.inf_timeframe}": {},
                f"s2_{self.inf_timeframe}": {},
            }
        }
    