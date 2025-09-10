
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from datetime import datetime

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Scalp_7(IStrategy):
    """
        this strategy is based around the idea of generating a lot of potentatils entrys and make tiny profits on each trade

        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses.

        Recommended is to only exit based on ROI for this strategy
    """


    minimal_roi = {
        "0": 0.01
    }




    entryControlDict = {}

    stoploss = -0.04


    timeframe = '1m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=5, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=5, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=5, price='low')
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] < dataframe['ema_low']) &
                (dataframe['adx'] > 30) &
                (
                    (dataframe['fastk'] < 30) &
                    (dataframe['fastd'] < 30) &
                    (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
                )
            ),
            'enter_long'] = 1
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:

        if pair not in self.entryControlDict:
            self.entryControlDict[pair] = []

        self.entryControlDict[pair].append(current_time)
        if (len(self.entryControlDict[pair]) < 3):
            return False

        startDatetime = self.entryControlDict[pair][0]
        endDatetime = self.entryControlDict[pair][-1]
        diff = endDatetime - startDatetime
        diff_in_minutes = diff.total_seconds() / 60

        self.entryControlDict[pair] = []
        if (diff_in_minutes > 20):            
            self.entryControlDict[pair].append(current_time)
            return False

        return True

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] >= dataframe['ema_high'])
            ) |
            (
                (qtpylib.crossed_above(dataframe['fastk'], 70)) |
                (qtpylib.crossed_above(dataframe['fastd'], 70))
            ),
            'exit_long'] = 1
        return dataframe
