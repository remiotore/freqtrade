
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

class Scalp_2(IStrategy):
    """
        this strategy is based around the idea of generating a lot of potentatils buys and make tiny profits on each trade
        we recommend to have at least 60 parallel trades at any time to cover non avoidable losses.
        Recommended is to only sell based on ROI for this strategy
        Backtest result on top 20 coins in 2022 from 20210131-20211231
        - timeframe, winrate, roi avg profit, stop_loss avg profit, total profit
        - 1d, 50%, 1%, -4.19%, -2.39%
        - 4h, 67.3%, 1%, -4.19%, -9.18%
        - 30m, 67.8%, 1%, -4.19%, -25.41%
        - 5m, 77.5%, 1%, -4.19%, -63.49%
        - 1m, 77.5%, 1%, -4.19%, -93.14%
        - hyperopt 
        - hyperopt 1m, 46%, 0.54%, -33.63%, 82.66%
    """


    minimal_roi = {
        "0": 0.015
    }




    stoploss = -0.03


    timeframe = '1m'

    buy_fastd = IntParameter(low=10, high=40, default=30, space='buy')
    buy_fastk = IntParameter(low=10, high=40, default=30, space='buy')
    buy_adx = IntParameter(low=10, high=40, default=30, space='buy')

    sell_fastd = IntParameter(low=60, high=90, default=70, space='sell')
    sell_fastk = IntParameter(low=60, high=90, default=70, space='sell')

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

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] < dataframe['ema_low']) &
                (dataframe['adx'] > self.buy_adx.value) &
                (
                    (dataframe['fastk'] < self.buy_fastk.value) &
                    (dataframe['fastd'] < self.buy_fastd.value) &
                    (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd']))
                )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['open'] >= dataframe['ema_high'])
            ) |
            (
                (qtpylib.crossed_above(dataframe['fastk'], self.sell_fastk.value)) |
                (qtpylib.crossed_above(dataframe['fastd'], self.sell_fastd.value))
            ),
            'sell'] = 1
        return dataframe