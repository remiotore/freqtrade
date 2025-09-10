
from pandas.core.dtypes.missing import notna, notnull
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy

class heikin_419(IStrategy):

    timeframe = '5m'

    minimal_roi = {
        "5": 0.004,
        "15": 0.008,
        "25": 0.013,
        "31": 0
    }
    stoploss = -0.031

    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 500

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }



























    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:   



        dataframe['ohlc4']=(dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ohlc4-1']=(dataframe['open'].shift(1) + dataframe['high'].shift(1) + dataframe['low'].shift(1) + dataframe['close'].shift(1)) / 4
        dataframe['hlc3']=(dataframe['high'] + dataframe['low'] + dataframe['close']) / 3


        dataframe['ohlc4-1']=dataframe['ohlc4-1'].fillna(dataframe['ohlc4'])
        dataframe['haOpen'] = (dataframe['ohlc4'] + dataframe['ohlc4-1']) / 2


       
        dataframe['haC']= (dataframe['ohlc4'] + dataframe['haOpen'] + dataframe[['high', 'haOpen']].max(axis=1) + dataframe[['low', 'haOpen']].min(axis=1) ) / 4

        dataframe['ema1'] = ta.EMA(dataframe['haC'], timeperiod=15)

        dataframe['ema2'] = ta.EMA(dataframe['ema1'], timeperiod=15)

        dataframe['ema3'] = ta.EMA(dataframe['ema2'], timeperiod=15)

        dataframe['TMA1'] = 3 * dataframe['ema1'] - 3 * dataframe['ema2'] + dataframe['ema3']

        dataframe['ema4'] = ta.EMA(dataframe['TMA1'], timeperiod=15)

        dataframe['ema5'] = ta.EMA(dataframe['ema4'], timeperiod=15)

        dataframe['ema6'] = ta.EMA(dataframe['ema5'], timeperiod=15)

        dataframe['TMA2'] = 3 * dataframe['ema4'] - 3 * dataframe['ema5'] + dataframe['ema6']

        dataframe['ipek'] = dataframe['TMA1'] - dataframe['TMA2']

        dataframe['yasin'] = dataframe['TMA1'] - dataframe['ipek']

        dataframe['ema7'] = ta.EMA(dataframe['hlc3'], timeperiod=15)

        dataframe['ema8'] = ta.EMA(dataframe['ema7'], timeperiod=15)

        dataframe['ema9'] = ta.EMA(dataframe['ema8'], timeperiod=15)

        dataframe['TMA3'] = 3 * dataframe['ema7'] - 3 * dataframe['ema8'] + dataframe['ema9']

        dataframe['ema10'] = ta.EMA(dataframe['TMA3'], timeperiod=15)

        dataframe['ema11'] = ta.EMA(dataframe['ema10'], timeperiod=15)

        dataframe['ema12'] = ta.EMA(dataframe['ema11'], timeperiod=15)

        dataframe['TMA4'] = 3 * dataframe['ema10'] - 3 * dataframe['ema11'] + dataframe['ema12']

        dataframe['ipek1'] = dataframe['TMA3'] - dataframe['TMA4']

        dataframe['yasin1'] = dataframe['TMA3'] - dataframe['ipek1']






        dataframe['ohlc4']=(dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['hlc3']=(dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['hl2']=(dataframe['high'] + dataframe['low'] ) / 2
        dataframe['hma16'] = qtpylib.hma(dataframe['ohlc4'], 20)
        dataframe['hma8'] = qtpylib.hma(dataframe['hl2'], 8)
        return dataframe
        

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['hma16']))&
                (dataframe['yasin1'] > dataframe['yasin']) &
                (dataframe['yasin1'].shift(1) <= dataframe['yasin'].shift(1))
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['yasin1'] < dataframe['yasin']) &
                (dataframe['yasin1'].shift(1) >= dataframe['yasin'].shift(1))
            ),
            'sell'] = 0
        return dataframe
