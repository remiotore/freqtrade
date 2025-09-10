import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta

class VolumeSupertrendStrategy(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.10
    timeframe = '5m'
    inf_timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        period = 7
        factor = 3

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=period)

        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=period)

        dataframe['vol_factor'] = dataframe['volume'] / dataframe['volume_sma']

        dataframe['upperband'] = ((dataframe['high'] + dataframe['low']) / 2) + (factor * dataframe['atr'] * dataframe['vol_factor'])
        dataframe['lowerband'] = ((dataframe['high'] + dataframe['low']) / 2) - (factor * dataframe['atr'] * dataframe['vol_factor'])

        dataframe['in_uptrend'] = True
        dataframe['in_uptrend'] = np.where(dataframe['close'] > dataframe['lowerband'], True, dataframe['in_uptrend'])
        dataframe['in_uptrend'] = np.where(dataframe['close'] < dataframe['upperband'], False, dataframe['in_uptrend'])

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['in_uptrend'] == True) &
            (dataframe['in_uptrend'].shift(1) == False),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['in_uptrend'] == False) &
            (dataframe['in_uptrend'].shift(1) == True),
            'sell'] = 1
        return dataframe
