
import numpy as np
import pandas as pd
from functools import reduce
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, RealParameter, informative, merge_informative_pair
import pandas_ta as ta
from datetime import datetime
from freqtrade.persistence import Trade



class VWAPStrategy(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '5m'

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.07 # 7% as default

    take_profit = 1.5

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = False

    use_custom_stoploss = True


    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, dataframe: DataFrame, **kwargs) -> float:
        
        dataframe['ATR'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=7)

        stoploss = 1.2*dataframe['ATR'][-1]

        return stoploss

    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.rsi(dataframe['close'], length=16)

        dataframe.set_index(pd.DatetimeIndex(dataframe["date"]), inplace=True)
        dataframe['VWAP'] = ta.vwap(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'], anchor='D', offset=None)

        b_bands = ta.bbands(dataframe['close'], length=14, std=2.0)
        dataframe=dataframe.join(b_bands)

        VWAP_signal = [0]*len(dataframe)
        backcandles = 15

        for row in range(backcandles, len(dataframe)):
            up_trend = 1
            down_trend = 1
            for i in range(row-backcandles, row+1):
                if max(dataframe['open'][i], dataframe['close'][i]) >= dataframe['VWAP'][i]:
                    down_trend = 0
                if min(dataframe['open'][i], dataframe['close'][i]) <= dataframe['VWAP'][i]:
                    up_trend = 0
            if up_trend == 1 and down_trend == 1:
                VWAP_signal[row] = 3
            elif up_trend == 1:
                VWAP_signal[row] = 2
            elif down_trend == 1:
                VWAP_signal[row] = 1

        dataframe['VWAP_signal'] = VWAP_signal

        return dataframe
    

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        conditions = []

        conditions.append(
            (dataframe['volume'] > 0) &
            (dataframe['VWAP_signal'] == 1) & 
            (dataframe['close'] <= dataframe['BBL_14_2.0']) & 
            (dataframe['rsi'] < 45)
        )

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        conditions = []

        conditions.append(
            (dataframe['volume'] > 0) &
            (dataframe['VWAP_signal'] == 2) & 
            (dataframe['close'] >= dataframe['BBU_14_2.0']) & 
            (dataframe['rsi'] > 55) &
            (dataframe['rsi'] <= 90)
        )

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
