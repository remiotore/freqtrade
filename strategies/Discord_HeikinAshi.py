
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series, concat
# --------------------------------
from freqtrade.strategy import (merge_informative_pair,DecimalParameter, IntParameter, CategoricalParameter)
import numpy as np

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class HeikinAshi(IStrategy):
    """Tradingview heikin ashi smoothed v4
    author@: 
    """

    INTERFACE_VERSION = 2
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = '5m'
    inf_1h = '1h'

    # trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True


    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        informative_1h['ohlc4'] = informative_1h[['open', 'high', 'low', 'close']].mean(axis=1)
        informative_1h['hlc3'] = informative_1h[['high', 'low', 'close']].mean(axis=1)      

        #Heikin Ashi Smoothed V4
        informative_1h['has'] = heikin_ashi_smoothed(informative_1h)

        return informative_1h

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            (
                (dataframe['has_1h'] == 1) &
                (dataframe['volume'] > 0)
            )
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            (
                (dataframe['has_1h'] == -1) &
                (dataframe['volume'] > 0)
            )
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1
        return dataframe

def heikin_ashi_smoothed(df, EMAlength=55):
    df = df.copy()
    df = df.fillna(0)

    haOpen = df['ohlc4'].rolling(2, min_periods=1).mean().fillna(0)
    haC = df['ohlc4'] + haOpen \
        + concat([df['high'], haOpen]).max() \
        + concat([df['low'], haOpen]).min()
    haC *= 0.25
    EMA1 = ta.EMA(haC, timeperiod=EMAlength)
    EMA2 = ta.EMA(EMA1, timeperiod=EMAlength)
    EMA3 = ta.EMA(EMA2, timeperiod=EMAlength)
    TMA1 = 3 * EMA1 - 3 * EMA2 + EMA3
    EMA4 = ta.EMA(TMA1, timeperiod=EMAlength)
    EMA5 = ta.EMA(EMA4, timeperiod=EMAlength)
    EMA6 = ta.EMA(EMA5, timeperiod=EMAlength)
    TMA2 = 3 * EMA4 - 3 * EMA5 + EMA6
    IPEK = TMA1 - TMA2
    YASIN = TMA1 + IPEK
    EMA7 = ta.EMA(df['hlc3'], timeperiod=EMAlength)
    EMA8 = ta.EMA(EMA7, timeperiod=EMAlength)
    EMA9 = ta.EMA(EMA8, timeperiod=EMAlength)
    TMA3 = 3 * EMA7 - 3 * EMA8 + EMA9
    EMA10 = ta.EMA(TMA3, timeperiod=EMAlength)
    EMA11 = ta.EMA(EMA10, timeperiod=EMAlength)
    EMA12 = ta.EMA(EMA11, timeperiod=EMAlength)
    TMA4 = 3 * EMA10 - 3 * EMA11 + EMA12
    IPEK1 = TMA3 - TMA4
    YASIN1 = TMA3 + IPEK1

    mavi = YASIN1
    kirmizi = YASIN

    # Signal trade
    longCond = np.logical_and(mavi > kirmizi, shift(mavi, -1) <= shift(kirmizi, -1))
    shortCond = np.logical_and(mavi < kirmizi, shift(mavi, -1) >= shift(kirmizi, -1))
    
    state = 0
    for i, _ in enumerate(longCond):
        if(longCond[i]):
            state += 1
        if(shortCond[i]):
            state -= 1
        if(state > 1):
            longCond[i] = 0
        elif(state < 0):
            shortCond[i] = 0


    #import pdb; pdb.set_trace()
    return Series(longCond.astype(int) - shortCond.astype(int), name ='has')
    

def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e
