
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)
from typing import Dict, List
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class WaveTrendStra_v0(IStrategy):
    """
    author@: Gert Wohlgemuth

    just a skeleton
    """

    minimal_roi = {
        "0": 0.757,
        "1100": 0.22,
        "2257": 0.073,
        "6699": 0
    }

    stoploss = -0.269

    timeframe = '4h'

    trailing_stop = True
    trailing_stop_positive = 0.312
    trailing_stop_positive_offset = 0.385
    trailing_only_offset_is_reached = False

    ema_period = IntParameter(5, 50, default=23, space='buy')
    constant = DecimalParameter(0.001, 0.03, default=0.03, space='buy')
    tci_ema_period = IntParameter(5, 50, default=35, space='buy')
    sma_period = IntParameter(2, 20, default=18, space='buy')
    rsi_period = IntParameter(5, 50, default=7, space='buy')
    macd_fastperiod = IntParameter(5, 50, default=7, space='buy')
    macd_slowperiod = IntParameter(10, 100, default=52, space='buy')
    macd_signalperiod = IntParameter(5, 50, default=22, space='buy')
    adx_period = IntParameter(5, 50, default=33, space='buy')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        ap = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["ap"] = ap
        dataframe['volume_ema'] = ta.EMA(dataframe['volume'], timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ap = dataframe["ap"]
        
        esa = ta.EMA(ap, int(self.ema_period.value))
        d = ta.EMA(abs(ap - esa), int(self.ema_period.value))
        ci = (ap - esa) / (self.constant.value * d)
        tci = ta.EMA(ci, int(self.tci_ema_period.value))

        dataframe["wt1"] = tci
        dataframe["wt2"] = ta.SMA(dataframe["wt1"], int(self.sma_period.value))

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=int(self.rsi_period.value))
        macd = ta.MACD(dataframe, fastperiod=int(self.macd_fastperiod.value), 
                       slowperiod=int(self.macd_slowperiod.value), 
                       signalperiod=int(self.macd_signalperiod.value))
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=int(self.adx_period.value))

        dataframe.loc[
            (qtpylib.crossed_above(dataframe["wt1"], dataframe["wt2"])) &
            (dataframe['rsi'] > 50) &
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['volume'] > dataframe['volume_ema']) &
            (dataframe['atr'] > dataframe['atr'].mean()) &
            (dataframe['adx'] > 25),  # ADX filter for trend strength
            'buy'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['wt1'], dataframe['wt2'])),
            'sell'] = 1
        return dataframe
