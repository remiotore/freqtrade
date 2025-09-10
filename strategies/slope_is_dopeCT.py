



















from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter, RealParameter)
from scipy.spatial.distance import cosine
import numpy as np

class slope_is_dopeCT(IStrategy):

    minimal_roi = {
        "0": 0.6
    }

    stoploss = -0.9

    timeframe = '4h'

    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    slope_length = IntParameter(5, 20, default=11, optimize=True)
    stoploss_length = IntParameter(5, 15, default=10, optimize=True)
    rsi_buy = IntParameter(30, 60, default=55, space="buy", optimize=True)
    fslope_buy = IntParameter(-5, 5, default=0, space="buy", optimize=True)
    sslope_buy = IntParameter(-5, 5, default=0, space="buy", optimize=True)
    fslope_sell = IntParameter(-5, 5, default=0, space="sell", optimize=True)

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.28

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 1,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": True
            })

        return prot
        

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['marketMA'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['fastMA'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['slowMA'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['entryMA'] = ta.SMA(dataframe, timeperiod=3)


        dataframe['sy1'] = dataframe['slowMA'].shift(+(self.slope_length.value))
        dataframe['sy2'] = dataframe['slowMA'].shift(+1)
        sx1 = 1
        sx2 = (self.slope_length.value)
        dataframe['sy'] = dataframe['sy2'] - dataframe['sy1']
        dataframe['sx'] = sx2 - sx1
        dataframe['slow_slope'] = dataframe['sy']/dataframe['sx']
        dataframe['fy1'] = dataframe['fastMA'].shift(+(self.slope_length.value))
        dataframe['fy2'] = dataframe['fastMA'].shift(+1)
        fx1 = 1
        fx2 = (self.slope_length.value)
        dataframe['fy'] = dataframe['fy2'] - dataframe['fy1']
        dataframe['fx'] = fx2 - fx1
        dataframe['fast_slope'] = dataframe['fy']/dataframe['fx']


        dataframe['last_lowest'] = dataframe['low'].rolling((self.stoploss_length.value)).min().shift(1)

        return dataframe

    plot_config = {
        "main_plot": {

            "fastMA": {"color": "red"},
            "slowMA": {"color": "blue"},
        },
        "subplots": {

            "rsi": {"rsi": {"color": "blue"}},
            "fast_slope": {"fast_slope": {"color": "red"}, "slow_slope": {"color": "blue"}},
        },
    }


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (


                (dataframe['fast_slope'] > self.fslope_buy.value) &

                (dataframe['slow_slope'] > self.sslope_buy.value) &



                (dataframe['close'] > dataframe['close'].shift(+(self.slope_length.value))) &

                (dataframe['rsi'] > self.rsi_buy.value) 




                )
            ),
            'buy'] = 1


        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['fast_slope'] < self.fslope_sell.value)


                | (dataframe['close'] < dataframe['last_lowest'])

            ),
            'sell'] = 1
        return dataframe
























































































































































































































































































