import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class yoyo_action_zone_strategy(IStrategy):


    minimal_roi = {
        "0": 10
    }

    timeframe = '4h'

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }
    
    emaFast = 6
    emaSlow = 18
    rsiPeriod = 14
    overBought = 80
    overSold = 30

    stoploss = -0.10

    trailing_stop = False

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        atrFast = 6
        atrFM = 0.5 # fast ATR multiplier

        atrSlow = 18 # Slow ATR perod
        atrSM = 2 # Slow ATR multiplier



        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.emaFast)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.emaSlow)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsiPeriod)
        dataframe['macd'] = dataframe['ema_fast'] - dataframe['ema_slow']
        dataframe['bullish'] = dataframe['macd'] > 0
        dataframe['bearish'] = dataframe['macd'] < 0

        dataframe.dropna(inplace=True)
        dataframe.head()

        dataframe['green'] = False
        dataframe.loc[
        (
            (dataframe['bearish'] & dataframe['close'] > dataframe['ema_fast'])
        ),
        'green'] = True

        dataframe['blue'] = False
        dataframe.loc[
        (
            dataframe['bearish'] & (dataframe['close'] > dataframe['ema_fast']) & (dataframe['close'] < dataframe['ema_slow'])
        ),
        'blue'] = True

        dataframe['yellow'] = False
        dataframe.loc[
        (
            dataframe['bearish'] & (dataframe['close'] < dataframe['ema_fast']) & (dataframe['close'] > dataframe['ema_slow'])
        ),
        'yellow'] = True

        dataframe['brown'] = False
        dataframe.loc[
                (
                    dataframe['bearish'] & (dataframe['close'] < dataframe['ema_fast']) & (dataframe['close'] < dataframe['ema_slow'])
                ),
                'brown'] = True

        dataframe['red'] = False
        dataframe.loc[
                (
                    dataframe['bearish'] & dataframe['close'] < dataframe['ema_fast']
                ),
                'red'] = True

        dataframe['sl1'] = atrFM*ta.ATR(dataframe.high, dataframe.low, dataframe.close,timeperiod=atrFast)  # Stop Loss
        dataframe['sl2'] = atrSM*ta.ATR(dataframe.high, dataframe.low, dataframe.close,timeperiod=atrSlow) 

        dataframe['long'] = False
        dataframe['preBuy'] = False
        dataframe['short'] = False
        dataframe['preSell'] = False
        dataframe['trail2'] = 0.0

        """
        Buy = bullish and bearish[1]
        PreBuy = Blue and Blue[1] and Blue[2] and Blue[3] and mainSource<mainSource[2]
        BuyMore = barssince(bullish)<26 and Yellow and mainSource==lowest(mainSource,9)
        Sell = bearish and bullish[1]
        PreSell = Yellow and barssince(Buy)>25 and mainSource<mainSource[2]
        SellMore = Yellow and barssince(Yellow)>2 and mainSource<mainSource[2]
        """
        for index in range(len(dataframe)):

            if dataframe.iloc[index].close > dataframe.iloc[index - 1].trail2 and dataframe.iloc[index - 1].close > dataframe.iloc[index - 1].trail2:
                dataframe.trail2.iloc[index] = max(dataframe.iloc[index - 1].trail2, dataframe.iloc[index].close - dataframe.iloc[index].sl2)

            elif dataframe.iloc[index].close < dataframe.iloc[index - 1].trail2 and dataframe.iloc[index - 1].close < dataframe.iloc[index - 1].trail2: 
                dataframe.trail2.iloc[index] = min(dataframe.iloc[index - 1].trail2, dataframe.iloc[index - 1].close +  dataframe.iloc[index - 1].sl2)

            elif dataframe.iloc[index].close > dataframe.iloc[index - 1].trail2:
                dataframe.trail2.iloc[index] = dataframe.iloc[index].close - dataframe.iloc[index].sl2
            else:
                dataframe.trail2.iloc[index] = dataframe.iloc[index].close + dataframe.iloc[index].sl2

            dataframe.long.iloc[index] = dataframe.bullish.iloc[index] and dataframe.bullish.iloc[index - 1]
            dataframe.preBuy.iloc[index] = dataframe.bullish.iloc[index] and dataframe.bullish.iloc[index - 1]

            dataframe.short.iloc[index] = dataframe.bearish.iloc[index] and dataframe.bearish.iloc[index - 1]

        dataframe['greenLine'] = False
        dataframe.loc[
                (
                    dataframe["greenLine"] & (dataframe["close"] > dataframe['trail2'])
                ),
                'greenLine'] = True
        dataframe['greenLine_last'] = dataframe.greenLine.shift(-1)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

             (dataframe["long"] | dataframe["green"] & dataframe['greenLine']) | (dataframe["blue"] & dataframe['greenLine']) # later          
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

             dataframe['short'].shift(-1) == 1 & ((dataframe['greenLine_last'] & dataframe['greenLine'] == False) | dataframe["red"]) # later          
            ),
            'sell'] = 1
        return dataframe