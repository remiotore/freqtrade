import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class yoyo_action_strategy(IStrategy):


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


    emaFast = 24
    emaSlow = 112
    rsiPeriod = 14
    overBought = 80
    overSold = 30


    atrFast = 6
    atrFM = 0.5 # fast ATR multiplier

    atrSlow = 18 # Slow ATR perod
    atrSM = 2 # Slow ATR multiplier

    trailing_stop = False

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ohlc4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.emaFast)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.emaSlow)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsiPeriod)
        dataframe['macd'] = dataframe['ema_fast'] - dataframe['ema_slow']
        dataframe['bullish'] = dataframe['macd'] > 0
        dataframe['bearish'] = dataframe['macd'] < 0

        dataframe['sl1'] = self.atrFM*ta.ATR(dataframe.high, dataframe.low, dataframe.close,timeperiod=self.atrFast)  # Stop Loss
        dataframe['sl2'] = self.atrSM*ta.ATR(dataframe.high, dataframe.low, dataframe.close,timeperiod=self.atrSlow) 
        dataframe.dropna(inplace=True)

        dataframe['red'] = False
        dataframe['brown'] = False
        dataframe['yellow'] = False
        dataframe['blue'] = False
        dataframe['green'] = False
        dataframe['long'] = False
        dataframe['preBuy'] = False
        dataframe['short'] = False
        dataframe['preSell'] = False
        dataframe['trail2'] = 0.0
        
        for index in range(len(dataframe)):

            dataframe.green.iloc[index] = dataframe.bullish.iloc[index] and (dataframe.ohlc4.iloc[index] > dataframe.ema_fast.iloc[index])

            dataframe.blue.iloc[index] = dataframe.bearish.iloc[index] and dataframe.ohlc4.iloc[index] > dataframe.ema_fast.iloc[index]

            dataframe.yellow.iloc[index] = dataframe.bullish.iloc[index] and dataframe.ohlc4.iloc[index] < dataframe.ema_slow.iloc[index]

            dataframe.brown.iloc[index] = dataframe.bullish.iloc[index] and dataframe.ohlc4.iloc[index] < dataframe.ema_fast.iloc[index] and dataframe.ohlc4.iloc[index] < dataframe.ema_slow.iloc[index]

            dataframe.red.iloc[index] = dataframe.bearish.iloc[index] and dataframe.ohlc4.iloc[index] < dataframe.ema_fast.iloc[index]

            if dataframe.close.iloc[index] > dataframe.trail2.iloc[index - 1] and dataframe.close.iloc[index - 1] > dataframe.trail2.iloc[index - 1]:
                dataframe.trail2.iloc[index] = max(dataframe.trail2.iloc[index - 1], dataframe.close.iloc[index] - dataframe.sl2.iloc[index])

            elif dataframe.close.iloc[index] < dataframe.trail2.iloc[index - 1] and dataframe.close.iloc[index - 1] < dataframe.trail2.iloc[index - 1]: 
                dataframe.trail2.iloc[index] = min(dataframe.trail2.iloc[index - 1], dataframe.close.iloc[index - 1] +  dataframe.sl2.iloc[index - 1])

            elif dataframe.close.iloc[index] > dataframe.trail2.iloc[index - 1]:
                dataframe.trail2.iloc[index] = dataframe.close.iloc[index] - dataframe.sl2.iloc[index]
            else:
                dataframe.trail2.iloc[index] = dataframe.close.iloc[index] + dataframe.sl2.iloc[index]

            dataframe.long.iloc[index] = dataframe.bullish.iloc[index] and dataframe.bullish.iloc[index - 1]
            dataframe.preBuy.iloc[index] = dataframe.bullish.iloc[index] and dataframe.bullish.iloc[index - 1]

            dataframe.short.iloc[index] = dataframe.bearish.iloc[index] and dataframe.bearish.iloc[index - 1]

        dataframe['greenLine'] = False
        dataframe.loc[
                (
                    (dataframe["close"] > dataframe['trail2'])
                ),
                'greenLine'] = True
        dataframe['greenLine_last'] = dataframe.greenLine.shift(1)

        dataframe['short_last'] = dataframe.short.shift(1)
        dataframe['green_last'] = dataframe.green.shift(1)
        dataframe['red_last'] = dataframe.red.shift(1)
        dataframe['hold_state'] = False
        dataframe.dropna(inplace=True)
        dataframe

        dataframe['greenLine'] = False
        dataframe.loc[
                (
                    (dataframe["close"] > dataframe['trail2'])
                ),
                'greenLine'] = True
        dataframe['greenLine_last'] = dataframe.greenLine.shift(1)
        dataframe['short_last'] = dataframe.short.shift(1)
        dataframe['green_last'] = dataframe.green.shift(1)
        dataframe['red_last'] = dataframe.red.shift(1)
        dataframe['hold_state'] = False
        dataframe.dropna(inplace=True)

        dataframe.loc[(
            ((dataframe['green_last'] == False) & (dataframe['green'] == True)) # Green buy
            | ((dataframe['greenLine'] == True) & (dataframe['blue'] == True)) # Over ATR and blue
        ), 'signal_buy'] = True

        dataframe.loc[(
            ((dataframe['red_last'] == False) & (dataframe['red'] == True)) # Red Sell

        ), 'signal_sell'] = True

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['signal_buy'] == True) , 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['signal_sell'] == True), 'sell'] = 1
        return dataframe