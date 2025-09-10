
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Cuncontrading(IStrategy):
    """
    Sample strategy implementing Informative Pairs - compares stake_currency with USDT.
    Not performing very well - but should serve as an example how to use a referential pair against USDT.
    author@: xmatthias
    github@: https://github.com/freqtrade/freqtrade-strategies
    How to use it?
    > python3 freqtrade -s InformativeSample
    """


    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }


    stoploss = -0.10

    ticker_interval = '5m'

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    ta_on_candle = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['short'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['long'] = ta.SMA(dataframe, timeperiod=6)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_low'] = bollinger['lower']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']

        dataframe['bb_perc'] = (dataframe['close'] - dataframe['bb_low']) / (dataframe['bb_upper'] - dataframe['bb_low'])
        return dataframe
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    ((dataframe['adx'] > 25) & 
                        qtpylib.crossed_above(dataframe['short'], dataframe['long'])) |
                    (
                        (dataframe['macd'] > 0) & 
                        ((dataframe['macd'] > dataframe['macdsignal']) | 
                            ((dataframe['ao'] > 0) & (dataframe['ao'].shift() < 0))) | 
                        (dataframe['bb_perc'] < 0.1)
                    )
            ),
            'buy'] = 1
        return dataframe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe['adx'] < 25) &
                    (qtpylib.crossed_above(dataframe['long'], dataframe['short']))) |
                (
                        (dataframe['macd'] < 0) & 
                        ((dataframe['macd'] < dataframe['macdsignal']) | 
                            ((dataframe['ao'] < 0) & 
                                (dataframe['ao'].shift() > 0)))
                        (dataframe['close'] > dataframe['high'].rolling(60).max().shift())
                )
            ),
            'sell'] = 1
        return dataframe
        