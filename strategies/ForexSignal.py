


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IStrategy, merge_informative_pair


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ForexSignal(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "60": 0.01,
        "30": 0.03,
        "0": 0.04
    }


    stoploss = -0.03

    trailing_stop = False




    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

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

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()

        informative_pairs = [(pair, '1h') for pair in pairs]

        
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        if not self.dp:

            return dataframe

        inf_tf = '1h'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)
        
        informative['ema8'] = ta.EMA(informative, timeperiod=8)
        informative['ema21'] = ta.EMA(informative, timeperiod=21)





        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        dataframe['ema8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['ema8_1h'] > dataframe['ema21_1h']) &  # Onko nousutrendi
                (dataframe['ema8'] > dataframe['ema13']) &  
                (dataframe['ema13'] > dataframe['ema21']) &
                (dataframe['low'] < dataframe['ema8']) 
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['ema8'] < dataframe['ema8'].shift(1))
            ),
            'sell'] = 1
        return dataframe
