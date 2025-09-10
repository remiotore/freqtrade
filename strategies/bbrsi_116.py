


from decimal import Decimal
from email.policy import default
import numpy as np  # noqa
from functools import reduce

import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class bbrsi_116(IStrategy):


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.064,
        "28": 0.048,
        "56": 0.031,
        "85": 0
    }

    buy_rsi_value = DecimalParameter(5,50 , default = 5.091)
    buy_rsi_enabled = CategoricalParameter([True, False] , default=False)
    buy_bb_value= CategoricalParameter(['tr_bb_lower_1sd', 'tr_bb_lower_2sd', 'tr_bb_lower_3sd', 'tr_bb_lower_4sd'] , default='tr_bb_lower_1sd')

    sell_rsi_value = DecimalParameter(30,100 , default = 83.571)
    sell_rsi_enabled = CategoricalParameter([True, False] , default=True)
    sell_bb_value = CategoricalParameter(['sell_tr_bb_lower_1sd',
                         'sell_tr_bb_mid_1sd',
                         'sell_tr_bb_upper_1sd'] , default='sell_tr_bb_lower_1sd')


    stoploss =  -0.327

    trailing_stop = False




    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
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
            'bb_upperband': {'color': 'green'},
            'bb_midband': {'color': 'orange'},
            'bb_lowerband': {'color': 'red'},
        },
        'subplots': {
            "RSI": {
                'rsi': {'color': 'yellow'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe)

        bollinger_1sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband_1sd'] = bollinger_1sd['lower']
        dataframe['bb_middleband_1sd'] = bollinger_1sd['mid']
        dataframe['bb_upperband_1sd'] = bollinger_1sd['upper']

        bollinger_2sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband_2sd'] = bollinger_2sd['lower']
        dataframe['bb_middleband_2sd'] = bollinger_2sd['mid']
        dataframe['bb_upperband_2sd'] = bollinger_2sd['upper']

        bollinger_3sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband_3sd'] = bollinger_3sd['lower']
        dataframe['bb_middleband_3sd'] = bollinger_3sd['mid']
        dataframe['bb_upperband_3sd'] = bollinger_3sd['upper']

        bollinger_4sd = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband_4sd'] = bollinger_4sd['lower']
        dataframe['bb_middleband_4sd'] = bollinger_4sd['mid']
        dataframe['bb_upperband_4sd'] = bollinger_4sd['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        conditions =[]
        
        if self.buy_rsi_enabled.value :
            conditions.append((dataframe['rsi'] < self.buy_rsi_value.value))

        if self.buy_bb_value.value == 'tr_bb_lower_1sd':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband_1sd'])
        if self.buy_bb_value.value == 'tr_bb_lower_2sd':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband_2sd'])
        if self.buy_bb_value.value == 'tr_bb_lower_3sd':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband_3sd'])
        if self.buy_bb_value.value == 'tr_bb_lower_4sd':
            conditions.append(dataframe['close'] < dataframe['bb_lowerband_4sd'])




        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        collections =[]

        if self.sell_rsi_enabled.value :
            collections.append((dataframe['rsi'] > self.sell_rsi_value.value))

        if self.sell_bb_value == "sell_tr_bb_lower_1sd":
            collections.append((dataframe['close'] > dataframe['bb_lowerband_1sd']))
        if self.sell_bb_value == "sell_tr_bb_mid_1sd":
            collections.append((dataframe['close'] > dataframe['bb_middleband_1sd']))
        if self.sell_bb_value == "sell_tr_bb_upper_1sd":
            collections.append((dataframe['close'] > dataframe['bb_upperband_1sd']))
        
        if collections:
            dataframe.loc[reduce(lambda x,y: x&y , collections),'sell'] = 1

        return dataframe