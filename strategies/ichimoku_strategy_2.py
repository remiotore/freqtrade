from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from technical.indicators import ichimoku

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class ichimoku_strategy_2(IStrategy):
    """
    Ichimoku Strategy
    """

    minimal_roi = {
        "90": 0.04,
        "60": 0.05,
        "30": 0.06
    }

    timeframe = '5m'

    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    ta_on_candle = False

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_buy_signal = False

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    plot_config = {
        'main_plot': {
            'senkou_a': {'color': 'green'},
            'senkou_b': {'color': 'red'},
            'tenkan': {'color': 'orange'},
            'kijun': {'color': 'blue'},
        },
        'subplots': {
            "Moving Avarages": {
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        """

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """

        ichi = ichimoku(dataframe)
        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']
        dataframe['cloud'] = dataframe['senkou_a'] - dataframe['senkou_b']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        (dataframe['close'] > dataframe['senkou_a']) &
        
        
        """

        dataframe.loc[
            (dataframe['tenkan'] > dataframe['kijun']) & # Tekan-sen (Conversion line) boven Kijun-sen (Base line)
            (dataframe['close'] > dataframe['cloud_green']) & # Boven de cloud
            (dataframe['close'] > dataframe['senkou_a']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean()),  # Koop als het volume boven het gemiddelde ligt
            'enter_long'
        ] = 1

        dataframe.loc[dataframe['enter_long'] == 1, 'buy_marker'] = dataframe['close']
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['tenkan'] < dataframe['kijun']) &
            (dataframe['close'] < dataframe['cloud']) &
            (dataframe['close'] < dataframe['senkou_a']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean()),  # Verkoop als het volume boven het gemiddelde ligt
            'exit_long'
        ] = 1

        dataframe.loc[dataframe['exit_long'] == 1, 'sell_marker'] = dataframe['close']
        
        return dataframe








