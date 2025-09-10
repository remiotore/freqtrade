
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta


class MACD00001Strategy(IStrategy):
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
          "0": 0.1128,
          "22": 0.05443,
          "54": 0.02719,
          "114": 0
          }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.25025

    # Optimal timeframe for the strategy
    timeframe = '3m'

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['adx'] = ta.ADX(dataframe) 
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['di+'] = ta.PLUS_DI(dataframe) 
        dataframe['di-'] = ta.MINUS_DI(dataframe) 

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['macd'] > dataframe['macdsignal']) &
                (dataframe['di+'] > dataframe['di-']) &
                (dataframe['adx'] > 20)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['di-'] > dataframe['di+']) &
                (dataframe['adx'] > 98)
            ),
            'sell'] = 1

        return dataframe
