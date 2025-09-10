
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Roth03_2(IStrategy):


    buy_params = {
        'adx-enabled': False,
        'adx-value': 50,
        'cci-enabled': False,
        'cci-value': -196,
        'fastd-enabled': True,
        'fastd-value': 37,
        'mfi-enabled': True,
        'mfi-value': 20,
        'rsi-enabled': False,
        'rsi-value': 26,
        'trigger': 'bb_lower'
    }

    sell_params = {
        'sell-adx-enabled': False,
        'sell-adx-value': 73,
        'sell-cci-enabled': False,
        'sell-cci-value': 189,
        'sell-fastd-enabled': True,
        'sell-fastd-value': 79,
        'sell-mfi-enabled': True,
        'sell-mfi-value': 86,
        'sell-rsi-enabled': True,
        'sell-rsi-value': 69,
        'sell-trigger': 'sell-sar_reversal'
    }

    minimal_roi = {
        "0": 0.24553,
        "33": 0.07203,
        "90": 0.01452,
        "111": 0
    }

    stoploss = -0.31939

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_low'] = bollinger['lower']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_perc'] = (dataframe['close'] - dataframe['bb_low']) / (
                    dataframe['bb_upper'] - dataframe['bb_low'])

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_low']) &
                (dataframe['fastd'] > 37) &
                (dataframe['mfi'] < 20.0)

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
                (dataframe['sar'] > dataframe['close']) &




                (dataframe['rsi'] > 69) &
                (dataframe['mfi'] > 86) &
                (dataframe['fastd'] > 79)
            ),
            'sell'] = 1

        return dataframe
