
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime
from freqtrade.persistence import Trade
from typing import Optional, Union


class VMCipherBDivergencesStrategy(IStrategy):
    """
    This is a custom strategy based on the VuManChu B Divergences indicator from TradingView.
    """
    stoploss = -0.1

    timeframe = '4h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        dataframe['wt1'], dataframe['wt2'] = self.wavetrend(dataframe, 9, 12, 3)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=60)

        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14)
        dataframe['stoch_rsi'] = stoch_rsi['fastk']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """
        dataframe.loc[
            (
                (dataframe['wt1'] < -53) &  # WT oversold
                (dataframe['wt1'] > dataframe['wt2'])  # WT crossing up
            ),
            ['enter_long', 'enter_tag']] = (1, 'long')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """

        dataframe.loc[
            (
                (dataframe['wt1'] > 53) &  # WT overbought
                (dataframe['wt1'] < dataframe['wt2'])  # WT crossing down
            ),
            'exit_long'] = 1
        return dataframe

    def wavetrend(self, dataframe: DataFrame, chlen: int, avg: int, malen: int):
        """
        WaveTrend indicator
        """

        hlc3 = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        esa = ta.EMA(hlc3, timeperiod=chlen)
        de = ta.EMA(abs(hlc3 - esa), timeperiod=chlen)
        ci = (hlc3 - esa) / (0.015 * de)
        wt1 = ta.EMA(ci, timeperiod=avg)
        wt2 = ta.SMA(wt1, timeperiod=malen)
        return wt1, wt2
