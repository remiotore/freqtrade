
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from typing import Dict, List
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open



# --------------------------------
class asem1(IStrategy):


        ########################################
#
# Ichimoku Cloud
#
    def ichimoku(
         dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30
        ):
        df = dataframe.copy()
        
        tenkan_sen = (
            dataframe["high"].rolling(window=conversion_line_period).max()
            + dataframe["low"].rolling(window=conversion_line_period).min()
              ) / 2

        kijun_sen = (
                  dataframe["high"].rolling(window=base_line_periods).max()
                + dataframe["low"].rolling(window=base_line_periods).min()
                   ) / 2

        leading_senkou_span_a = (tenkan_sen + kijun_sen) / 2

        leading_senkou_span_b = (
                  dataframe["high"].rolling(window=laggin_span).max()
              + dataframe["low"].rolling(window=laggin_span).min()
           ) / 2

        senkou_span_a = leading_senkou_span_a.shift(displacement - 1)

        senkou_span_b = leading_senkou_span_b.shift(displacement - 1)

        chikou_span = dataframe["close"].shift(-displacement + 1)

        cloud_green = senkou_span_a > senkou_span_b
        cloud_red = senkou_span_b > senkou_span_a

        return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "leading_senkou_span_a": leading_senkou_span_a,
        "leading_senkou_span_b": leading_senkou_span_b,
        "chikou_span": chikou_span,
        "cloud_green": cloud_green,
        "cloud_red": cloud_red,
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ma200'] = ta.MA(dataframe, timeperiod=200)
        dataframe['tenkan'] = df.ichimoku(dataframe)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                   (dataframe['leading_senkou_span_a'] < dataframe['close']) 

            ),
            'buy'] = 1
        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

            ),
            'sell'] = 1

        return dataframe
