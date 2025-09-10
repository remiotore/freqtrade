

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from technical.indicators import ichimoku

def zema(dataframe: dataframe, period, field='ha_close'):
  """
  Source: https://github.com/freqtrade/technical/blob/master/technical/indicators/overlap_studies.py#L79
  Modified slightly to use ta.EMA instead of technical ema
  """
  dataframe['ema1'] = ta.EMA(dataframe[field], timeperiod=period)
  dataframe['ema2'] = ta.EMA(dataframe['ema1'], timeperiod=period)
  dataframe['d'] = df['ema1'] - dataframe['ema2']
  dataframe['zema'] = dataframe['ema1'] + dataframe['d']

return dataframe['zema']

class Ichimoku_v31_Heikin_227(IStrategy):

  minimal_roi = {
    "0": 100
  }

  stoploss = -0.99

  timeframe = '1h'

  inf_tf = '4h'

  process_only_new_candles = True

  use_sell_signal = True
  sell_profit_only = False
  ignore_roi_if_buy_signal = True

  startup_candle_count = 150

  order_types = {
    'buy': 'market',
    'sell': 'market',
    'stoploss': 'market',
    'stoploss_on_exchange': False
  }


  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

    if not self.dp:

      return dataframe

    dataframe= self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe)

    heikinashi = qtpylib.heikinashi(dataframe)

    dataframe['ha_open'] = heikinashi['open']
    dataframe['ha_close'] = heikinashi['close']
    dataframe['ha_high'] = heikinashi['high']
    dataframe['ha_low'] = heikinashi['low']

    dataframe['open'] = heikinashi['open']
    dataframe['close'] = heikinashi['close']
    dataframe['high'] = heikinashi['high']
    dataframe['low'] = heikinashi['low']


    ha_ichi = ichimoku(heikinashi,
      conversion_line_period=20,
      base_line_periods=60,
      laggin_span=120,
      displacement=30
    )

    dataframe['senkou_a'] = ha_ichi['senkou_span_a']
    dataframe['senkou_b'] = ha_ichi['senkou_span_b']
    dataframe['cloud_green'] = ha_ichi['cloud_green']
    dataframe['cloud_red'] = ha_ichi['cloud_red']

    """
    Senkou Span A > Senkou Span B = Cloud Green
    Senkou Span B > Senkou Span A = Cloud Red
    """

    dataframe['fastMA'] = ta.SMA(dataframe, timeperiod=13)
    dataframe['slowMA'] = ta.SMA(dataframe, timeperiod=26)
    dataframe['signalMA'] = ta.SMA(dataframe, timeperiod=9)
    
    dataframe['zema'] = zema(dataframe, 20)






    return dataframe

  def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
      (
        (dataframe['zema'] < dataframe['zema'].shift()) #up
        & dataframe['open'] < dataframe['close'] #srcOPen<....
        & dataframe['slowMA'] > dataframe['fastMA'] #not fast< slow

      ),
      'buy'] = 1

    return dataframe

  def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
      (
        (dataframe['ha_close'] < dataframe['senkou_a']) |
        (dataframe['ha_close'] < dataframe['senkou_b'])
      ),
        'sell'] = 1
    return dataframe
