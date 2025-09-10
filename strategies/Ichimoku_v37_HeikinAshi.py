

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import IStrategy, merge_informative_pair
from technical.indicators import ichimoku

class Ichimoku_v37_HeikinAshi(IStrategy):

  minimal_roi = {
    "0": 100
  }

  stoploss = -0.99

  timeframe = '4h'

  inf_tf = '1d'

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

  def informative_pairs(self):
    if not self.dp:

      return []

    pairs = self.dp.current_whitelist()

    informative_pairs =  [(pair, '1d') for pair in pairs]
    return informative_pairs

  def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

    if not self.dp:

      return dataframe

    dataframe_inf = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_tf)

    heikinashi = qtpylib.heikinashi(dataframe_inf)
    heik = qtpylib.heikinashi(dataframe)

    dataframe_inf['ha_open'] = heikinashi['open']
    dataframe_inf['ha_close'] = heikinashi['close']
    dataframe_inf['ha_high'] = heikinashi['high']
    dataframe_inf['ha_low'] = heikinashi['low']

    dataframe['ha_4h_open'] = heik['open']
    dataframe['ha_4h_close'] = heik['close']
    dataframe['ha_4h_high'] = heik['high']
    dataframe['ha_4h_low'] = heik['low']

    ha_ichi = ichimoku(heikinashi,
      conversion_line_period=20,
      base_line_periods=60,
      laggin_span=120,
      displacement=30
    )

    dataframe_inf['senkou_a'] = ha_ichi['senkou_span_a']
    dataframe_inf['senkou_b'] = ha_ichi['senkou_span_b']
    dataframe_inf['cloud_green'] = ha_ichi['cloud_green']
    dataframe_inf['cloud_red'] = ha_ichi['cloud_red']

    dataframe = merge_informative_pair(dataframe, dataframe_inf, self.timeframe, self.inf_tf, ffill=True)

    """
    Senkou Span A > Senkou Span B = Cloud Green
    Senkou Span B > Senkou Span A = Cloud Red
    """
    return dataframe

  def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
      (
        (
          (dataframe['ha_4h_close'].crossed_above(dataframe['senkou_a_1d'])) &
          (dataframe['ha_4h_close'].shift() < (dataframe['senkou_a_1d'])) &
          (dataframe['cloud_green_1d'] == True)
        ) |
        (
          (dataframe['ha_4h_close'].crossed_above(dataframe['senkou_b_1d'])) &
          (dataframe['ha_4h_close'].shift() < (dataframe['senkou_b_1d'])) &
          (dataframe['cloud_red_1d'] == True)
        )
      ),
      'buy'] = 1

    return dataframe

  def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    dataframe.loc[
      (
        (dataframe['ha_4h_close'] < dataframe['senkou_a_1d']) |
        (dataframe['ha_4h_close'] < dataframe['senkou_b_1d'])
      ),
        'sell'] = 1
        
    return dataframe
