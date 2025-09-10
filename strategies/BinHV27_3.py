from freqtrade.strategy.interface import IStrategy



from freqtrade.strategy import IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge



import numpy  # noqa


class BinHV27_3(IStrategy):
    """

        strategy sponsored by user BinH from slack

    """

    minimal_roi = {
        "0": 1
    }

    stoploss = -0.50
    timeframe = '5m'

    buy_low_sma = IntParameter(low=50, high=120, default=60, optimize= False)
    buy_high_sma = IntParameter(low=100, high=300, default=120, optimize= False)
    buy_fast_sma = IntParameter(low=50, high=120, default=120, optimize= False)
    buy_slow_sma = IntParameter(low=100, high=300, default=240, optimize= False)
    buy_minusdi_ema = IntParameter(low=5, high=50, default=25, optimize= False)
    buy_plusdi_ema = IntParameter(low=5, high=50, default=5, optimize= False)

    buy_adx_1 = IntParameter(low=5, high=50, default=25, optimize= False)
    buy_emarsi_1 = IntParameter(low=5, high=50, default=20, optimize= False)

    buy_adx_2 = IntParameter(low=5, high=50, default=30, optimize= False)
    buy_emarsi_2 = IntParameter(low=5, high=50, default=20, optimize= False)

    buy_adx_3 = IntParameter(low=5, high=50, default=35, optimize= False)
    buy_emarsi_3 = IntParameter(low=5, high=50, default=20, optimize= False)

    buy_adx_4 = IntParameter(low=5, high=50, default=30, optimize= False)
    buy_emarsi_4 = IntParameter(low=5, high=50, default=25, optimize= False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = numpy.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        rsiframe = DataFrame(dataframe['rsi']).rename(columns={'rsi': 'close'})
        dataframe['emarsi'] = numpy.nan_to_num(ta.EMA(rsiframe, timeperiod=5))
        dataframe['adx'] = numpy.nan_to_num(ta.ADX(dataframe))
        dataframe['minusdi'] = numpy.nan_to_num(ta.MINUS_DI(dataframe))
        minusdiframe = DataFrame(dataframe['minusdi']).rename(columns={'minusdi': 'close'})
        
        dataframe['plusdi'] = numpy.nan_to_num(ta.PLUS_DI(dataframe))
        plusdiframe = DataFrame(dataframe['plusdi']).rename(columns={'plusdi': 'close'})
        
        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & ((dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe['slowsma'].shift().gt(dataframe['slowsma'].shift(2))
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())
        
        if not self.config['runmode'].value == 'hyperopt':
          dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=self.buy_plusdi_ema.value))
          dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=self.buy_minusdi_ema.value))
          dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=self.buy_low_sma.value))
          dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=self.buy_high_sma.value))
          dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=self.buy_fast_sma.value))
          dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=self.buy_slow_sma.value))

        return dataframe
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
          dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=self.buy_plusdi_ema.value))
          dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=self.buy_minusdi_ema.value))
          dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=self.buy_low_sma.value))
          dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=self.buy_high_sma.value))
          dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=self.buy_fast_sma.value))
          dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=self.buy_slow_sma.value))
        dataframe.loc[
            dataframe['slowsma'].gt(0) &
            dataframe['close'].lt(dataframe['highsma']) &
            dataframe['close'].lt(dataframe['lowsma']) &
            dataframe['minusdi'].gt(dataframe['minusdiema']) &
            dataframe['rsi'].ge(dataframe['rsi'].shift()) &
            (
              (
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx_1.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx_2.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx_3.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(20)
              ) |
              (
                dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx_4.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(25)
              )
            ),
            'buy'] = 1
        return dataframe
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
          dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=self.buy_plusdi_ema.value))
          dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=self.buy_minusdi_ema.value))
          dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=self.buy_low_sma.value))
          dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=self.buy_high_sma.value))
          dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=self.buy_fast_sma.value))
          dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=self.buy_slow_sma.value))
        dataframe.loc[
            (
              (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                (dataframe['close'].gt(dataframe['lowsma']) | dataframe['close'].gt(dataframe['highsma'])) &
                dataframe['highsma'].gt(0) &
                dataframe['bigdown']
              ) |
              (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                (dataframe['emarsi'].ge(75) | dataframe['close'].gt(dataframe['slowsma'])) &
                dataframe['bigdown']
              ) |
              (
                ~dataframe['preparechangetrendconfirm'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                dataframe['adx'].gt(30) &
                dataframe['emarsi'].ge(80) &
                dataframe['bigup']
              ) |
              (
                dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['slowingdown'] &
                dataframe['emarsi'].ge(75) &
                dataframe['slowsma'].gt(0)
              ) |
              (
                dataframe['preparechangetrendconfirm'] &
                dataframe['minusdi'].lt(dataframe['plusdi']) &
                dataframe['close'].gt(dataframe['lowsma']) &
                dataframe['slowsma'].gt(0)
              )
            ),
            'sell'] = 1
        return dataframe
