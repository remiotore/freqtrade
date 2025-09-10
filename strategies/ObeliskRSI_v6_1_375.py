
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'

from technical.util import resample_to_interval, resampled_merge
from freqtrade.exchange import timeframe_to_minutes

from functools import reduce
from datetime import datetime, timedelta





















































def easeInCubic(t):
    return t * t * t

def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)

def clamp01(num):
    return clamp(num, 0, 1)


class ObeliskRSI_v6_1_375(IStrategy):

    timeframe = '5m'

    startup_candle_count = 240
    process_only_new_candles = True

    minimal_roi = {
        "0": 0.15,
        "35": 0.04,
        "65": 0.01,
        "120": 0
    }

    buy_params = {
     'bear-buy-rsi-value': 21,
     'bull-buy-rsi-value': 35
    }

    sell_params = {
     'bear-sell-rsi-value': 55,
     'bull-sell-rsi-value': 69
    }

    stoploss = -0.30

    use_custom_stoploss = True
    custom_stop_ramp_minutes = 110

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        resample_rsi_interval = timeframe_to_minutes(self.timeframe) * 12
        resample_rsi_key = 'resample_{}_rsi'.format(resample_rsi_interval)

        dataframe_long = resample_to_interval(dataframe, resample_rsi_interval)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=14)
        dataframe = resampled_merge(dataframe, dataframe_long)
        dataframe[resample_rsi_key].fillna(method='ffill', inplace=True)


        dataframe['bull'] = dataframe[resample_rsi_key].gt(60).astype('int')


        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []
        conditions.append(dataframe['volume'] > 0)

        conditions.append(
            ((dataframe['bull'] > 0) & qtpylib.crossed_below(dataframe['rsi'], params['bull-buy-rsi-value'])) |
            (~(dataframe['bull'] > 0) & qtpylib.crossed_below(dataframe['rsi'], params['bear-buy-rsi-value']))
            )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []
        conditions.append(dataframe['volume'] > 0)

        conditions.append(
            ((dataframe['bull'] > 0) & (dataframe['rsi'] > params['bull-sell-rsi-value'])) |
            (~(dataframe['bull'] > 0) & (dataframe['rsi'] > params['bear-sell-rsi-value']))
            )

        dataframe.loc[
            reduce(lambda x, y: x & y, conditions),
            'sell'] = 1

        return dataframe



    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:



        since_open = current_time.replace(tzinfo=trade.open_date.tzinfo) - trade.open_date

        sl_pct = 1 - easeInCubic( clamp01( since_open / timedelta(minutes=self.custom_stop_ramp_minutes) ) )
        sl_ramp = abs(self.stoploss) * sl_pct

        return sl_ramp + 0.001 # we can't go all the way to zero