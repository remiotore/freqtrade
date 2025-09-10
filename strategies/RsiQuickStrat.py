
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from datetime import datetime
from datetime import timedelta

import pandas as pd
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from technical.indicators import atr
from freqtrade.strategy import DecimalParameter, IntParameter, BooleanParameter, merge_informative_pair, stoploss_from_open
from technical.util import resample_to_interval, resampled_merge
from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)


rangeUpper = 60
rangeLower = 5

use_sell_signal = True
sell_profit_only = False
ignore_roi_if_buy_signal = False



def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma1_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif


def valuewhen(dataframe, condition, source, occurrence):
    copy = dataframe.copy()
    copy['colFromIndex'] = copy.index
    copy = copy.sort_values(
        by=[condition, 'colFromIndex'], ascending=False).reset_index(drop=True)
    copy['valuewhen'] = np.where(
        copy[condition] > 0, copy[source].shift(-occurrence), copy[source])
    copy['barrsince'] = copy['colFromIndex'] - \
        copy['colFromIndex'].shift(-occurrence)
    copy.loc[
        (
            (rangeLower <= copy['barrsince']) &
            (copy['barrsince'] <= rangeUpper)
        ), "in_range"] = 1
    copy['in_range'] = copy['in_range'].fillna(0)
    copy = copy.sort_values(by=['colFromIndex'],
                            ascending=True).reset_index(drop=True)
    return copy['valuewhen'], copy['in_range']


class RsiQuickStrat(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '5m'
    inf_tf = '1h'

    buy_params = {
        'use_bull': False,
        'use_hidden_bull': False,
        'use_rsi_smma_buy': True,
        "high_rsi_buy": 49,
        "high_smma_rsi_buy": 42,
        "ewo_rol_bull_buy": 1,
        "ewo_rol_smma_buy": 2,
    }

    sell_params = {
        'low_ewo_sell': 1.9,
        'ewo_rol_sell': 4,
    }

    minimal_roi = {
        "0": 10,
    }

    stoploss = -0.272

    trailing_stop = False
    trailing_stop_positive = 0.33
    trailing_stop_positive_offset = 0.42200000000000004
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True

    use_bull = BooleanParameter(
        default=buy_params['use_bull'], space='buy', optimize=False)
    use_hidden_bull = BooleanParameter(
        default=buy_params['use_hidden_bull'], space='buy', optimize=False)
    use_rsi_smma_buy = BooleanParameter(
        default=buy_params['use_rsi_smma_buy'], space='buy', optimize=False)

    low_ewo_sell = DecimalParameter(
        0, 2, decimals=1, default=sell_params['low_ewo_sell'], space='sell', optimize=False)
    ewo_rol_sell = IntParameter(
        1, 4, default=sell_params['ewo_rol_sell'], space='sell', optimize=False)

    high_smma_rsi_buy = IntParameter(
        15, 50, default=buy_params['high_smma_rsi_buy'], space='buy', optimize=True)
    ewo_rol_smma_buy = IntParameter(
        1, 4, default=buy_params['ewo_rol_smma_buy'], space='buy', optimize=True)

    ewo_rol_bull_buy = IntParameter(
        1, 4, default=buy_params['ewo_rol_bull_buy'], space='buy', optimize=False)
    high_rsi_buy = IntParameter(
        20, 50, default=buy_params['high_rsi_buy'], space='buy', optimize=False)

    fast_ewo = 5
    slow_ewo = 35

    startup_candle_count: int = 50

    osc = 'RSI'
    src = 'close'
    lbL = 40
    lbR = 5

    @property
    def plot_config(self):
        return {

            'main_plot': {







            },
            'subplots': {

                "RSI": {
                    'RSI': {'color': 'red'},
                    'SMMA': {'color': 'yellow'}
                }
            }
        }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        informative_pairs.extend([(pair, self.inf_tf) for pair in pairs])
        return informative_pairs

    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['SMMA'] = ta.SMA(
            ta.RSI(dataframe, timeperiod=5).rolling(3).mean(), period=14
        )

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
























































































































































































        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []


















        conditions.append(
            (dataframe['EWO'] > dataframe['EWO'].shift(1).rolling(self.ewo_rol_smma_buy.value).max()) &
            (dataframe['RSI'] < self.high_smma_rsi_buy.value) &
            (qtpylib.crossed_above(dataframe['RSI'], dataframe['SMMA']))
        )

        conditions.append(
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        conditions.append(
            (dataframe['EWO'] > self.low_ewo_sell.value) &
            (dataframe['EWO'] < dataframe['EWO'].shift(1).rolling(self.ewo_rol_sell.value).min()) &
            (dataframe['RSI'] < dataframe['SMMA']) &
            (dataframe['volume'] > 0)
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1



        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        sl_new = 1

        if (current_profit > 0.2):
            sl_new = 0.05
        elif (current_profit > 0.1):
            sl_new = 0.03
        elif (current_profit > 0.05):
            sl_new = 0.02
        elif (current_profit > 0.02):
            sl_new = 0.01

        return sl_new
