
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


class RSIDivergence_2(IStrategy):
    INTERFACE_VERSION = 2

    timeframe = '15m'
    inf_tf = '4h'

    buy_params = {
        'use_bull': True,
        'use_hidden_bull': False,
        'use_rsi_smma_buy': True,
        "ewo_high": 5.835,
        "low_rsi_buy": 40,
        "high_rsi_buy": 40,
        "high_smma_rsi_buy": 35,
        "low_adx_buy": 30,
        "high_adx_buy": 30,
        "low_stoch_buy": 20,
        "high_stoch_buy": 80,
        "low_osc_buy": 80,
        "high_osc_buy": 80,
        "ewo_rol": 7,
    }

    sell_params = {
        'low_rsi_sell': 65,
        'low_ewo_sell': 0.5
    }

    minimal_roi = {
        "0": 10,
    }

    stoploss = -0.24

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = False
    use_custom_stoploss = False

    use_bull = BooleanParameter(
        default=buy_params['use_bull'], space='buy', optimize=True)
    use_hidden_bull = BooleanParameter(
        default=buy_params['use_hidden_bull'], space='buy', optimize=True)
    use_rsi_smma_buy = BooleanParameter(
        default=buy_params['use_rsi_smma_buy'], space='buy', optimize=True)

    fast_ewo = 5
    slow_ewo = 35
    ewo_high = DecimalParameter(
        0, 7.0, default=buy_params['ewo_high'], space='buy', optimize=False)
    low_ewo_sell = DecimalParameter(
        -0.5, 3, decimals=1, default=sell_params['low_ewo_sell'], space='sell', optimize=True)
    ewo_rol = IntParameter(
        0, 10, default=buy_params['ewo_rol'], space='buy', optimize=False)
    low_rsi_buy = IntParameter(
        0, 100, default=buy_params['low_rsi_buy'], space='buy', optimize=False)
    low_rsi_sell = IntParameter(
        40, 100, default=sell_params['low_rsi_sell'], space='sell', optimize=True)
    high_rsi_buy = IntParameter(
        15, 60, default=buy_params['high_rsi_buy'], space='buy', optimize=True)
    high_smma_rsi_buy = IntParameter(
        15, 60, default=buy_params['high_smma_rsi_buy'], space='buy', optimize=True)
    low_adx_buy = IntParameter(
        0, 100, default=buy_params['low_adx_buy'], space='buy', optimize=False)
    high_adx_buy = IntParameter(
        0, 100, default=buy_params['high_adx_buy'], space='buy', optimize=False)
    low_stoch_buy = IntParameter(
        0, 100, default=buy_params['low_stoch_buy'], space='buy', optimize=False)
    high_stoch_buy = IntParameter(
        0, 100, default=buy_params['high_stoch_buy'], space='buy', optimize=False)
    low_osc_buy = IntParameter(
        0, 100, default=buy_params['low_osc_buy'], space='buy', optimize=False)
    high_osc_buy = IntParameter(
        0, 100, default=buy_params['high_osc_buy'], space='buy', optimize=False)

    startup_candle_count: int = 50

    osc = 'RSI'
    src = 'close'
    rsiLen = 14
    lbL = 40
    lbR = 5

    @property
    def plot_config(self):
        return {

            'main_plot': {
                f'sar_{self.inf_tf}': {'color': 'lightgreen'},







            },
            'subplots': {

                "RSI": {
                    'RSI': {'color': 'red'},
                    'SMMA': {'color': 'yellow'}
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """

        pairs = self.dp.current_whitelist()

        informative_pairs = [(pair, self.inf_tf)
                             for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.inf_tf)

        informative['sar'] = ta.SAR(informative)

        dataframe = merge_informative_pair(
            dataframe, informative, self.timeframe, self.inf_tf, ffill=True)








        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=14)
        stoch = ta.STOCH(dataframe, fastk_period=10, slowk_period=3,
                         slowk_matype=0, slowd_period=3, slowd_matype=0)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        dataframe['osc'] = dataframe[self.osc]

        dataframe['SMMA'] = ta.SMA(
            ta.RSI(dataframe, timeperiod=5).rolling(3).mean(), period=14
        )

        dataframe['min'] = dataframe['osc'].rolling(self.lbL).min()
        dataframe['prevMin'] = np.where(dataframe['min'] > dataframe['min'].shift(
        ), dataframe['min'].shift(), dataframe['min'])
        dataframe.loc[
            (
                (dataframe['osc'].shift(1) == dataframe['prevMin'].shift(1)) &
                (dataframe['osc'] != dataframe['prevMin'])
            ), 'plFound'] = 1

        dataframe['max'] = dataframe['osc'].rolling(self.lbL).max()
        dataframe['prevMax'] = np.where(dataframe['max'] < dataframe['max'].shift(
        ), dataframe['max'].shift(), dataframe['max'])
        dataframe.loc[
            (
                (dataframe['osc'].shift(1) == dataframe['prevMax'].shift(1)) &
                (dataframe['osc'] != dataframe['prevMax'])
            ), 'phFound'] = 1




        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(
            dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
            ), 'oscHL'] = 1


        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(
            dataframe, 'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] < dataframe['valuewhen_plFound_low']), 'priceLL'] = 1

        dataframe.loc[
            (
                (dataframe['priceLL'] == 1) &
                (dataframe['oscHL'] == 1) &
                (dataframe['plFound'] == 1)
            ), 'bullCond'] = 1






















        dataframe['valuewhen_plFound_osc'], dataframe['inrange_plFound_osc'] = valuewhen(
            dataframe, 'plFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_plFound_osc']) &
                (dataframe['inrange_plFound_osc'] == 1)
            ), 'oscLL'] = 1




        dataframe['valuewhen_plFound_low'], dataframe['inrange_plFound_low'] = valuewhen(
            dataframe, 'plFound', 'low', 1)
        dataframe.loc[
            (dataframe['low'] > dataframe['valuewhen_plFound_low']), 'priceHL'] = 1

        dataframe.loc[
            (
                (dataframe['priceHL'] == 1) &
                (dataframe['oscLL'] == 1) &
                (dataframe['plFound'] == 1)
            ), 'hiddenBullCond'] = 1

























        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(
            dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] < dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
            ), 'oscLH'] = 1




        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(
            dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] > dataframe['valuewhen_phFound_high']), 'priceHH'] = 1


        dataframe.loc[
            (
                (dataframe['priceHH'] == 1) &
                (dataframe['oscLH'] == 1) &
                (dataframe['phFound'] == 1)
            ), 'bearCond'] = 1

























        dataframe['valuewhen_phFound_osc'], dataframe['inrange_phFound_osc'] = valuewhen(
            dataframe, 'phFound', 'osc', 1)
        dataframe.loc[
            (
                (dataframe['osc'] > dataframe['valuewhen_phFound_osc']) &
                (dataframe['inrange_phFound_osc'] == 1)
            ), 'oscHH'] = 1




        dataframe['valuewhen_phFound_high'], dataframe['inrange_phFound_high'] = valuewhen(
            dataframe, 'phFound', 'high', 1)
        dataframe.loc[
            (dataframe['high'] < dataframe['valuewhen_phFound_high']), 'priceLH'] = 1


        dataframe.loc[
            (
                (dataframe['priceLH'] == 1) &
                (dataframe['oscHH'] == 1) &
                (dataframe['phFound'] == 1)
            ), 'hiddenBearCond'] = 1




















        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)





        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.use_bull.value:
            conditions.append(
                (
                    (dataframe['bullCond'] > 0) &

                    (dataframe['EWO'] > dataframe['EWO'].shift(1).rolling(self.ewo_rol.value).max()) &




                    (dataframe['RSI'] < self.high_rsi_buy.value) &





                    (dataframe['volume'] > 0)
                )
            )

        if self.use_hidden_bull.value:
            conditions.append(
                (
                    (dataframe['hiddenBullCond'] > 0) &

                    (dataframe['EWO'] > dataframe['EWO'].shift(1).rolling(self.ewo_rol.value).max()) &



                    (dataframe['RSI'] < self.high_rsi_buy.value) &





                    (dataframe['volume'] > 0)
                )
            )

        if self.use_rsi_smma_buy.value:
            conditions.append(
                (

                    (dataframe['EWO'] > dataframe['EWO'].shift(1).rolling(self.ewo_rol.value).max()) &
                    (dataframe['RSI'] < self.high_smma_rsi_buy.value) &
                    (qtpylib.crossed_above(dataframe['RSI'], dataframe['SMMA'])) &
                    (dataframe['volume'] > 0)
                )
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (dataframe['EWO'] > self.low_ewo_sell.value) &
            (dataframe['EWO'] < dataframe['EWO'].shift(1)) &


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
        elif (current_profit > 0.06):
            sl_new = 0.02
        elif (current_profit > 0.03):
            sl_new = 0.01

        return sl_new
