import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (IStrategy, merge_informative_pair, stoploss_from_open,
                                IntParameter, DecimalParameter, CategoricalParameter)

from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import custom_indicators as cta

"""
Solipsis - By @werkkrew

Credits - 
@JimmyNixx for many of the ideas used throughout as well as helping me stay motivated throughout development!
@rk for submitting many PR's that have made this strategy possible! 

I ask for nothing in return except that if you make changes which bring you greater success than what has been provided, you share those ideas back to 
the community. Also, please don't nag me with a million questions and especially don't blame me if you lose a ton of money using this.

I take no responsibility for any success or failure you have using this strategy.

VERSION: 5.2.1
"""


class Solipsis_v5(IStrategy):


    base_mp = IntParameter(10, 50, default=30, space='buy', load=True, optimize=True)
    base_rmi_max = IntParameter(30, 60, default=50, space='buy', load=True, optimize=True)
    base_rmi_min = IntParameter(0, 30, default=20, space='buy', load=True, optimize=True)
    base_ma_streak = IntParameter(1, 4, default=1, space='buy', load=True, optimize=True)
    base_rmi_streak = IntParameter(3, 8, default=3, space='buy', load=True, optimize=True)
    base_trigger = CategoricalParameter(['pcc', 'rmi', 'none'], default='rmi', space='buy', load=True, optimize=True)
    inf_pct_adr = DecimalParameter(0.70, 0.99, default=0.80, space='buy', load=True, optimize=True)

    xbtc_guard = CategoricalParameter(['strict', 'lazy', 'none'], default='lazy', space='buy', optimize=True)
    xbtc_base_rmi = IntParameter(20, 70, default=40, space='buy', load=True, optimize=True)

    xtra_base_stake_rmi = IntParameter(10, 50, default=50, space='buy', load=True, optimize=True)
    xtra_base_fiat_rmi = IntParameter(30, 70, default=50, space='buy', load=True, optimize=True)


    csell_roi_type = CategoricalParameter(['static', 'decay', 'step'], default='step', space='sell', load=True,
                                          optimize=True)
    csell_roi_time = IntParameter(720, 1440, default=720, space='sell', load=True, optimize=True)
    csell_roi_start = DecimalParameter(0.01, 0.05, default=0.01, space='sell', load=True, optimize=True)
    csell_roi_end = DecimalParameter(0.0, 0.01, default=0, space='sell', load=True, optimize=True)
    csell_trend_type = CategoricalParameter(['rmi', 'ssl', 'candle', 'any', 'none'], default='any', space='sell',
                                            load=True, optimize=True)
    csell_pullback = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)
    csell_pullback_amount = DecimalParameter(0.005, 0.03, default=0.01, space='sell', load=True, optimize=True)
    csell_pullback_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)
    csell_endtrend_respect_roi = CategoricalParameter([True, False], default=False, space='sell', load=True,
                                                      optimize=True)

    cstop_loss_threshold = DecimalParameter(-0.05, -0.01, default=-0.03, space='sell', load=True, optimize=True)
    cstop_bail_how = CategoricalParameter(['roc', 'time', 'any', 'none'], default='none', space='sell', load=True,
                                          optimize=True)
    cstop_bail_roc = DecimalParameter(-5.0, -1.0, default=-3.0, space='sell', load=True, optimize=True)
    cstop_bail_time = IntParameter(60, 1440, default=720, space='sell', load=True, optimize=True)
    cstop_bail_time_trend = CategoricalParameter([True, False], default=True, space='sell', load=True, optimize=True)

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {}

    sell_params = {}

    minimal_roi = {
        "0": 100
    }

    stoploss = -0.99
    use_custom_stoploss = True

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 233
    process_only_new_candles = False

    custom_trade_info = {}
    custom_fiat = "USD"  # Only relevant if stake is BTC or ETH
    custom_btc_inf = False  # Don't change this.

    """
    Informative Pair Definitions
    """

    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]

        if self.config['stake_currency'] in ('BTC', 'ETH'):
            for pair in pairs:
                coin, stake = pair.split('/')
                coin_fiat = f"{coin}/{self.custom_fiat}"
                informative_pairs += [(coin_fiat, self.timeframe)]

            stake_fiat = f"{self.config['stake_currency']}/{self.custom_fiat}"
            informative_pairs += [(stake_fiat, self.timeframe)]

        else:
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                informative_pairs += [(btc_stake, self.timeframe)]

        return informative_pairs

    """
    Indicator Definitions
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not metadata['pair'] in self.custom_trade_info:
            self.custom_trade_info[metadata['pair']] = {}
            if not 'had-trend' in self.custom_trade_info[metadata["pair"]]:
                self.custom_trade_info[metadata['pair']]['had-trend'] = False


        dataframe['kama'] = ta.KAMA(dataframe, length=233)

        dataframe['rmi'] = cta.RMI(dataframe, length=24, mom=5)

        dataframe['roc-mp'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['mp'] = ta.RSI(dataframe['roc-mp'], timeperiod=3)

        dataframe['mastreak'] = cta.mastreak(dataframe, period=4)

        upper, mid, lower = cta.pcc(dataframe, period=40, mult=3)
        dataframe['pcc-lowerband'] = lower
        dataframe['pcc-upperband'] = upper

        lookup_idxs = dataframe.index.values - (abs(dataframe['mastreak'].values) + 1)
        valid_lookups = lookup_idxs >= 0
        dataframe['sbc'] = np.nan
        dataframe.loc[valid_lookups, 'sbc'] = dataframe['close'].to_numpy()[lookup_idxs[valid_lookups].astype(int)]

        dataframe['streak-roc'] = 100 * (dataframe['close'] - dataframe['sbc']) / dataframe['sbc']

        dataframe['candle-up'] = np.where(dataframe['close'] >= dataframe['open'], 1, 0)
        dataframe['candle-up-trend'] = np.where(dataframe['candle-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-up'] = np.where(dataframe['rmi'] >= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-up-trend'] = np.where(dataframe['rmi-up'].rolling(5).sum() >= 3, 1, 0)

        dataframe['rmi-dn'] = np.where(dataframe['rmi'] <= dataframe['rmi'].shift(), 1, 0)
        dataframe['rmi-dn-count'] = dataframe['rmi-dn'].rolling(8).sum()

        dataframe['streak-bo'] = np.where(dataframe['streak-roc'] < dataframe['pcc-lowerband'], 1, 0)
        dataframe['streak-bo-count'] = dataframe['streak-bo'].rolling(8).sum()

        ssldown, sslup = cta.SSLChannels_ATR(dataframe, length=21)
        dataframe['sroc'] = cta.SROC(dataframe, roclen=21, emalen=13, smooth=21)
        dataframe['ssl-dir'] = np.where(sslup > ssldown, 'up', 'down')

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        informative['1d-high'] = informative['close'].rolling(24).max()
        informative['1d-low'] = informative['close'].rolling(24).min()
        informative['adr'] = informative['1d-high'] - informative['1d-low']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)


        if self.config['stake_currency'] in ('BTC', 'ETH'):
            coin, stake = metadata['pair'].split('/')
            fiat = self.custom_fiat
            coin_fiat = f"{coin}/{fiat}"
            stake_fiat = f"{stake}/{fiat}"

            coin_fiat_tf = self.dp.get_pair_dataframe(pair=coin_fiat, timeframe=self.timeframe)
            dataframe[f"{fiat}_rmi"] = cta.RMI(coin_fiat_tf, length=55, mom=5)

            stake_fiat_tf = self.dp.get_pair_dataframe(pair=stake_fiat, timeframe=self.timeframe)
            dataframe[f"{stake}_rmi"] = cta.RMI(stake_fiat_tf, length=55, mom=5)

        else:
            pairs = self.dp.current_whitelist()
            btc_stake = f"BTC/{self.config['stake_currency']}"
            if not btc_stake in pairs:
                self.custom_btc_inf = True

                btc_stake_tf = self.dp.get_pair_dataframe(pair=btc_stake, timeframe=self.timeframe)
                dataframe['BTC_rmi'] = cta.RMI(btc_stake_tf, length=55, mom=5)
                dataframe['BTC_close'] = btc_stake_tf['close']
                dataframe['BTC_kama'] = ta.KAMA(btc_stake_tf, length=144)

        return dataframe

    """
    Buy Signal
    """

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (dataframe['close'] <= dataframe[f"1d-low_{self.inf_timeframe}"] +
             (self.inf_pct_adr.value * dataframe[f"adr_{self.inf_timeframe}"]))
        )

        conditions.append(
            (dataframe['rmi-dn-count'] >= self.base_rmi_streak.value) &
            (dataframe['streak-bo-count'] >= self.base_ma_streak.value) &
            (dataframe['rmi'] <= self.base_rmi_max.value) &
            (dataframe['rmi'] >= self.base_rmi_min.value) &
            (dataframe['mp'] <= self.base_mp.value)
        )

        if self.base_trigger.value == 'pcc':
            conditions.append(qtpylib.crossed_above(dataframe['streak-roc'], dataframe['pcc-lowerband']))

        if self.base_trigger.value == 'rmi':
            conditions.append(dataframe['rmi-up-trend'] == 1)

        if self.config['stake_currency'] in ('BTC', 'ETH'):
            conditions.append(
                (dataframe[f"{self.custom_fiat}_rmi"] > self.xtra_base_fiat_rmi.value) |
                (dataframe[f"{self.config['stake_currency']}_rmi"] < self.xtra_base_stake_rmi.value)
            )

        else:
            if self.custom_btc_inf:
                if self.xbtc_guard.value == 'strict':
                    conditions.append(
                        (
                                (dataframe['BTC_rmi'] > self.xbtc_base_rmi.value) &
                                (dataframe['BTC_close'] > dataframe['BTC_kama'])
                        )
                    )
                if self.xbtc_guard.value == 'lazy':
                    conditions.append(
                        (dataframe['close'] > dataframe['kama']) |
                        (
                                (dataframe['BTC_rmi'] > self.xbtc_base_rmi.value) &
                                (dataframe['BTC_close'] > dataframe['BTC_kama'])
                        )
                    )

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    """
    Sell Signal
    """

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sell'] = 0

        return dataframe

    """
    Custom Stoploss
    """

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        in_trend = self.custom_trade_info[trade.pair]['had-trend']

        if current_profit < self.cstop_loss_threshold.value:
            if self.cstop_bail_how.value == 'roc' or self.cstop_bail_how.value == 'any':

                if last_candle['sroc'] <= self.cstop_bail_roc.value:
                    return 0.01
            if self.cstop_bail_how.value == 'time' or self.cstop_bail_how.value == 'any':

                if trade_dur > self.cstop_bail_time.value:
                    if self.cstop_bail_time_trend.value == True and in_trend == True:
                        return 1
                    else:
                        return 0.01
        return 1

    """
    Custom Sell
    """

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)
        max_profit = max(0, trade.calc_profit_ratio(trade.max_rate))
        pullback_value = max(0, (max_profit - self.csell_pullback_amount.value))
        in_trend = False

        if self.csell_roi_type.value == 'static':
            min_roi = self.csell_roi_start.value
        elif self.csell_roi_type.value == 'decay':
            min_roi = cta.linear_decay(self.csell_roi_start.value, self.csell_roi_end.value, 0,
                                       self.csell_roi_time.value, trade_dur)
        elif self.csell_roi_type.value == 'step':
            if trade_dur < self.csell_roi_time.value:
                min_roi = self.csell_roi_start.value
            else:
                min_roi = self.csell_roi_end.value

        if self.csell_trend_type.value == 'rmi' or self.csell_trend_type.value == 'any':
            if last_candle['rmi-up-trend'] == 1:
                in_trend = True
        if self.csell_trend_type.value == 'ssl' or self.csell_trend_type.value == 'any':
            if last_candle['ssl-dir'] == 'up':
                in_trend = True
        if self.csell_trend_type.value == 'candle' or self.csell_trend_type.value == 'any':
            if last_candle['candle-up-trend'] == 1:
                in_trend = True

        if in_trend == True and current_profit > 0:

            self.custom_trade_info[trade.pair]['had-trend'] = True

            if self.csell_pullback.value == True and (current_profit <= pullback_value):
                if self.csell_pullback_respect_roi.value == True and current_profit > min_roi:
                    return 'intrend_pullback_roi'
                elif self.csell_pullback_respect_roi.value == False:
                    if current_profit > min_roi:
                        return 'intrend_pullback_roi'
                    else:
                        return 'intrend_pullback_noroi'

            return None

        elif in_trend == False:
            if self.custom_trade_info[trade.pair]['had-trend']:
                if current_profit > min_roi:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_roi'
                elif self.csell_endtrend_respect_roi.value == False:
                    self.custom_trade_info[trade.pair]['had-trend'] = False
                    return 'trend_noroi'
            elif current_profit > min_roi:
                return 'notrend_roi'
        else:
            return None