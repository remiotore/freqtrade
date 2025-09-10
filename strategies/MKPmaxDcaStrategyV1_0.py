


import os
import sys
from types import SimpleNamespace

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
import datetime
from datetime import timedelta  # noqa
from typing import Optional, Union  # noqa

from technical.indicators import PMAX
import json

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.optimize.space import SKDecimal
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib

def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print('{} - {}'.format(exc_type, exc_tb.tb_lineno))
            return None

    return safe_f

class MKPmaxDcaStrategyV1_0(IStrategy):
    INTERFACE_VERSION = 3

    buy_dca_koef = DecimalParameter(1, 20, decimals=2, default=1.25, space="buy")
    buy_dca_percent = DecimalParameter(-0.10, -0.001, decimals=3, default=-0.02, space="buy")
    buy_exit_profit_only = BooleanParameter(default=True, space="buy")
    buy_ignore_roi_if_entry_signal = BooleanParameter(default=False, space="buy")
    buy_max_dca_multiplier = DecimalParameter(1, 30, decimals=2, default=5.5, space="buy")
    buy_max_dca_orders = IntParameter(1, 30, default=3, space="buy")
    buy_rsi = IntParameter(1, 55, default=30, space="buy")
    stoploss_percent = DecimalParameter(-0.70, -0.30, decimals=2, default=-0.50, space="buy")
    trailing_only_offset_is_reached = BooleanParameter(default=False, space="buy")
    trailing_stop = BooleanParameter(default=True, space="buy")
    trailing_stop_positive = DecimalParameter(0.001, 0.01, decimals=3, default=0.005, space="buy")
    trailing_stop_positive_offset = DecimalParameter(0.011, 0.03, decimals=3, default=0.015, space="buy")



    if os.path.exists(f"{sys.path[0]}/MKPmaxDcaStrategyV1_0.json"):
        x = json.loads(open(f'{sys.path[0]}/MKPmaxDcaStrategyV1_0.json',mode='r').read())
        buy = x['params']['buy']
        buy_dca_koef.value = buy['buy_dca_koef']
        buy_exit_profit_only.value = buy['buy_exit_profit_only']
        buy_ignore_roi_if_entry_signal.value = buy['buy_ignore_roi_if_entry_signal']
        buy_max_dca_multiplier.value = buy['buy_max_dca_multiplier']
        buy_max_dca_orders.value = buy['buy_max_dca_orders']
        buy_rsi.value = buy['buy_rsi']
        buy_dca_percent.value = buy['buy_dca_percent']
        trailing = x['params']['trailing']
        trailing_only_offset_is_reached.value = trailing['trailing_only_offset_is_reached']
        trailing_stop.value = trailing['trailing_stop']
        trailing_stop_positive.value = trailing['trailing_stop_positive']
        trailing_stop_positive_offset.value = trailing['trailing_stop_positive_offset']

        stoploss = x['params']['stoploss']
        stoploss_percent.value = stoploss['stoploss']

    timeframe = '1m'
    position_adjustment_enable = True

    can_short: bool = False



    timeframe_mins = timeframe_to_minutes(timeframe)




    minimal_roi = {
        "0": 0.246,
        "304": 0.078,
        "851": 0.032,
        "1452": 0
    }


    stoploss = stoploss_percent.value

    trailing_stop = trailing_stop.value
    trailing_stop_positive = trailing_stop_positive.value
    trailing_stop_positive_offset = trailing_stop_positive_offset.value
    trailing_only_offset_is_reached = trailing_only_offset_is_reached.value

    process_only_new_candles = False


    use_exit_signal = False #buy_use_exit_signal.value
    exit_profit_only = buy_exit_profit_only.value
    ignore_roi_if_entry_signal = buy_ignore_roi_if_entry_signal.value

    startup_candle_count: int = 0

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {}
        plot_config['main_plot']['MA_1_9'] = {'color': 'yellow'}
        plot_config['main_plot']['pm_10_27_9_1'] = {'color': 'green'}
        plot_config['subplots'] = {}
        plot_config['subplots']['RSI'] = {"rsi": {"color": "red"}}
        plot_config['subplots']['PMAX'] = {"PMAXRES": {"color": "red"}}
        return plot_config

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = PMAX(dataframe, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['PMAXRES'] = dataframe['MA_1_9'] / dataframe['pm_10_27_9_1']

        dataframe['rsi'] = ta.RSI(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                 ((qtpylib.crossed_above(dataframe['MA_1_9'], dataframe['pm_10_27_9_1']))
                 & (dataframe['MA_1_9'] > dataframe['pm_10_27_9_1'])
                 & (dataframe['volume'] > 0) | (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)))
            ),
            ['enter_long', 'enter_tag']] = (1, 'buy_signal')

        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1
        """

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((qtpylib.crossed_below(dataframe['MA_1_9'], dataframe['pm_10_27_9_1'])) &
                 (dataframe['volume'] > 0))
            ),
            ['exit_long', 'exit_tag']] = (1, 'sell_signal')

        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
                (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1
        """
        return dataframe

    def bot_loop_start(self, **kwargs) -> None:
        pass

    def custom_entry_price(self, pair: str, current_time: 'datetime', proposed_rate: float,
                           entry_tag: 'Optional[str]', side: str, **kwargs) -> float:
        return proposed_rate

    def adjust_entry_price(self, trade: 'Trade', order: 'Optional[Order]', pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        return current_order_rate

    def custom_exit_price(self, pair: str, trade: 'Trade',
                          current_time: 'datetime', proposed_rate: float,
                          current_profit: float, exit_tag: Optional[str], **kwargs) -> float:
        return proposed_rate

    @safe
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            entry_tag: Optional[str], side: str, **kwargs) -> float:


        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        return proposed_stake / self.buy_max_dca_multiplier.value

    @safe
    def block_pair(self, pair, sell_reason, minutes):
        _block_year = datetime.datetime.now() + timedelta(minutes=minutes)
        self.lock_pair(pair=pair, until=_block_year, reason=sell_reason)

    @safe
    def obtain_last_prev_candles(self, pair, timeframe):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            return last_candle, previous_candle
        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            return None, None

    @safe
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):









        try:

            filled_buys = trade.select_filled_orders('buy')
            count_of_buys = len(filled_buys)
            last_candle, previous_candle = self.obtain_last_prev_candles(trade.pair, self.timeframe)

            if last_candle is not None and previous_candle is not None:

                if last_candle['close'] < previous_candle['close']:
                    return None

                if current_profit >= self.buy_dca_percent.value:
                    return None

                if 0 < count_of_buys <= self.buy_max_dca_orders.value:
                    try:

                        stake_amount = filled_buys[0].cost

                        stake_amount = stake_amount * (1 + (count_of_buys * self.buy_dca_koef.value))
                        return stake_amount
                    except:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
                        return None

        except:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print('{} - {} - {}'.format(exc_type, fname, exc_tb.tb_lineno))
            pass

        return None

    use_custom_stoploss = True

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs) -> 'Optional[Union[str, bool]]':
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: 'datetime', **kwargs) -> bool:
        return True

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                            current_time: datetime, **kwargs) -> bool:
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                           current_time: datetime, **kwargs) -> bool:
        return False

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return 1.0
