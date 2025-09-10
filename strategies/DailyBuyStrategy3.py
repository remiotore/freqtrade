import json
import logging
import os
import sys
from functools import reduce

import numpy
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame, Series
from typing import Optional, Union, List, Tuple

from pandas_ta import stdev

from freqtrade.enums import ExitCheckTuple
from freqtrade.persistence import Trade, Order, CustomDataWrapper
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter,
                                informative)
import datetime
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas_ta as pta


class DailyBuyStrategy3(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.03
    }

    leverage_value = 2

    timeframe_hierarchy = {
        '1m': '5m',
        '5m': '15m',
        '15m': '1h',
        '1h': '4h',
        '4h': '1d',
        '1d': '1w',
        '1w': '1M'
    }

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    stoploss = -0.03  # Basic stop loss

    use_exit_signal = True
    exit_profit_only = False

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.008

    dca_attempts = {}
    position_adjustment_enable = True
    candle_open_prices = {}
    last_dca_candle_index = {}

    last_dca_price = {}
    csl = {}

    new_sl_coef = DecimalParameter(0.3, 0.9, default=0.75, space='sell', optimize=False)

    buy_rsi = IntParameter(25, 60, default=55, space='buy', optimize=True)
    sell_rsi = IntParameter(50, 70, default=70, space='sell', optimize=True)

    atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space='stoploss', optimize=True)

    swing_window = IntParameter(10, 50, default=50, space='buy', optimize=True)
    swing_min_periods = IntParameter(1, 10, default=10, space='buy', optimize=True)
    swing_buffer = DecimalParameter(0.01, 0.1, default=0.03, space='buy', optimize=True)

    buy_macd = DecimalParameter(-0.02, 0.02, default=0.00, space='buy', optimize=True)
    buy_ema_short = IntParameter(5, 50, default=10, space='buy', optimize=True)
    buy_ema_long = IntParameter(50, 200, default=50, space='buy', optimize=True)

    sell_macd = DecimalParameter(-0.02, 0.02, default=-0.005, space='sell', optimize=True)
    sell_ema_short = IntParameter(5, 50, default=10, space='sell', optimize=True)
    sell_ema_long = IntParameter(50, 200, default=50, space='sell', optimize=True)

    volume_dca_int = IntParameter(1, 30, default=7, space='buy', optimize=True)

    def __init__(self, config):
        return super().__init__(config)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, timeframe) for pair in pairs for timeframe in self.timeframe_hierarchy.keys()]
        return informative_pairs

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        return self.leverage_value

    def calculate_swing(self, dataframe):
        swing_low = pd.Series(
            dataframe['low'].rolling(window=self.swing_window.value, min_periods=self.swing_min_periods.value).min(),
            index=dataframe.index
        )
        swing_high = pd.Series(
            dataframe['high'].rolling(window=self.swing_window.value, min_periods=self.swing_min_periods.value).max(),
            index=dataframe.index
        )
        return swing_low, swing_high

    def calculate_pivots(self, dataframe: DataFrame) -> Tuple[Series, Series, Series]:

        dataframe['pp'] = (dataframe['high'].shift(1) + dataframe['low'].shift(1) + dataframe['close'].shift(1)) / 3

        dataframe['r1'] = 2 * dataframe['pp'] - dataframe['low'].shift(1)

        dataframe['s1'] = 2 * dataframe['pp'] - dataframe['high'].shift(1)
        return dataframe['pp'], dataframe['r1'], dataframe['s1']

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        return proposed_stake / 4

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)
        dataframe['previous_close'] = dataframe['close'].shift(1)
        dataframe['max_since_buy'] = dataframe['high'].cummax()
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        pp, r1, s1 = self.calculate_pivots(dataframe)
        dataframe['pivot_point'] = pp
        dataframe['resistance_1'] = r1
        dataframe['support_1'] = s1

        swing_low, swing_high = self.calculate_swing(dataframe)
        dataframe['swing_low'] = swing_low
        dataframe['swing_high'] = swing_high

        dataframe['resistance_signal'] = ((dataframe['close'] > dataframe['resistance_1']) & (
                dataframe['close'] > dataframe['previous_close']))

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = [

            (dataframe['macd'] > dataframe['macdsignal']) & (dataframe['ema_short'] > dataframe['ema_long']) |
            (dataframe['resistance_signal']) & (dataframe['volume'] > 0)
        ]

        level = self.timeframe_hierarchy[self.timeframe]
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=level)

        if not informative.empty:
            last_close_informative = informative['close'].iat[-1]
            last_close = dataframe['close'].iat[-1]

            conditions.append(dataframe['close'] < last_close_informative)
        else:
            logging.info(f"No data available for {metadata['pair']} in '{level}' timeframe. Skipping this condition.")

        if all(isinstance(cond, pd.Series) for cond in conditions):
            final_condition = np.logical_and.reduce(conditions)
            dataframe.loc[final_condition, ['enter_long', 'enter_tag']] = (1, 'multi_timeframe_cross')
        else:
            logging.error("Not all conditions are pandas Series.")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = [
            (
                    (dataframe['close'] > dataframe['swing_high']) |
                    (
                            (dataframe['macd'] < dataframe['macdsignal']) &
                            (dataframe['ema_short'] < dataframe['ema_long'])
                    )
            ),
            (dataframe['volume'] > 0)
        ]
        exit_condition = np.logical_and.reduce([cond.values for cond in conditions if isinstance(cond, pd.Series)])
        dataframe.loc[exit_condition, ['exit_long', 'exit_tag']] = (1, 'macd_ema_exit')
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        profit_ratio = trade.calc_profit_ratio(rate)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if ('macd_ema_exit' in exit_reason) and (profit_ratio >= 0.005):

            return True

        if (('trailing' in exit_reason) or ('roi' in exit_reason)) and (profit_ratio >= 0.005):

            return True

        if 'force' in exit_reason:
            return True

        if 'stop_loss' in exit_reason:
            if len(self.get_dca_list(trade)) < 3:
                return False  # Pokračování v obchodování
            else:
                return True  # Ukončení obchodu po 3 pokusech
        return False

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:
        sl = self.get_mk_sl(trade)
        if current_rate <= sl:
            return f"custom_stop_loss_{sl}"
        pass

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        if current_rate > self.get_mk_sl(trade):
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]

        if len(self.get_dca_list(trade)) < 3:
            if (last_candle['close'] > last_candle['bb_lowerband'] or last_candle['rsi'] > 70):
                return None

            num_days = self.volume_dca_int.value
            weights = np.exp(-np.arange(num_days) / 5)  # Adjust the decay factor according to your strategy needs
            weighted_volumes = dataframe['volume'].iloc[-num_days:].multiply(weights[::-1])
            weighted_average_volume = weighted_volumes.sum() / weights.sum()

            if dataframe['volume'].iloc[-1] <= weighted_average_volume:
                return None

            dca_list = self.get_dca_list(trade)
            if dca_list and current_rate > dca_list[-1]:
                logging.info(f"Aktuální cena {current_rate} je vyšší než poslední DCA cena {dca_list[-1]}. DCA nebude provedeno.")
                return None

            logging.info(
                f"{current_time} - DCA triggered for {trade.pair}. Adjusting position with additional stake {trade.stake_amount * 2}")

            self.confirm_dca(current_rate, trade)

            return trade.stake_amount * 2

        return None

    def get_dca_list(self, trade):
        try:
            dcas = CustomDataWrapper.get_custom_data(trade_id=trade.id, key="DCA")[0].value
            return dcas
        except Exception as ex:
            pass
        return []

    def get_mk_sl(self, trade):
        try:
            sl = CustomDataWrapper.get_custom_data(trade_id=trade.id, key="SL")[0].value
            return sl
        except Exception as ex:
            pass
        return trade.stop_loss

    def set_mk_sl(self, trade):
        sl = self.get_mk_sl(trade)
        sl = sl * self.new_sl_coef.value
        if sl > trade.liquidation_price:
            CustomDataWrapper.set_custom_data(trade_id=trade.id, key="SL", value=sl)
        else:
            logging.info(f"ERROR SL")
            pass

    def confirm_dca(self, current_rate, trade):
        dcas = self.get_dca_list(trade)
        dcas.append(current_rate)
        self.set_mk_sl(trade)
        CustomDataWrapper.set_custom_data(trade_id=trade.id, key="DCA", value=dcas)
