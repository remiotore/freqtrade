import logging
from typing import Optional

import pandas as pd
from technical import candles
from technical.candles import doji

from freqtrade.strategy.interface import IStrategy
from functools import reduce
from datetime import timedelta, datetime
from pandas import DataFrame, errors

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, informative
import talib.abstract as ta
from warnings import simplefilter

simplefilter(action="ignore", category=errors.PerformanceWarning)


class HPStrategyLeverage(IStrategy):
    INTERFACE_VERSION = 3
    leverage_value = 3

    minimal_roi = {
        "0": 0.215 * leverage_value,
        "40": 0.032 * leverage_value,
        "87": 0.016 * leverage_value,
        "201": 0 * leverage_value
    }

    stoploss = -0.3 * leverage_value

    trailing_stop = True
    trailing_stop_positive = 0.01 * leverage_value
    trailing_stop_positive_offset = 0.02 * leverage_value
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01 * leverage_value
    ignore_roi_if_entry_signal = False

    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'ioc'
    }

    timeframe = '15m'

    process_only_new_candles = True
    startup_candle_count = 400







    rolling_ha_treshold = IntParameter(1, 10, default=7, space='buy', optimize=True)
    rsi_upper_limit = IntParameter(30, 70, default=50, space='buy', optimize=True)
    rsi_buy_limit = IntParameter(5, 60, default=15, space='buy', optimize=True)
    rsi_sell_limit = IntParameter(60, 95, default=75, space='sell', optimize=True)
    roc_threshold = IntParameter(1, 10, default=2, space='buy', optimize=True)
    doji_lastx = IntParameter(1, 10, default=10, space='buy', optimize=True)
    doji_diff_threshold = DecimalParameter(0.003, 0.05, default=0.01, space='buy', optimize=True)


    @informative('15m')
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['roc'] = ((dataframe['close'] / dataframe['close'].shift(self.roc_threshold.value)) - 1) * 100

        dataframe['doji'] = (dataframe["open"] - dataframe["close"]).abs() <= (
                (dataframe["high"] - dataframe["close"]) * self.doji_diff_threshold.value).abs().astype("float32")
        dataframe['doji_last_x'] = dataframe['doji'].rolling(window=self.doji_lastx.value).sum() > 0

        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        return dataframe


    def calculate_heiken_ashi(self, dataframe):
        if dataframe.empty:
            raise ValueError("DataFrame je prázdný")

        heiken_ashi = pd.DataFrame(index=dataframe.index)
        heiken_ashi['HA_Close'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        heiken_ashi['HA_Open'] = heiken_ashi['HA_Close'].shift(1)
        heiken_ashi['HA_Open'].iloc[0] = heiken_ashi['HA_Close'].iloc[0]

        heiken_ashi['HA_High'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe['high'], how='inner').max(axis=1)
        heiken_ashi['HA_Low'] = heiken_ashi[['HA_Open', 'HA_Close']].join(dataframe['low'], how='inner').min(axis=1)

        heiken_ashi['HA_Close'] = heiken_ashi['HA_Close'].rolling(window=self.rolling_ha_treshold.value).mean()
        heiken_ashi['HA_Open'] = heiken_ashi['HA_Open'].rolling(window=self.rolling_ha_treshold.value).mean()
        heiken_ashi['HA_High'] = heiken_ashi['HA_High'].rolling(window=self.rolling_ha_treshold.value).mean()
        heiken_ashi['HA_Low'] = heiken_ashi['HA_Low'].rolling(window=self.rolling_ha_treshold.value).mean()

        return heiken_ashi

    def should_buy(self, dataframe):
        heiken_ashi = self.calculate_heiken_ashi(dataframe)
        last_candle = heiken_ashi.iloc[-1]

        if last_candle['HA_Close'] > last_candle['HA_Open']:

            return True
        else:

            return False

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        uptrend = dataframe['close'] > dataframe['ema_50']
        doji_in_uptrend = uptrend & (dataframe['doji_last_x'] > 0)

        rsi_cond = (
                (dataframe['rsi'] < self.rsi_buy_limit.value) & self.should_buy(dataframe)
        )
        conditions.append(rsi_cond)
        momentum_cond = (
                (dataframe['rsi_fast_15m'] < self.rsi_upper_limit.value) &
                (dataframe['roc'] > self.roc_threshold.value) &
                (dataframe['volume_15m'] > 0)
        )
        conditions.append(momentum_cond)

        combined_conditions = reduce(lambda x, y: x | y, conditions)

        final_conditions = (combined_conditions & ~doji_in_uptrend)

        dataframe.loc[final_conditions, ['enter_long', 'enter_tag']] = (1, 'rsi_momentum_ha_enter_long')

        dont_buy_conditions = [
            (dataframe['enter_long'].shift(1) == 1)
        ]

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'enter_long'] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(dataframe['rsi'] >= self.rsi_sell_limit.value), 'exit_long'] = 1
        dont_exit_conditions = [
        ]
        if dont_exit_conditions:
            for condition in dont_exit_conditions:
                dataframe.loc[condition, 'exit_long'] = 0
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        default_reasons = ['force_sell', 'force_exit', 'roi', 'stop_loss', 'sell_signal', 'sell_profit_only',
                           'sell_profit_offset', 'sell_custom', 'sell_signal', 'trailing_stop_loss']
        if (exit_reason in default_reasons):
            logging.info(f"CTEN {pair} - {order_type} - {amount} - {rate} - {time_in_force}")
            return True
        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        logging.info(f"CTEN {pair} - {order_type} - {amount} - {rate} - {time_in_force} - {side}")
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                 **kwargs) -> float:
        return self.leverage_value
