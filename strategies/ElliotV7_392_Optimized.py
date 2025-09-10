from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter, IntParameter

# --------------------------------------
# Helper functions
# --------------------------------------

def EWO(dataframe: DataFrame, ema_length: int = 20, ema2_length: int = 200) -> DataFrame:
    """Elliot Wave Oscillator"""
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe['close'] * 100

# --------------------------------------
# Strategy class
# --------------------------------------

class ElliotV7_392_Optimized(IStrategy):
    """5‑minute trend‑following / pullback hybrid, optimised for reduced drawdown"""

    INTERFACE_VERSION = 3

    timeframe = '5m'
    inf_1h = '1h'

    # --- Performance targets -------------------------------------------------
    minimal_roi = {
        "0": 0.03,   # 3 % immediately
        "30": 0.02,  # after 30 min allow 2 %
        "60": 0.01,  # after 60 min allow 1 %
        "120": 0     # after 120 min free‑ride
    }

    stoploss = -0.20  # emergency SL – custom_stoploss is primary

    trailing_stop = True
    trailing_stop_positive = 0.008   # 0.8 %
    trailing_stop_positive_offset = 0.08  # start trailing after 8 %
    trailing_only_offset_is_reached = True

    # --- Hyperopt parameters -------------------------------------------------
    base_nb_candles_buy = IntParameter(5, 80, default=14, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(5, 80, default=24, space='sell', optimize=True)
    low_offset = DecimalParameter(0.9, 0.99, default=0.975, space='buy', optimize=True)
    high_offset = DecimalParameter(0.95, 1.1, default=0.991, space='sell', optimize=True)
    high_offset_2 = DecimalParameter(0.99, 1.5, default=0.997, space='sell', optimize=True)

    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0, default=-19.988, space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=2.327, space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=69, space='buy', optimize=True)

    # --- Order execution -----------------------------------------------------
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'ioc'
    }

    # --- Protections ---------------------------------------------------------
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 60

    plot_config = {
        'main_plot': {
            'ema_fast': {},
            'ema_slow': {},
            'hma_50': {'color': 'orange'},
            'atr': {'color': 'grey'}
        }
    }

    # ---------------------------------------------------------------------
    # Informative pairs / higher timeframe
    # ---------------------------------------------------------------------
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.inf_1h) for pair in pairs]

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider required"
        df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        df['ema_fast'] = ta.EMA(df, timeperiod=20)
        df['ema_slow'] = ta.EMA(df, timeperiod=60)
        df['uptrend'] = ((df['ema_fast'] > df['ema_slow'] * 1.002)).astype('int')
        df['rsi_100'] = ta.RSI(df, timeperiod=100)
        return df

    # ---------------------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # merge informative 1h
        inf = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, inf, self.timeframe, self.inf_1h, ffill=True)

        # Moving averages for adaptive channels
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Volatility – ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close'] * 100

        # Trend metrics
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # Hull and SMA
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        return dataframe

    # ---------------------------------------------------------------------
    # Buy Logic
    # ---------------------------------------------------------------------
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Common filters
        cond_common = (
            (dataframe['volume'] > 0) &
            (dataframe['uptrend_1h'] > 0) &
            (dataframe['rsi_fast'] < 35) &
            (dataframe['atr_percent'] < 5)
        )

        # Pullback‑in‑uptrend (EWO high)
        conditions.append(
            cond_common &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) &
            (dataframe['close'] < dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)
        )

        # Deep pullback (EWO low)
        conditions.append(
            cond_common &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value) &
            (dataframe['close'] < dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)
        )

        if conditions:
            dataframe.loc[reduce(lambda a, b: a | b, conditions), 'buy'] = 1
        return dataframe

    # ---------------------------------------------------------------------
    # Sell Logic
    # ---------------------------------------------------------------------
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        cond_hma_cross = (
            (dataframe['sma_9'] > dataframe['hma_50']) &
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)
        ) | (
            (dataframe['sma_9'] < dataframe['hma_50']) &
            (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)
        )

        conditions.append(
            cond_hma_cross &
            (dataframe['volume'] > 0) &
            (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        )

        if conditions:
            dataframe.loc[reduce(lambda a, b: a | b, conditions), 'sell'] = 1
        return dataframe

    # ---------------------------------------------------------------------
    # Custom Stoploss – ATR & time based
    # ---------------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Dynamic SL: tighten if position ages or deep in loss."""
        # Hard tighten after 90 min if still red beyond ‑5 %
        if current_profit < -0.05 and (current_time - trade.open_date_utc) > timedelta(minutes=90):
            return -0.015  # cut quickly

        # Universal ATR‑based fallback (approx 1.2 × ATR%)
        try:
            pair_df: DataFrame = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            atr_pct = (ta.ATR(pair_df, timeperiod=14).iloc[-1] / current_rate)
            dynamic_sl = -1.2 * float(atr_pct)
            # Cap to emergency
            return max(dynamic_sl, self.stoploss)
        except Exception:
            return self.stoploss
