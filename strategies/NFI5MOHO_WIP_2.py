import logging
from functools import reduce
from typing import Dict
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter)
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib

logger = logging.getLogger(__name__)

class NFI5MOHO_WIP_2(IStrategy):
    protections = [
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 1,
            "stop_duration": 60,
            "required_profit": -0.05
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        }
    ]

    INTERFACE_VERSION = 2

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'trailing_stop_loss': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    minimal_roi = {
        "0": 0.08,
        "10": 0.04,
        "30": 0.02,
        "60": 0.01
    }

    stoploss = -0.15

    base_nb_candles_buy = IntParameter(5, 80, default=20, load=True, space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(5, 80, default=20, load=True, space='sell', optimize=False)
    low_offset_sma = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_sma = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)
    low_offset_ema = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_ema = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)
    low_offset_trima = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_trima = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)
    low_offset_t3 = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_t3 = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)
    low_offset_kama = DecimalParameter(0.9, 0.99, default=0.958, load=True, space='buy', optimize=False)
    high_offset_kama = DecimalParameter(0.99, 1.1, default=1.012, load=True, space='sell', optimize=False)

    ewo_low = DecimalParameter(-20.0, -8.0, default=-20.0, load=True, space='buy', optimize=False)
    ewo_high = DecimalParameter(2.0, 12.0, default=6.0, load=True, space='buy', optimize=False)
    fast_ewo = IntParameter(10, 50, default=50, load=True, space='buy', optimize=False)
    slow_ewo = IntParameter(100, 200, default=200, load=True, space='buy', optimize=False)

    ma_types = ['sma', 'ema', 'trima', 't3', 'kama']
    ma_map = {
        'sma': {
            'low_offset': low_offset_sma.value,
            'high_offset': high_offset_sma.value,
            'calculate': ta.SMA
        },
        'ema': {
            'low_offset': low_offset_ema.value,
            'high_offset': high_offset_ema.value,
            'calculate': ta.EMA
        },
        'trima': {
            'low_offset': low_offset_trima.value,
            'high_offset': high_offset_trima.value,
            'calculate': ta.TRIMA
        },
        't3': {
            'low_offset': low_offset_t3.value,
            'high_offset': high_offset_t3.value,
            'calculate': ta.T3
        },
        'kama': {
            'low_offset': low_offset_kama.value,
            'high_offset': high_offset_kama.value,
            'calculate': ta.KAMA
        }
    }

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.04

    use_custom_stoploss = False
    timeframe = '5m'
    inf_1h = '1h'
    process_only_new_candles = True
    use_exie_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    startup_candle_count: int = 300

    plot_config = {
        'main_plot': {
            'ma_offset_buy': {'color': 'orange'},
            'ma_offset_sell': {'color': 'orange'},
        },
    }

    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_11_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_13_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_14_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_15_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_16_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_17_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_18_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_19_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_20_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_condition_21_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    buy_dip_threshold_1 = DecimalParameter(0.001, 0.05, default=0.02, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.01, 0.2, default=0.14, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.05, 0.4, default=0.32, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_4 = DecimalParameter(0.2, 0.5, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_5 = DecimalParameter(0.001, 0.05, default=0.015, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_6 = DecimalParameter(0.01, 0.2, default=0.06, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_7 = DecimalParameter(0.05, 0.4, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_8 = DecimalParameter(0.2, 0.5, default=0.4, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_9 = DecimalParameter(0.001, 0.05, default=0.026, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10 = DecimalParameter(0.01, 0.2, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_11 = DecimalParameter(0.05, 0.4, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_12 = DecimalParameter(0.2, 0.5, default=0.66, space='buy', decimals=3, optimize=False, load=True)

    buy_pump_pull_threshold_1 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_1 = DecimalParameter(0.4, 1.0, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_2 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_2 = DecimalParameter(0.4, 1.0, default=0.56, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_3 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_3 = DecimalParameter(0.4, 1.0, default=0.85, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_4 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_4 = DecimalParameter(0.4, 1.0, default=0.4, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_5 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_5 = DecimalParameter(0.4, 1.0, default=0.56, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_6 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_6 = DecimalParameter(0.4, 1.0, default=0.68, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_7 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_7 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_8 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_8 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    buy_pump_pull_threshold_9 = DecimalParameter(1.5, 3.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_9 = DecimalParameter(0.4, 1.8, default=1.3, space='buy', decimals=3, optimize=False, load=True)

    buy_min_inc_1 = DecimalParameter(0.01, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_1 = DecimalParameter(25.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_1 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_1 = DecimalParameter(20.0, 40.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_2 = DecimalParameter(1.0, 10.0, default=2.6, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_min_2 = DecimalParameter(30.0, 40.0, default=32.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_2 = DecimalParameter(70.0, 95.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_diff_2 = DecimalParameter(30.0, 50.0, default=39.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_2 = DecimalParameter(30.0, 56.0, default=49.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_2 = DecimalParameter(0.97, 0.999, default=0.983, space='buy', decimals=3, optimize=False, load=True)
    buy_bb40_bbdelta_close_3 = DecimalParameter(0.005, 0.06, default=0.057, space='buy', optimize=False, load=True)
    buy_bb40_closedelta_close_3 = DecimalParameter(0.01, 0.03, default=0.023, space='buy', optimize=False, load=True)
    buy_bb40_tail_bbdelta_3 = DecimalParameter(0.15, 0.45, default=0.418, space='buy', optimize=False, load=True)
    buy_ema_rel_3 = DecimalParameter(0.97, 0.999, default=0.986, space='buy', decimals=3, optimize=False, load=True)
    buy_bb20_close_bblowerband_4 = DecimalParameter(0.96, 0.99, default=0.979, space='buy', optimize=False, load=True)
    buy_bb20_volume_4 = DecimalParameter(1.0, 20.0, default=10.0, space='buy', decimals=2, optimize=False, load=True)
    buy_ema_open_mult_5 = DecimalParameter(0.016, 0.03, default=0.019, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_5 = DecimalParameter(0.98, 1.0, default=0.999, space='buy', decimals=3, optimize=False, load=True)
    buy_ema_rel_5 = DecimalParameter(0.97, 0.999, default=0.982, space='buy', decimals=3, optimize=False, load=True)
    buy_ema_open_mult_6 = DecimalParameter(0.02, 0.03, default=0.025, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_6 = DecimalParameter(0.98, 0.999, default=0.984, space='buy', decimals=3, optimize=False, load=True)
    buy_volume_7 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_open_mult_7 = DecimalParameter(0.02, 0.04, default=0.03, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_7 = DecimalParameter(24.0, 50.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_rel_7 = DecimalParameter(0.97, 0.999, default=0.986, space='buy', decimals=3, optimize=False, load=True)
    buy_volume_8 = DecimalParameter(1.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_8 = DecimalParameter(36.0, 40.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)
    buy_tail_diff_8 = DecimalParameter(3.0, 10.0, default=3.5, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_9 = DecimalParameter(1.0, 4.0, default=1.0, space='buy', decimals=2, optimize=False, load=True)
    buy_ma_offset_9 = DecimalParameter(0.94, 0.99, default=0.97, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_9 = DecimalParameter(0.97, 0.99, default=0.985, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_9 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_9 = DecimalParameter(70.0, 90.0, default=88.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_9 = DecimalParameter(36.0, 65.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_10 = DecimalParameter(1.0, 8.0, default=2.4, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_10 = DecimalParameter(0.93, 0.97, default=0.944, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_10 = DecimalParameter(0.97, 0.99, default=0.994, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_10 = DecimalParameter(20.0, 40.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_11 = DecimalParameter(0.93, 0.99, default=0.939, space='buy', decimals=3, optimize=False, load=True)
    buy_min_inc_11 = DecimalParameter(0.005, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_11 = DecimalParameter(40.0, 60.0, default=56.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_11 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_11 = DecimalParameter(30.0, 48.0, default=48.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_11 = DecimalParameter(36.0, 56.0, default=38.0, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_12 = DecimalParameter(1.0, 10.0, default=1.7, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_12 = DecimalParameter(0.93, 0.97, default=0.936, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_12 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_12 = DecimalParameter(2.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_13 = DecimalParameter(1.0, 10.0, default=1.6, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_13 = DecimalParameter(0.93, 0.98, default=0.978, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_13 = DecimalParameter(-14.0, -7.0, default=-10.4, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_14 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_open_mult_14 = DecimalParameter(0.01, 0.03, default=0.014, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_14 = DecimalParameter(0.98, 1.0, default=0.986, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_14 = DecimalParameter(0.93, 0.99, default=0.97, space='buy', decimals=3, optimize=False, load=True)
    buy_volume_15 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_open_mult_15 = DecimalParameter(0.02, 0.04, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.954, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_15 = DecimalParameter(30.0, 50.0, default=28.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=False, load=True)
    buy_volume_16 = DecimalParameter(1.0, 10.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_16 = DecimalParameter(0.93, 0.97, default=0.952, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_16 = DecimalParameter(26.0, 50.0, default=31.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_16 = DecimalParameter(4.0, 8.0, default=2.8, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_17 = DecimalParameter(0.5, 8.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ma_offset_17 = DecimalParameter(0.93, 0.98, default=0.958, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_17 = DecimalParameter(-18.0, -10.0, default=-12.0, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_18 = DecimalParameter(1.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_18 = DecimalParameter(16.0, 32.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_18 = DecimalParameter(0.98, 1.0, default=0.982, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_19 = DecimalParameter(40.0, 70.0, default=50.0, space='buy', decimals=1, optimize=False, load=True)
    buy_chop_min_19 = DecimalParameter(20.0, 60.0, default=24.1, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_20 = DecimalParameter(0.5, 6.0, default=1.2, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_20 = DecimalParameter(20.0, 36.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_20 = DecimalParameter(14.0, 30.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)
    buy_volume_21 = DecimalParameter(0.5, 6.0, default=3.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_21 = DecimalParameter(10.0, 28.0, default=23.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_21 = DecimalParameter(18.0, 40.0, default=24.0, space='buy', decimals=1, optimize=False, load=True)

    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, default=79.5, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_bb_2 = DecimalParameter(72.0, 90.0, default=81, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_main_3 = DecimalParameter(77.0, 90.0, default=82, space='sell', decimals=1, optimize=False, load=True)
    sell_dual_rsi_rsi_4 = DecimalParameter(72.0, 84.0, default=73.4, space='sell', decimals=1, optimize=False, load=True)
    sell_dual_rsi_rsi_1h_4 = DecimalParameter(78.0, 92.0, default=79.6, space='sell', decimals=1, optimize=False, load=True)
    sell_ema_relative_5 = DecimalParameter(0.005, 0.05, default=0.024, space='sell', optimize=False, load=True)
    sell_rsi_diff_5 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)
    sell_rsi_under_6 = DecimalParameter(72.0, 90.0, default=79.0, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_1h_7 = DecimalParameter(80.0, 95.0, default=81.7, space='sell', decimals=1, optimize=False, load=True)
    sell_bb_relative_8 = DecimalParameter(1.05, 1.3, default=1.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_0 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=33.0, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=38.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_2 = DecimalParameter(34.0, 50.0, default=43.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_3 = DecimalParameter(0.06, 0.30, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_3 = DecimalParameter(38.0, 55.0, default=48.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_4 = DecimalParameter(0.3, 0.6, default=0.25, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_4 = DecimalParameter(40.0, 58.0, default=50.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=60.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.6, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=62.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_dec_profit_1 = DecimalParameter(0.01, 0.10, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_2 = DecimalParameter(0.05, 0.2, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.15, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.46, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.2, default=0.18, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_min_2 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.12, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.14, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_min_3 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.04, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=False, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)
    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.004, space='sell', optimize=False, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=8.0, space='sell', optimize=False, load=True)

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        max_profit = (trade.calc_profit_ratio(trade.max_rate) * 100)
        sell_tag = None

        if current_profit > self.sell_custom_profit_4.value and last_candle['rsi'] < self.sell_custom_rsi_4.value:
            return (f'custom_sell_profit_4_qtpylib_rsi_{last_candle["rsi"]}', current_profit)
        if current_profit > self.sell_custom_profit_3.value and last_candle['rsi'] < self.sell_custom_rsi_3.value:
            return (f'custom_sell_profit_3_qtpylib_rsi_{last_candle["rsi"]}', current_profit)
        if current_profit > self.sell_custom_profit_2.value and last_candle['rsi'] < self.sell_custom_rsi_2.value:
            return (f'custom_sell_profit_2_qtpylib_rsi_{last_candle["rsi"]}', current_profit)
        if current_profit > self.sell_custom_profit_1.value and last_candle['rsi'] < self.sell_custom_rsi_1.value:
            return (f'custom_sell_profit_1_qtpylib_rsi_{last_candle["rsi"]}', current_profit)
        if current_profit > self.sell_custom_profit_0.value and last_candle['rsi'] < self.sell_custom_rsi_0.value:
            return (f'custom_sell_profit_0_qtpylib_rsi_{last_candle["rsi"]}', current_profit)

        if current_profit < self.sell_custom_under_profit_3.value and last_candle['rsi'] > self.sell_custom_under_rsi_3.value:
            return (f'custom_sell_under_profit_3_qtpylib_rsi_{last_candle["rsi"]}', current_profit)
        if current_profit < self.sell_custom_under_profit_2.value and last_candle['rsi'] > self.sell_custom_under_rsi_2.value:
            return (f'custom_sell_under_profit_2_qtpylib_rsi_{last_candle["rsi"]}', current_profit)
        if current_profit < self.sell_custom_under_profit_1.value and last_candle['rsi'] > self.sell_custom_under_rsi_1.value:
            return (f'custom_sell_under_profit_1_qtpylib_rsi_{last_candle["rsi"]}', current_profit)

        if max_profit > self.sell_trail_profit_min_1.value and \
                current_profit < (max_profit - self.sell_trail_down_1.value):
            return (f'custom_sell_trail_qtpylib_profit_max_{max_profit}_current_profit_{current_profit}', current_profit)

        if max_profit > self.sell_trail_profit_min_2.value and \
                current_profit < (max_profit - self.sell_trail_down_2.value):
            return (f'custom_sell_trail_qtpylib_profit_max_{max_profit}_current_profit_{current_profit}', current_profit)

        if max_profit > self.sell_trail_profit_min_3.value and \
                current_profit < (max_profit - self.sell_trail_down_3.value):
            return (f'custom_sell_trail_qtpylib_profit_max_{max_profit}_current_profit_{current_profit}', current_profit)

        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        sell_reason = None

        if current_profit < self.sell_custom_stoploss_under_rel_1.value:
            if last_candle['rsi'] > (self.sell_custom_stoploss_under_rsi_diff_1.value + last_candle['rsi']):
                sell_reason = f'custom_stoploss_qtpylib_profit_max_{current_profit}_current_profit_{current_profit}'

        return sell_reason

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, 
                            current_time, entry_tag, side: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        dataframe['sma_offset_buy'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_buy.value) * self.low_offset_sma.value
        dataframe['sma_offset_sell'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_sell.value) * self.high_offset_sma.value
        dataframe['ema_offset_buy'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value) * self.low_offset_ema.value
        dataframe['ema_offset_sell'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value) * self.high_offset_ema.value
        dataframe['trima_offset_buy'] = ta.TRIMA(dataframe, timeperiod=self.base_nb_candles_buy.value) * self.low_offset_trima.value
        dataframe['trima_offset_sell'] = ta.TRIMA(dataframe, timeperiod=self.base_nb_candles_sell.value) * self.high_offset_trima.value
        dataframe['t3_offset_buy'] = ta.T3(dataframe, timeperiod=self.base_nb_candles_buy.value) * self.low_offset_t3.value
        dataframe['t3_offset_sell'] = ta.T3(dataframe, timeperiod=self.base_nb_candles_sell.value) * self.high_offset_t3.value
        dataframe['kama_offset_buy'] = ta.KAMA(dataframe, timeperiod=self.base_nb_candles_buy.value) * self.low_offset_kama.value
        dataframe['kama_offset_sell'] = ta.KAMA(dataframe, timeperiod=self.base_nb_candles_sell.value) * self.high_offset_kama.value

        dataframe['bb_upperband'] = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        dataframe['bb_middleband'] = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        dataframe['bb_lowerband'] = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        dataframe['bbpercent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=50)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe, timeperiod=14)
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['adxr'] = ta.ADXR(dataframe, timeperiod=14)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        dataframe['ultosc'] = ta.ULTOSC(dataframe)
        dataframe['kst'] = ta.KST(dataframe)
        dataframe['macd'] = ta.MACD(dataframe)[0]
        dataframe['macdsignal'] = ta.MACD(dataframe)[1]
        dataframe['macdhist'] = ta.MACD(dataframe)[2]
        dataframe['ppo'] = ta.PPO(dataframe)
        dataframe['pposignal'] = ta.PPO(dataframe)[1]
        dataframe['ppohist'] = ta.PPO(dataframe)[2]
        dataframe['fastk'], dataframe['fastd'] = ta.STOCHRSI(dataframe, timeperiod=14)
        dataframe['slowk'], dataframe['slowd'] = ta.STOCH(dataframe)
        dataframe['fisher'] = 0.5 * np.log((1 + dataframe['fastk']) / (1 - dataframe['fastk']))
        dataframe['fisher'] = dataframe['fisher'].fillna(0)
        dataframe['ao'] = ta.AO(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        dataframe['rocp'] = ta.ROCP(dataframe, timeperiod=14)
        dataframe['apo'] = ta.APO(dataframe)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe)
        dataframe['bop'] = ta.BOP(dataframe)
        dataframe['cmo'] = ta.CMO(dataframe)
        dataframe['dx'] = ta.DX(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['mom'] = ta.MOM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['rvi'] = ta.RVI(dataframe)
        dataframe['stoch_k'] = ta.STOCH(dataframe)[0]
        dataframe['stoch_d'] = ta.STOCH(dataframe)[1]
        dataframe['atr'] = ta.ATR(dataframe)
        dataframe['trix'] = ta.TRIX(dataframe)
        dataframe['ht_trendline'] = ta.HT_TRENDLINE(dataframe)
        dataframe['ht_sine'], dataframe['ht_leadsine'] = ta.HT_SINE(dataframe)
        dataframe['ht_phasor_inphase'], dataframe['ht_phasor_quadrature'] = ta.HT_PHASOR(dataframe)

        return dataframe

    def informative_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert isinstance(dataframe, DataFrame)
        assert isinstance(metadata, dict)

        dataframe = self.normal_tf_indicators(dataframe, metadata)
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        informative = self.informative_tf_indicators(informative, metadata)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_1h, ffill=True)
        skip_columns = [(s, s + "_" + self.inf_1h) for s in list(informative.columns) if s != 'date']
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_1h, ffill=True, skip_columns=skip_columns)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.buy_condition_1_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value)
            )
        if self.buy_condition_2_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value)
            )
        if self.buy_condition_3_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_4_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_5_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_6_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_7_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_8_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_9_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_10_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_11_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_12_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_13_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_14_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_15_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_16_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_17_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_18_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_19_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['sma_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_1.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_20_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )
        if self.buy_condition_21_enable.value:
            conditions.append(
                (dataframe['close'] < dataframe['ema_offset_buy']) &
                (dataframe['rsi'] < self.buy_rsi_1.value) &
                (dataframe['mfi'] < self.buy_mfi_2.value) &
                (dataframe['volume'] > self.buy_volume_2.value)
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        if self.sell_condition_1_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['sma_offset_sell']) &
                (dataframe['rsi'] > self.sell_rsi_bb_1.value)
            )
        if self.sell_condition_2_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['rsi'] > self.sell_rsi_bb_2.value)
            )
        if self.sell_condition_3_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['rsi'] > self.sell_rsi_main_3.value)
            )
        if self.sell_condition_4_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['rsi'] > self.sell_dual_rsi_rsi_4.value) &
                (dataframe['rsi'] > self.sell_dual_rsi_rsi_1h_4.value)
            )
        if self.sell_condition_5_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['rsi'] > self.sell_ema_relative_5.value) &
                (dataframe['rsi'] > self.sell_rsi_diff_5.value)
            )
        if self.sell_condition_6_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['sma_offset_sell']) &
                (dataframe['rsi'] > self.sell_rsi_under_6.value)
            )
        if self.sell_condition_7_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['ema_offset_sell']) &
                (dataframe['rsi'] > self.sell_rsi_1h_7.value)
            )
        if self.sell_condition_8_enable.value:
            conditions.append(
                (dataframe['close'] > dataframe['sma_offset_sell']) &
                (dataframe['rsi'] > self.sell_bb_relative_8.value)
            )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'] = 1

        return dataframe

def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif
