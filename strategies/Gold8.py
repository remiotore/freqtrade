# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import stoploss_from_open
import pandas_ta as pta
from pandas import DataFrame, Series
import talib.abstract as ta
from functools import reduce
import logging
import time
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IStrategy, IntParameter)
import math
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import merge_informative_pair
from datetime import datetime
from calendar import monthrange

from freqtrade.exchange import timeframe_to_minutes

from datetime import timedelta
from technical.indicators import SSLChannels
from technical.indicators import RMI, zema, ichimoku
import numpy as np
from scipy.signal import argrelextrema
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------

def top_percent_change_dca(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def ha_typical_price(bars):

            # Heiken Ashi
    heikinashi = qtpylib.heikinashi(bars)
    res = (heikinashi['high'] + heikinashi['low'] + heikinashi['close']) / 3.
    return Series(index=bars.index, data=res)

def ewo(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

class Gold8(IStrategy):
    """
    Gold1
    Indicators: 200 SMA, 50 SMA, RSI, Bollinger Bands
    author@: Karol Sokolowski (sokoow@gmail.com)
    version@: 2024-01-20

    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.


    # Optimal timeframe for the strategy.
    timeframe = '5m'
    inf_1h = '1h'
    inf_1d = '1d'
    can_short = False
    # Number of candles the strategy requires before producing valid signals
    # considering the 1h timeframe, 200hs = 8 days
    startup_candle_count: int = 200
    position_adjustment_enable = False
    timeframe_mins = timeframe_to_minutes(timeframe)

    # Minimal ROI designed for the strategy.
    #minimal_roi = {
    #    "0": DecimalParameter(0.02, 1.0, default=0.5, space='roi'), # exit immediately if achieved x% profit
    #    str(timeframe_mins * 3): DecimalParameter(0.02, 1.0, default=0.5, space='roi'),  # after 3 candles
    #    str(timeframe_mins * 6): DecimalParameter(0.02, 1.0, default=0.2, space='roi'), 
    #    str(timeframe_mins * 9): DecimalParameter(0.02, 1.0, default=0.3, space='roi'),  
    #    str(timeframe_mins * 12): DecimalParameter(0.02, 1.0, default=0.1, space='roi')
    #}
    minimal_roi = {
      "0": 0.168,
      "23": 0.041,
      "42": 0.024,
      "55": 0
    }
    # Stoploss:
    stoploss = -0.99  # Stoploss at 70%

    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.03  # Trailing stop activates at 3% profit
    # 5% from peak, trailing stop starts at 2% profit
    # should be lower than minimal_roi
    trailing_stop_positive_offset = 0.05 # stop will actually be 10% below the peak price once the price has moved 2% in your favor.
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle instead of running for past candles on every loop
    # @tip: during development, set to false to plot points for past candles also.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.05
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True

    # Define hyperopt parameters
    
    buy_buffer = DecimalParameter(0.50, 1.0, default=0.766, space='buy')
    sell_buffer = DecimalParameter(0.50, 1.0, default=0.98, space='sell')

    #sma50_crossed_sma200_above = IntParameter(1, 30, default=48, space='buy')  # Window of validity in hours
    close_crossed_above_sma200_validity_period = IntParameter(1, 72, default=22, space='buy')  # Window of validity in hours
    close_crossed_below_sma200_validity_period = IntParameter(1, 72, default=72, space='sell')  # Window of validity in hours
    #bands_crossover_validity_period = IntParameter(1, 30, default=24, space='buy')  # Window of validity in hours

    #buy_rsi = IntParameter(10, 40, default=30, space='buy')
    #sell_rsi = IntParameter(60, 90, default=70, space='sell')

    # The percentage of profit at which the trailing profit starts to activate.
    # Start trailing when 3% profit is reached
    trailing_profit_start_percent = 0.03

    # The distance (in percentage) from the maximum price that the trailing profit level should be set.
    # Trail by 1% below the highest price reached
    trailing_profit_percent = 0.01 

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'upper_bband': {'color': 'green'},
            'middle_bband': {'color': 'orange'},
            'lower_bband': {'color': 'red'},
            'sma200': {'color': '#00e1ff'},
            'sma50': {'color': '#fff700'}
        },
        'subplots': {
            'RSI': {
                'rsi': {'color': 'red'}
            },
            'MACD': {
                'macd': {'color': 'orange'},
                'macdsignal': {'color': 'red'},
                'macdhist': {'color': 'blue'},
            },
            'Stochastic': {
                'slowk': {'color': 'blue'},
                'slowd': {'color': 'orange'},
            },
            'ADX': {
                'adx': {'color': 'green'}
            }
        }
    }

    use_max_drawdown_protection = BooleanParameter(default=True, space='protection', optimize=True)
    use_stoploss_protection = BooleanParameter(default=True, space='protection', optimize=True)

    cooldown_lookback = IntParameter(2, 48, default=3, space="protection", optimize=False)

    maxdrawdown_loopback = IntParameter(12, 48, default=48, space='protection', optimize=True)
    maxdrawdown_trade_limit = IntParameter(1, 20, default=20, space='protection', optimize=True)
    maxdrawdown_stop_duration = IntParameter(12, 200, default=12, space='protection', optimize=True)
    maxdrawdown_max_allowed_drawdown = DecimalParameter(0.01, 0.2, default=0.2, space='protection', optimize=True)

    stoploss_lookback = IntParameter(2, 60, default=10, space="protection", optimize=True)
    stoploss_trade_limit = IntParameter(1, 2, default=1, space="protection", optimize=True)
    stoploss_stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    stoploss_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True)

    # is_optimize_32 = True
    # buy_rsi_fast_32 = IntParameter(20, 70, default=46, space='buy', optimize=is_optimize_32)
    # buy_rsi_32 = IntParameter(15, 50, default=19, space='buy', optimize=is_optimize_32)
    # buy_sma15_32 = DecimalParameter(0.900, 1, default=0.942, decimals=3, space='buy', optimize=is_optimize_32)
    # buy_cti_32 = DecimalParameter(-1, 0, default=-0.86, decimals=2, space='buy', optimize=is_optimize_32)

    is_optimize_ewo = True
    buy_rsi_fast = IntParameter(35, 50, default=47, space='buy', optimize=is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=27, space='buy', optimize=is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=0.946, space='buy', optimize=is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.952, space='buy', optimize=is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.126, space='buy', optimize=is_optimize_ewo)


    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0, space='sell', optimize=is_optimize_deadfish)

    sell_fastx = IntParameter(50, 100, default=75, space='sell', optimize=True)
    # short_ema_period = IntParameter(5, 20, default=12, space="buy")  # Short EMA period
    # long_ema_period = IntParameter(20, 50, default=26, space="buy")  # Long EMA period
    # signal_period = IntParameter(5, 15, default=9, space="buy")  # Signal EMA period
    # ppo_threshold_buy = DecimalParameter(0.0, 1.0, default=0.1, space="buy")  # PPO buy threshold
    # ppo_threshold_sell = DecimalParameter(-1.0, 0.0, default=-0.1, space="sell")  # PPO sell threshold

    # # sell params
    # is_optimize_sell_ewo3 = True
    # sell_cmf = DecimalParameter(-0.4, 0.0, default=-0.15, optimize = is_optimize_sell_ewo3)
    # #sell_ema = DecimalParameter(0.97, 0.99, default=0.987 , optimize = is_optimize_sell_ewo3)
    # sell_ewo3 = DecimalParameter(-3.0, 8.0, default=3 , optimize = is_optimize_sell_ewo3)
    # sell_r14_ewo3 = IntParameter(-40, 10, default=-20, optimize = is_optimize_sell_ewo3)
    # sell_rsi_ewo3 = IntParameter(20, 90, default=79, optimize = is_optimize_sell_ewo3)
    # sell_ema3_high = DecimalParameter(0, 0.99, default=0.987 , optimize = is_optimize_sell_ewo3)
    # sell_ema3_low = DecimalParameter(0, 0.99, default=0.987 , optimize = is_optimize_sell_ewo3)
    # sell_bb_delta3 = DecimalParameter(0.01, 0.04, default=0.025, optimize = is_optimize_sell_ewo3)
    # sell_bb_width3 = DecimalParameter(0.065, 0.135, default=0.095, optimize = is_optimize_sell_ewo3)
    # sell_closedelta3 = DecimalParameter(0.0, 18.0, default=3.0, optimize = is_optimize_sell_ewo3)
    # sell_bb_factor3 = DecimalParameter(0.5, 0.999, default=0.995, optimize = is_optimize_sell_ewo3)

    buy_params = {
        "base_nb_candles_buy": 24,
        "rsi_buy": 50,
        "rsi_buy2": 48,
        "ewo_high": 7.367,
        "ewo_low": -15.701,
        "low_offset": 0.995,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179,
        "low_offset_1": 0.995,
        "ewo_high2": 3.233,
        "high_offset_1": 0.969,
        "ewo_high": 5.262,
        "ewo_low": -8.164,
        "nasos_base_nb_candles_buy": 4,
        "nasos_ewo_high": 2.403,
        "nasos_ewo_high_2": -5.585,
        "nasos_ewo_low": -14.378,
        "nasos_lookback_candles": 19,
        "nasos_low_offset": 0.984,
        "nasos_low_offset_2": 0.942,
        "nasos_profit_threshold": 1.024,
        "nasos_rsi_buy": 72
        }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 16,
        "high_offset": 1.084,
        "high_offset_2": 1.401,
        "nasos_base_nb_candles_sell": 16,
        "nasos_high_offset": 1.084,
        "nasos_high_offset_2": 1.401,
        "nasos_pHSL": -0.15,
        "nasos_pPF_1": 0.016,
        "nasos_pPF_2": 0.024,
        "nasos_pSL_1": 0.014,
        "nasos_pSL_2": 0.022
    }

    base_nb_candles_buy = IntParameter(8, 30, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 30, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)
    move = IntParameter(35, 60, default=48, space='buy', optimize=True)
    rsi_buy = IntParameter(35, 60, default=buy_params['rsi_buy'], space='buy', optimize=True)
    mms = IntParameter(6, 20, default=12, space='buy', optimize=True)
    mml = IntParameter(300, 400, default=360, space='buy', optimize=True)

    fast_ewo = 50
    slow_ewo = 200

    is_optimize_ewo = False 
    low_offset_1 = DecimalParameter(0.985, 0.995, default=buy_params['low_offset_1'], space='buy', optimize=is_optimize_ewo)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high2'], space='buy', optimize=is_optimize_ewo)
    high_offset_1 = DecimalParameter(0.95, 1.10, default=buy_params['high_offset_1'], space='buy', optimize=is_optimize_ewo)
    rsi_buy2 = IntParameter(35, 60, default=buy_params['rsi_buy2'], space='buy', optimize=is_optimize_ewo)
    buy_rsi_fast_32 = IntParameter(20, 70, default=60, space='buy', optimize=True)
    buy_rsi_32 = IntParameter(15, 50, default=50, space='buy', optimize=True)
    buy_sma15_32 = DecimalParameter(0.900, 1, default=0.901, decimals=3, space='buy', optimize=True)
    buy_cti_32 = DecimalParameter(-1, 1, default=-0.85, decimals=2, space='buy', optimize=True)

    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4

    ewo_low = DecimalParameter(
        -20.0, -8.0, default=-20.0, load=True, space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=6.0, load=True, space='buy', optimize=True)


    # Multi Offset
    base_nb_candles_buy = IntParameter(
        5, 80, default=20, load=True, space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=20, load=True, space='sell', optimize=True)
    low_offset_sma = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_sma = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_ema = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_ema = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_trima = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_trima = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_t3 = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_t3 = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)
    low_offset_kama = DecimalParameter(
        0.9, 0.99, default=0.958, load=True, space='buy', optimize=True)
    high_offset_kama = DecimalParameter(
        0.99, 1.1, default=1.012, load=True, space='sell', optimize=True)

    ma_types = ['ema', 'kama']
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
        # 't3': {
        #     'low_offset': low_offset_t3.value,
        #     'high_offset': high_offset_t3.value,
        #     'calculate': ta.T3
        # },
        'kama': {
            'low_offset': low_offset_kama.value,
            'high_offset': high_offset_kama.value,
            'calculate': ta.KAMA
        }
    }

    # Hyperoptable Parameters
    tsi_short = DecimalParameter(5, 20, default=13, space="buy")  # Short EMA for TSI
    tsi_long = DecimalParameter(20, 50, default=25, space="buy")  # Long EMA for TSI
    tsi_signal = DecimalParameter(5, 15, default=7, space="buy")  # Signal EMA for TSI
    tsi_buy_threshold = DecimalParameter(-50.0, 0.0, default=-20.0, space="buy")  # Buy threshold
    tsi_sell_threshold = DecimalParameter(0.0, 50.0, default=20.0, space="sell")  # Sell threshold

    ftc_ma_period = IntParameter(10, 50, default=47, space="buy")  # Moving average period for FTC
    ftc_atr_multiplier = DecimalParameter(1.0, 3.0, default=2.096, space="buy")  # ATR multiplier for FTC boundaries


    smi_k_period = IntParameter(5, 20, default=14, space="buy")  # K period
    smi_d_period = IntParameter(3, 10, default=3, space="buy")  # D period
    smi_smooth_period = IntParameter(3, 10, default=3, space="buy")  # Smoothing period
    smi_buy_threshold = DecimalParameter(-50.0, 0.0, default=-30.0, space="buy")  # Buy threshold
    smi_sell_threshold = DecimalParameter(0.0, 50.0, default=30.0, space="sell") 

    # Hyperoptable Parameters
    kc_ema_period = IntParameter(10, 50, default=19, space="buy")  # EMA period
    kc_atr_multiplier = DecimalParameter(1.0, 3.0, default=2.664, space="buy")


    nasos_base_nb_candles_buy = IntParameter(
        2, 20, default=buy_params['nasos_base_nb_candles_buy'], space='buy', optimize=True)
    nasos_base_nb_candles_sell = IntParameter(
        2, 25, default=sell_params['nasos_base_nb_candles_sell'], space='sell', optimize=True)
    nasos_low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['nasos_low_offset'], space='buy', optimize=False)
    nasos_low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['nasos_low_offset_2'], space='buy', optimize=False)
    nasos_high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['nasos_high_offset'], space='sell', optimize=True)
    nasos_high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['nasos_high_offset_2'], space='sell', optimize=True)

      # Protection
    nasos_fast_ewo = 50
    nasos_slow_ewo = 200

    nasos_lookback_candles = IntParameter(
        1, 24, default=buy_params['nasos_lookback_candles'], space='buy', optimize=True)

    nasos_profit_threshold = DecimalParameter(1.0, 1.03,
                                        default=buy_params['nasos_profit_threshold'], space='buy', optimize=True)

    nasos_ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['nasos_ewo_low'], space='buy', optimize=False)
    nasos_ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['nasos_ewo_high'], space='buy', optimize=False)

    nasos_ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['nasos_ewo_high_2'], space='buy', optimize=False)

    nasos_rsi_buy = IntParameter(50, 100, default=buy_params['nasos_rsi_buy'], space='buy', optimize=False)

    # trailing stoploss hyperopt parameters
    # hard stoploss profit
    nasos_pHSL = DecimalParameter(-0.200, -0.040, default=-0.15, decimals=3,
                            space='sell', optimize=False, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    nasos_pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3,
                             space='sell', optimize=False, load=True)
    nasos_pSL_1 = DecimalParameter(0.008, 0.020, default=0.014, decimals=3,
                             space='sell', optimize=False, load=True)

    # profit threshold 2, SL_2 is used
    nasos_pPF_2 = DecimalParameter(0.040, 0.100, default=0.024, decimals=3,
                             space='sell', optimize=False, load=True)
    nasos_pSL_2 = DecimalParameter(0.020, 0.070, default=0.022, decimals=3,
                             space='sell', optimize=False, load=True)

    info_timeframes = ["1h", "1d"]

    ma_types = ['trima', 't3']
    ma_map = {
        'trima': {
            'low_offset': 0.932,
            'high_offset': 1.084,
            'calculate': ta.TRIMA
        },
        't3': {
            'low_offset': 0.935,
            'high_offset': 1.072,
            'calculate': ta.T3
        },
    }

    @staticmethod
    def calculate_smi(dataframe: DataFrame, k_period: int, d_period: int, smooth_period: int) -> DataFrame:
        """
        Calculate Stochastic Momentum Index (SMI) and Signal Line.
        """
        high_low_mean = (dataframe['high'] + dataframe['low']) / 2
        min_low = dataframe['low'].rolling(window=k_period).min()
        max_high = dataframe['high'].rolling(window=k_period).max()
        distance = max_high - min_low

        dataframe['smi_diff'] = high_low_mean - (min_low + (distance / 2))
        dataframe['smi_hl'] = distance

        dataframe['smi'] = (
            dataframe['smi_diff'].rolling(window=smooth_period).mean() /
            dataframe['smi_hl'].rolling(window=smooth_period).mean()
        ) * 100

        dataframe['smi_signal'] = dataframe['smi'].rolling(window=d_period).mean()

        return dataframe.copy()

    # def informative_pairs(self):
    #     pairs = self.dp.current_whitelist()
    #     informative_pairs = [(pair, '1h') for pair in pairs]

    #     # informative_pairs += [("BTC/USDT", "5m"),]
    #     # informative_pairs += [("SHIB/USDT", "5m"),]
    #     return informative_pairs


    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.

        informative_pairs = []
        for info_timeframe in self.info_timeframes:
          informative_pairs.extend([(pair, info_timeframe) for pair in pairs])
        
        return informative_pairs

    def informative_1d_indicators(self, metadata: dict, info_timeframe) -> DataFrame:


        informative_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)
        informative_1d["SMA10"] = ta.SMA(informative_1d['close'], timeperiod=10)
    
        return informative_1d


    def informative_1h_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h['safe_pump_24'] = ((((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / informative_1h['close'].rolling(24).min()) < 0.5) | (((informative_1h['open'].rolling(24).max() - informative_1h['close'].rolling(24).min()) / 1.75) > (informative_1h['close'] - informative_1h['close'].rolling(24).min())))
        

        return informative_1h


    def info_switcher(self, metadata: dict, info_timeframe) -> DataFrame:
        if info_timeframe == "1d":
          return self.informative_1d_indicators(metadata, info_timeframe)
        elif info_timeframe == "1h":
          return self.informative_1h_indicators(metadata, info_timeframe)
        else:
          raise RuntimeError(f"{info_timeframe} not supported as informative timeframe for BTC pair.")

    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })

        if self.use_max_drawdown_protection.value:
            prot.append(({
                "method": "MaxDrawdown",
                "lookback_period_candles": self.maxdrawdown_loopback.value,
                "trade_limit": self.maxdrawdown_trade_limit.value,
                "stop_duration_candles": self.maxdrawdown_stop_duration.value,
                "max_allowed_drawdown": self.maxdrawdown_max_allowed_drawdown.value,
            }))


        if self.use_stoploss_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_lookback.value,
                "trade_limit": self.stoploss_trade_limit.value,
                "stop_duration_candles": self.stoploss_stop_duration.value,
                "only_per_pair": self.stoploss_only_per_pair.value,
            })

        return prot
    @staticmethod
    def calculate_tsi(dataframe: DataFrame, short_period: int, long_period: int, signal_period: int) -> DataFrame:
        """
        Calculate TSI and TSI Signal Line.
        """
        close_diff = dataframe['close'].diff(1)
        abs_close_diff = close_diff.abs()

        ema1 = close_diff.ewm(span=short_period, adjust=False).mean()
        ema2 = ema1.ewm(span=long_period, adjust=False).mean()

        abs_ema1 = abs_close_diff.ewm(span=short_period, adjust=False).mean()
        abs_ema2 = abs_ema1.ewm(span=long_period, adjust=False).mean()

        dataframe['tsi'] = (ema2 / abs_ema2) * 100  # TSI as percentage
        dataframe['tsi_signal'] = dataframe['tsi'].ewm(span=signal_period, adjust=False).mean()

        return dataframe

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # def isFriday(date):
        #     return date.weekday() == 4
        
        # def isFridayNight(date):
        #     return date.weekday() == 4 & date.hour == 23 & date.minute == 59
        
        # def isEndOfMonth(date):
        #     return date.day == monthrange(date.year, date.month)[1]
        
        # dataframe['isFriday'] = dataframe['date'].apply(isFriday)
        # dataframe['isFridayNight'] = dataframe['date'].apply(isFridayNight)
        # dataframe['isEndOfMonth'] = dataframe['date'].apply(isEndOfMonth)

        # # Calculate the N-period SMA
        dataframe['sma200'] = ta.SMA(dataframe['close'], timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe['close'], timeperiod=50)

        # # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # # MACD
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        
        dataframe['upper_bband'] = bollinger['upperband']
        dataframe['middle_bband'] = bollinger['middleband']
        dataframe['lower_bband'] = bollinger['lowerband']

        # # Stochastic Oscillator
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowk'] = stoch['slowk']
        # dataframe['slowd'] = stoch['slowd']

        # # ADX
        # dataframe['adx'] =lower_bband ta.ADX(dataframe)

        # # sma / price crossover events
        dataframe['golden_cross'] = qtpylib.crossed_above(dataframe['sma50'], dataframe['sma200'])
        # dataframe['death_cross'] = qtpylib.crossed_below(dataframe['sma50'], dataframe['sma200'])
        
        dataframe['close_crossed_above_sma200'] = qtpylib.crossed_above(dataframe['close'], dataframe['sma200'])
        # dataframe['close_crossed_below_sma200'] = qtpylib.crossed_below(dataframe['close'], dataframe['sma200'])

        dataframe['upper_bband_crossed'] = qtpylib.crossed_above(dataframe['close'], dataframe['upper_bband'])
        dataframe['lower_bband_crossed'] = qtpylib.crossed_below(dataframe['close'], dataframe['lower_bband'])

        # # track the maximum favorable price movement since entry
        # dataframe['highest_price_since_entry'] = dataframe['close'].cummax()

        # # Identify downtrend by checking if previous 'n' candles had lower close prices
        n = 3  # Number of previous candles to check for a downtrend
        dataframe['downtrend'] = (
             (dataframe['close'].shift(n) > dataframe['close'].shift(n-1)) &
             (dataframe['close'].shift(n-1) > dataframe['close'].shift(n-2))
        )

        # # Define the number of candles to check for a reversal
        # n = 3  # Number of candles to check for reversal

        # # Check if the close price has started to increase after the downtrend
        # dataframe['downtrend_reversal'] = (
        #     (dataframe['close'] > dataframe['close'].shift(1)) &
        #     (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &
        #     (dataframe['close'].shift(2) <= dataframe['close'].shift(3))  # Ensure the previous trend was a downtrend
        # )
        
        # # Calculate the rolling mean of volume and fill NaNs with 0
        great_volume_threshold = 2.5  # This represents 2.5x the average volume

        dataframe['average_volume'] = dataframe['volume'].rolling(window=20, min_periods=1).mean().fillna(0)

        # # Check if the last volume is greater than 2.5 times the average volume for the  last 1 candle
        dataframe['has_great_volume'] = dataframe['volume'].shift(1) > (dataframe['average_volume'].shift(1) * great_volume_threshold)

        dataframe['price_trending_up_above_sma200_end_of_month'] = (
             (dataframe['close_crossed_above_sma200']) & (dataframe['close'] > dataframe['close'].shift(1)) & dataframe['has_great_volume']
        )
        
        # # % distance from sma200    
        # # ex: 10 (%) distant from sma200
        # # use the values of the last candle
        dataframe['current_price_distance_from_sma200'] = (dataframe['sma200'].shift(1) - dataframe['close'].shift(1)) / dataframe['sma200'].shift(1) * 100

        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])
  
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma2880'] = ta.SMA(dataframe, timeperiod=2880)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        # stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        # dataframe['fastk'] = stoch_fast['fastk']

        # dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        # dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        # dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        # dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)
        # dataframe['EWO'] = ewo(dataframe, 50, 200)

        # stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']

        stoch2 = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch2['fastk']
        dataframe['srsi_fd'] = stoch2['fastd']

        # bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband2'] = bollinger2['lower']
        # dataframe['bb_middleband2'] = bollinger2['mid']
        # dataframe['bb_upperband2'] = bollinger2['upper']

        # dataframe['bb_width'] = (
        #         (dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        # dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        # dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ewo'] = ewo(dataframe, 50, 200)
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['rmi_length_13'] = RMI(dataframe, length=13, mom=4)
        dataframe[f'cci_length_27'] = ta.CCI(dataframe, 27)

        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe['bb_lowerband'] = bollinger['lower']
        # dataframe['bb_middleband'] = bollinger['mid']
        # dataframe['bb_upperband'] = bollinger['upper']
        # dataframe["bb_percent"] = (
        #     (dataframe["close"] - dataframe["bb_lowerband"]) /
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        # )

        # # Bollinger bands
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        # bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        # dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        # dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        # dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        # ### Other BB checks
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        # dataframe['zero'] = 0
        # # Elliot
        dataframe['EWO'] = ewo(dataframe, self.fast_ewo, self.slow_ewo)
        dataframe.loc[dataframe['EWO'] > 0, "EWO_UP"] = dataframe['EWO']
        # dataframe.loc[dataframe['EWO'] < 0, "EWO_DN"] = dataframe['EWO']
        dataframe['EWO_UP'].ffill()
        # dataframe['EWO_DN'].ffill()
        # dataframe['EWO_MEAN_UP'] = dataframe['EWO_UP'].mean()
        # dataframe['EWO_MEAN_DN'] = dataframe['EWO_DN'].mean()
        # dataframe['EWO_UP_FIB'] = dataframe['EWO_MEAN_UP'] * 1.618
        # dataframe['EWO_DN_FIB'] = dataframe['EWO_MEAN_DN'] * 1.618

        for val in self.nasos_base_nb_candles_buy.range:
             dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # # Calculate all ma_sell values
        # for val in self.base_nb_candles_sell.range:
        #     dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.nasos_base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        #dataframe['ma_lo'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * (self.low_offset.value)
        # dataframe['ma_hi'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (self.high_offset.value)
        # dataframe['ma_hi_2'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * (self.high_offset_2.value)

        dataframe['OHLC4'] = (dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close']) / 4

        # # Check how far we are from min and max 
        # dataframe['max'] = dataframe['OHLC4'].rolling(self.mms.value).max() / dataframe['OHLC4'] - 1
        dataframe['min'] = abs(dataframe['OHLC4'].rolling(self.mms.value).min() / dataframe['OHLC4'] - 1)

        # dataframe['max_l'] = dataframe['OHLC4'].rolling(self.mml.value).max() / dataframe['OHLC4'] - 1
        # dataframe['min_l'] = abs(dataframe['OHLC4'].rolling(self.mml.value).min() / dataframe['OHLC4'] - 1)

        # # Apply rolling window operation to the 'OHLC4'column
        rolling_window = dataframe['OHLC4'].rolling(self.move.value) 
        rolling_max = rolling_window.max()
        # rolling_min = rolling_window.min()

        # # Calculate the peak-to-peak value on the resulting rolling window data
        ptp_value = rolling_window.apply(lambda x: np.ptp(x))

        # # Assign the calculated peak-to-peak value to the DataFrame column
        dataframe['move'] = ptp_value / dataframe['OHLC4']
        # dataframe['move_mean'] = dataframe['move'].mean()
        # dataframe['move_mean_x'] = dataframe['move'].mean() * 1.6
        # dataframe['exit_mean'] = rolling_min * (1 + dataframe['move_mean'])
        # dataframe['exit_mean_x'] = rolling_min * (1 + dataframe['move_mean_x'])
        # dataframe['enter_mean'] = rolling_max * (1 - dataframe['move_mean'])
        #dataframe['enter_mean_x'] = rolling_max * (1 - dataframe['move_mean_x'])
        dataframe['atr_pcnt'] = (ta.ATR(dataframe, timeperiod=5) / dataframe['OHLC4'])
        dataframe['EWO2'] = ewo(dataframe, 60, 220)

        dataframe = self.pump_dump_protection(dataframe, metadata)

        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['adx_100'] = ta.ADX(dataframe, timeperiod=100)
        dataframe['adx_200'] = ta.ADX(dataframe, timeperiod=200)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        dataframe['atr'] = ta.ATR(dataframe)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        
        dataframe['vwap_upperband'] = vwap_high
        dataframe['vwap_middleband'] = vwap
        dataframe['vwap_lowerband'] = vwap_low
        dataframe['vwap_width'] = ( (dataframe['vwap_upperband'] - dataframe['vwap_lowerband']) / dataframe['vwap_middleband'] ) * 100

        dataframe['ema_vwap_diff_50'] = ( ( dataframe['ema_50'] - dataframe['vwap_lowerband'] ) / dataframe['ema_50'] )

       # Offset
        for i in self.ma_types:
            dataframe[f'{i}_offset_buy'] = self.ma_map[f'{i}']['calculate'](
                dataframe, self.base_nb_candles_buy.value) * \
                self.ma_map[f'{i}']['low_offset']

        dataframe['daily_return'] = dataframe['close'].pct_change()

        # Check if each of the last 5 days had positive returns
        dataframe['positive_days'] = dataframe['daily_return'] > 0

        # Rolling sum of positive days over the last 5 days
        dataframe['positive_last_5_days'] = dataframe['positive_days'].rolling(5).sum()

        # info_tf = '5m'

        # informative = self.dp.get_pair_dataframe('BTC/USDT', timeframe=info_tf)
        # informative_btc = informative.copy().shift(1)

        # dataframe['btc_close'] = informative_btc['close']

        # informative = self.dp.get_pair_dataframe('SHIB/USDT', timeframe=info_tf)
        # informative_shib = informative.copy().shift(1)

        # dataframe['shib_close'] = informative_shib['close']

        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)
        dataframe['ftc_ma'] = ta.SMA(dataframe['close'], timeperiod=self.ftc_ma_period.value)

        # Average True Range (ATR)
        dataframe['atr_ftc'] = ta.ATR(dataframe, timeperiod=self.ftc_ma_period.value)

        # Future Trend Channel boundaries
        dataframe['ftc_upper'] = dataframe['ftc_ma'] + (dataframe['atr_ftc'] * self.ftc_atr_multiplier.value)
        dataframe['ftc_lower'] = dataframe['ftc_ma'] - (dataframe['atr_ftc'] * self.ftc_atr_multiplier.value)

        dataframe["DI_values"] = ta.PLUS_DI(dataframe) - ta.MINUS_DI(dataframe)
        dataframe["DI_cutoff"] = 0
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)
        
        dataframe = dataframe.copy()

        dataframe = dataframe.copy()
        dataframe = self.calculate_smi(
            dataframe,
            k_period=self.smi_k_period.value,
            d_period=self.smi_d_period.value,
            smooth_period=self.smi_smooth_period.value
        )

        def isWorkday(date):
            return date.weekday() >=0 and date.weekday() <=4

        dataframe['isWorkday'] = dataframe['date'].apply(isWorkday)


        window_size = 50  # Lookback period for Markov Chain computation
        dataframe['return'] = dataframe['close'].pct_change()

        # Define states (price movement)
        conditions = [
            (dataframe['return'] > 0.002),  # Up
            (dataframe['return'] < -0.002),  # Down
        ]
        choices = [1, -1]  # State 1 (Up), State -1 (Down)
        dataframe['state'] = np.select(conditions, choices, default=0)  # 0 = Sideways

        # Rolling transition matrix computation
        prob_up_list = []
        prob_down_list = []

        unique_states = [-1, 0, 1]

        for i in range(len(dataframe)):
            if i < window_size:
                prob_up_list.append(None)
                prob_down_list.append(None)
                continue

            # Extract rolling window of states
            window_states = dataframe['state'].iloc[i - window_size:i].dropna().values

            if len(window_states) < 2:
                prob_up_list.append(None)
                prob_down_list.append(None)
                continue

            # Build transition matrix
            transition_matrix = np.zeros((3, 3))

            for j in range(len(window_states) - 1):
                current_state = unique_states.index(window_states[j])
                next_state = unique_states.index(window_states[j + 1])
                transition_matrix[current_state, next_state] += 1

            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

            # Get last state
            last_state = unique_states.index(window_states[-1])
            next_state_probabilities = transition_matrix[last_state]

            prob_up_list.append(next_state_probabilities[2] * 100)
            prob_down_list.append(next_state_probabilities[0] * 100)

        dataframe['prob_up'] = prob_up_list
        dataframe['prob_down'] = prob_down_list

        for i in self.ma_types:
            dataframe[f'{i}_offset_buy'] = self.ma_map[f'{i}']['calculate'](
                dataframe, self.base_nb_candles_buy.value) * \
                self.ma_map[f'{i}']['low_offset']

        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['safe_dips'] = ((((dataframe['open'] - dataframe['close']) / dataframe['close']) < 0.02) &
                            (((dataframe['open'].rolling(2).max() - dataframe['close']) / dataframe['close']) < 0.14) &
                            (((dataframe['open'].rolling(12).max() - dataframe['close']) / dataframe['close']) < 0.32) &
                            (((dataframe['open'].rolling(144).max() - dataframe['close']) / dataframe['close']) < 0.5))
        
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        return dataframe

    """
    Add TA indicators to the given dataframe
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        tik = time.perf_counter()
  
        dataframe = self.normal_tf_indicators(dataframe, metadata)
        

        # inf_tf = '1d'
        # informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1d)
        # informative = self.informative_1d_indicators(informative,metadata)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        info_indicators = self.info_switcher(metadata, self.inf_1d)
        dataframe = merge_informative_pair(dataframe, info_indicators, self.timeframe, self.inf_1d, ffill=True)

        info_indicators = self.info_switcher(metadata, self.inf_1h)
        dataframe = merge_informative_pair(dataframe, info_indicators, self.timeframe, self.inf_1h, ffill=True)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] populate_indicators took: {tok - tik:0.4f} seconds.")

        return dataframe

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        tik = time.perf_counter()
        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe
        
        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()
        
        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)

        dataframe['tpct_change_0']   = top_percent_change_dca(dataframe,0)
        dataframe['tpct_change_1']   = top_percent_change_dca(dataframe,1)
        dataframe['tcp_percent_4'] =   top_percent_change_dca(dataframe , 4)
        dataframe = self.calculate_tsi(
            dataframe,
            short_period=int(self.tsi_short.value),
            long_period=int(self.tsi_long.value),
            signal_period=int(self.tsi_signal.value)
        )

        dataframe['kc_middle'] = ta.EMA(dataframe['close'], timeperiod=self.kc_ema_period.value)

        # Calculate ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.kc_ema_period.value)

        # Calculate upper and lower bands
        dataframe['kc_upper'] = dataframe['kc_middle'] + (dataframe['atr'] * self.kc_atr_multiplier.value)
        dataframe['kc_lower'] = dataframe['kc_middle'] - (dataframe['atr'] * self.kc_atr_multiplier.value)


        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] pump_dump_protection took: {tok - tik:0.4f} seconds.")
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # btc_dump = (
        #         (dataframe['btc_close'].rolling(24).max() >= (dataframe['btc_close'] * 1.03 ))
        # )
        
        # shib_dump = (
        #         (dataframe['shib_close'].rolling(12).max() >= (dataframe['shib_close'] * 1.03 ))
        # )    
        
        rsi_check = (
                (dataframe['rsi_84'].shift(1) < 60) &
                (dataframe['rsi_112'].shift(1) < 60)
        )

        short_rsi_check = (
                (dataframe['rsi_84'] > 70) &
                (dataframe['rsi_112'] > 70)
        )

        drop_percentage = 0.03
        conditions = []
        conditions_short = []
        dataframe.loc[:, 'enter_tag'] = ''
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0


        #dataframe['enter_tag'] = None
        #informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
        #informative = self.populate_indicators_informative(informative, metadata)
        # dataframe = dataframe.merge(
        #     informative[['price_drop']],
        #     left_index=True,
        #     right_index=True,
        #     how='left',
        #     suffixes=('', '_1h')
        # )
        # Define an entry condition based on golden cross

        #four_three = (
            #(dataframe['close'].rolling(17).max() >= (dataframe['close'] * 1.03)) &
            #(dataframe['close'] > dataframe['close'].shift(1)) &
            #(dataframe['close'] > dataframe['sma200']) &
            #(rsi_check) &
            #(dataframe['tsi'] < -5 ) &
            #(dataframe['tsi'].rolling(12).max() < 15) &
            #(dataframe['close'] > dataframe['ftc_lower']) &
            #(dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
            #(dataframe['srsi_fk'].shift(1) < 20) &
            #(dataframe['srsi_fd'].shift(1) < 15) &   # Price above lower band
            # (dataframe['r_14'] < -80) &
            #(dataframe['close'].shift(1) < dataframe['vwap_low'].shift(1)) &
         #   (dataframe['&s-extrema'] == -1) &
          #  (dataframe['minima'] == 1) &
            #(dataframe['cci_length_27'] > 0) &
            # (dataframe['close'] > dataframe['ema_16']) &
           # (dataframe['adx_100'] > 5) &
           # (dataframe['adx_200'] > 3)
        #) 
        #dataframe.loc[four_three, 'enter_tag'] += 'four_three'
        #dataframe.loc[four_three, 'enter_long'] = 1
        #conditions.append(four_three)

        volume_buy = (
           (dataframe['has_great_volume']) &
           (dataframe['has_great_volume'].shift(1) == 0) &
           (dataframe['adx_100'] > 5) &
           (dataframe['adx_200'] > 3) &
           (dataframe['srsi_fk'] >= 90) &
           (dataframe['srsi_fd'] >= 90) &
           (dataframe['r_14'] <= -10) &
           (dataframe['bb_delta'] > 0.005) &
           (dataframe['rsi'] > 50)
        ) 
        dataframe.loc[volume_buy, 'enter_tag'] += 'volume_buy'
        dataframe.loc[volume_buy, 'enter_long'] = 0
        conditions.append(volume_buy)


        # maxima_check = (
        #         (dataframe["DI_catch"] == 1)  # Condição DI_catch
        #         & (dataframe["maxima_check"] == 0)  # Condição maxima_check
        #         & (dataframe["maxima_check"].shift(5) == 1)  # Condição maxima_check anterior
        #         & (dataframe["volume"] > 0)  # Volume maior que 0
        #         #& (dataframe["rsi"] > 70)  # RSI acima de 70 (condição adicional para limitar entradas)
        # ) 
        # dataframe.loc[maxima_check, 'enter_tag'] += 'maxima_check'
        # dataframe.loc[maxima_check, 'enter_short'] = 1
        # conditions_short.append(maxima_check)

        # longdi = (
        #             (dataframe['dx']  > dataframe['mdi']) &
        #             (dataframe['adx'] > dataframe['mdi']) &
        #             (dataframe['pdi'] > dataframe['mdi']) &
        #             (dataframe['obv'] > dataframe['obv'].shift(1)) &
        #             (dataframe['mfi'] > 50) &
        #             (dataframe['atr'] < 0.2) &
        #             # (dataframe['downtrend']) &
        #             # (dataframe['has_great_volume']) &
        #             # (dataframe['positive_last_5_days'] == 3) &
        #             # (rsi_check)
        #             (shib_dump ==0)
        #     )
        # dataframe.loc[longdi, 'enter_tag'] += 'longdi'
        # conditions.append(longdi)

        # cross_sma = (
        #         (dataframe['golden_cross']) &  # Confirm a golden cross occurred
        #         (dataframe['close'] > dataframe['sma200']) &  # Ensure price is above SMA200
        #         (dataframe['close'] < dataframe['upper_bband']) &  # Price below upper Bollinger Band
        #         (dataframe['volume'] > dataframe['average_volume'] * 2.5)  # Ensure volume is significantly high
        # )
        # dataframe.loc[cross_sma, 'enter_tag'] += 'sgolden_cross_price_above_sma200_high_volume'
        # conditions.append(cross_sma)

        # superboost = (
        #         (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
        #         (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
        #         (dataframe['rsi'] > self.buy_rsi_32.value) &
        #         (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
        #         (dataframe['cti'] < self.buy_cti_32.value)
        # )
        # dataframe.loc[superboost, 'enter_tag'] += 'superboost'
        # dataframe.loc[superboost, 'enter_long'] = 1
        # conditions.append(superboost)

        #ewo = (
        #        (dataframe['rsi_fast'] < self.buy_rsi_fast.value) &
        #        (dataframe['close'] < dataframe['ema_8'] * self.buy_ema_low.value) &
        #        (dataframe['EWO'] > self.buy_ewo.value) &
        #        (dataframe['close'] < dataframe['ema_16'] * self.buy_ema_high.value) &
        #        (dataframe['rsi'] < self.buy_rsi.value)
        #)
        #dataframe.loc[ewo, 'enter_tag'] += 'ewo'
        #dataframe.loc[ewo, 'enter_long'] = 1
        #conditions.append(ewo)

        # buy1 = (
        #         (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
        #         (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
        #         (dataframe['rsi'] > self.buy_rsi_32.value) &
        #         (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
        #         (dataframe['cti'] < self.buy_cti_32.value) &
        #         (dataframe['has_great_volume']) &
        #         (btc_dump == 0)
        # )
        # dataframe.loc[buy1, 'enter_tag'] += 'buy1'
        # conditions.append(buy1)



        # Entry condition for price far from SMA200 and ending of a downtrend
        # sma_far = (
        #         (dataframe['current_price_distance_from_sma200'] >= 2) &
        #         (dataframe['close'] > dataframe['close'].shift(3)) &  # Ensure price is trending up
        #         (dataframe['close'] < dataframe['sma200']) &  # Ensure price is below SMA200
        #         (dataframe['downtrend'])  # Confirm previous downtrend
        # )
        # dataframe.loc[sma_far, 'enter_tag'] += 'price_far_from_sma200_great_volume'
        # conditions.append(sma_far)
        
        
        # if dataframe['price_trending_up_above_sma200_end_of_month'].all():
        #     dataframe.loc[:, ['enter_long', 'enter_tag']] = (1, 'price_trending_up_price_above_sma200_great_volume')

        #cross_price = (
        #        #(dataframe['price_drop_1h'] <= -drop_percentage ) &
        #        (dataframe['golden_cross'])
        #        & (dataframe['close'] > dataframe['close'].shift(1)) # Ensure price is trending up
        #        & (dataframe['close'] < dataframe['upper_bband']) # price below upper band
        #        & (dataframe['has_great_volume'])
        #)
        #dataframe.loc[cross_price, 'enter_tag'] += 'golden_cross_price_trending_up_price_below_upper_band_great_volume'
        #dataframe.loc[cross_price, 'enter_long'] = 1
        #conditions.append(cross_price)

        # # Stochastic and ADX
        # dataframe.loc[
        #     (
        #         (informative['price_drop'] <= -drop_percentage ) &
        #         (dataframe['slowk'] < 20) &  # Stochastic K below 20 (oversold)
        #         (dataframe['slowd'] < 20) &  # Stochastic D below 20 (oversold)
        #         (dataframe['adx'] > 25) &  # Strong trend
        #         (dataframe['close'] > dataframe['sma200'])  # Price above SMA200
        #     ),
        #     ['enter_long', 'enter_tag']
        # ] = (1, 'stochastic_oversold_adx_strong_trend')


        # RSI and price above SMA200
        # dataframe.loc[
        #     (
        #         (informative['price_drop'] <= -drop_percentage ) &
        #         (dataframe['rsi'] < 30) &  # RSI below 30 (oversold)
        #         (dataframe['close'] > dataframe['sma200'])  # Price above SMA200
        #     ),
        #     ['enter_long', 'enter_tag']
        # ] = (1, 'rsi_oversold_price_above_sma200')


        # dataframe.loc[
        #    (
        #        (dataframe['close'] > dataframe['trend']) &
        #        (informative['price_drop'] <= -drop_percentage ) &
        #        (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
        #        (dataframe['close'].shift(2) <= dataframe['trend'].shift(2)) &
        #        (dataframe['close'].shift(3) <= dataframe['trend'].shift(3)) &
        #        (dataframe['obv'] > dataframe['obv'].shift(1)) &
        #        (dataframe['obv'].shift(1) > dataframe['obv'].shift(2))
        #    ), 
        #    ['enter_long', 'enter_tag' ]
        # ] = ( 1, 'trend_entry')

        # high_drop = (
        #        (informative['price_drop_1'] + informative['price_drop_2'] <= -0.05 ) &
        #        (dataframe['obv'] > dataframe['obv'].shift(1)) &
        #        (dataframe['close'] < dataframe['trend']) &
        #        (dataframe['has_great_volume'])
        # )
        # dataframe.loc[high_drop, 'enter_tag'] += 'high_drop'
        # conditions.append(high_drop)


        nfi_33 = (
                (dataframe['close'] < (dataframe['ema_13'] * 0.978)) &
                (dataframe['ewo'] > 8) &
                (dataframe['cti'] < -0.88) &
                (dataframe['rsi'] < 32) &
                (dataframe['r_14'] < -98.0) &
                (dataframe['volume'] < (dataframe['volume_mean_4'] * 2.5)) &
                #(dataframe['isWorkday'] > 0) &
                (dataframe['adx'] > 30)
        )
        dataframe.loc[nfi_33, 'enter_tag'] += 'nfi_33'
        dataframe.loc[nfi_33, 'enter_long'] = 1
        conditions.append(nfi_33)

        bb = (
                (dataframe[f'rmi_length_13'] < 31) &
                (dataframe[f'cci_length_27'] <= -127) &
                (dataframe['srsi_fk'] < 32) &
                (dataframe['bb_delta'] > 0.019) &
                (dataframe['bb_width'] > 0.096) &
                (dataframe['closedelta'] > dataframe['close'] * 13.43 / 1000 ) &    # from BinH
                (dataframe['close'] < dataframe['bb_lowerband3'] * 0.995) &
                #(dataframe['isWorkday'] > 0) &
                (dataframe['adx'] > 30)
        ) 
        dataframe.loc[bb, 'enter_tag'] += 'bb'
        dataframe.loc[bb, 'enter_long'] = 1
        conditions.append(bb)

        ewo2 = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.nasos_base_nb_candles_buy.value}'] * self.nasos_low_offset_2.value)) &
                (dataframe['EWO'] > self.nasos_ewo_high_2.value) &
                (dataframe['rsi'] < self.nasos_rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[f'ma_sell_{self.nasos_base_nb_candles_sell.value}'] * self.nasos_high_offset.value)) &
                (dataframe['rsi'] < 25) &
                #(dataframe['isWorkday'] > 0) &
                (dataframe['adx'] > 30)
        ) 
        dataframe.loc[ewo2, 'enter_tag'] += 'ewo2'
        dataframe.loc[ewo2, 'enter_long'] = 1
        conditions.append(ewo2) 

        ewo1 = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.nasos_base_nb_candles_buy.value}'] * self.nasos_low_offset.value)) &
                (dataframe['EWO'] > self.nasos_ewo_high.value) &
                (dataframe['rsi'] < self.nasos_rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (
                    dataframe[f'ma_sell_{self.nasos_base_nb_candles_sell.value}'] * self.nasos_high_offset.value)) &
                #(dataframe['isWorkday'] > 0) &
                (dataframe['adx'] > 30) &
                (dataframe[f'rmi_length_13'] < 31) &
                (dataframe[f'cci_length_27'] <= -127) &
                (dataframe['srsi_fk'] < 32) &
                (
                    (dataframe['ewo'] < -8.114) |
                    (dataframe['ewo'] > 4.994)
                )
 
        )        
        dataframe.loc[ewo1, 'enter_tag'] += 'ewo1'
        dataframe.loc[ewo1, 'enter_long'] = 1
        conditions.append(ewo1)

        average = (
                (dataframe['close'] < dataframe['SMA10_1d']) &
                (dataframe['adx'] > 30) &
                (dataframe['lower_bband_crossed']) &
                (dataframe['bb_delta'] > 0.019) &
                (dataframe['bb_width'] > 0.096)
        )        
        dataframe.loc[average, 'enter_tag'] += 'average'
        dataframe.loc[average, 'enter_long'] = 1
        conditions.append(average)

        markov = (
                (dataframe['prob_up'] > 50) &
                # (dataframe['bb_delta'] > 0.019) &
                # (dataframe['bb_width'] > 0.096) &
                (dataframe['adx'] > 30) &
                (dataframe[f'rmi_length_13'] < 31) &
                (dataframe['lower_bband_crossed'])
                #(dataframe[f'cci_length_27'] <= -127) &
                #(dataframe['srsi_fk'] < 32)
        )        
        dataframe.loc[markov, 'enter_tag'] += 'markov'
        dataframe.loc[markov, 'enter_long'] = 1
        conditions.append(markov)
        
        # b1ewo = (
        #         (dataframe['rsi_fast'] < 35 ) &
        #         (dataframe['close'] < dataframe['ma_lo']) &
        #         (dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
        #         (dataframe['close'] < dataframe['enter_mean_x']) &
        #         (dataframe['close'].shift() < dataframe['enter_mean_x'].shift()) &
        #         (dataframe['rsi'] < self.rsi_buy.value) &
        #         (dataframe['atr_pcnt'] > dataframe['min']) &
        #         (rsi_check)
        # ) 
        # dataframe.loc[b1ewo, 'enter_tag'] += 'b1ewo'
        # conditions.append(b1ewo)

        # eworsi = (
        #     (dataframe['rsi_fast'] < 35) &
        #     (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_1.value)) &
        #     (dataframe['EWO2'] > self.ewo_high.value) &
        #     (dataframe['rsi'] < self.rsi_buy2.value) &
        #     (dataframe['volume'] > 0) &
        #     (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_1.value))
        # ) 
        # dataframe.loc[eworsi, 'enter_tag'] += 'eworsi'
        # conditions.append(eworsi)

        # cond_104 = (
        #     (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
        #     (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
        #     (dataframe['rsi'] > self.buy_rsi_32.value) &
        #     (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
        #     (dataframe['cti'] < self.buy_cti_32.value)
        # ) 
        # dataframe.loc[cond_104, 'enter_tag'] += '104'
        # dataframe.loc[cond_104, 'enter_long'] = 1
        # conditions.append(cond_104)

        wvap = (
                (dataframe['close'] < dataframe['vwap_lowerband']) &
                (dataframe['tpct_change_1'] > 0.04) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (rsi_check) &
                (dataframe['adx'] > 30)
                #(dataframe['isWorkday'])
        ) 
        dataframe.loc[wvap, 'enter_tag'] += 'wvap'
        dataframe.loc[wvap, 'enter_long'] = 1
        conditions.append(wvap)

        t3_offset = (
            (dataframe['close'] < dataframe['t3_offset_buy']) &
                (
                    (dataframe['ewo'] < -8.114) |
                    (dataframe['ewo'] > 4.994)
                ) &
                (dataframe['volume'] > 0) &
                (dataframe['adx'] > 30)
        )
        dataframe.loc[t3_offset, 'enter_tag'] += 't3_offset'
        dataframe.loc[t3_offset, 'enter_long'] = 1
        conditions.append(t3_offset)

        trima_offset = (
            (dataframe['close'] < dataframe['trima_offset_buy']) &
                (
                    (dataframe['ewo'] < -8.114) |
                    (dataframe['ewo'] > 4.994)
                ) &
                (dataframe['volume'] > 0) &
                (dataframe['adx'] > 30)
        )
        dataframe.loc[trima_offset, 'enter_tag'] += 'trima_offset'
        dataframe.loc[trima_offset, 'enter_long'] = 1
        conditions.append(trima_offset)

        sig19 = (
                (dataframe['ema_100_1h'] > dataframe['ema_200_1h']) &

                (dataframe['sma_200'] > dataframe['sma_200'].shift(36)) &
                (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                (dataframe['safe_dips']) &
                (dataframe['safe_pump_24_1h']) &

                (dataframe['close'].shift(1) > dataframe['ema_100_1h']) &
                (dataframe['low'] < dataframe['ema_100_1h']) &
                (dataframe['close'] > dataframe['ema_100_1h']) &
                (dataframe['rsi_1h'] > 45.0) &
                (dataframe['chop'] < 56.6) &
                (dataframe['volume'] > 0) &
                (dataframe['adx'] > 30)
        )
        dataframe.loc[sig19, 'enter_tag'] += 'sig19'
        dataframe.loc[sig19, 'enter_long'] = 1
        conditions.append(sig19)

        # for i in self.ma_types:
        #     cond = (
        #             (dataframe['close'] < dataframe[f'{i}_offset_buy']) &
        #         (
        #             (dataframe['ewo'] < self.ewo_low.value) |
        #             (dataframe['ewo'] > self.ewo_high.value)
        #         ) &
        #         (dataframe['volume'] > 0)
        #     )
        #     dataframe.loc[cond, 'enter_tag'] += f'{i}_offset'
        #     conditions.append(cond)
        
        # Applying buy conditions
        # if conditions:
        #     dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1

        # if conditions_short:
        #     dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1

        # Additional conditions to avoid buying
        #dont_buy_conditions = []
    
        # don't buy if there seems to be a Pump and Dump event.
        #dont_buy_conditions.append((dataframe['pnd_volume_warn'] < 0.0))
    
        # BTC price protection
        #dont_buy_conditions.append((dataframe['btc_rsi_8_1h'] < 35.0))
    
        # Applying don't buy conditions
        #if dont_buy_conditions:
        #    for condition in dont_buy_conditions:
        #        print('PUMP PROTECTION')
        #        dataframe.loc[condition, 'enter_long'] = 0

        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit >= 0.05:
            return -0.002

        if str(trade.enter_tag) == "buy_new" and current_profit >= 0.03:
            return -0.003

        return None

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
 
        last_candle = dataframe.iloc[-1].squeeze()
        candle_before_last = dataframe.iloc[-2].squeeze()
        candle_before_before_last = dataframe.iloc[-3].squeeze()
        trade_duration = (current_time - trade.open_date_utc).total_seconds()

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag

        strategy_tags = [ 'b1ewo', 'four_three' ]
        if buy_tag == 'sfour_short' and current_profit >= 0.03:
            return "short_sell"
        elif buy_tag == 'sfour_short' and trade_duration >= 24*3600:
            return "sshort_sell"
        elif buy_tag == 'four_three' and trade_duration >= 5*3600:
             return 'declutter'
        elif buy_tag in strategy_tags and current_profit >= 0.08:
            return 'b1sell'
        elif buy_tag in strategy_tags and current_profit >= 0.04 and trade_duration >= 3*24*3600:
            return 'tb1sell'
        elif current_profit  >= 0.06 and buy_tag not in strategy_tags: 
            return 'RR6'
        elif buy_tag not in strategy_tags and last_candle['close'] < candle_before_last['close'] and candle_before_last['close'] > candle_before_before_last['close'] and current_profit >= 0.05:
             return 'pullback'


    def adjust_entry_price(self, trade: Trade, order: Order | None, pair: str,
                           current_time: datetime, proposed_rate: float, current_order_rate: float,
                           entry_tag: str | None, side: str, **kwargs) -> float:
    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        return current_order_rate
    

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        print("Current SMI: %f", current_candle['smi'])
        
        return trade.stake_amount

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        conditions_short = []
        dataframe.loc[:, 'exit_tag'] = ''
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        rsi_check = (
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60)
        )

        # four_three_short = (
        #         #(dataframe['&s-extrema'] == 1) &
        #         (dataframe["high"] > dataframe["maxima"]) &
        #         (dataframe['adx_100'] > 5) &
        #         (dataframe['adx_200'] > 3) &
        #         (dataframe['rsi'] > 50)
        #         #(dataframe['r_14'] >= -3) &
        #         #(dataframe['rsi'] > 70) &
        #         # (dataframe['srsi_fd'] > 80) &
        #         # (dataframe['srsi_fk'] > 80)
        #         #(dataframe['rsi_84'] >=65) &
        #         #(dataframe['rsi_112'] >=60) 
        #         # (dataframe['close'].rolling(17).min() < (dataframe['close'] * 1.03)) &
        #         # (dataframe['srsi_fk'] > 80 ) &
        #         # (dataframe['srsi_fd'] > 80) &   # Price above lower band
        #         # (dataframe['r_14'] > -15) &
        #         # (dataframe['cci_length_27'] > 240) &
        #         # (dataframe['close'] > dataframe['vwap_low']) &
        #         # (dataframe['close'] > dataframe['trend']) &
        #         # (dataframe['bb_width'] < 0.05) &
        #         # (dataframe['bb_width'] > 0.03)
        #         # (dataframe['EWO'] > 1.0)
        #     ) 
        # dataframe.loc[four_three_short, 'exit_tag'] += 'exitlong'
        # dataframe.loc[four_three_short, 'exit_long'] = 1
        # conditions_short.append(four_three_short)


        markov_sell = (
            (dataframe['prob_down'] > 49)
            # (dataframe['close'].rolling(17).max() >= (dataframe['close'] * 1.03)) &
            # (dataframe['close'] > dataframe['close'].shift(1)) &
            # #(dataframe['close'] > dataframe['sma200']) &
            # (rsi_check) &
            # #(dataframe['tsi'] < -5 ) &
            # #(dataframe['tsi'].rolling(12).max() < 15) &
            # #(dataframe['close'] > dataframe['ftc_lower']) &
            # #(dataframe['EWO'] > dataframe['EWO_MEAN_UP']) &
            # (dataframe['srsi_fk'] < 5) &
            # (dataframe['srsi_fd'] < 5) &   # Price above lower band
            # (dataframe['r_14'] < -80) &
            # (dataframe['close'] < dataframe['vwap_low'])
        ) 
        dataframe.loc[markov_sell, 'exit_tag'] += 'markov_sell'
        dataframe.loc[markov_sell, 'exit_long'] = 1
        conditions.append(markov_sell)

        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        if trade.calc_profit_ratio(rate) < 0.01:
            return False
        return True
