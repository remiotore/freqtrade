
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter, informative
import technical.indicators as ftt
import pandas_ta as pta

import math
import logging

logger = logging.getLogger(__name__)


buy_params = {
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "ewo_high": 3.001,
        "ewo_low": -10.289,
        "low_offset": 0.987,
        "lambo2_ema_14_factor": 0.979,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
        "downtrend_limit": 0.91,

        "buy_clucha_bbdelta_close": 0.049,
        "buy_clucha_bbdelta_tail": 1.146,
        "buy_clucha_close_bblower": 0.018,
        "buy_clucha_closedelta_close": 0.017,
        "buy_clucha_rocr_1h": 0.526,
    }

sell_params = {
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01,

        "sell_deadfish_profit": -0.063,
        "sell_deadfish_bb_factor": 0.954,
        "sell_deadfish_bb_width": 0.043,
        "sell_deadfish_volume_factor": 2.37
    }

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""

    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

def zlema(dataframe, timeperiod):
    lag =  int(math.floor((timeperiod - 1) / 2) )
    if isinstance(dataframe, Series):
        ema_data = dataframe  + (dataframe  - dataframe.shift(lag))
    else:
        ema_data = dataframe['close']  + (dataframe['close']  - dataframe['close'] .shift(lag))
    return ta.EMA(ema_data, timeperiod = timeperiod)

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']


def top_percent_change(dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
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





class price_down_latest_v25(IStrategy):
    INTERFACE_VERSION = 2
    """

    minimal_roi = {
        "0": 0.08,
        "20": 0.04,
        "40": 0.032,
        "87": 0.016,
        "201": 0,
        "202": -1
    }
    """
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01
            }
        ]

    minimal_roi = {
        "0": 0.50,
        
    }
   
    

    def informative_pairs(self):
        pairs = self.dp.current_whitelist

        informative_pairs.extend = [(pair, '1h') for pair in pairs]
        informative_pairs.extend = [(pair, '1d') for pair in pairs]
        informative_pairs = [(pair, '1d') for pair in pairs]
        
        
     

        return informative_pairs

    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data)-1):
            if row < len(row_data)/2:
                conditions.append(row_data[row] > row_data[row+1])
            else:
                conditions.append(row_data[row] < row_data[row+1])
        return reduce(lambda x, y: x & y, conditions)


    stoploss = -0.99


    buy_31_ma_offset = 0.962
    buy_31_ewo = -10.4
    buy_31_wr = -90.0
    buy_31_cti = -0.89

    buy_34_ma_offset = 0.93
    buy_34_dip = 0.005
    buy_34_ewo = -6.0
    buy_34_cti = -0.88
    buy_34_volume = 2.0

    buy_44_ma_offset = 0.982
    buy_44_ewo = -18.143
    buy_44_cti = -0.8
    buy_44_r_1h = -75.0

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=0.02206, optimize=is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, optimize=is_optimize_clucha)
    buy_clucha_close_bblower = DecimalParameter(0.001, 0.05, default=0.03669, optimize=is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, optimize=is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, optimize=is_optimize_clucha)

    is_optimize_local_uptrend = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_uptrend)

    sell_rsi_bb_2 = DecimalParameter(55.0, 75.0, default=70, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_main_3 = DecimalParameter(60.0, 90.0, default=82, space='sell', decimals=1, optimize=False, load=True)
    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, default=79.5, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_profit_11 = DecimalParameter(0.16, 0.45, default=0.20, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_11 = DecimalParameter(28.0, 40.0, default=34.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_0 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=34.0, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=35.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_2 = DecimalParameter(30.0, 50.0, default=37.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_3 = DecimalParameter(0.01, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_3 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_4 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_4 = DecimalParameter(35.0, 50.0, default=43.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_5 = DecimalParameter(0.01, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_5 = DecimalParameter(35.0, 50.0, default=45.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_6 = DecimalParameter(0.01, 0.1, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_6 = DecimalParameter(38.0, 55.0, default=48.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_7 = DecimalParameter(0.01, 0.1, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_7 = DecimalParameter(40.0, 58.0, default=54.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_8 = DecimalParameter(0.06, 0.1, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_8 = DecimalParameter(40.0, 50.0, default=55.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_9 = DecimalParameter(0.05, 0.14, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_9 = DecimalParameter(40.0, 60.0, default=54.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_10 = DecimalParameter(0.1, 0.14, default=0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_10 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_11 = DecimalParameter(0.16, 0.45, default=0.20, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_11 = DecimalParameter(28.0, 40.0, default=34.0, space='sell', decimals=2, optimize=False, load=True)



    sell_custom_roi_profit_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_rsi_1 = DecimalParameter(40.0, 56.0, default=50, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_2 = DecimalParameter(0.01, 0.20, default=0.04, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_rsi_2 = DecimalParameter(42.0, 56.0, default=50, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_3 = DecimalParameter(0.15, 0.30, default=0.08, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_rsi_3 = DecimalParameter(44.0, 58.0, default=56, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_profit_4 = DecimalParameter(0.3, 0.7, default=0.14, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_roi_rsi_4 = DecimalParameter(44.0, 60.0, default=58, space='sell', decimals=2, optimize=False, load=True)

    sell_custom_roi_profit_5 = DecimalParameter(0.01, 0.1, default=0.04, space='sell', decimals=2, optimize=False, load=True)

    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.11, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_stoploss_1 = DecimalParameter(-0.15, -0.03, default=-0.05, space='sell', decimals=2, optimize=False, load=True)

    sell_rsi_main = DecimalParameter(72.0, 90.0, default=80, space='sell', decimals=2, optimize=True, load=True)

    sell_custom_under_profit_0 = DecimalParameter(0.01, 0.4, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_0 = DecimalParameter(28.0, 40.0, default=35.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=57.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=58.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_4 = DecimalParameter(0.02, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_4 = DecimalParameter(50.0, 68.0, default=59.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_5 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_5 = DecimalParameter(46.0, 62.0, default=60.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_6 = DecimalParameter(0.03, 0.1, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_6 = DecimalParameter(44.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_7 = DecimalParameter(0.04, 0.1, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_7 = DecimalParameter(46.0, 60.0, default=54.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_8 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_8 = DecimalParameter(40.0, 58.0, default=55.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_9 = DecimalParameter(0.08, 0.14, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_9 = DecimalParameter(40.0, 60.0, default=54.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_10 = DecimalParameter(0.1, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_10 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_11 = DecimalParameter(0.16, 0.3, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_11 = DecimalParameter(24.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=buy_params['lambo2_ema_14_factor'], space='buy', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'], space='buy', optimize=True)

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(-20.0, -8.0,default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_deadfish)

    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.025
    ignore_roi_if_buy_signal = False

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    timeframe = '5m'
    inf_1h = '1h'
    

    process_only_new_candles = True
    startup_candle_count = 400

    

   

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }


    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle_1 = dataframe.iloc[-1].squeeze()
        previous_candle_2 = dataframe.iloc[-2].squeeze()
        previous_candle_3 = dataframe.iloc[-3].squeeze()
        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)



        if (last_candle is not None):













            if (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_1.value)):
                return 'trail_target_1'
            elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_2.value)):
                return 'trail_target_2'
            elif (current_profit > 3) & (last_candle['rsi'] > 85):
                 return 'RSI-85 target'




        


   
            
            if (current_profit > 0) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi']>50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell signal1'
            if (current_profit > 0) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &  (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                return 'sell signal2'

                return 'sell stoploss1'

            


            if (    (current_profit < self.sell_deadfish_profit.value)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value)
            ):
                return f"sell_stoploss_deadfish"


            if (-0.12 <= current_profit < -0.08):
                if (last_candle['close'] < last_candle['atr_high_thresh_1']) and (last_candle['cmf'] < -0.0):
                    return  'sell_stoploss_atr_1'
            if (-0.16 <= current_profit < -0.12):
                if (last_candle['close'] < last_candle['atr_high_thresh_2']) and  (last_candle['cmf'] < -0.0):
                    return  'sell_stoploss_atr_2'
            if (-0.2 <= current_profit < -0.16):
                if (last_candle['close'] < last_candle['atr_high_thresh_3']) and  (last_candle['cmf'] < -0.0):
                    return  'sell_stoploss_atr_3'
            if (-0.3 <= current_profit < -0.2):
                if (last_candle['close'] < last_candle['atr_high_thresh_4']) and (last_candle['cmf'] < -0.0):
                    return  'sell_stoploss_atr_4'
            if (current_profit < -0.3):
                if (last_candle['close'] < last_candle['atr_high_thresh_5']) and  (last_candle['cmf'] < -0.0):
                    return  'sell_stoploss_atr_5'
           

            

   
    


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1d') for pair in pairs]

        if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))






        return informative_pairs

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['sma_200_dec'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        sup_series = dataframe['low'].rolling(window = 5, center=True).apply(lambda row: self.is_support(row), raw=True).shift(2)
        dataframe['sup_level'] = Series(np.where(sup_series, np.where(dataframe['close'] < dataframe['open'], dataframe['close'], dataframe['open']), float('NaN'))).ffill()
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        crsi_closechange = dataframe['close'] / dataframe['close'].shift(1)
        crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        dataframe['crsi'] =  (ta.RSI(dataframe['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(dataframe['close'], 100)) / 3
        dataframe['r_480'] = williams_r(dataframe, period=480)

        inf_heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_close'] = inf_heikinashi['close']
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=168)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        
        
        
        
 
        return dataframe

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        df36h = dataframe.copy().shift( 432 ) # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift( 288 ) # TODO FIXME: This assumes 5m timeframe

        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (dataframe['volume_mean_long'] / dataframe['volume_mean_base'])

        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()

        dataframe['pnd_volume_warn'] = np.where((dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)

        return dataframe


    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['price_trend_long'] = (dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())


        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)


        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['stake_currency'] in ['USDT','BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()

        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_high_thresh_1'] = (dataframe['high'] - (dataframe['atr'] * 7.0))
        dataframe['atr_high_thresh_2'] = (dataframe['high'] - (dataframe['atr'] * 5.6))
        dataframe['atr_high_thresh_3'] = (dataframe['high'] - (dataframe['atr'] * 5.0))
        dataframe['atr_high_thresh_4'] = (dataframe['high'] - (dataframe['atr'] * 3.8))
        dataframe['atr_high_thresh_5'] = (dataframe['high'] - (dataframe['atr'] * 2.0))

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        dataframe['r_480'] = williams_r(dataframe, period=480)

        dataframe['tpct_change__2']   = self.top_percent_change_dca(dataframe,2)
        dataframe['tpct_change__12']  = self.top_percent_change_dca(dataframe,12)
        dataframe['tpct_change__144'] = self.top_percent_change_dca(dataframe,144)


        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['tcp_percent_4'] = top_percent_change(dataframe , 4)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)



        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)

        dataframe['moderi_32'] = moderi(dataframe, 32)
        dataframe['moderi_64'] = moderi(dataframe, 64)
        dataframe['moderi_96'] = moderi(dataframe, 96)

        dataframe['zlema_68'] = zlema(dataframe, 68)

        inf_heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_close'] = inf_heikinashi['close']
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=168)
        
        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h, ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        inf_tf = '1d'

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        informative['rsi'] = ta.RSI(informative, timeperiod=7)

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['sma_28'] = ta.SMA(dataframe, timeperiod=28)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)



        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['cci_25'] = ta.CCI(dataframe, source='hlc3', timeperiod=25) # doplnene na testovanie z NFIX
        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15) # doplnene na testovanie z NFIX

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)

        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()

        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

        dataframe['ema_13'] = ta.EMA(dataframe, timeperiod=13)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)


        dataframe['is_pump'] = (dataframe['high'] / dataframe['low'].shift(36) > 1.2).astype('int')
        dataframe['pump_warning'] = (dataframe['is_pump'].rolling(192).max() > 0).astype('int')

        dataframe['downtrend1']=(dataframe['high'] / dataframe['low'].shift(36))
        dataframe['downtrend2']=(dataframe['high'] / dataframe['low'].shift(200))
        dataframe['downtrend_final']=dataframe['downtrend1'] - dataframe['downtrend2']

        dataframe['long_term_price_drop_4_20'] = np.where((dataframe['close'].rolling(4).max() * 0.96 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_4_25'] = np.where((dataframe['close'].rolling(5).max() * 0.96 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_5_30'] = np.where((dataframe['close'].rolling(6).max() * 0.95 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_5_40'] = np.where((dataframe['close'].rolling(8).max() * 0.95 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_6_40'] = np.where((dataframe['close'].rolling(8).max() * 0.94 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_8_60'] = np.where((dataframe['close'].rolling(12).max() * 0.94 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_10_60'] = np.where((dataframe['close'].rolling(12).max() * 0.90 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_15_8h'] = np.where((dataframe['close'].rolling(96).max() * 0.85 > dataframe['close'].rolling(6).mean()), 1, 0)
        dataframe['long_term_price_drop_20_12h'] = np.where((dataframe['close'].rolling(144).max() * 0.8 > dataframe['close'].rolling(8).mean()), 1, 0)
        dataframe['long_term_price_drop_25_24h'] = np.where((dataframe['close'].rolling(288).max() * 0.75 > dataframe['close'].rolling(12).mean()), 1, 0)
        dataframe['long_term_price_drop_30_36h'] = np.where((dataframe['close'].rolling(432).max() * 0.7 > dataframe['close'].rolling(24).mean()), 1, 0)



        dataframe['long_term_price_drop'] = np.where(
            (

                (
                    dataframe['close'].rolling(4).max() * 0.96 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                 (
                    dataframe['close'].rolling(6).max() * 0.96 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                 (
                    dataframe['close'].rolling(6).max() * 0.95 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                 (
                    dataframe['close'].rolling(8).max() * 0.95 > dataframe['close'].rolling(6).mean()  #0 buys 20220101
                ) |

                (
                    dataframe['close'].rolling(8).max() * 0.94 > dataframe['close'].rolling(6).mean()
                ) |

                (
                    dataframe['close'].rolling(12).max() * 0.92 > dataframe['close'].rolling(6).mean()
                ) |

                (
                    dataframe['close'].rolling(24).max() * 0.90 > dataframe['close'].rolling(6).mean()
                ) |

                (
                    dataframe['close'].rolling(96).max() * 0.85 > dataframe['close'].rolling(6).mean()
                ) |

                (
                    dataframe['close'].rolling(144).max() * 0.8 > dataframe['close'].rolling(8).mean()
                ) |

                (
                    dataframe['close'].rolling(288).max() * 0.75 > dataframe['close'].rolling(12).mean()
                ) |

                (
                    dataframe['close'].rolling(432).max() * 0.7 > dataframe['close'].rolling(24).mean()
                )
            ), 1, 0)


        dataframe = self.pump_dump_protection(dataframe, metadata)


        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)



        return dataframe

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''

        lambo2_drop_4_20 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_4_20'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_4_20 , 'buy_tag'] += 'L_d_4_20 '
        conditions.append(lambo2_drop_4_20 )

        lambo2_drop_4_25 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_4_25'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_4_25 , 'buy_tag'] += 'L_d_4_25 '
        conditions.append(lambo2_drop_4_25 )

        lambo2_drop_5_30 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_5_30'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_5_30 , 'buy_tag'] += 'L_d_5_30 '
        conditions.append(lambo2_drop_5_30 )

        lambo2_drop_5_40 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_5_40'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_5_40 , 'buy_tag'] += 'L_d_5_40 '
        conditions.append(lambo2_drop_5_40 )

        lambo2_drop_6_40 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_6_40'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_6_40 , 'buy_tag'] += 'L_d_6_40 '
        conditions.append(lambo2_drop_6_40 )

        lambo2_drop_8_60 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_8_60'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_8_60 , 'buy_tag'] += 'L_d_8_60 '
        conditions.append(lambo2_drop_8_60 )

        lambo2_drop_10_60 = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_10_60'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_10_60 , 'buy_tag'] += 'L_d_10_60 '
        conditions.append(lambo2_drop_10_60 )

        lambo2_drop_15_8h = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_15_8h'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_15_8h , 'buy_tag'] += 'L_d_15_8h '
        conditions.append(lambo2_drop_15_8h )

        lambo2_drop_20_12h = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_20_12h'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_20_12h , 'buy_tag'] += 'L_d_20_12h '
        conditions.append(lambo2_drop_20_12h )

        lambo2_drop_25_24h = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_25_24h'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_25_24h , 'buy_tag'] += 'L_d_25_24h '
        conditions.append(lambo2_drop_25_24h )

        lambo2_drop_30_36h = (

            (dataframe['pump_warning'] == 0) &
            (dataframe['long_term_price_drop_30_36h'] == 1) &
            (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_1d'] > 20) &
            (dataframe['cci_25'] < -120.0) &

            (dataframe['close'] < dataframe['sma_15'] * 0.936) &  # doplnene na testovanie z NFIX
            (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value))

            
        )
        dataframe.loc[lambo2_drop_30_36h , 'buy_tag'] += 'L_d_30_36h '
        conditions.append(lambo2_drop_30_36h )


        

        
        buy1ewo = (
                (dataframe['pump_warning'] == 0) &
                (dataframe['rsi_fast'] <35)&
                (dataframe['long_term_price_drop'] == 1) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_1d'] > 20) &
                (dataframe['cci_25'] < -120.0) &

                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'buy_tag'] += 'buy1eworsi_'
        conditions.append(buy1ewo)

        NFIX29 = (

                (dataframe['close'] > (dataframe['sup_level_1h'] * 0.72)) &
                (dataframe['close'] < (dataframe['ema_16'] * 0.982)) &
                (dataframe['EWO'] < -10.0) &
                (dataframe['cti'] < -0.9)

        )
        dataframe.loc[NFIX29, 'buy_tag'] += 'NFIX29_'
        conditions.append(NFIX29)


        NFIX9= (

           (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
           (dataframe['close'] > (dataframe['sup_level_1h'] * 0.88)) &
           (dataframe['close'] < dataframe['sma_30'] * 0.99) &
           (dataframe['cti'] < -0.92) &
           (dataframe['EWO'] < -5.9) &
           (dataframe['cti_1h'] < -0.75) &
           (dataframe['crsi_1h'] > 8.0) &
           (dataframe['volume_mean_12'] > (dataframe['volume_mean_24'] * 1.05))


        )
        dataframe.loc[NFIX9, 'buy_tag'] += 'NFIX9_'
        conditions.append(NFIX9)


        NFINext31= (

            (dataframe['moderi_64'] == False) &
            (dataframe['close'] < dataframe['zlema_68'] * self.buy_31_ma_offset ) &
            (dataframe['EWO'] < self.buy_31_ewo) &
            (dataframe['r_480'] < self.buy_31_wr) &
            (dataframe['cti'] < self.buy_31_cti)




        )
        dataframe.loc[NFINext31, 'buy_tag'] += 'NFINext31_'
        conditions.append(NFINext31)


        NFINext34= (

            (dataframe['cti'] < self.buy_34_cti) &
            ((dataframe['open'] - dataframe['close']) / dataframe['close'] < self.buy_34_dip) &
            (dataframe['close'] < dataframe['ema_13'] * self.buy_34_ma_offset) &
            (dataframe['EWO'] < self.buy_34_ewo) &
            (dataframe['volume'] < (dataframe['volume_mean_4'] * self.buy_34_volume))

        )
        dataframe.loc[NFINext34, 'buy_tag'] += 'NFINext34_'
        conditions.append(NFINext34)

        NFINext44= (

            (dataframe['close'] < (dataframe['ema_16'] * self.buy_44_ma_offset)) &
            (dataframe['EWO'] < self.buy_44_ewo) &
            (dataframe['cti'] < self.buy_44_cti) &
            (dataframe['r_480_1h'] < self.buy_44_r_1h)

         )
        dataframe.loc[NFINext44, 'buy_tag'] += 'NFINext44_'
        conditions.append(NFINext44)

        clucHA = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value ) &
                
                        (dataframe['bb_lowerband2_40'].shift() > 0) &
                        (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value) &
                        (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value) &
                        (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value) &
                        (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()) &
                        (dataframe['close'] > (dataframe['sup_level_1h'] * 0.88)) &
                        (dataframe['ha_close'] < dataframe['ha_close'].shift()) 
                                      
                        
            )
        dataframe.loc[clucHA, 'buy_tag'] += 'clucHA_'
        conditions.append(clucHA)

        clucHA2 = (
                (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value ) &
                
                                                    
                (dataframe['ha_close'] < dataframe['ema_slow']) &
                (dataframe['ha_close'] < self.buy_clucha_close_bblower.value * dataframe['bb_lowerband2'])
            
            )
        dataframe.loc[clucHA2, 'buy_tag'] += 'clucHA2_'
        conditions.append(clucHA2) 

        local_uptrend = (
                (dataframe['ema_26'] > dataframe['ema_12']) &
                (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value) &
                (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100) &
                (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value) &
                (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000 ) 
            
            )
        dataframe.loc[local_uptrend, 'buy_tag'] += 'local_uptrend_'
        conditions.append(local_uptrend)       

          


                   



        

        buy2ewo = (
                (dataframe['pump_warning'] == 0) &
                (dataframe['rsi_fast'] < 35)&
                (dataframe['long_term_price_drop'] == 1) &
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_1d'] > 20) &
                (dataframe['cci_25'] < -120.0) &

                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'buy_tag'] += 'buy2ewo_'
        conditions.append(buy2ewo)

        vwap = (
                (dataframe['pump_warning'] == 0) &
                (dataframe['rsi_1d'] > 20) &
                (dataframe['close'] < dataframe['vwap_low']) &
                (dataframe['tcp_percent_4'] > 0.05) &
                (dataframe['cti'] < -0.8) &
                (dataframe['rsi'] < 35) &
                (dataframe['rsi_84'] < 60) &
                (dataframe['rsi_112'] < 60) &

                (dataframe['volume'] > 0)

        )
        dataframe.loc[vwap, 'buy_tag'] += 'vwap_'
        conditions.append(vwap)











        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1


       

        return dataframe



    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (   (dataframe['close']>dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']>50)&
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']>dataframe['rsi_slow'])

            )
            |
            (
                (dataframe['close']<dataframe['hma_50'])&
                (dataframe['close'] > (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                
                (dataframe['rsi_fast']>dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=0


        return dataframe


    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        trade.sell_reason = sell_reason + "_" + trade.buy_tag

        return True

def pct_change(a, b):
    return (b - a) / a

class Price_down_dca_latest_v25(price_down_latest_v25):
   

    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.1
    safety_order_volume_scale = 1.4

    def top_percent_change_dca(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['tpct_change_0']   = self.top_percent_change_dca(dataframe,0)
        dataframe['tpct_change_2']   = self.top_percent_change_dca(dataframe,2)
        dataframe['tpct_change_12']  = self.top_percent_change_dca(dataframe,12)
        dataframe['tpct_change_144'] = self.top_percent_change_dca(dataframe,144)
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None


        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        previous2_candle = dataframe.iloc[-3].squeeze()


        

        if (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']):
            return None

        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'buy':
                continue
            if order.status == "closed":
                count_of_buys += 1



        if 1 <= count_of_buys <= self.max_safety_orders:
            
            safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale,(count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale,(count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(f"Initiating safety order buy #{count_of_buys} for {trade.pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}') 
                    return None

        return None

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
        name="{0} Williams %R".format(period),
        )

    return WR * -100
