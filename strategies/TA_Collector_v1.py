
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
import pandas_ta as pta

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, DatetimeIndex, merge
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open
from freqtrade.exchange import timeframe_to_prev_date
from functools import reduce
from technical.indicators import RMI, zema, ichimoku

import json 
import requests

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""

    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

def SROC(dataframe, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()

    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)

    return sroc

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe['low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe['close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

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

class TA_Collector_v1(IStrategy):




    buy_params = {
        "max_slip": 0.983,

        "buy_bb_width_1h": 0.954,
        "buy_roc_1h": 86,

        "buy_threshold": 0.003,
        "buy_bb_factor": 0.999,

        "buy_bb_delta": 0.025,
        "buy_bb_width": 0.095,

        "buy_cci": -116,
        "buy_cci_length": 25,
        "buy_rmi": 49,
        "buy_rmi_length": 17,
        "buy_srsi_fk": 32,

        "buy_closedelta": 17.922,
        "buy_ema_diff": 0.026,

        "buy_ema_high": 0.968,
        "buy_ema_low": 0.935,
        "buy_ewo": -5.001,
        "buy_rsi": 23,
        "buy_rsi_fast": 44,

        "buy_ema_high_2": 1.087,
        "buy_ema_low_2": 0.970,
        "buy_ewo_high_2": 4.179,
        "buy_rsi_ewo_2": 35,
        "buy_rsi_fast_ewo_2": 45,

        "buy_closedelta_local_dip": 12.044,
        "buy_ema_diff_local_dip": 0.024,
        "buy_ema_high_local_dip": 1.014,
        "buy_rsi_local_dip": 21,

        "buy_r_deadfish_bb_factor": 1.014,
        "buy_r_deadfish_bb_width": 0.299,
        "buy_r_deadfish_ema": 1.054,
        "buy_r_deadfish_volume_factor": 1.59,
        "buy_r_deadfish_cti": -0.115,
        "buy_r_deadfish_r14": -44.34,

        "buy_clucha_bbdelta_close": 0.049,
        "buy_clucha_bbdelta_tail": 1.146,
        "buy_clucha_close_bblower": 0.018,
        "buy_clucha_closedelta_close": 0.017,
        "buy_clucha_rocr_1h": 0.526,

        "buy_adx": 13,
        "buy_cofi_r14": -85.016,
        "buy_cofi_cti": -0.892,
        "buy_ema_cofi": 1.147,
        "buy_ewo_high": 8.594,
        "buy_fastd": 28,
        "buy_fastk": 39,

        "buy_gumbo_ema": 1.121,
        "buy_gumbo_ewo_low": -9.442,
        "buy_gumbo_cti": -0.374,
        "buy_gumbo_r14": -51.971,

        "buy_sqzmom_ema": 0.981,
        "buy_sqzmom_ewo": -3.966,
        "buy_sqzmom_r14": -45.068,

        "buy_nfix_39_ema": 0.912,

        "buy_nfix_49_cti": -0.105,
        "buy_nfix_49_r14": -81.827,
    }

    sell_params = {

        "sell_cmf": -0.046,
        "sell_ema": 0.988,
        "sell_ema_close_delta": 0.022,

        "sell_deadfish_profit": -0.063,
        "sell_deadfish_bb_factor": 0.954,
        "sell_deadfish_bb_width": 0.043,
        "sell_deadfish_volume_factor": 2.37,

        "sell_cti_r_cti": 0.844,
        "sell_cti_r_r": -19.99,
    }

    minimal_roi = {
        "0": 0.205,
        "81": 0.038,
        "292": 0.005,
    }

    timeframe = '1m'
    inf_5m = '5m'
    inf_1h = '1h'

    process_only_new_candles = True

    stoploss = -0.99

    use_custom_stoploss = True
    use_sell_signal = True



    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize= is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, optimize = is_optimize_break)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, optimize = is_optimize_break)

    is_optimize_local_uptrend = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize = False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_uptrend)

    is_optimize_local_dip = False
    buy_ema_diff_local_dip = DecimalParameter(0.022, 0.027, default=0.025, optimize = is_optimize_local_dip)
    buy_ema_high_local_dip = DecimalParameter(0.90, 1.2, default=0.942 , optimize = is_optimize_local_dip)
    buy_closedelta_local_dip = DecimalParameter(12.0, 18.0, default=15.0, optimize = is_optimize_local_dip)
    buy_rsi_local_dip = IntParameter(15, 45, default=28, optimize = is_optimize_local_dip)
    buy_crsi_local_dip = IntParameter(10, 18, default=10, optimize = False)

    is_optimize_ewo = False
    buy_rsi_fast = IntParameter(35, 50, default=45, optimize = is_optimize_ewo)
    buy_rsi = IntParameter(15, 35, default=35, optimize = is_optimize_ewo)
    buy_ewo = DecimalParameter(-6.0, 5, default=-5.585, optimize = is_optimize_ewo)
    buy_ema_low = DecimalParameter(0.9, 0.99, default=0.942 , optimize = is_optimize_ewo)
    buy_ema_high = DecimalParameter(0.95, 1.2, default=1.084 , optimize = is_optimize_ewo)

    is_optimize_ewo_2 = False
    buy_rsi_fast_ewo_2 = IntParameter(15, 50, default=45, optimize = is_optimize_ewo_2)
    buy_rsi_ewo_2 = IntParameter(15, 50, default=35, optimize = is_optimize_ewo_2)
    buy_ema_low_2 = DecimalParameter(0.90, 1.2, default=0.970 , optimize = is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(0.90, 1.2, default=1.087 , optimize = is_optimize_ewo_2)
    buy_ewo_high_2 = DecimalParameter(2, 12, default=4.179, optimize = is_optimize_ewo_2)

    is_optimize_r_deadfish = False
    buy_r_deadfish_ema = DecimalParameter(0.90, 1.2, default=1.087 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_bb_factor = DecimalParameter(0.90, 1.2, default=1.0 , optimize = is_optimize_r_deadfish)
    buy_r_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_r_deadfish)

    is_optimize_r_deadfish_protection = False
    buy_r_deadfish_cti = DecimalParameter(-0.6, -0.0, default=-0.5 , optimize = is_optimize_r_deadfish_protection)
    buy_r_deadfish_r14 = DecimalParameter(-60, -44, default=-60 , optimize = is_optimize_r_deadfish_protection)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.01,0.05, default=0.02206, optimize = is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, optimize = is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, optimize = is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, optimize = is_optimize_clucha)

    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.94, 1.2, default=0.97 , optimize = is_optimize_cofi)
    buy_fastk = IntParameter(0, 40, default=20, optimize = is_optimize_cofi)
    buy_fastd = IntParameter(0, 40, default=20, optimize = is_optimize_cofi)
    buy_adx = IntParameter(0, 30, default=30, optimize = is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    is_optimize_cofi_protection = False
    buy_cofi_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_cofi_protection)
    buy_cofi_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_cofi_protection)

    is_optimize_gumbo = False
    buy_gumbo_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_gumbo)
    buy_gumbo_ewo_low = DecimalParameter(-12.0, 5, default=-5.585, optimize = is_optimize_gumbo)

    is_optimize_gumbo_protection = False
    buy_gumbo_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_gumbo_protection)
    buy_gumbo_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_gumbo_protection)

    is_optimize_sqzmom_protection = False
    buy_sqzmom_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_sqzmom_protection)
    buy_sqzmom_ewo = DecimalParameter(-12 , 12, default= 0 , optimize = is_optimize_sqzmom_protection)
    buy_sqzmom_r14 = DecimalParameter(-100, -22, default=-50 , optimize = is_optimize_sqzmom_protection)

    is_optimize_nfix_39 = True
    buy_nfix_39_ema = DecimalParameter(0.9, 1.2, default=0.97 , optimize = is_optimize_nfix_39)

    is_optimize_nfix_49_protection = False
    buy_nfix_49_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_nfix_49_protection)
    buy_nfix_49_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_nfix_49_protection)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)

    is_optimize_check = False
    buy_roc_1h = IntParameter(-25, 200, default=10, optimize = is_optimize_check)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=0.3, optimize = is_optimize_check)


    is_optimize_slip = False
    max_slip = DecimalParameter(0.33, 1.00, default=0.33, decimals=3, optimize=is_optimize_slip , space='buy', load=True)


    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = False)

    is_optimize_sell_stoploss = False
    sell_cmf = DecimalParameter(-0.4, 0.0, default=0.0, optimize = is_optimize_sell_stoploss)
    sell_ema_close_delta = DecimalParameter(0.022, 0.027, default= 0.024, optimize = is_optimize_sell_stoploss)
    sell_ema = DecimalParameter(0.97, 0.99, default=0.987 , optimize = is_optimize_sell_stoploss)

    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_deadfish)

    is_optimize_bleeding = False
    sell_bleeding_cti = DecimalParameter(-0.9, -0.0, default=-0.5 , optimize = is_optimize_bleeding)
    sell_bleeding_r14 = DecimalParameter(-100, -44, default=-60 , optimize = is_optimize_bleeding)
    sell_bleeding_volume_factor = DecimalParameter(1, 2.5, default=1.0 , optimize = is_optimize_bleeding)

    is_optimize_cti_r = False
    sell_cti_r_cti = DecimalParameter(0.55, 1, default=0.5 , optimize = is_optimize_cti_r)
    sell_cti_r_r = DecimalParameter(-15, 0, default=-20 , optimize = is_optimize_cti_r)


    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
        informative_pairs.extend([(pair, self.inf_5m) for pair in pairs])

        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb_lowerband2'] = bollinger2['lower']
        informative_1h['bb_middleband2'] = bollinger2['mid']
        informative_1h['bb_upperband2'] = bollinger2['upper']
        
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=3)
        informative_1h['bb_lowerband3'] = bollinger3['lower']
        informative_1h['bb_middleband3'] = bollinger3['mid']
        informative_1h['bb_upperband3'] = bollinger3['upper']

        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h['rsi_4'] = ta.RSI(informative_1h, timeperiod=4)
        informative_1h['rsi_20'] = ta.RSI(informative_1h, timeperiod=20)

        inf_heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_open'] = inf_heikinashi['open']
        informative_1h['ha_high'] = inf_heikinashi['high']
        informative_1h['ha_low'] = inf_heikinashi['low']
        informative_1h['ha_close'] = inf_heikinashi['close']

        informative_1h['vwap'] = qtpylib.rolling_vwap(informative_1h)

        informative_1h['obv'] = ta.OBV(informative_1h['close'], informative_1h['volume'])

        macd = ta.MACD(informative_1h, 20, 5)
        informative_1h['macd'] = macd['macd']
        informative_1h['macd_signal'] = macd['macdsignal']
        informative_1h['macd_histogram'] = macd['macdhist']

        return informative_1h
        
    def informative_5m_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."

        informative_5m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_5m)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_5m), window=20, stds=2)
        informative_5m['bb_lowerband2'] = bollinger2['lower']
        informative_5m['bb_middleband2'] = bollinger2['mid']
        informative_5m['bb_upperband2'] = bollinger2['upper']
        
        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(informative_5m), window=20, stds=3)
        informative_5m['bb_lowerband3'] = bollinger3['lower']
        informative_5m['bb_middleband3'] = bollinger3['mid']
        informative_5m['bb_upperband3'] = bollinger3['upper']

        informative_5m['cmf'] = chaikin_money_flow(informative_5m, 20)

        informative_5m['rsi_14'] = ta.RSI(informative_5m, timeperiod=14)
        informative_5m['rsi_4'] = ta.RSI(informative_5m, timeperiod=4)
        informative_5m['rsi_20'] = ta.RSI(informative_5m, timeperiod=20)

        inf_heikinashi = qtpylib.heikinashi(informative_5m)
        informative_5m['ha_open'] = inf_heikinashi['open']
        informative_5m['ha_high'] = inf_heikinashi['high']
        informative_5m['ha_low'] = inf_heikinashi['low']
        informative_5m['ha_close'] = inf_heikinashi['close']

        informative_5m['vwap'] = qtpylib.rolling_vwap(informative_5m)

        informative_5m['obv'] = ta.OBV(informative_5m['close'], informative_5m['volume'])

        macd = ta.MACD(informative_5m, 20, 5)
        informative_5m['macd'] = macd['macd']
        informative_5m['macd_signal'] = macd['macdsignal']
        informative_5m['macd_histogram'] = macd['macdhist']

        return informative_5m



    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        return 1











    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        return None








































































































    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:












        return False


    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_20'] = ta.RSI(dataframe, timeperiod=20)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['vwap'] = qtpylib.rolling_vwap(dataframe)









        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

        macd = ta.MACD(dataframe, 20, 5)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_histogram'] = macd['macdhist']












































































































        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        informative_5m = self.informative_5m_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_5m, self.timeframe, self.inf_5m, ffill=True)

        dataframe = self.normal_tf_indicators(dataframe, metadata)

        postData = {
            "pair": metadata['pair'],
            "strategy": self.config['strategy'],
            "data": dataframe.iloc[-1:].to_json(orient = "records")
        }
        requests.post('http://192.168.1.7:9000/data-gateway/submit-data', json=postData)

        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe.loc[:, 'buy'] = 0
        return dataframe
































































































































































































































    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'sell'] = 0

        return dataframe

def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'









    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx

def momdiv(dataframe: DataFrame, mom_length: int = 10, bb_length: int = 20, bb_dev: float = 2.0, lookback: int = 30) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    hh = dataframe['high'].rolling(lookback).max()
    ll = dataframe['low'].rolling(lookback).min()
    coh = dataframe['high'] >= hh
    col = dataframe['low'] <= ll
    df = DataFrame({
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            "momdiv_sell": sell,
            "momdiv_coh": coh,
            "momdiv_col": col,
        }, index=dataframe['close'].index)
    return df

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']
