# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
from freqtrade.strategy import (IStrategy,Trade,Order,PairLocks,informative,BooleanParameter,CategoricalParameter,DecimalParameter,IntParameter,RealParameter,timeframe_to_minutes,timeframe_to_next_date,timeframe_to_prev_date,merge_informative_pair,stoploss_from_absolute,stoploss_from_open)
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
from pandas import Series , DataFrame
from collections import namedtuple


def rmi(dataframe:DataFrame , length=20, mom=5):
    df = dataframe.copy()
    df['maxup'] = (df['close'] - df['close'].shift(mom)).clip(lower=0)
    df['maxdown'] = (df['close'].shift(mom) - df['close']).clip(lower=0)
    df.fillna(0, inplace=True)
    df["emaInc"] = ta.EMA(df, price='maxup', timeperiod=length)
    df["emaDec"] = ta.EMA(df, price='maxdown', timeperiod=length)
    df['RMI'] = np.where(df['emaDec'] == 0, 0, 100 - 100 / (1 + df["emaInc"] / df["emaDec"]))
    return df["RMI"]

def zema(dataframe:DataFrame, period, field='close'):
    df = dataframe.copy()
    df['ema1'] = ta.EMA(df[field], timeperiod=period)
    df['ema2'] = ta.EMA(df['ema1'], timeperiod=period)
    df['d'] = df['ema1'] - df['ema2']
    df['zema'] = df['ema1'] + df['d']
    return df['zema']

def pcc(dataframe: DataFrame, period: int = 20, mult: int = 2):
    df = dataframe.copy()
    df['previous_close'] = df['close'].shift()
    df['close_change'] = (df['close'] - df['previous_close']) / df['previous_close'] * 100
    df['high_change'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_change'] = (df['low'] - df['close']) / df['close'] * 100
    df['delta'] = df['high_change'] - df['low_change']
    mid = zema(df, period, 'close_change')
    rangema = zema(df, period, 'delta')
    upper = mid + rangema * mult
    lower = mid - rangema * mult
    return upper, rangema, lower

def ssl_channel_atr(dataframe:DataFrame, length=7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['ssl_Down'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['ssl_Up'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['ssl_Down'], df['ssl_Up']

def sroc(dataframe:DataFrame, roclen=21, emalen=13, smooth=21):
    df = dataframe.copy()
    roc = ta.ROC(df, timeperiod=roclen)
    ema = ta.EMA(df, timeperiod=emalen)
    sroc = ta.ROC(ema, timeperiod=smooth)
    return sroc

def cmf(dataframe:DataFrame, length=20, fillna=False) -> Series:
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(length, min_periods=0).sum() / df['volume'].rolling(length, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def ssl_channel(dataframe:DataFrame, length=7):
    df = dataframe.copy()
    df["ATR"] = ta.ATR(df, timeperiod=14)
    df["smaHigh"] = df["high"].rolling(length).mean() + df["ATR"]
    df["smaLow"] = df["low"].rolling(length).mean() - df["ATR"]
    df["hlv"] = np.where(df["close"] > df["smaHigh"], 1, np.where(df["close"] < df["smaLow"], -1, np.NAN))
    df["hlv"] = df["hlv"].ffill()
    df["ssl_Down"] = np.where(df["hlv"] < 0, df["smaHigh"], df["smaLow"])
    df["ssl_Up"] = np.where(df["hlv"] < 0, df["smaLow"], df["smaHigh"])
    return df["ssl_Down"], df["ssl_Up"]

def ewo(dataframe:DataFrame, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif

def wavetrend(dataframe:DataFrame, chlen:int= 10, avg:int= 21, smalen:int= 4) ->Series:
    df = dataframe.copy()
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['esa'] = ta.EMA(df['hlc3'], timeperiod=chlen)
    df['d'] = ta.EMA((df['hlc3'] - df['esa']).abs(), timeperiod=chlen)
    df['ci'] = (df['hlc3'] - df['esa']) / (0.015 * df['d'])
    df['tci'] = ta.EMA(df['ci'], timeperiod=avg)
    df['wt1'] = df['tci']
    df['wt2'] = ta.SMA(df['wt1'], timeperiod=smalen)
    df['wt1-wt2'] = df['wt1'] - df['wt2']
    return df['wt1'], df['wt2']

def t3(dataframe:DataFrame, length:int= 5) -> Series:
    df = dataframe.copy()
    df['xe1'] = ta.EMA(df['close'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b*b*b
    c2 = 3*b*b+3*b*b*b
    c3 = -6*b*b-3*b-3*b*b*b
    c4 = 1+3*b+b*b*b+3*b*b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']
    return df['T3Average']

def tsi(dataframe: DataFrame, slow_len: int, fast_len: int, fillna=False) -> Series:
    df = dataframe.copy()
    min_periods_slow = 0 if fillna else slow_len
    min_periods_fast = 0 if fillna else fast_len
    close_diff            = df['close'].diff()
    close_diff_abs        = close_diff.abs()
    smooth_close_diff     = close_diff.ewm(span=slow_len, min_periods=min_periods_slow, adjust=False).mean().ewm(span=fast_len, min_periods=min_periods_fast, adjust=False).mean()
    smooth_close_diff_abs = close_diff_abs.ewm(span=slow_len, min_periods=min_periods_slow, adjust=False).mean().ewm(span=fast_len, min_periods=min_periods_fast, adjust=False).mean()
    tsi = smooth_close_diff / smooth_close_diff_abs * 100
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return tsi

def trend(dataframe:DataFrame , length:int = 10 , multiplier:float = 3):
    df = dataframe.copy()
    t = pta.supertrend(high=df['high'] , low=df['low'] , close= df['close'] , length=length , multiplier= multiplier)
    df['trend'] = t[t.columns[0]]
    df['dir'] = t[t.columns[1]]
    df['long'] = t[t.columns[2]]
    df['short'] = t[t.columns[3]]
    return df['trend'] , df['dir'] , df['long'] , df['short']

def heiken(dataframe:DataFrame):
    df = dataframe.copy()
    heikinashi = qtpylib.heikinashi(df)
    df['ha_open'] = heikinashi['open']
    df['ha_close'] = heikinashi['close']  
    return df['ha_open'] , df['ha_close']

def vwma(dataframe: DataFrame, length: int = 10):
    pv = dataframe['close'] * dataframe['volume']
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe['volume'], timeperiod=length))
    return vwma

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series((highest_high - dataframe["close"]) / (highest_high - lowest_low),name=f"{period} Williams %R")
    return WR * -100

def qqe(dataframe:DataFrame , length:int = 20 , smooth:int = 5 , factor:float = 4.7):
    df = dataframe.copy()
    q = pta.qqe(close= df['close'] , length= length , smooth= smooth , factor= factor)
    df['qqe'] = q[q.columns[0]]
    df['qqel'] = q[q.columns[2]]
    df['qqes'] = q[q.columns[3]]
    return df['qqe'] , df['qqel'] , df['qqes']

def range_percent_change(dataframe: DataFrame, method, length: int) -> float:
    if method == 'HL':
        return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe[
            'low'].rolling(length).min()
    elif method == 'OC':
        return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe[
            'close'].rolling(length).min()
    else:
        raise ValueError(f"Method {method} not defined!")

def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend

def supertrend(dataframe:pd.DataFrame , period:int = 5 , multiplyer:float = 3):
    t = pta.supertrend(high=dataframe['high'] , low=dataframe['low'] , close=dataframe['close'] , length= period , multiplier= multiplyer)
    trend = t[t.columns[0]]
    direction = t[t.columns[1]]
    long = t[t.columns[2]]
    short = t[t.columns[3]]

    return trend , direction , long , short

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df,window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def top_percent_change(dataframe: DataFrame, length: int) -> float:
        if length == 0:
            return (dataframe['open'] - dataframe['close']) / dataframe['close']
        else:
            return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

_trend_length = 14
_bb_smooth_length=4

def iff(a,b,c):
    if a:
        return b
    else:
        return c


class pol313(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short: bool = True
    max_open_trades = 1

    # minimal_roi = {
    #     "0": 0.04
    # }

    # stoploss = -0.9
    #######################################
    # Buy hyperspace params:
    buy_params = {
        "buy_cmf": -0.162,
        "buy_percent": -0.776,
        "buy_width": 0.034,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_cmf": 0.044,
        "sell_percent": 1.495,
        "sell_width": 0.016,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.203,
        "17": 0.082,
        "37": 0.032,
        "104": 0
    }

    # Stoploss:
    stoploss = -0.349
    #######################################
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 400

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "kc_upper": {"color": "#ff0000"},
                "kc_lower": {"color": "#ffff00"},
            },
            "subplots": {
                "CMF": {
                    "cmf": {"color": "red"},
                }
            }
        }

    buy_percent = DecimalParameter(-2.0 , 0 , default= -1 , decimals= 3 , space='buy' , optimize= True)
    sell_percent = DecimalParameter(0 , 2.0 , default= 1 , decimals= 3 , space='sell' , optimize= True)

    buy_width = DecimalParameter(0.0 , 0.1 , default= 0.05 , decimals= 3 , space='buy' , optimize= True)
    sell_width = DecimalParameter(0.0 , 0.1 , default= 0.05 , decimals= 3 , space='sell' , optimize= True)

    buy_cmf = DecimalParameter(-0.6 , 0 , default= -0.05 , decimals= 3 , space='buy' , optimize= True)
    sell_cmf = DecimalParameter(0 , 0.6 , default= 0.05 , decimals= 3 , space='sell' , optimize= True)

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['tema'] = pta.tema(df['close'] , 6)
        df['ema5'] = pta.ema(df['close'] , 5)
        df['ema10'] = pta.ema(df['close'] , 10)
        df['ema20'] = pta.ema(df['close'] , 20)
        df['ema25'] = pta.ema(df['close'] , 25)
        df['ema30'] = pta.ema(df['close'] , 30)
        df['ema40'] = pta.ema(df['close'] , 40)
        df['ema50'] = pta.ema(df['close'] , 50)
        df['ema50'] = pta.ema(df['close'] , 50)
        df['ema100'] = pta.ema(df['close'] , 100)
        df['ema150'] = pta.ema(df['close'] , 150)
        df['ema200'] = pta.ema(df['close'] , 200)

        df['cti'] = pta.cti(df["close"], length=20)
        df["mfi"] = ta.MFI(df)
        df['roc'] = ta.ROC(df, timeperiod=4)
        df['mp']  = ta.RSI(df['roc'], timeperiod=8)
        df['sar'] = ta.SAR(df)
        df['rocr'] = ta.ROCR(df['close'], timeperiod=168)
        df['mom'] = ta.MOM(df, timeperiod=14)
        df['adx'] = ta.ADX(df, timeperiod=14)
        df['cci'] = ta.CCI(df)

        df["ao"] = qtpylib.awesome_oscillator(df)
        df['cmf'] = cmf(dataframe= df , length= 20)
        df['sroc'] = sroc(dataframe= df , roclen= 21 , emalen= 13 , smooth= 21)
        df['ewo'] = ewo(df , 5 ,35)
        df['tsi'] = tsi(df , slow_len= 26 , fast_len= 6)
        df['rmi'] = rmi(df)
        df['t3'] = t3(dataframe= df , length= 7)
        df['mac'] = self.mac(df, 20, 50)
        df['streak'] = self.ma_streak(df, period=4)

        df["ssl_Down"], df["ssl_Up"] = ssl_channel(dataframe= df , length= 7)
        df['wt1'], df['wt2'] = wavetrend(dataframe= df , chlen=10 , avg= 21 , smalen= 4)
        

        heikinashi = qtpylib.heikinashi(df)
        df['ha_open'] = heikinashi['open']
        df['ha_close'] = heikinashi['close']
        df['ha_high'] = heikinashi['high']
        df['ha_low'] = heikinashi['low']

        t = pta.supertrend(high=df['high'] , low=df['low'] , close= df['close'] , length=24 , multiplier= 2.5)
        df['long'] = t[t.columns[2]]
        df['short'] = t[t.columns[3]]

        streak = abs(int(df['streak'].iloc[-1]))
        streak_back_close = df['close'].shift(streak + 1)
        df['streak_roc'] = 100 * (df['close'] - streak_back_close) / streak_back_close

        pcc = self.pcc(df, period=20, mult=2)
        df['pcc_lower'] = pcc.lowerband
        df['pcc_upper'] = pcc.upperband

        df['r_14'] = williams_r(df, period=14)
        
        stoch_fast = ta.STOCHF(df, 5, 3, 0, 3, 0)
        df['fastd'] = stoch_fast['fastd']
        df['fastk'] = stoch_fast['fastk']

        df['plus_di'] = ta.PLUS_DI(df, timeperiod=25)
        df['minus_di'] = ta.MINUS_DI(df, timeperiod=25)

        bollinger2_40 = qtpylib.bollinger_bands(ha_typical_price(df), window=40, stds=2)
        df['lower40'] = bollinger2_40['lower']
        df['mid40'] = bollinger2_40['mid']
        df['upper40'] = bollinger2_40['upper']
        df['delta'] = ((df['mid40'] - df['lower40']) / (df['mid40'])).abs()   # distance between middle and lower band
        df['tail'] = (df['ha_close'] - df['ha_low']).abs()      # distance between close and low
        df['head'] = (df['ha_high'] - df['ha_close']).abs()     # distance between high and close

        vwap_low, vwap, vwap_high = VWAPB(df, 20, 1)
        df['vwap_upper'] = vwap_high
        df['vwap_mid'] = vwap
        df['vwap_lower'] = vwap_low
        df['vwap_width'] = ( (df['vwap_upper'] - df['vwap_lower']) / df['vwap_mid'] ) * 100

        weighted_bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
        df["wbb_upper"] = weighted_bollinger["upper"]
        df["wbb_lower"] = weighted_bollinger["lower"]
        df["wbb_mid"] = weighted_bollinger["mid"]
        df["wbb_percent"] = ((df["close"] - df["wbb_lower"]) / (df["wbb_upper"] - df["wbb_lower"]))
        df["wbb_width"] = ((df["wbb_upper"] - df["wbb_lower"]) / df["wbb_mid"])

        keltner = qtpylib.keltner_channel(df , window= 24 , atrs=1.6)
        df["kc_upper"] = keltner["upper"]
        df["kc_lower"] = keltner["lower"]
        df["kc_mid"] = keltner["mid"]
        df["kc_percent"] = ((df["close"] - df["kc_lower"]) / ((df["kc_upper"] - df["kc_lower"])))
        df["kc_width"] = ((df["kc_upper"] - df["kc_lower"]) / df["kc_mid"])

        hilbert = ta.HT_SINE(df)
        df["htsine"] = hilbert["sine"]
        df["htleadsine"] = hilbert["leadsine"]

        stoch = ta.STOCH(df)
        df["slowd"] = stoch["slowd"]
        df["slowk"] = stoch["slowk"]

        stoch_fast = ta.STOCHF(df)
        df["fastd"] = stoch_fast["fastd"]
        df["fastk"] = stoch_fast["fastk"]

        stoch_rsi = ta.STOCHRSI(df)
        df["fastd_rsi"] = stoch_rsi["fastd"]
        df["fastk_rsi"] = stoch_rsi["fastk"]

        df["rsi"] = ta.RSI(df)
        rsi = 0.1 * (df["rsi"] - 50)
        df["fisher_rsi"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        df["fisher_rsi_norma"] = 50 * (df["fisher_rsi"] + 1)

        
        df["plus_dm"] = ta.PLUS_DM(df)
        df["plus_di"] = ta.PLUS_DI(df)

        df["minus_dm"] = ta.MINUS_DM(df)
        df["minus_di"] = ta.MINUS_DI(df)

        
        df['tpct_0'] = top_percent_change(df , 0)
        df['tpct_3'] = top_percent_change(df , 3)
        df['tpct_9'] = top_percent_change(df , 9)
        return df
    
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        df.loc[(
                (df['close'] < df['kc_lower']) &
                (df['kc_percent'] > self.buy_percent.value) &
                (df['kc_width'] > self.buy_width.value) &
                (df['cmf'] < self.buy_cmf.value)&
                (df['volume'] > 0)),'enter_long'] = 1

        df.loc[(
                (df['close'] > df['kc_upper']) &
                (df['kc_percent'] > self.sell_percent.value) &
                (df['kc_width'] > self.sell_width.value) &
                (df['cmf'] > self.sell_cmf.value)&
                (df['volume'] > 0)),'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[((df['enter_long'] == 1)) , 'exit_short'] = 1
        df.loc[(df['enter_short'] == 1) , 'exit_long'] = 1
        return df

    def leverage(self,pair: str,current_time: datetime,current_rate: float,proposed_leverage: float,max_leverage: float,entry_tag: str,side: str,**kwargs) -> float:
        return 5.0

    def mac(self, dataframe: DataFrame, fast: int = 20, slow: int = 50) -> Series:
        dataframe = dataframe.copy()
        upper_fast = ta.EMA(dataframe['high'], timeperiod=fast)
        lower_fast = ta.EMA(dataframe['low'], timeperiod=fast)
        upper_slow = ta.EMA(dataframe['high'], timeperiod=slow)
        lower_slow = ta.EMA(dataframe['low'], timeperiod=slow)
        crosses_lf_us = qtpylib.crossed_above(lower_fast, upper_slow) | qtpylib.crossed_below(lower_fast, upper_slow)
        crosses_uf_ls = qtpylib.crossed_above(upper_fast, lower_slow) | qtpylib.crossed_below(upper_fast, lower_slow)
        dir_1 = np.where(crosses_lf_us, 1, np.nan)
        dir_2 = np.where(crosses_uf_ls, -1, np.nan)
        dir = np.where(dir_1 == 1, dir_1, np.nan)
        dir = np.where(dir_2 == -1, dir_2, dir_1)
        res = Series(dir).fillna(method='ffill').to_numpy()
        return res

    def ma_streak(self, dataframe: DataFrame, period: int = 4, source_type='close') -> Series:
        dataframe = dataframe.copy()
        avgval = self.zlema(dataframe[source_type], period)
        arr = np.diff(avgval)
        pos = np.clip(arr, 0, 1).astype(bool).cumsum()
        neg = np.clip(arr, -1, 0).astype(bool).cumsum()
        streak = np.where(arr >= 0, pos - np.maximum.accumulate(np.where(arr <= 0, pos, 0)),-neg + np.maximum.accumulate(np.where(arr >= 0, neg, 0)))
        res = np.concatenate((np.full((dataframe.shape[0] - streak.shape[0]), np.nan), streak))
        return res
    
    def zlema(self, series: Series, period):
        ema1 = ta.EMA(series, period)
        ema2 = ta.EMA(ema1, period)
        d = ema1 - ema2
        zlema = ema1 + d
        return zlema
    
    def pcc(self, dataframe: DataFrame, period: int = 20, mult: int = 2):
        PercentChangeChannel = namedtuple('PercentChangeChannel', ['upperband', 'middleband', 'lowerband'])
        dataframe = dataframe.copy()
        close = dataframe['close']
        previous_close = close.shift()
        low = dataframe['low']
        high = dataframe['high']
        close_change = (close - previous_close) / previous_close * 100
        high_change = (high - close) / close * 100
        low_change = (low - close) / close * 100
        mid = self.zlema(close_change, period)
        rangema = self.zlema(high_change - low_change, period)
        upper = mid + rangema * mult
        lower = mid - rangema * mult
        return PercentChangeChannel(upper, rangema, lower)



