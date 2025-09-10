import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List, Optional
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, timeframe_to_minutes, stoploss_from_open, RealParameter)
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import zema
import math
import pandas_ta as pta
import technical.indicators as ftt





























class MultiMaX(IStrategy):
    def version(self) -> str:
        return "v1p"

    buy_params = {
        "buy_ema_fast": 50,
        "buy_ema_push": 1.000,
        "buy_ema_slow": 100,
    
        "buy_tema_fast": 50,
        "buy_tema_push": 1.000,
        "buy_tema_slow": 100,

        "buy_kama_fast": 50,
        "buy_kama_push": 1.000,
        "buy_kama_slow": 100,


        "buy_rsx": 60,
    }


    sell_params = {

        "pHSL": -0.185,
        "pPF_1": 0.014,
        "pPF_2": 0.061,
        "pSL_1": 0.016,
        "pSL_2": 0.056,

    }

    minimal_roi = {
        "0": 999
    }

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.017

    use_custom_stoploss = True

    protection_params = {
        "low_profit_lookback": 48,
        "low_profit_min_req": 0.04,
        "low_profit_stop_duration": 14,

        "cooldown_lookback": 2,  # value loaded from strategy
        "stoploss_lookback": 72,  # value loaded from strategy
        "stoploss_stop_duration": 20,  # value loaded from strategy
    }

    stoploss = -0.04

    timeframe = '5m'


    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 120

    rsi_buy_optimize = True
    buy_rsx = IntParameter(15, 90, default=buy_params['buy_rsx'], space='buy', optimize = rsi_buy_optimize)


    buy_opt_ema = False
    buy_ema_fast = IntParameter(2, 50, default=9, space='buy', optimize = buy_opt_ema)
    buy_ema_slow = IntParameter(12, 100, default=18, space='buy', optimize = buy_opt_ema)
    buy_ema_push = DecimalParameter(0.8, 1.2, decimals=3, default=1, space='buy', optimize = buy_opt_ema)

    buy_opt_tema = True
    buy_tema_fast = IntParameter(2, 50, default=9, space='buy', optimize = buy_opt_tema)
    buy_tema_slow = IntParameter(12, 100, default=18, space='buy', optimize = buy_opt_tema)
    buy_tema_push = DecimalParameter(0.8, 1.2, decimals=3, default=1, space='buy', optimize = buy_opt_tema)

    buy_opt_kama = False
    buy_kama_fast = IntParameter(2, 50, default=9, space='buy', optimize = buy_opt_kama)
    buy_kama_slow = IntParameter(12, 100, default=18, space='buy', optimize = buy_opt_kama)
    buy_kama_push = DecimalParameter(0.8, 1.2, decimals=3, default=1, space='buy', optimize = buy_opt_kama)


    buy_condition_enable_optimize = False

    buy_condition_ema_enable = BooleanParameter(default=False, space='buy', optimize=buy_condition_enable_optimize)

    buy_condition_tema_enable = BooleanParameter(default=False, space='buy', optimize=buy_condition_enable_optimize)

    buy_condition_kama_enable = BooleanParameter(default=False, space='buy', optimize=buy_condition_enable_optimize)

    pHSL = DecimalParameter(-0.200, -0.040, default=-0.08, decimals=3, space='sell', load=True)

    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', load=True)

    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', load=True)
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['tema_bb'] = ta.TEMA(dataframe, timeperiod=9)

        dataframe["rsx"] = pta.rsx(dataframe['close'], timeperiod=14)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
       
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy_copy'] = 0
        dataframe.loc[:, 'buy'] = 0

        if (self.buy_condition_ema_enable.value):

            dataframe['buy_ema_fast'] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_fast.value))
            dataframe['buy_ema_slow'] = ta.EMA(dataframe, timeperiod=int(self.buy_ema_slow.value))
            buy_offset_ema = (qtpylib.crossed_below(dataframe['buy_ema_fast'],dataframe['buy_ema_slow']*self.buy_ema_push.value))

            dataframe.loc[buy_offset_ema, 'buy_tag'] += 'emaX '
            conditions.append(buy_offset_ema)


        if (self.buy_condition_tema_enable.value):

            dataframe['buy_tema_fast'] = ta.TEMA(dataframe, timeperiod=int(self.buy_tema_fast.value))
            dataframe['buy_tema_slow'] = ta.TEMA(dataframe, timeperiod=int(self.buy_tema_slow.value))
            buy_offset_tema = (qtpylib.crossed_below(dataframe['buy_tema_fast'],dataframe['buy_tema_slow']*self.buy_tema_push.value))

            dataframe.loc[buy_offset_tema, 'buy_tag'] += 'temaX '
            conditions.append(buy_offset_tema)


        if (self.buy_condition_kama_enable.value):

            dataframe['buy_kama_fast'] = ta.KAMA(dataframe, timeperiod=int(self.buy_kama_fast.value))
            dataframe['buy_kama_slow'] = ta.KAMA(dataframe, timeperiod=int(self.buy_kama_slow.value))
            buy_offset_kama = (qtpylib.crossed_below(dataframe['buy_kama_fast'],dataframe['buy_kama_slow']*self.buy_kama_push.value))

            dataframe.loc[buy_offset_kama, 'buy_tag'] += 'kamaX '
            conditions.append(buy_offset_kama)

        buy_check = ((dataframe['rsx'] < self.buy_rsx.value))


        buy_bb = ((qtpylib.crossed_above(dataframe['tema_bb'], dataframe['bb_middleband'])) &
                (dataframe['rsi_slow'] < dataframe['rsi_fast']) &
                (dataframe['tema_bb'] > dataframe['tema_bb'].shift(1)) &  # Guard: tema is raising
                (dataframe['rsi_fast'] > dataframe['rsi_fast'].shift(1))&
                (dataframe['volume'] > 0))

        dataframe.loc[buy_bb, 'buy_tag'] += 'bbX '
        conditions.append(buy_bb)


        if conditions:
            dataframe.loc[
                (buy_check & reduce(lambda x, y: x | y, conditions)),
                'buy'
            ]=1
    
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'exit_tag'] = ''
        conditions = []

        sell_cond_1 = (
            (dataframe['volume'] > 0)
            &
            (dataframe['rsx'] > 100)
        )

        conditions.append(sell_cond_1)
        dataframe.loc[sell_cond_1, 'exit_tag'] += 'EMA '

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1


        return dataframe



    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value




        if (current_profit > PF_2):
            sl_profit = SL_2 + (current_profit - PF_2)
        elif (current_profit > PF_1):
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if (sl_profit >= current_profit):
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)


def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma1_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


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


def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)
            
    return df

def pump_warning(dataframe, perc=15):
    df = dataframe.copy()    
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"]/df["low"]) > (perc/100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']

def tv_wma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + dataframe[field].shift(i) * weight

    dataframe["tv_wma"] = (sum / norm) if norm > 0 else 0
    return dataframe["tv_wma"]

def tv_hma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    dataframe["h"] = 2 * tv_wma(dataframe, math.floor(length / 2), field) - tv_wma(dataframe, length, field)

    dataframe["tv_hma"] = tv_wma(dataframe, math.floor(math.sqrt(length)), "h")


    return dataframe["tv_hma"]

def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

def ssl_atr(dataframe, length = 7):
    df = dataframe.copy()
    df['smaHigh'] = df['high'].rolling(length).mean() + df['atr']
    df['smaLow'] = df['low'].rolling(length).mean() - df['atr']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']
