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
from technical.indicators import zema, RMI, ichimoku
import math
import pandas_ta as pta
import technical.indicators as ftt
import logging
logger = logging.getLogger(__name__)

























class iRNWE_dca(IStrategy):
    def version(self) -> str:
        return "L1V3"

    buy_params = {
        "buy_ewo_high": 2.055,
        "buy_cci": -129,
        "buy_cci_length": 34,
        "buy_ema_high": 1.028,
        "buy_ema_high_2": 1.197,
        "buy_ema_low": 0.987,
        "buy_ema_low_2": 0.975,
        "buy_ewo": -4.07,
        "buy_rmi": 36,
        "buy_rmi_length": 18,
        "buy_rsx": 33,
        "buy_rsi_fast": 45,
        "rsi_buy": 60,
        "rsi_buy2": 45,
        "buy_btc_safe": -289,
        "buy_btc_safe_1d": -0.05,
        "mult_buy": 3,
        "bandwidth_buy": 8,
        "window_buy": 500
    }


    sell_params = {

        "base_nb_candles_ema_sell": 6,
        "high_offset_sell_ema": 0.991,
        "sell_btc_safe": -38,
    }

    minimal_roi = {
        "0": 999
    }

    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.017

    use_custom_stoploss = True

    stoploss = -0.04

    timeframe = '5m'

    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 200

    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize=True)



    is_optimize_dip = True
    buy_rmi = IntParameter(30, 50, default=35, optimize= is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize= is_optimize_dip)

    buy_cci_length = IntParameter(25, 45, default=25, optimize = is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize = is_optimize_dip)
    buy_condition_enable_optimize = False



    is_optimize_ewo_2 = True
    buy_ema_low_2 = DecimalParameter(0.96, 0.978, default=0.96 , optimize = is_optimize_ewo_2)
    buy_ema_high_2 = DecimalParameter(1.05, 1.2, default=1.09 , optimize = is_optimize_ewo_2)



    window_buy = IntParameter(60, 1000, default=500, space='buy', optimize=True)
    bandwidth_buy = IntParameter(2, 15, default=8, space='buy', optimize=True)
    mult_buy = DecimalParameter(1, 20, default=3, space='buy', optimize=True)

    ewo_check_optimize = True
    ewo_low = DecimalParameter(-20.0, -8.0, default=-20.0, decimals = 1, space='buy', optimize=ewo_check_optimize)
    ewo_high = DecimalParameter(2.0, 12.0, default=6.0, decimals = 1, space='buy', optimize=ewo_check_optimize)
    ewo_low2 = DecimalParameter(-20.0, -8.0, default=-20.0, decimals = 1, space='buy', optimize=ewo_check_optimize)
    ewo_high2 = DecimalParameter(2.0, 12.0, default=6.0, decimals = 1, space='buy', optimize=ewo_check_optimize)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = ewo_check_optimize)
    
    rsi_buy_optimize = True
    rsi_buy = IntParameter(30, 70, default=50, space='buy', optimize=rsi_buy_optimize)
    rsi_buy2 = IntParameter(30, 70, default=50, space='buy', optimize=rsi_buy_optimize)
    buy_rsi_fast = IntParameter(0, 50, default=35, space='buy', optimize=False)
    buy_rsx = IntParameter(15, 30, default=35, optimize = rsi_buy_optimize)

    fast_ewo = IntParameter(10, 50, default=50, space='buy', optimize=False)
    slow_ewo = IntParameter(100, 200, default=200, space='buy', optimize=False)

    is_optimize_btc_safe = False
    buy_btc_safe = IntParameter(-300, 50, default=-200, optimize = is_optimize_btc_safe)
    buy_btc_safe_1d = DecimalParameter(-0.075, -0.025, default=-0.05, optimize = is_optimize_btc_safe)
    buy_threshold = DecimalParameter(0.003, 0.012, default=0.008, optimize = is_optimize_btc_safe)


    sell_btc_safe = IntParameter(-400, -300, default=-365, optimize = False)



    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "30m") for pair in pairs]
        informative_pairs.extend([(pair, "1h") for pair in pairs])
        informative_pairs.extend([(pair, "4h") for pair in pairs])
        informative_pairs.extend([("BTC/USDT", "5m")])
        return informative_pairs

    @informative('30m')
    def populate_indicators_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)

        return dataframe

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        inf_heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['ha_close'] = inf_heikinashi['close']
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=168)
        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )

        dataframe['chikou_span'] = ichimoku['chikou_span']

        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        dataframe.loc[:, 'cloud_top'] = dataframe.loc[:, ['senkou_a', 'senkou_b']].max(axis=1)
        dataframe.loc[:, 'cloud_bottom'] = dataframe.loc[:, ['senkou_a', 'senkou_b']].min(axis=1)


        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2
        dataframe['future_red'] = (dataframe['leading_senkou_span_a'] < dataframe['leading_senkou_span_b']).astype('int') * 2



        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['cloud_top'])
            ).shift(displacement).fillna(0).astype('int')

        dataframe['chikou_low'] = (
                (dataframe['chikou_span'] < dataframe['cloud_bottom'])
            ).shift(displacement).fillna(0).astype('int')


        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        ssl_down, ssl_up = ssl_atr(dataframe, 10)
        dataframe['ssl_down'] = ssl_down
        dataframe['ssl_up'] = ssl_up
        dataframe['ssl_ok'] = (
                (ssl_up > ssl_down) 
            ).astype('int') * 3
        dataframe['ssl_bear'] = (
                (ssl_up < ssl_down) 
            ).astype('int') * 3

        dataframe['ichimoku_ok'] = (
                (dataframe['tenkan_sen'] > dataframe['kijun_sen'])
                & (dataframe['close'] > dataframe['cloud_top'])
                & (dataframe['future_green'] > 0) 
                & (dataframe['chikou_high'] > 0) 
            ).astype('int') * 4

        dataframe['ichimoku_bear'] = (
                (dataframe['tenkan_sen'] < dataframe['kijun_sen'])
                & (dataframe['close'] < dataframe['cloud_bottom'])
                & (dataframe['future_red'] > 0) 
                & (dataframe['chikou_low'] > 0) 
            ).astype('int') * 4

        dataframe['ichimoku_valid'] = (
                (dataframe['leading_senkou_span_b'] == dataframe['leading_senkou_span_b']) # not NaN
            ).astype('int') * 1

        dataframe['trend_pulse'] = (
                (dataframe['ichimoku_ok'] > 0) 
                & (dataframe['ssl_ok'] > 0)
            ).astype('int') * 2



        dataframe['trend_over'] = (
                (dataframe['ssl_ok'] == 0)
                | (dataframe['close'] < dataframe['cloud_top'])
            ).astype('int') * 1


        dataframe.loc[ (dataframe['trend_pulse'] > 0), 'trending'] = 3
        dataframe.loc[ (dataframe['trend_over'] > 0) , 'trending'] = 0
        dataframe['trending'].fillna(method='ffill', inplace=True)


        return dataframe


    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )

        dataframe['chikou_span'] = ichimoku['chikou_span']

        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        dataframe.loc[:, 'cloud_top'] = dataframe.loc[:, ['senkou_a', 'senkou_b']].max(axis=1)
        dataframe.loc[:, 'cloud_bottom'] = dataframe.loc[:, ['senkou_a', 'senkou_b']].min(axis=1)


        dataframe['future_green'] = (dataframe['leading_senkou_span_a'] > dataframe['leading_senkou_span_b']).astype('int') * 2
        dataframe['future_red'] = (dataframe['leading_senkou_span_a'] < dataframe['leading_senkou_span_b']).astype('int') * 2



        dataframe['chikou_high'] = (
                (dataframe['chikou_span'] > dataframe['cloud_top'])
            ).shift(displacement).fillna(0).astype('int')

        dataframe['chikou_low'] = (
                (dataframe['chikou_span'] < dataframe['cloud_bottom'])
            ).shift(displacement).fillna(0).astype('int')


        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        ssl_down, ssl_up = ssl_atr(dataframe, 10)
        dataframe['ssl_down'] = ssl_down
        dataframe['ssl_up'] = ssl_up
        dataframe['ssl_ok'] = (
                (ssl_up > ssl_down) 
            ).astype('int') * 3
        dataframe['ssl_bear'] = (
                (ssl_up < ssl_down) 
            ).astype('int') * 3

        dataframe['ichimoku_bear'] = (
                (dataframe['tenkan_sen'] < dataframe['kijun_sen'])
                & (dataframe['close'] < dataframe['cloud_bottom'])
                & (dataframe['future_red'] > 0) 
                & (dataframe['chikou_low'] > 0) 
            ).astype('int') * 4

        dataframe['ichimoku_valid'] = (
                (dataframe['leading_senkou_span_b'] == dataframe['leading_senkou_span_b']) # not NaN
            ).astype('int') * 1


        dataframe['bear_trend_pulse'] = (
                (dataframe['ichimoku_bear'] > 0) 
                & (dataframe['ssl_bear'] > 0)
            ).astype('int') * 2

        dataframe['bear_trend_over'] = (
                (dataframe['ssl_bear'] == 0)
                | (dataframe['close'] > dataframe['cloud_bottom'])
            ).astype('int') * 1

        dataframe.loc[ (dataframe['bear_trend_pulse'] > 0), 'bear_trending'] = 3
        dataframe.loc[ (dataframe['bear_trend_over'] > 0) , 'bear_trending'] = 0
        dataframe['bear_trending'].fillna(method='ffill', inplace=True)
        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe('BTC/USDT', timeframe=inf_tf)
        informative_past = informative.copy().shift(1)                                                                                                   # Get recent BTC info

        informative_past_source = (informative_past['open'] + informative_past['close'] + informative_past['high'] + informative_past['low']) / 4        # Get BTC price
        informative_threshold = informative_past_source * self.buy_threshold.value                                                                       # BTC dump n% in 5 min
        informative_past_delta = informative_past['close'].shift(1) - informative_past['close']                                                          # should be positive if dump
        informative_diff = informative_threshold - informative_past_delta                                                                                # Need be larger than 0
        dataframe['btc_threshold'] = informative_threshold
        dataframe['btc_diff'] = informative_diff

        informative_past_1d = informative.copy().shift(288)
        informative_past_source_1d = (informative_past_1d['open'] + informative_past_1d['close'] + informative_past_1d['high'] + informative_past_1d['low']) / 4
        dataframe['btc_5m'] = informative_past_source
        dataframe['btc_1d'] = informative_past_source_1d


        dataframe['rsx'] = pta.rsx(dataframe['close'], timeperiod=14)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]


        dataframe = HA(dataframe, 4)
        dataframe[['nwe_up','nwe_down']] = funcNadarayaWatsonEnvelope(dataframe, source = 'close', bandwidth = self.bandwidth_buy.value, window = self.window_buy.value, mult = self.mult_buy.value)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        dataframe.loc[:, 'buy_tag'] = ''
        dataframe.loc[:, 'buy_copy'] = 0
        dataframe.loc[:, 'buy'] = 0

        ichi_check = (
                        (dataframe['ichimoku_valid_1h'] > 0)
                        & (dataframe['trending_1h'] > 0)
                        & (dataframe['bear_trending_4h'] == 0)
                        & (dataframe['ichimoku_valid_4h'] > 0))

        is_btc_safe = (
                        (dataframe['btc_5m'] - dataframe['btc_1d'] > dataframe['btc_1d'] * self.buy_btc_safe_1d.value)
                        &(dataframe['volume'] > 0))

        buy_check = (   (dataframe['rsx'] < self.buy_rsx.value)
                        &(dataframe['close'] < dataframe['Smooth_HA_L']))

        is_rsx_sh = (   dataframe['rsx'] > dataframe['rsx'].shift(2))
        
        conditions.append(is_rsx_sh)                                                  # ~2.19 / 92.6% / 28.12%
        dataframe.loc[is_rsx_sh, 'buy_tag'] += 'RSX '

        buy_NWE = (
                        (qtpylib.crossed_below(dataframe['close'], dataframe['nwe_down']))
                        )
        dataframe.loc[buy_NWE, 'buy_tag'] += 'NWE '
        conditions.append(buy_NWE)



        if conditions:
            dataframe.loc[
                (ichi_check & buy_check & reduce(lambda x, y: x | y, conditions)),
                'buy'
            ]=1
    
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (dataframe['trending_1h'] <= 0)          
        ,
        'sell'] = 0

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        sl_frac = 0.158

        profit_threshold = 0.0137


        profit_sl_frac = ((current_profit - profit_threshold) * 0.25) + profit_threshold 
        
        if (current_profit > profit_threshold):

            return profit_sl_frac
        else:

            return sl_frac

        return sl_frac

class iRNWE_dca(iRNWE_dca):

   

    initial_safety_order_trigger = -0.021
    max_safety_orders = 2
    safety_order_step_scale = 0.7
    safety_order_volume_scale = 0.5

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

def funcNadarayaWatsonEnvelope(dtloc, source = 'close', bandwidth = 8, window = 500, mult = 3):
    """
    // This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/
    // Nadaraya-Watson Envelope [LUX]
      https://www.tradingview.com/script/Iko0E2kL-Nadaraya-Watson-Envelope-LUX/
     :return: up and down   
     translated for freqtrade: viksal1982  viktors.s@gmail.com  
    """ 
    dtNWE = dtloc.copy()
    dtNWE['nwe_up'] = np.nan
    dtNWE['nwe_down'] =  np.nan
    wn = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
                wn[i,j] = math.exp(-(math.pow(i-j,2)/(bandwidth*bandwidth*2)))
    sumSCW = wn.sum(axis = 1)
    def calc_nwa(dfr, init=0):
        global calc_src_value
        if init == 1:
            calc_src_value = list()
            return
        calc_src_value.append(dfr[source])
        mae = 0.0
        y2_val = 0.0
        y2_val_up = np.nan
        y2_val_down = np.nan
        if len(calc_src_value) > window:
            calc_src_value.pop(0)
        if len(calc_src_value) >= window:
            src = np.array(calc_src_value)
            sumSC = src * wn
            sumSCS = sumSC.sum(axis = 1)
            y2 = sumSCS / sumSCW
            sum_e = np.absolute(src - y2)
            mae = sum_e.sum()/window*mult 
            y2_val = y2[-1]
            y2_val_up = y2_val + mae
            y2_val_down = y2_val - mae
        return y2_val_up,y2_val_down
    calc_nwa(None, init=1)
    dtNWE[['nwe_up','nwe_down']] = dtNWE.apply(calc_nwa, axis = 1, result_type='expand')
    return dtNWE[['nwe_up','nwe_down']]
