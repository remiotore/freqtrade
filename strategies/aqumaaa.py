import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
from functools import reduce
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, stoploss_from_open, RealParameter, IntParameter, informative
from pandas import DataFrame, Series
from datetime import datetime
import math
import logging
from freqtrade.persistence import Trade
import pandas_ta as pta
from technical.indicators import RMI
import pandas as pd
import time

logger = logging.getLogger(__name__)

def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe['close'] * 100
    return smadif

def top_percent_change_dca(dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe['open'] - dataframe['close']) / dataframe['close']
    else:
        return (dataframe['open'].rolling(length).max() - dataframe['close']) / dataframe['close']

def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name="{0} Williams %R".format(period),
    )
    return WR * -100

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')

def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)

class aquma3(IStrategy):
    can_short = True

    buy_params = {
        "bbdelta_close": 0.00082,
        "bbdelta_tail": 0.85788,
        "close_bblower": 0.00128,
        "closedelta_close": 0.00987,
        "low_offset": 0.991,
        "rocr1_1h": 0.9346,
        "rocr_1h": 0.65666,
        "base_nb_candles_buy": 12,
        "buy_bb_delta": 0.025,
        "buy_bb_factor": 0.995,
        "buy_bb_width": 0.095,
        "buy_bb_width_1h": 1.074,
        "buy_cci": -116,
        "buy_cci_length": 25,
        "buy_closedelta": 15.0,
        "buy_clucha_bbdelta_close": 0.049,
        "buy_clucha_bbdelta_tail": 1.146,
        "buy_clucha_close_bblower": 0.018,
        "buy_clucha_closedelta_close": 0.017,
        "buy_clucha_rocr_1h": 0.526,
        "buy_ema_diff": 0.025,
        "buy_rmi": 49,
        "buy_rmi_length": 17,
        "buy_roc_1h": 10,
        "buy_srsi_fk": 32,
    }

    sell_params = {
        "high_offset": 1.012,
        "high_offset_2": 1.016,
        "sell_deadfish_bb_factor": 1.089,
        "sell_deadfish_bb_width": 0.11,
        "sell_deadfish_profit": -0.107,
        "sell_deadfish_volume_factor": 1.761,
        "base_nb_candles_sell": 22,
        "pHSL": -0.397,
        "pPF_1": 0.012,
        "pPF_2": 0.07,
        "pSL_1": 0.015,
        "pSL_2": 0.068,
        "sell_bbmiddle_close": 1.09092,
        "sell_fisher": 0.46406,
        "sell_trail_down_1": 0.03,
        "sell_trail_down_2": 0.015,
        "sell_trail_profit_max_1": 0.4,
        "sell_trail_profit_max_2": 0.11,
        "sell_trail_profit_min_1": 0.1,
        "sell_trail_profit_min_2": 0.04,
    }

    short_params = {
        "short_bb_delta": 0.025,
        "short_bb_factor": 1.005,
        "short_bb_width": 0.095,
        "short_bb_width_1h": 1.074,
        "short_cci": 116,
        "short_cci_length": 25,
        "short_closedelta": 15.0,
        "short_ema_diff": 0.025,
        "short_rmi": 51,
        "short_rmi_length": 17,
        "short_roc_1h": -10,
        "short_srsi_fk": 68,
        "short_low_offset": 1.009,
        "short_high_offset": 0.991,
        "short_clucha_bbdelta_close": 0.049,
        "short_clucha_bbdelta_tail": 1.146,
        "short_clucha_close_bbupper": 0.018,
        "short_clucha_closedelta_close": 0.017,
        "short_clucha_rocr_1h": 0.526,
        "short_44_ma_offset": 1.018,
        "short_44_ewo": 18.143,
        "short_44_cti": 0.8,
        "short_44_r_1h": 75.0,
        "short_37_ma_offset": 1.02,
        "short_37_ewo": -9.8,
        "short_37_rsi": 44.0,
        "short_37_cti": 0.7,
        "short_ema_open_mult_7": 0.030,
        "short_cti_7": 0.89,
    }

    minimal_roi = {
        "0": 0.276,
        "32": 0.105,
        "88": 0.037,
        "208": 0
    }

    position_adjustment_enable = True

    stoploss = -0.99

    trailing_stop = False
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True

    timeframe = '5m'

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count = 168

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
    }

    def is_support(self, row_data) -> bool:
        conditions = []
        for row in range(len(row_data)-1):
            if row < len(row_data)/2:
                conditions.append(row_data[row] > row_data[row+1])
            else:
                conditions.append(row_data[row] < row_data[row+1])
        return reduce(lambda x, y: x & y, conditions)

    fast_ewo = 50
    slow_ewo = 200

    buy_44_ma_offset = 0.982
    buy_44_ewo = -18.143
    buy_44_cti = -0.8
    buy_44_r_1h = -75.0

    buy_37_ma_offset = 0.98
    buy_37_ewo = 9.8
    buy_37_rsi = 56.0
    buy_37_cti = -0.7

    buy_ema_open_mult_7 = 0.030
    buy_cti_7 = -0.89

    is_optimize_dip = False
    buy_rmi = IntParameter(30, 50, default=35, optimize=is_optimize_dip)
    buy_cci = IntParameter(-135, -90, default=-133, optimize=is_optimize_dip)
    buy_srsi_fk = IntParameter(30, 50, default=25, optimize=is_optimize_dip)
    buy_cci_length = IntParameter(25, 45, default=25, optimize=is_optimize_dip)
    buy_rmi_length = IntParameter(8, 20, default=8, optimize=is_optimize_dip)

    is_optimize_break = False
    buy_bb_width = DecimalParameter(0.065, 0.135, default=0.095, optimize=is_optimize_break)
    buy_bb_delta = DecimalParameter(0.018, 0.035, default=0.025, optimize=is_optimize_break)

    is_optimize_check = False
    buy_roc_1h = IntParameter(-25, 200, default=10, optimize=is_optimize_check)
    buy_bb_width_1h = DecimalParameter(0.3, 2.0, default=0.3, optimize=is_optimize_check)

    is_optimize_clucha = False
    buy_clucha_bbdelta_close = DecimalParameter(0.01, 0.05, default=0.02206, optimize=is_optimize_clucha)
    buy_clucha_bbdelta_tail = DecimalParameter(0.7, 1.2, default=1.02515, optimize=is_optimize_clucha)
    buy_clucha_close_bblower = DecimalParameter(0.001, 0.05, default=0.03669, optimize=is_optimize_clucha)
    buy_clucha_closedelta_close = DecimalParameter(0.001, 0.05, default=0.04401, optimize=is_optimize_clucha)
    buy_clucha_rocr_1h = DecimalParameter(0.1, 1.0, default=0.47782, optimize=is_optimize_clucha)

    is_optimize_local_uptrend = False
    buy_ema_diff = DecimalParameter(0.022, 0.027, default=0.025, optimize=is_optimize_local_uptrend)
    buy_bb_factor = DecimalParameter(0.990, 0.999, default=0.995, optimize=False)
    buy_closedelta = DecimalParameter(12.0, 18.0, default=15.0, optimize=is_optimize_local_uptrend)

    rocr_1h = RealParameter(0.5, 1.0, default=0.54904, space='buy', optimize=True)
    rocr1_1h = RealParameter(0.5, 1.0, default=0.72, space='buy', optimize=True)
    bbdelta_close = RealParameter(0.0005, 0.02, default=0.01965, space='buy', optimize=True)
    closedelta_close = RealParameter(0.0005, 0.02, default=0.00556, space='buy', optimize=True)
    bbdelta_tail = RealParameter(0.7, 1.0, default=0.95089, space='buy', optimize=True)
    close_bblower = RealParameter(0.0005, 0.02, default=0.00799, space='buy', optimize=True)

    sell_fisher = RealParameter(0.1, 0.5, default=0.38414, space='sell', optimize=False)
    sell_bbmiddle_close = RealParameter(0.97, 1.1, default=1.07634, space='sell', optimize=False)

    is_optimize_deadfish = True
    sell_deadfish_bb_width = DecimalParameter(0.03, 0.75, default=0.05, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.15, -0.05, default=-0.08, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0, space='sell', optimize=is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1, 2.5, default=1.5, space='sell', optimize=is_optimize_deadfish)

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.25, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.04, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)

    sell_trail_profit_min_2 = DecimalParameter(0.04, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.08, 0.25, default=0.11, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.04, 0.2, default=0.015, space='sell', decimals=3, optimize=False, load=True)

    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell', optimize=False, load=True)
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=False, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=False, load=True)
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell', optimize=False, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell', optimize=False, load=True)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = lena = len(filled_buys)
        filled_sells = trade.select_filled_orders('sell')
        count_of_sells = len(filled_sells)

        if last_candle is not None:
            if trade.is_short == False:
                if (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_1.value)):
                    return 'trail_target_1'
                elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (((trade.max_rate - trade.open_rate) / 100) > (current_profit + self.sell_trail_down_2.value)):
                    return 'trail_target_2'
                elif (current_profit > 3) & (last_candle['rsi'] > 85):
                    return 'RSI-85 target'
                elif (current_profit > 0) & (count_of_buys < 4) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi'] > 50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                    return 'sell signal1'
                elif (current_profit > 0) & (count_of_buys >= 4) & (last_candle['close'] > last_candle['hma_50'] * 1.01) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) & (last_candle['rsi'] > 50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                    return 'sell signal1 * 1.01'
                elif (current_profit > 0) & (last_candle['close'] > last_candle['hma_50']) & (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] > last_candle['rsi_slow']):
                    return 'sell signal2'
                elif (current_profit < self.sell_deadfish_profit.value) & (last_candle['close'] < last_candle['ema_200']) & (last_candle['bb_width'] < self.sell_deadfish_bb_width.value) & (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value) & (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value) & (last_candle['cmf'] < 0.0):
                    return 'sell_stoploss_deadfish'
            else:
                if (current_profit > self.sell_trail_profit_min_1.value) & (current_profit < self.sell_trail_profit_max_1.value) & (((trade.open_rate - trade.min_rate) / 100) > (current_profit + self.sell_trail_down_1.value)):
                    return 'short_trail_target_1'
                elif (current_profit > self.sell_trail_profit_min_2.value) & (current_profit < self.sell_trail_profit_max_2.value) & (((trade.open_rate - trade.min_rate) / 100) > (current_profit + self.sell_trail_down_2.value)):
                    return 'short_trail_target_2'
                elif (current_profit > 3) & (last_candle['rsi'] < 15):
                    return 'short_RSI-15 target'
                elif (current_profit > 0) & (count_of_sells < 4) & (last_candle['close'] < last_candle['hma_50']) & (last_candle['close'] < (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.short_params['short_high_offset'])) & (last_candle['rsi'] < 50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] < last_candle['rsi_slow']):
                    return 'short_sell_signal1'
                elif (current_profit > 0) & (count_of_sells >= 4) & (last_candle['close'] < last_candle['hma_50'] * 0.99) & (last_candle['close'] < (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.short_params['short_high_offset'])) & (last_candle['rsi'] < 50) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] < last_candle['rsi_slow']):
                    return 'short_sell_signal1 * 0.99'
                elif (current_profit > 0) & (last_candle['close'] < last_candle['hma_50']) & (last_candle['close'] < (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.short_params['short_low_offset'])) & (last_candle['volume'] > 0) & (last_candle['rsi_fast'] < last_candle['rsi_slow']):
                    return 'short_sell_signal2'
                elif (current_profit < self.sell_deadfish_profit.value) & (last_candle['close'] > last_candle['ema_200']) & (last_candle['bb_width'] < self.sell_deadfish_bb_width.value) & (last_candle['close'] < last_candle['bb_middleband2'] * (1 / self.sell_deadfish_bb_factor.value)) & (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value) & (last_candle['cmf'] > 0.0):
                    return 'short_sell_stoploss_deadfish'

        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        if trade.is_short:
            if current_profit > PF_2 + 0.02:  # Tighter stoploss for shorts in high profit
                sl_profit = SL_2 + (current_profit - (PF_2 + 0.02)) * 0.5
            elif current_profit > PF_1:
                sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
            else:
                sl_profit = HSL
        else:
            if current_profit > PF_2:
                sl_profit = SL_2 + (current_profit - PF_2)
            elif current_profit > PF_1:
                sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
            else:
                sl_profit = HSL

        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.info(f"Using provided dataframe for {metadata['pair']}, last_candle_date={dataframe['date'].iloc[-1]}")

        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe('BTC/USDT:USDT', timeframe=inf_tf)
        if informative.empty:
            logger.warning(f"No data available for BTC/USDT:USDT on {inf_tf} timeframe, using provided dataframe")
            informative = dataframe.copy()
        informative_btc = informative.copy().shift(1)

        dataframe['btc_close'] = informative_btc['close']
        dataframe['btc_ema_fast'] = ta.EMA(informative_btc, timeperiod=20)
        dataframe['btc_ema_slow'] = ta.EMA(informative_btc, timeperiod=25)
        dataframe['down'] = (dataframe['btc_ema_fast'] < dataframe['btc_ema_slow']).astype('int')

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)

        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])

        bollinger2_40 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=2)
        dataframe['bb_lowerband2_40'] = bollinger2_40['lower']
        dataframe['bb_middleband2_40'] = bollinger2_40['mid']
        dataframe['bb_upperband2_40'] = bollinger2_40['upper']

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['bb_delta_cluc'] = (dataframe['bb_middleband2_40'] - dataframe['bb_lowerband2_40']).abs()
        dataframe['ha_closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()

        stoch = ta.STOCHRSI(dataframe, 15, 20, 2, 2)
        dataframe['srsi_fk'] = stoch['fastk']
        dataframe['srsi_fd'] = stoch['fastd']

        mid, lower = bollinger_bands(ha_typical_price(dataframe), window_size=40, num_of_std=2)
        dataframe['lower'] = lower
        dataframe['mid'] = mid

        dataframe['bbdelta'] = (mid - dataframe['lower']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['bb_lowerband'] = dataframe['lower']
        dataframe['bb_middleband'] = dataframe['mid']

        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']
        dataframe['bb_delta'] = ((dataframe['bb_lowerband2'] - dataframe['bb_lowerband3']) / dataframe['bb_lowerband2'])

        dataframe['ema_fast'] = ta.EMA(dataframe['ha_close'], timeperiod=3)
        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=50)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['rocr'] = ta.ROCR(dataframe['ha_close'], timeperiod=28)

        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['vwap_low'] = vwap_low
        dataframe['vwap'] = vwap
        dataframe['vwap_high'] = vwap_high
        dataframe['vwap_width'] = ((dataframe['vwap_high'] - dataframe['vwap_low']) / dataframe['vwap']) * 100

        dataframe['ema_vwap_diff_50'] = ((dataframe['ema_50'] - dataframe['vwap_low']) / dataframe['ema_50'])

        dataframe['tpct_change_0'] = top_percent_change_dca(dataframe, 0)
        dataframe['tpct_change_1'] = top_percent_change_dca(dataframe, 1)
        dataframe['tcp_percent_4'] = top_percent_change_dca(dataframe, 4)

        dataframe['ewo'] = ewo(dataframe, 50, 200)

        for val in self.buy_rmi_length.range:
            dataframe[f'rmi_length_{val}'] = RMI(dataframe, length=val, mom=4)

        for val in self.buy_cci_length.range:
            dataframe[f'cci_length_{val}'] = ta.CCI(dataframe, val)

        dataframe['cti'] = pta.cti(dataframe["close"], length=20)

        dataframe['ema_16'] = ta.EMA(dataframe, timeperiod=16)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)

        dataframe['r_14'] = williams_r(dataframe, period=14)

        dataframe['ema_5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema_10'] = ta.EMA(dataframe, timeperiod=10)

        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)
        dataframe['sma_75'] = ta.SMA(dataframe, timeperiod=75)

        rsi = ta.RSI(dataframe)
        dataframe["rsi"] = rsi
        rsi = 0.1 * (rsi - 50)
        dataframe["fisher"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)

        inf_heikinashi = qtpylib.heikinashi(informative)
        informative['ha_close'] = inf_heikinashi['close']
        informative['rocr'] = ta.ROCR(informative['ha_close'], timeperiod=168)
        informative['rsi_14'] = ta.RSI(informative, timeperiod=14)
        informative['cmf'] = chaikin_money_flow(informative, 20)

        sup_series = informative['low'].rolling(window=5).apply(lambda row: self.is_support(row), raw=True)
        informative['sup_level'] = Series(np.where(sup_series, np.where(informative['close'] < informative['open'], informative['close'], informative['open']), float('NaN'))).ffill()
        informative['roc'] = ta.ROC(informative, timeperiod=9)

        informative['r_480'] = williams_r(informative, period=480)

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
        informative['bb_lowerband2'] = bollinger2['lower']
        informative['bb_middleband2'] = bollinger2['mid']
        informative['bb_upperband2'] = bollinger2['upper']
        informative['bb_width'] = ((informative['bb_upperband2'] - informative['bb_lowerband2']) / informative['bb_middleband2'])

        informative['r_84'] = williams_r(informative, period=84)
        informative['cti_40'] = pta.cti(informative["close"], length=40)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        last_candle = dataframe.iloc[-1]
        logger.info(f"Indicators for {metadata['pair']}: rmi={last_candle[f'rmi_length_{self.buy_rmi_length.value}']}, cci={last_candle[f'cci_length_{self.buy_cci_length.value}']}, srsi_fk={last_candle['srsi_fk']}, bb_delta={last_candle['bb_delta']}, bb_width={last_candle['bb_width']}, last_candle_date={last_candle['date']}")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        desired_leverage = 1.0
        max_leverage = self.dp.get_pair_leverage(metadata['pair'], 1.0) if hasattr(self.dp, 'get_pair_leverage') else 1.0
        leverage = min(desired_leverage, max_leverage)
        logger.info(f"Setting leverage for {metadata['pair']}: {leverage}, Max leverage: {max_leverage}")

        # Long entry signals - DIP signal
        dip_conditions = [
            (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] < self.buy_rmi.value),
            (dataframe[f'cci_length_{self.buy_cci_length.value}'] <= self.buy_cci.value),
            (dataframe['srsi_fk'] < self.buy_srsi_fk.value),
            (dataframe['bb_delta'] > self.buy_bb_delta.value),
            (dataframe['bb_width'] > self.buy_bb_width.value),
            (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000),
            (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value),
            (dataframe['roc_1h'] < self.buy_roc_1h.value),
            (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value),
        ]
        if dip_conditions[0].any():
            logger.info(f"DIP signal conditions for {metadata['pair']}: {list(zip(['rmi', 'cci', 'srsi_fk', 'bb_delta', 'bb_width', 'closedelta', 'close_bblower', 'roc_1h', 'bb_width_1h'], [cond.iloc[-1] for cond in dip_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, dip_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'DIP signal', leverage)

        # Long entry signals - Break signal
        break_conditions = [
            (dataframe['bb_delta'] > self.buy_bb_delta.value),
            (dataframe['bb_width'] > self.buy_bb_width.value),
            (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000),
            (dataframe['close'] < dataframe['bb_lowerband3'] * self.buy_bb_factor.value),
            (dataframe['roc_1h'] < self.buy_roc_1h.value),
            (dataframe['bb_width_1h'] < self.buy_bb_width_1h.value),
        ]
        if break_conditions[0].any():
            logger.info(f"Break signal conditions for {metadata['pair']}: {list(zip(['bb_delta', 'bb_width', 'closedelta', 'close_bblower', 'roc_1h', 'bb_width_1h'], [cond.iloc[-1] for cond in break_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, break_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'Break signal', leverage)

        # Long entry signals - cluc_HA
        clucha_conditions = [
            (dataframe['rocr_1h'] > self.buy_clucha_rocr_1h.value),
            (dataframe['bb_lowerband2_40'].shift() > 0),
            (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.buy_clucha_bbdelta_close.value),
            (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.buy_clucha_closedelta_close.value),
            (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.buy_clucha_bbdelta_tail.value),
            (dataframe['ha_close'] < dataframe['bb_lowerband2_40'].shift()),
            (dataframe['close'] > (dataframe['sup_level_1h'] * 0.88)),
            (dataframe['ha_close'] < dataframe['ha_close'].shift()),
        ]
        if clucha_conditions[0].any():
            logger.info(f"cluc_HA signal conditions for {metadata['pair']}: {list(zip(['rocr_1h', 'bb_lowerband2_40', 'bb_delta_cluc', 'ha_closedelta', 'tail', 'ha_close_bblower', 'close_sup_level', 'ha_close_shift'], [cond.iloc[-1] for cond in clucha_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, clucha_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'cluc_HA', leverage)

        # Long entry signals - NFIX39
        nfix39_conditions = [
            (dataframe['ema_200'] > (dataframe['ema_200'].shift(12) * 1.01)),
            (dataframe['ema_200'] > (dataframe['ema_200'].shift(48) * 1.07)),
            (dataframe['bb_lowerband2_40'].shift().gt(0)),
            (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)),
            (dataframe['closedelta'].gt(dataframe['close'] * 0.01)),
            (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)),
            (dataframe['close'].lt(dataframe['bb_lowerband2_40'].shift())),
            (dataframe['close'].le(dataframe['close'].shift())),
            (dataframe['close'] > dataframe['ema_50'] * 0.912),
        ]
        if nfix39_conditions[0].any():
            logger.info(f"NFIX39 signal conditions for {metadata['pair']}: {list(zip(['ema_200_12', 'ema_200_48', 'bb_lowerband2_40', 'bb_delta_cluc', 'closedelta', 'tail', 'close_bblower', 'close_shift', 'close_ema50'], [cond.iloc[-1] for cond in nfix39_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, nfix39_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'NFIX39', leverage)

        # Long entry signals - NFIX29
        nfix29_conditions = [
            (dataframe['close'] > (dataframe['sup_level_1h'] * 0.72)),
            (dataframe['close'] < (dataframe['ema_16'] * 0.982)),
            (dataframe['EWO'] < -10.0),
            (dataframe['cti'] < -0.9),
        ]
        if nfix29_conditions[0].any():
            logger.info(f"NFIX29 signal conditions for {metadata['pair']}: {list(zip(['close_sup_level', 'close_ema16', 'EWO', 'cti'], [cond.iloc[-1] for cond in nfix29_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, nfix29_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'NFIX29', leverage)

        # Long entry signals - local_uptrend
        local_uptrend_conditions = [
            (dataframe['ema_26'] > dataframe['ema_12']),
            (dataframe['ema_26'] - dataframe['ema_12'] > dataframe['open'] * self.buy_ema_diff.value),
            (dataframe['ema_26'].shift() - dataframe['ema_12'].shift() > dataframe['open'] / 100),
            (dataframe['close'] < dataframe['bb_lowerband2'] * self.buy_bb_factor.value),
            (dataframe['closedelta'] > dataframe['close'] * self.buy_closedelta.value / 1000),
        ]
        if local_uptrend_conditions[0].any():
            logger.info(f"local_uptrend signal conditions for {metadata['pair']}: {list(zip(['ema_26_12', 'ema_diff', 'ema_shift', 'close_bblower', 'closedelta'], [cond.iloc[-1] for cond in local_uptrend_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, local_uptrend_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'local_uptrend', leverage)

        # Long entry signals - vwap
        vwap_conditions = [
            (dataframe['close'] < dataframe['vwap_low']),
            (dataframe['tcp_percent_4'] > 0.053),
            (dataframe['cti'] < -0.8),
            (dataframe['rsi'] < 35),
            (dataframe['rsi_84'] < 60),
            (dataframe['rsi_112'] < 60),
            (dataframe['volume'] > 0),
        ]
        if vwap_conditions[0].any():
            logger.info(f"vwap signal conditions for {metadata['pair']}: {list(zip(['close_vwap_low', 'tcp_percent_4', 'cti', 'rsi', 'rsi_84', 'rsi_112', 'volume'], [cond.iloc[-1] for cond in vwap_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, vwap_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'vwap', leverage)

        # Long entry signals - insta_signal
        insta_conditions = [
            (dataframe['bb_width_1h'] > 0.131),
            (dataframe['r_14'] < -51),
            (dataframe['r_84_1h'] < -70),
            (dataframe['cti'] < -0.845),
            (dataframe['cti_40_1h'] < -0.735),
            ((dataframe['close'].rolling(48).max() >= (dataframe['close'] * 1.1))),
            (dataframe['btc_close'].rolling(24).max() >= (dataframe['btc_close'] * 1.03)),
        ]
        if insta_conditions[0].any():
            logger.info(f"insta_signal conditions for {metadata['pair']}: {list(zip(['bb_width_1h', 'r_14', 'r_84_1h', 'cti', 'cti_40_1h', 'close_rolling', 'btc_close_rolling'], [cond.iloc[-1] for cond in insta_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, insta_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'insta_signal', leverage)

        # Long entry signals - NFINext44
        nfin44_conditions = [
            (dataframe['close'] < (dataframe['ema_16'] * self.buy_44_ma_offset)),
            (dataframe['ewo'] < self.buy_44_ewo),
            (dataframe['cti'] < self.buy_44_cti),
            (dataframe['r_480_1h'] < self.buy_44_r_1h),
            (dataframe['volume'] > 0),
        ]
        if nfin44_conditions[0].any():
            logger.info(f"NFINext44 signal conditions for {metadata['pair']}: {list(zip(['close_ema16', 'ewo', 'cti', 'r_480_1h', 'volume'], [cond.iloc[-1] for cond in nfin44_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, nfin44_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'NFINext44', leverage)

        # Long entry signals - NFINext37
        nfin37_conditions = [
            (dataframe['pm'] > dataframe['pmax_thresh']),
            (dataframe['close'] < dataframe['sma_75'] * self.buy_37_ma_offset),
            (dataframe['ewo'] > self.buy_37_ewo),
            (dataframe['rsi'] < self.buy_37_rsi),
            (dataframe['cti'] < self.buy_37_cti),
        ]
        if nfin37_conditions[0].any():
            logger.info(f"NFINext37 signal conditions for {metadata['pair']}: {list(zip(['pm_pmax', 'close_sma75', 'ewo', 'rsi', 'cti'], [cond.iloc[-1] for cond in nfin37_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, nfin37_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'NFINext37', leverage)

        # Long entry signals - NFINext7
        nfin7_conditions = [
            (dataframe['ema_26'] > dataframe['ema_12']),
            ((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7)),
            ((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100)),
            (dataframe['cti'] < self.buy_cti_7),
        ]
        if nfin7_conditions[0].any():
            logger.info(f"NFINext7 signal conditions for {metadata['pair']}: {list(zip(['ema_26_12', 'ema_diff', 'ema_shift', 'cti'], [cond.iloc[-1] for cond in nfin7_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, nfin7_conditions), ['enter_long', 'enter_tag', 'leverage']] = (1, 'NFINext7', leverage)

        # Short entry signals - short_DIP_signal
        short_dip_conditions = [
            (dataframe[f'rmi_length_{self.buy_rmi_length.value}'] > self.short_params['short_rmi']),
            (dataframe[f'cci_length_{self.buy_cci_length.value}'] >= self.short_params['short_cci']),
            (dataframe['srsi_fk'] > self.short_params['short_srsi_fk']),
            (dataframe['bb_delta'] > self.short_params['short_bb_delta']),
            (dataframe['bb_width'] > self.short_params['short_bb_width']),
            (dataframe['closedelta'] > dataframe['close'] * self.short_params['short_closedelta'] / 1000),
            (dataframe['close'] > dataframe['bb_upperband3'] * self.short_params['short_bb_factor']),
            (dataframe['roc_1h'] > self.short_params['short_roc_1h']),
            (dataframe['bb_width_1h'] < self.short_params['short_bb_width_1h']),
        ]
        if short_dip_conditions[0].any():
            logger.info(f"short_DIP_signal conditions for {metadata['pair']}: {list(zip(['rmi', 'cci', 'srsi_fk', 'bb_delta', 'bb_width', 'closedelta', 'close_bbupper', 'roc_1h', 'bb_width_1h'], [cond.iloc[-1] for cond in short_dip_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_dip_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_DIP_signal', leverage)

        # Short entry signals - short_Break_signal
        short_break_conditions = [
            (dataframe['bb_delta'] > self.short_params['short_bb_delta']),
            (dataframe['bb_width'] > self.short_params['short_bb_width']),
            (dataframe['closedelta'] > dataframe['close'] * self.short_params['short_closedelta'] / 1000),
            (dataframe['close'] > dataframe['bb_upperband3'] * self.short_params['short_bb_factor']),
            (dataframe['roc_1h'] > self.short_params['short_roc_1h']),
            (dataframe['bb_width_1h'] < self.short_params['short_bb_width_1h']),
        ]
        if short_break_conditions[0].any():
            logger.info(f"short_Break_signal conditions for {metadata['pair']}: {list(zip(['bb_delta', 'bb_width', 'closedelta', 'close_bbupper', 'roc_1h', 'bb_width_1h'], [cond.iloc[-1] for cond in short_break_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_break_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_Break_signal', leverage)

        # Short entry signals - short_local_downtrend
        short_downtrend_conditions = [
            (dataframe['ema_26'] < dataframe['ema_12']),
            (dataframe['ema_12'] - dataframe['ema_26'] > dataframe['open'] * self.short_params['short_ema_diff']),
            (dataframe['ema_12'].shift() - dataframe['ema_26'].shift() > dataframe['open'] / 100),
            (dataframe['close'] > dataframe['bb_upperband2'] * self.short_params['short_bb_factor']),
            (dataframe['closedelta'] > dataframe['close'] * self.short_params['short_closedelta'] / 1000),
        ]
        if short_downtrend_conditions[0].any():
            logger.info(f"short_local_downtrend conditions for {metadata['pair']}: {list(zip(['ema_26_12', 'ema_diff', 'ema_shift', 'close_bbupper', 'closedelta'], [cond.iloc[-1] for cond in short_downtrend_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_downtrend_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_local_downtrend', leverage)

        # Short entry signals - short_vwap
        short_vwap_conditions = [
            (dataframe['close'] > dataframe['vwap_high']),
            (dataframe['tcp_percent_4'] < -0.053),
            (dataframe['cti'] > 0.8),
            (dataframe['rsi'] > 65),
            (dataframe['rsi_84'] > 60),
            (dataframe['rsi_112'] > 60),
            (dataframe['volume'] > 0),
        ]
        if short_vwap_conditions[0].any():
            logger.info(f"short_vwap conditions for {metadata['pair']}: {list(zip(['close_vwap_high', 'tcp_percent_4', 'cti', 'rsi', 'rsi_84', 'rsi_112', 'volume'], [cond.iloc[-1] for cond in short_vwap_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_vwap_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_vwap', leverage)

        # Short entry signals - short_cluc_HA
        short_clucha_conditions = [
            (dataframe['rocr_1h'] < self.short_params['short_clucha_rocr_1h']),
            (dataframe['bb_upperband2_40'].shift() > 0),
            (dataframe['bb_delta_cluc'] > dataframe['ha_close'] * self.short_params['short_clucha_bbdelta_close']),
            (dataframe['ha_closedelta'] > dataframe['ha_close'] * self.short_params['short_clucha_closedelta_close']),
            (dataframe['tail'] < dataframe['bb_delta_cluc'] * self.short_params['short_clucha_bbdelta_tail']),
            (dataframe['ha_close'] > dataframe['bb_upperband2_40'].shift()),
            (dataframe['close'] < (dataframe['sup_level_1h'] * 1.12)),
            (dataframe['ha_close'] > dataframe['ha_close'].shift()),
        ]
        if short_clucha_conditions[0].any():
            logger.info(f"short_cluc_HA conditions for {metadata['pair']}: {list(zip(['rocr_1h', 'bb_upperband2_40', 'bb_delta_cluc', 'ha_closedelta', 'tail', 'ha_close_bbupper', 'close_sup_level', 'ha_close_shift'], [cond.iloc[-1] for cond in short_clucha_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_clucha_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_cluc_HA', leverage)

        # Short entry signals - short_NFIX39
        short_nfix39_conditions = [
            (dataframe['ema_200'] < (dataframe['ema_200'].shift(12) * 0.99)),
            (dataframe['ema_200'] < (dataframe['ema_200'].shift(48) * 0.93)),
            (dataframe['bb_upperband2_40'].shift().gt(0)),
            (dataframe['bb_delta_cluc'].gt(dataframe['close'] * 0.056)),
            (dataframe['closedelta'].gt(dataframe['close'] * 0.01)),
            (dataframe['tail'].lt(dataframe['bb_delta_cluc'] * 0.5)),
            (dataframe['close'].gt(dataframe['bb_upperband2_40'].shift())),
            (dataframe['close'].ge(dataframe['close'].shift())),
            (dataframe['close'] < dataframe['ema_50'] * 1.088),
        ]
        if short_nfix39_conditions[0].any():
            logger.info(f"short_NFIX39 conditions for {metadata['pair']}: {list(zip(['ema_200_12', 'ema_200_48', 'bb_upperband2_40', 'bb_delta_cluc', 'closedelta', 'tail', 'close_bbupper', 'close_shift', 'close_ema50'], [cond.iloc[-1] for cond in short_nfix39_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_nfix39_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_NFIX39', leverage)

        # Short entry signals - short_NFIX29
        short_nfix29_conditions = [
            (dataframe['close'] < (dataframe['sup_level_1h'] * 1.28)),
            (dataframe['close'] > (dataframe['ema_16'] * 1.018)),
            (dataframe['EWO'] > 10.0),
            (dataframe['cti'] > 0.9),
        ]
        if short_nfix29_conditions[0].any():
            logger.info(f"short_NFIX29 conditions for {metadata['pair']}: {list(zip(['close_sup_level', 'close_ema16', 'EWO', 'cti'], [cond.iloc[-1] for cond in short_nfix29_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_nfix29_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_NFIX29', leverage)

        # Short entry signals - short_insta_signal
        short_insta_conditions = [
            (dataframe['bb_width_1h'] > 0.131),
            (dataframe['r_14'] > 51),
            (dataframe['r_84_1h'] > 70),
            (dataframe['cti'] > 0.845),
            (dataframe['cti_40_1h'] > 0.735),
            ((dataframe['close'].rolling(48).min() <= (dataframe['close'] * 0.9))),
            (dataframe['btc_close'].rolling(24).min() <= (dataframe['btc_close'] * 0.97)),
        ]
        if short_insta_conditions[0].any():
            logger.info(f"short_insta_signal conditions for {metadata['pair']}: {list(zip(['bb_width_1h', 'r_14', 'r_84_1h', 'cti', 'cti_40_1h', 'close_rolling', 'btc_close_rolling'], [cond.iloc[-1] for cond in short_insta_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_insta_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_insta_signal', leverage)

        # Short entry signals - short_NFINext44
        short_nfin44_conditions = [
            (dataframe['close'] > (dataframe['ema_16'] * self.short_params['short_44_ma_offset'])),
            (dataframe['ewo'] > self.short_params['short_44_ewo']),
            (dataframe['cti'] > self.short_params['short_44_cti']),
            (dataframe['r_480_1h'] > self.short_params['short_44_r_1h']),
            (dataframe['volume'] > 0),
        ]
        if short_nfin44_conditions[0].any():
            logger.info(f"short_NFINext44 conditions for {metadata['pair']}: {list(zip(['close_ema16', 'ewo', 'cti', 'r_480_1h', 'volume'], [cond.iloc[-1] for cond in short_nfin44_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_nfin44_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_NFINext44', leverage)

        # Short entry signals - short_NFINext37
        short_nfin37_conditions = [
            (dataframe['pm'] < dataframe['pmax_thresh']),
            (dataframe['close'] > dataframe['sma_75'] * self.short_params['short_37_ma_offset']),
            (dataframe['ewo'] < self.short_params['short_37_ewo']),
            (dataframe['rsi'] > self.short_params['short_37_rsi']),
            (dataframe['cti'] > self.short_params['short_37_cti']),
        ]
        if short_nfin37_conditions[0].any():
            logger.info(f"short_NFINext37 conditions for {metadata['pair']}: {list(zip(['pm_pmax', 'close_sma75', 'ewo', 'rsi', 'cti'], [cond.iloc[-1] for cond in short_nfin37_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_nfin37_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_NFINext37', leverage)

        # Short entry signals - short_NFINext7
        short_nfin7_conditions = [
            (dataframe['ema_26'] < dataframe['ema_12']),
            ((dataframe['ema_12'] - dataframe['ema_26']) > (dataframe['open'] * self.short_params['short_ema_open_mult_7'])),
            ((dataframe['ema_12'].shift() - dataframe['ema_26'].shift()) > (dataframe['open'] / 100)),
            (dataframe['cti'] > self.short_params['short_cti_7']),
        ]
        if short_nfin7_conditions[0].any():
            logger.info(f"short_NFINext7 conditions for {metadata['pair']}: {list(zip(['ema_26_12', 'ema_diff', 'ema_shift', 'cti'], [cond.iloc[-1] for cond in short_nfin7_conditions]))}")
        dataframe.loc[reduce(lambda x, y: x & y, short_nfin7_conditions), ['enter_short', 'enter_tag', 'leverage']] = (1, 'short_NFINext7', leverage)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit signals
        long_exit_conditions = [
            (dataframe['fisher'] > self.sell_fisher.value),
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))),
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))),
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))),
            (dataframe['ema_fast'] > dataframe['ha_close']),
            ((dataframe['ha_close'] * self.sell_bbmiddle_close.value) > dataframe['bb_middleband']),
            (dataframe['volume'] > 0),
            (dataframe['rsi_14_1h'] < 30),  # Added safety condition
        ]
        dataframe.loc[reduce(lambda x, y: x & y, long_exit_conditions), 'exit_long'] = 1

        # Short exit signals
        short_exit_conditions = [
            (dataframe['fisher'] < -self.sell_fisher.value),
            (dataframe['ha_low'].ge(dataframe['ha_low'].shift(1))),
            (dataframe['ha_low'].shift(1).ge(dataframe['ha_low'].shift(2))),
            (dataframe['ha_close'].ge(dataframe['ha_close'].shift(1))),
            (dataframe['ema_fast'] < dataframe['ha_close']),
            ((dataframe['ha_close'] * (2 - self.sell_bbmiddle_close.value)) < dataframe['bb_middleband']),
            (dataframe['volume'] > 0),
            (dataframe['rsi_14_1h'] > 70),  # Added safety condition
        ]
        dataframe.loc[reduce(lambda x, y: x & y, short_exit_conditions), 'exit_short'] = 1

        return dataframe

    initial_safety_order_trigger = -0.05
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        desired_leverage = 1.0
        max_leverage = self.dp.get_pair_leverage(trade.pair, 1.0) if hasattr(self.dp, 'get_pair_leverage') else 1.0
        trade.leverage = min(desired_leverage, max_leverage)
        logger.info(f"DCA for {trade.pair}: Setting leverage to {trade.leverage}, Max leverage: {max_leverage}")

        logger.info(f"DCA for {trade.pair}: trade_id={trade.id}, is_open={trade.is_open}, is_short={trade.is_short}, orders={len(trade.orders)}, current_profit={current_profit}")
        logger.info(f"DCA for {trade.pair}: trade_orders={[order.__dict__ for order in trade.orders]}")

        if current_profit > self.initial_safety_order_trigger:
            logger.debug(f"DCA for {trade.pair}: Profit {current_profit} > {self.initial_safety_order_trigger}, skipping DCA")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = self.dp.get_pair_dataframe(pair=trade.pair, timeframe=self.timeframe)
                dataframe = ohlcv.copy()
                dataframe = self.populate_indicators(dataframe, {'pair': trade.pair})
                last_candle = dataframe.iloc[-1].squeeze()
                logger.info(f"D ASSOCIATE for {trade.pair}: Successfully fetched latest OHLCV on attempt {attempt + 1}, last_candle_date={last_candle['date']}")
                break
            except Exception as e:
                logger.warning(f"DCA for {trade.pair}: Attempt {attempt + 1}/{max_retries} failed to fetch OHLCV: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        if dataframe is None or last_candle is None:
            logger.error(f"DCA for {trade.pair}: Failed to fetch latest OHLCV, using cached dataframe")
            dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
            last_candle = dataframe.iloc[-1].squeeze()
            logger.info(f"DCA for {trade.pair}: Using cached dataframe, last_candle_date={last_candle['date']}")

        if trade.is_short:
            filled_sells = trade.select_filled_orders('sell')
            count_of_sells = len(filled_sells)
            logger.info(f"DCA for {trade.pair}: filled_sells={filled_sells}, count_of_sells={count_of_sells}, initial_cost={filled_sells[0].cost if filled_sells else 0}, volume_scale={self.safety_order_volume_scale}, leverage={trade.leverage}")

            if count_of_sells == 1 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']):
                logger.info(f"DCA for {trade.pair}: Skipped due to tpct_change_0={last_candle['tpct_change_0']} > 0.018 and close > open")
                return None
            elif count_of_sells == 2 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
                logger.info(f"DCA for {trade.pair}: Skipped due to tpct_change_0={last_candle['tpct_change_0']} > 0.018, close > open, ema_vwap_diff_50={last_candle['ema_vwap_diff_50']} < 0.215")
                return None
            elif count_of_sells == 3 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
                logger.info(f"DCA for {trade.pair}: Skipped due to tpct_change_0={last_candle['tpct_change_0']} > 0.018, close > open, ema_vwap_diff_50={last_candle['ema_vwap_diff_50']} < 0.215")
                return None
            elif count_of_sells == 4 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] > last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5'] >= last_candle['ema_10']):
                logger.info(f"DCA for {trade.pair}: Skipped due to tpct_change_0={last_candle['tpct_change_0']} > 0.018, close > open, ema_vwap_diff_50={last_candle['ema_vwap_diff_50']} < 0.215, ema_5 >= ema_10")
                return None
            elif count_of_sells >= 5 and (last_candle['cmf_1h'] > 0.0) and (last_candle['close'] > last_candle['open']) and (last_candle['rsi_14_1h'] > 30) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5'] >= last_candle['ema_10']):
                logger.info(f"DCA for {trade.pair}: Waiting for cmf_1h ({last_candle['cmf_1h']}) to fall below 0. Waiting for rsi_1h ({last_candle['rsi_14_1h']}) to fall below 30")
                return None

            if count_of_sells == 0 and trade.is_open:
                logger.warning(f"DCA for {trade.pair}: No filled sell orders, estimating initial sell from trade data")
                stake_amount = trade.amount * trade.open_rate
                count_of_sells = 1
            else:
                stake_amount = filled_sells[0].cost if filled_sells else min_stake

            if count_of_sells == 0 and trade.is_open:
                logger.warning(f"{trade.pair}의 DCA: 체결된 매도 주문 없음, 거래 데이터로 초기 매도 추정")
                stake_amount = trade.amount * trade.open_rate
                count_of_sells = 1
            else:
                stake_amount = filled_sells[0].cost if filled_sells else min_stake

            if count_of_sells <= self.max_safety_orders:
                safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_sells)
                if self.safety_order_step_scale > 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale, (count_of_sells - 1)) - 1) / (self.safety_order_step_scale - 1))
                elif self.safety_order_step_scale < 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale, (count_of_sells - 1))) / (1 - self.safety_order_step_scale))

                logger.info(f"{trade.pair}의 DCA: safety_order_trigger={safety_order_trigger}, current_profit={current_profit}, tpct_change_0={last_candle['tpct_change_0']}, rsi_14_1h={last_candle['rsi_14_1h']}")

                if current_profit <= (-1 * abs(safety_order_trigger)):
                    try:
                        stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, count_of_sells - 1)
                        amount = stake_amount / current_rate
                        logger.info(f"{trade.pair}의 안전 주문 매도 #{count_of_sells} 시작, 스테이크 금액={stake_amount}, 수량={amount}")
                        return stake_amount
                    except Exception as exception:
                        logger.error(f"{trade.pair}의 스테이크 금액 계산 중 오류 발생: {str(exception)}")
                        return None
        else:
            filled_buys = trade.select_filled_orders('buy')
            count_of_buys = len(filled_buys)
            logger.info(f"{trade.pair}의 DCA: filled_buys={filled_buys}, count_of_buys={count_of_buys}, initial_cost={filled_buys[0].cost if filled_buys else 0}, volume_scale={self.safety_order_volume_scale}, leverage={trade.leverage}")

            if count_of_buys == 1 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']):
                logger.info(f"{trade.pair}의 DCA: tpct_change_0={last_candle['tpct_change_0']} > 0.018 및 close < open으로 인해 스킵")
                return None
            elif count_of_buys == 2 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
                logger.info(f"{trade.pair}의 DCA: tpct_change_0={last_candle['tpct_change_0']} > 0.018, close < open, ema_vwap_diff_50={last_candle['ema_vwap_diff_50']} < 0.215로 인해 스킵")
                return None
            elif count_of_buys == 3 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215):
                logger.info(f"{trade.pair}의 DCA: tpct_change_0={last_candle['tpct_change_0']} > 0.018, close < open, ema_vwap_diff_50={last_candle['ema_vwap_diff_50']} < 0.215로 인해 스킵")
                return None
            elif count_of_buys == 4 and (last_candle['tpct_change_0'] > 0.018) and (last_candle['close'] < last_candle['open']) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5'] >= last_candle['ema_10']):
                logger.info(f"{trade.pair}의 DCA: tpct_change_0={last_candle['tpct_change_0']} > 0.018, close < open, ema_vwap_diff_50={last_candle['ema_vwap_diff_50']} < 0.215, ema_5 >= ema_10으로 인해 스킵")
                return None
            elif count_of_buys >= 5 and (last_candle['cmf_1h'] < 0.0) and (last_candle['close'] < last_candle['open']) and (last_candle['rsi_14_1h'] < 30) and (last_candle['tpct_change_0'] > 0.018) and (last_candle['ema_vwap_diff_50'] < 0.215) and (last_candle['ema_5'] >= last_candle['ema_10']):
                logger.info(f"{trade.pair}의 DCA: cmf_1h ({last_candle['cmf_1h']})가 0 초과로 상승하기를 대기. rsi_1h ({last_candle['rsi_14_1h']})가 30 초과로 상승하기를 대기")
                return None

            if count_of_buys == 0 and trade.is_open:
                logger.warning(f"{trade.pair}의 DCA: 체결된 매수 주문 없음, 거래 데이터로 초기 매수 추정")
                stake_amount = trade.amount * trade.open_rate
                count_of_buys = 1
            else:
                stake_amount = filled_buys[0].cost if filled_buys else min_stake

            if count_of_buys <= self.max_safety_orders:
                safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
                if self.safety_order_step_scale > 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
                elif self.safety_order_step_scale < 1:
                    safety_order_trigger = abs(self.initial_safety_order_trigger) + (abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / (1 - self.safety_order_step_scale))

                logger.info(f"{trade.pair}의 DCA: safety_order_trigger={safety_order_trigger}, current_profit={current_profit}, tpct_change_0={last_candle['tpct_change_0']}, rsi_14_1h={last_candle['rsi_14_1h']}")

                if current_profit <= (-1 * abs(safety_order_trigger)):
                    try:
                        stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, count_of_buys - 1)
                        amount = stake_amount / current_rate
                        logger.info(f"{trade.pair}의 안전 주문 매수 #{count_of_buys} 시작, 스테이크 금액={stake_amount}, 수량={amount}")
                        return stake_amount
                    except Exception as exception:
                        logger.error(f"{trade.pair}의 스테이크 금액 계산 중 오류 발생: {str(exception)}")
                        return None

        return None

def pmax(df, period, multiplier, length, MAtype, src):
    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = 'MA_' + str(MAtype) + '_' + str(length)
    atr = 'ATR_' + str(period)
    pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)

    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    else:
        mavalue = ta.EMA(masrc, timeperiod=length)

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
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down', 'up'), np.NaN)

    return pm, pmx