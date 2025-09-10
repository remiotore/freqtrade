# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
import pandas as pd
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, BooleanParameter, \
    DecimalParameter, IntParameter, CategoricalParameter
import math
import logging

logger = logging.getLogger(__name__)


########################################################################################################################################################
# EWO
########################################################################################################################################################
def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif


########################################################################################################################################################
# pct_change
########################################################################################################################################################
def pct_change(a, b):
    return (b - a) / a


########################################################################################################################################################

# 威科夫成交量扩散指标（检测背离）
def wyckoff_volume_divergence(dataframe, period=20):
    df = dataframe.copy()
    df['price_high'] = df['high'].rolling(period).max()
    df['vol_low'] = df['volume'].rolling(period).min()
    df['distribution_signal'] = ((df['high'] == df['price_high']) &
                                 (df['volume'] < df['vol_low'])).astype(int)
    df['price_low'] = df['low'].rolling(period).min()
    df['accumulation_signal'] = ((df['low'] == df['price_low']) &
                                 (df['volume'] < df['vol_low']*0.8)).astype(int)

    return df[['distribution_signal', 'accumulation_signal']]


class DS_Green_5m(IStrategy):

    ########################################################################################################################################################
    # Hyperopt
    ########################################################################################################################################################

    # 新增威科夫参数
    wyckoff_volume_threshold = DecimalParameter(1.3, 2.0, default=1.5, space='buy')
    dynamic_support_period = IntParameter(30, 100, default=50, space='sell')

    buy_params = {
        # 新增威科夫参数
        "wyckoff_volume_threshold": 1.7,
        "dynamic_support_period": 65,
        "lambo2_enabled": True,
        "ewo1_enabled": True,
        "ewo2_enabled": True,
        "cofi_enabled": True,
        # Ewo
        "base_nb_candles_buy": 12,
        # EWO 1
        "ewo_high": 3.009,
        "low_offset_1": 0.988,
        "high_offset_1": 0.969,
        "rsi_buy": 49,
        # Ewo 2
        "ewo_low": -8.929,
        "low_offset_2": 0.985,
        "high_offset_2": 1.01,
        # Lambo 2
        "lambo2_ema_14_factor": 0.95,
        "lambo2_rsi_14_limit": 53,
        "lambo2_rsi_4_limit": 46,
        # Cofi
        "buy_adx": 25,
        "buy_fastd": 20,
        "buy_fastk": 24,
        "buy_ema_cofi": 0.977,
        "buy_ewo_high": 3.767,
        # ?
        "dca_min_rsi": 64,
    }
    sell_params = {
        "distribution_rsi": 68,  # 派发阶段RSI阈值
        "dynamic_support_tolerance": 0.985,  # 支撑位容忍度
        "base_nb_candles_sell": 22,
        # Ewo
        "high_offset_above": 1.05,
        "high_offset_below": 1.01,
        # custom stoploss params, come from BB_RPB_TSL
        "pHSL": -0.397,
        "pPF_1": 0.012,
        "pPF_2": 0.07,
        "pSL_1": 0.015,
        "pSL_2": 0.068,
    }
    minimal_roi = {
        "0": 100,
    }
    stoploss = -0.99
    trailing_stop = True
    trailing_stop_positive = 0.001  # Positive offset for trailing stop.
    trailing_stop_positive_offset = 0.0135  # Offset for triggering the trailing stop.
    trailing_only_offset_is_reached = True  # Only trigger trailing stop if the offset is reached.
    ########################################################################################################################################################
    # Main
    ########################################################################################################################################################
    use_custom_stoploss = True
    timeframe = '5m'  # The primary timeframe for analysis.
    inf_1h = '1h'  # Informative timeframe to gather additional data.
    # Sell signal configuration.
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01  # Offset added to exit signal (profitable threshold).
    ignore_roi_if_entry_signal = False  # If True, ignore ROI when the buy signal is still present.
    # Number of past candles to consider upon startup.
    process_only_new_candles = True
    startup_candle_count = 400
    # Adjsut trade position
    initial_safety_order_trigger = -0.018  # Initial trigger for the first safety order.
    max_safety_orders = 8  # Maximum number of safety orders to prevent overexposure.
    safety_order_step_scale = 1.2  # How much to increase the trigger for each additional safety order.
    safety_order_volume_scale = 1.4  # How much to increase the volume of each safety order.
    # Configuration of order types.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'trailing_stop_loss': 'limit',
        'emergency_exit': 'market',
        'force_entry': 'limit',
        'force_exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }
    # Plotting configuration for visualizing indicators in backtesting.
    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},  # Color for the buy moving average.
            'ma_sell': {'color': 'orange'},  # Color for the sell moving average.
        },
    }
    # Order Time-In-Force defines how long an order will remain active before it is executed or expired.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    ########################################################################################################################################################
    # Parameters
    ########################################################################################################################################################
    # Enabled
    is_optimize_remove = False
    lambo2_enabled = BooleanParameter(default=buy_params['lambo2_enabled'], space='buy',
                                      optimize=is_optimize_remove)
    ewo1_enabled = BooleanParameter(default=buy_params['ewo1_enabled'], space='buy',
                                    optimize=is_optimize_remove)
    ewo2_enabled = BooleanParameter(default=buy_params['ewo2_enabled'], space='buy',
                                    optimize=is_optimize_remove)
    cofi_enabled = BooleanParameter(default=buy_params['cofi_enabled'], space='buy',
                                    optimize=is_optimize_remove)
    ############################################################################################################################################################################
    # Candles
    is_optimize_base_nb_candles = False
    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'],
                                       space='buy', optimize=is_optimize_base_nb_candles)
    base_nb_candles_sell = IntParameter(8, 20, default=sell_params['base_nb_candles_sell'],
                                        space='sell', optimize=is_optimize_base_nb_candles)
    # EWO Protection
    fast_ewo = 60
    slow_ewo = 220
    # EWO 1
    is_optimize_ewo = False
    low_offset_1 = DecimalParameter(0.985, 0.995, default=buy_params['low_offset_1'], space='buy',
                                    optimize=is_optimize_ewo)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy',
                           optimize=is_optimize_ewo)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy',
                                optimize=is_optimize_ewo)
    high_offset_1 = DecimalParameter(0.95, 1.10, default=buy_params['high_offset_1'], space='buy',
                                     optimize=is_optimize_ewo)
    # Ewo 2
    is_optimize_ewo2 = False
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy',
                               optimize=is_optimize_ewo2)
    low_offset_2 = DecimalParameter(0.985, 0.995, default=buy_params['low_offset_2'], space='buy',
                                    optimize=is_optimize_ewo2)
    high_offset_2 = DecimalParameter(0.95, 1.10, default=buy_params['high_offset_2'], space='buy',
                                     optimize=is_optimize_ewo2)
    # lambo2
    is_optimize_lambo2 = True
    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,
                                            default=buy_params['lambo2_ema_14_factor'], space='buy',
                                            optimize=is_optimize_lambo2)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_4_limit'], space='buy',
                                      optimize=is_optimize_lambo2)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=buy_params['lambo2_rsi_14_limit'],
                                       space='buy', optimize=is_optimize_lambo2)
    # cofi
    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, space='buy',
                                    optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, space='buy', optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, space='buy', optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, space='buy', optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, space='buy', default=3.553, optimize=is_optimize_cofi)
    # ?
    dca_min_rsi = IntParameter(35, 75, default=buy_params['dca_min_rsi'], space='buy',
                               optimize=False)
    # Sell
    is_optimize_offset_sell = True
    high_offset_above = DecimalParameter(1.00, 1.10, default=sell_params['high_offset_above'],
                                         space='sell', optimize=is_optimize_offset_sell)
    high_offset_below = DecimalParameter(0.95, 1.05, default=sell_params['high_offset_below'],
                                         space='sell', optimize=is_optimize_offset_sell)
    ############################################################################################################################################################################
    # Custom Stoploss
    is_optimize_stoploss = False
    # Hard stoploss profit
    pHSL = DecimalParameter(-0.500, -0.040, default=-0.08, decimals=3, space='sell',
                            optimize=is_optimize_stoploss, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell',
                             optimize=is_optimize_stoploss, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell',
                             optimize=is_optimize_stoploss, load=True)
    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.080, decimals=3, space='sell',
                             optimize=is_optimize_stoploss, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.040, decimals=3, space='sell',
                             optimize=is_optimize_stoploss, load=True)

    ########################################################################################################################################################
    # Informative Pairs
    ########################################################################################################################################################
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDT', 'BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USD',
                                             'EUR', 'GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))

        return informative_pairs

    ########################################################################################################################################################
    # Informative Pairs
    ########################################################################################################################################################
    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['price_trend_long'] = (
                    dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(
                144).mean())
        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s,
                         inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)
        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s,
                         inplace=True)

        return dataframe

    ########################################################################################################################################################
    # Indicators
    ########################################################################################################################################################
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['stake_currency'] in ['USDT', 'BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        btc_info_tf = self.info_tf_btc_indicators(btc_info_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_info_tf, self.timeframe, self.inf_1h,
                                           ffill=True)
        drop_columns = [f"{s}_{self.inf_1h}" for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(dataframe, btc_base_tf, self.timeframe, self.timeframe,
                                           ffill=True)
        drop_columns = [f"{s}_{self.timeframe}" for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        # lambo2
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        # Pump strength
        dataframe['dema_30'] = ta.DEMA(dataframe, period=30)
        dataframe['dema_200'] = ta.DEMA(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['dema_30'] - dataframe['dema_200']) / dataframe[
            'dema_30']
        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        # Dump Protection
        dataframe = self.pump_dump_protection(dataframe, metadata)

        # 新增威科夫指标
        wyckoff_signals = wyckoff_volume_divergence(dataframe)

        dataframe = pd.concat([dataframe, wyckoff_signals], axis=1)

        # 动态支撑位计算
        dataframe['dynamic_support'] = dataframe['low'].rolling(
            self.dynamic_support_period.value).min() * 0.97
        dataframe['phase'] = np.where(
            dataframe['accumulation_signal'] == 1, 'accumulation',
            np.where(
                dataframe['distribution_signal'] == 1, 'distribution',
                np.where(dataframe['close'] > dataframe['ema_50'], 'uptrend', 'downtrend')
            )
        )

        # RSI
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # dataframe['distribution_signal'] = ((dataframe['high'] == dataframe['price_high']) & (dataframe['volume'] < df['vol_low'])).astype(int)

        return dataframe

    ########################################################################################################################################################
    # Pump Dump Protection
    ########################################################################################################################################################
    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        df36h = dataframe.copy().shift(432)  # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift(288)  # TODO FIXME: This assumes 5m timeframe

        dataframe['volume_mean_short'] = dataframe['volume'].rolling(4).mean()
        dataframe['volume_mean_long'] = df24h['volume'].rolling(48).mean()
        dataframe['volume_mean_base'] = df36h['volume'].rolling(288).mean()

        dataframe['volume_change_percentage'] = (
                    dataframe['volume_mean_long'] / dataframe['volume_mean_base'])
        dataframe['rsi_mean'] = dataframe['rsi'].rolling(48).mean()
        dataframe['pnd_volume_warn'] = np.where(
            (dataframe['volume_mean_short'] / dataframe['volume_mean_long'] > 5.0), -1, 0)

        return dataframe

    ########################################################################################################################################################
    # Custom Stoploss
    ########################################################################################################################################################
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # 动态支撑止损
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_support = dataframe['dynamic_support'].iloc[-1]

        if current_rate < last_support * 0.985:
            return -1  # 立即止损

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PF_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        # Only for hyperopt invalid return
        if sl_profit >= current_profit:
            return -0.99

        return stoploss_from_open(sl_profit, current_profit)

    ########################################################################################################################################################
    # Buy Trend
    ########################################################################################################################################################
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # 新增威科夫积累买入条件
        wyckoff_buy = (
                (dataframe['accumulation_signal'] == 1) &
                (dataframe['volume'] > dataframe['volume'].rolling(
                    50).mean() * self.wyckoff_volume_threshold.value) &
                (dataframe['close'] > dataframe['high'].rolling(20).max())
        )
        dataframe.loc[wyckoff_buy, 'enter_tag'] += 'wyckoff_'
        conditions.append(wyckoff_buy)

        # Lambo2 condition
        lambo2 = (
                bool(self.lambo2_enabled) &
                (dataframe['close'] < (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
                (dataframe['rsi_4'] < int(self.lambo2_rsi_4_limit.value)) &
                (dataframe['rsi_14'] < int(self.lambo2_rsi_14_limit.value)) & (
                            dataframe['volume'] > dataframe['volume'].shift(5).mean() * 1.5)
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2_'
        # 禁用派发阶段的交易
        dataframe.loc[dataframe['distribution_signal'] == 1, 'enter_long'] = 0
        conditions.append(lambo2)

        # Buy1 EWO condition
        ewo = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[
                                           f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_1.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[
                                           f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_1.value))
        )
        dataframe.loc[ewo, 'enter_tag'] += 'eworsi_'
        conditions.append(ewo)

        # Buy2 EWO condition
        ewo2 = (
                (dataframe['rsi_fast'] < 35) &
                (dataframe['close'] < (dataframe[
                                           f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0) &
                (dataframe['close'] < (dataframe[
                                           f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value))
        )
        dataframe.loc[ewo2, 'enter_tag'] += 'ewo2_'
        conditions.append(ewo2)

        # COFI condition
        cofi = (
                (dataframe['open'] < dataframe['ema_8'] * self.buy_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] < self.buy_fastk.value) &
                (dataframe['fastd'] < self.buy_fastd.value) &
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['EWO'] > self.buy_ewo_high.value)
        )
        dataframe.loc[cofi, 'enter_tag'] += 'cofi_'
        conditions.append(cofi)

        # Applying buy conditions
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1

        # Additional conditions to avoid buying
        dont_buy_conditions = []

        # don't buy if there seems to be a Pump and Dump event.
        dont_buy_conditions.append((dataframe['pnd_volume_warn'] < 0.0))

        # BTC price protection
        dont_buy_conditions.append((dataframe['btc_rsi_8_1h'] < 35.0))

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, 'enter_long'] = 0

        return dataframe

    ########################################################################################################################################################
    # Adjust Trade Position
    ########################################################################################################################################################
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: float, max_stake: float, **kwargs):
        # 根据市场阶段调整补仓策略
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        market_phase = dataframe['phase'].iloc[-1]

        if market_phase == 'distribution':
            return None  # 派发阶段禁止补仓

        if market_phase == 'accumulation':
            self.safety_order_volume_scale = 1.8  # 积累阶段加大补仓力度

        return super().adjust_trade_position(trade, current_time, current_rate, current_profit,
                                             min_stake, max_stake, **kwargs)

    ########################################################################################################################################################
    # Sell Trend
    ########################################################################################################################################################
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize 'exit_long' to 0 for all rows
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = 'no_exit'  # Default tag

        # Define primary condition based on volume being greater than 0
        primary_condition = dataframe['volume'] > 0

        # Define exit conditions
        condition_hma50_above = (
                (dataframe['close'] > dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[
                                           f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_above.value)) &
                (dataframe['rsi'] > 50) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        )
        condition_hma50_below = (
                (dataframe['close'] < dataframe['hma_50']) &
                (dataframe['close'] > (dataframe[
                                           f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_below.value)) &
                (dataframe['rsi_fast'] > dataframe['rsi_slow'])
        )

        # Combine conditions with the primary condition
        combined_conditions_above = primary_condition & condition_hma50_above
        combined_conditions_below = primary_condition & condition_hma50_below

        # Apply the conditions to set 'exit_long' to 1
        dataframe.loc[combined_conditions_above, 'exit_long'] = 1
        dataframe.loc[combined_conditions_below, 'exit_long'] = 1

        # Tagging based on which specific condition was met
        dataframe.loc[combined_conditions_above, 'exit_tag'] = 'hma50_above'
        dataframe.loc[combined_conditions_below, 'exit_tag'] = 'hma50_below'
        # 新增威科夫派发卖出条件
        distribution_exit = (
                (dataframe['distribution_signal'] == 1) &
                (dataframe['rsi'] > 65) &
                (dataframe['close'] < dataframe['ema_20'])
        )
        dataframe.loc[distribution_exit, 'exit_long'] = 1
        dataframe.loc[distribution_exit, 'exit_tag'] = 'wyckoff_distribution'

        # 动态支撑止损
        support_exit = dataframe['close'] < dataframe['dynamic_support']
        dataframe.loc[support_exit, 'exit_long'] = 1
        return dataframe

    ########################################################################################################################################################
    # Confirm Trade Exit
    ########################################################################################################################################################
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        if trade and trade.exit_reason:
            trade.exit_reason = exit_reason + "_" + trade.enter_tag

        return True

    ########################################################################################################################################################
    # Custom to Sell unclog
    ########################################################################################################################################################
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # Sell any positions at a loss if they are held for more than 9.6 hours (0.4 days).
        time_held = current_time - trade.open_date_utc
        time_held_in_hours = time_held.total_seconds() / 3600  # Convert seconds to hours

        if current_profit < -0.04 and time_held_in_hours >= 6.5:
            return 'unclog'

    ########################################################################################################################################################
    # Trade Protections
    ########################################################################################################################################################
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
