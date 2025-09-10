
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)



def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class EI3v2_tag_cofi_green_Short(IStrategy):
    INTERFACE_VERSION = 3
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
    can_short = False

    enter_short_params = {
        "base_nb_candles_enter_short": 12,
        "rsi_enter_short": 42,
        "ewo_high": 3.001,
        "ewo_low": -10.289,
        "low_offset": 0.987,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 61,
        "lambo2_rsi_4_limit": 56,
        "enter_short_adx": 20,
        "enter_short_fastd": 80,
        "enter_short_fastk": 78,
        "enter_short_ema_cofi": 0.98,
        "enter_short_ewo_high": 4.179
    }

    exit_short_params = {
        "base_nb_candles_exit_short": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01
    }

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
        "0": 0.99,
        
    }

    stoploss = -0.99

    base_nb_candles_enter_short = IntParameter(8, 20, default=enter_short_params['base_nb_candles_enter_short'], space='enter_short', optimize=False)
    base_nb_candles_exit_short = IntParameter(8, 20, default=exit_short_params['base_nb_candles_exit_short'], space='exit_short', optimize=False)
    low_offset = DecimalParameter(0.985, 0.995, default=enter_short_params['low_offset'], space='enter_short', optimize=True)
    high_offset = DecimalParameter(1.005, 1.015, default=exit_short_params['high_offset'], space='exit_short', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=exit_short_params['high_offset_2'], space='exit_short', optimize=True)

    lambo2_ema_14_factor = DecimalParameter(0.8, 1.2, decimals=3,  default=enter_short_params['lambo2_ema_14_factor'], space='enter_short', optimize=True)
    lambo2_rsi_4_limit = IntParameter(5, 60, default=enter_short_params['lambo2_rsi_4_limit'], space='enter_short', optimize=True)
    lambo2_rsi_14_limit = IntParameter(5, 60, default=enter_short_params['lambo2_rsi_14_limit'], space='enter_short', optimize=True)

    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(-20.0, -8.0,default=enter_short_params['ewo_low'], space='enter_short', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=enter_short_params['ewo_high'], space='enter_short', optimize=True)
    rsi_enter_short = IntParameter(30, 70, default=enter_short_params['rsi_enter_short'], space='enter_short', optimize=False)

    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    is_optimize_cofi = False
    enter_short_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97 , optimize = is_optimize_cofi)
    enter_short_fastk = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    enter_short_fastd = IntParameter(20, 30, default=20, optimize = is_optimize_cofi)
    enter_short_adx = IntParameter(20, 30, default=30, optimize = is_optimize_cofi)
    enter_short_ewo_high = DecimalParameter(2, 12, default=3.553, optimize = is_optimize_cofi)

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

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
            'ma_enter_short': {'color': 'orange'},
            'ma_exit_short': {'color': 'orange'},
        },
    }


    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):

        if current_profit < -0.04 and (current_time - trade.open_date_utc).days >= 4:
            return 'unclog'


    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]

        if self.config['stake_currency'] in ['USDC:USDC','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDC:USDC"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1h))

        return informative_pairs

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


        dataframe['price_trend_short'] = (dataframe['close'].rolling(8).mean() / dataframe['close'].shift(8).rolling(144).mean())


        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        dataframe['rsi_8'] = ta.RSI(dataframe, timeperiod=8)


        ignore_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        dataframe.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.config['stake_currency'] in ['USDC','BUSD']:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDC"

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

        for val in self.base_nb_candles_enter_short.range:
            dataframe[f'ma_enter_short_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_exit_short.range:
            dataframe[f'ma_exit_short_{val}'] = ta.EMA(dataframe, timeperiod=val)

        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)


        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['rsi_4'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['zema_30'] = ftt.zema(dataframe, period=30)
        dataframe['zema_200'] = ftt.zema(dataframe, period=200)
        dataframe['pump_strength'] = (dataframe['zema_30'] - dataframe['zema_200']) / dataframe['zema_30']

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)



        dataframe = self.pump_dump_protection(dataframe, metadata)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = []
        short_conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        lambo2 = (


            (dataframe['close'] > (dataframe['ema_14'] * self.lambo2_ema_14_factor.value)) &
            (dataframe['rsi_4'] > int(self.lambo2_rsi_4_limit.value)) &
            (dataframe['rsi_14'] > int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, 'enter_tag'] += 'lambo2_'
        short_conditions.append(lambo2)

        buy1ewo = (
                (dataframe['rsi_fast'] >65)&
                (dataframe['close'] > (dataframe[f'ma_enter_short_{self.base_nb_candles_enter_short.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] < self.ewo_high.value) &
                (dataframe['rsi'] > self.rsi_enter_short.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] > (dataframe[f'ma_exit_short_{self.base_nb_candles_exit_short.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy1ewo, 'enter_tag'] += 'buy1eworsi_'
        short_conditions.append(buy1ewo)

        buy2ewo = (
                (dataframe['rsi_fast'] > 65)&
                (dataframe['close'] > (dataframe[f'ma_enter_short_{self.base_nb_candles_enter_short.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_low.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] > (dataframe[f'ma_exit_short_{self.base_nb_candles_exit_short.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy2ewo, 'enter_tag'] += 'buy2ewo'
        short_conditions.append(buy2ewo)

        is_cofi = (
                (dataframe['open'] > dataframe['ema_8'] * self.enter_short_ema_cofi.value) &
                (qtpylib.crossed_above(dataframe['fastk'], dataframe['fastd'])) &
                (dataframe['fastk'] > self.enter_short_fastk.value) &
                (dataframe['fastd'] > self.enter_short_fastd.value) &
                (dataframe['adx'] > self.enter_short_adx.value) &
                (dataframe['EWO'] < self.enter_short_ewo_high.value)
            )
        dataframe.loc[is_cofi, 'enter_tag'] += 'cofi_'
        short_conditions.append(is_cofi)

        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, short_conditions),
                'enter_short'] = 1
            
        else:
            dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'no_long_entry')

        dont_enter_short_conditions = []

        dont_enter_short_conditions.append((dataframe['pnd_volume_warn'] < 0.0))

        dont_enter_short_conditions.append((dataframe['btc_rsi_8_1h'] > 65.0))

        if dont_enter_short_conditions:
            for condition in dont_enter_short_conditions:
                dataframe.loc[condition, 'enter_short'] = 0
        
        ficcion_long = (
                    (dataframe['rsi_fast'] < 0)
            )
        dataframe.loc[ficcion_long, 'enter_tag'] += 'ficcion_long'
        long_conditions.append(ficcion_long)

        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, short_conditions),
                'enter_long'] = 1
        else:
            dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'no_long_entry')

        return dataframe



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (   (dataframe['close']<dataframe['hma_50'])&
                (dataframe['close'] < (dataframe[f'ma_exit_short_{self.base_nb_candles_exit_short.value}'] * self.high_offset_2.value)) &
                (dataframe['rsi']<50)&
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']<dataframe['rsi_slow'])

            )
            |
            (
                (dataframe['close']>dataframe['hma_50'])&
                (dataframe['close'] < (dataframe[f'ma_exit_short_{self.base_nb_candles_exit_short.value}'] * self.high_offset.value)) &
                (dataframe['volume'] > 0)&
                (dataframe['rsi_fast']<dataframe['rsi_slow'])
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_short'
            ]=1


        return dataframe
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        trade.exit_reason = exit_reason + "_" + trade.enter_tag

        return True

def pct_change(a, b):
    return (b - a) / a

class EI3v2_tag_cofi_dca_green_Future_Short(EI3v2_tag_cofi_green_Future_Short):
   

    initial_safety_order_trigger = -0.018
    max_safety_orders = 8
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.4

    enter_short_params = {
        "dca_max_rsi": 65,
    }

    enter_short_params.update(EI3v2_tag_cofi_green_Future_Short.enter_short_params)

    dca_min_rsi = IntParameter(35, 75, default=enter_short_params['dca_max_rsi'], space='enter_short', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None


        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()
        previous_candle = dataframe.iloc[-2].squeeze()
        if last_candle['close'] < previous_candle['close']:
            return None

        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'enter_short':
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

