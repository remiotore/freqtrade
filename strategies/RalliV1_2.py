

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


buy_params = {
      "base_nb_candles_buy": 14,
      "ewo_high": 2.327,
      "ewo_high_2": -2.327,
      "ewo_low": -20.988,
      "low_offset": 0.975,
      "low_offset_2": 0.955,
      "rsi_buy": 60,
      "rsi_buy_2": 45
    }

sell_params = {
      "base_nb_candles_sell": 24,
      "high_offset": 0.991,
      "high_offset_2": 0.997
    }

def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif



class RalliV1_2(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 0.04,
        "40": 0.032,
        "87": 0.018,
        "201": 0
    }

    stoploss = -0.3

    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset_2'], space='buy', optimize=True)        
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params['high_offset_2'], space='sell', optimize=True)        

    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)

    ewo_high_2 = DecimalParameter(
        -6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=True)       
    
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)
    rsi_buy_2 = IntParameter(30, 70, default=buy_params['rsi_buy_2'], space='buy', optimize=True)

    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    timeframe = '5m'
    inf_1h = '1h'

    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
        'main_plot': {
            'ma_buy': {'color': 'orange'},
            'ma_sell': {'color': 'orange'},
        },
    }

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2
        }
    ]

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        if ((rate > last_candle['close'])) : return False

        return True

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        else:
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            buy_signal = dataframe.loc[dataframe['date'] < trade_open_date]
            if not buy_signal.empty:
                buy_signal_candle = buy_signal.iloc[-1]
                buy_tag = buy_signal_candle['buy_tag'] if buy_signal_candle['buy_tag'] != '' else 'empty'
        buy_tags = buy_tag.split()

        current_profit = trade.calc_profit_ratio(current_rate)

        if (current_profit <= -0.3):
            return 'stoploss ( ' + buy_tag + ')'

        trade_time_40 = trade.open_date_utc + timedelta(minutes=40)
        trade_time_87 = trade.open_date_utc + timedelta(minutes=87)
        trade_time_201 = trade.open_date_utc + timedelta(minutes=201)

        if (current_time < trade_time_40):
            if(current_profit >= 0.04):
                return 'roi 0 ( ' + buy_tag + ')'
        elif (current_time >= trade_time_40) and (current_time < trade_time_87):
            if(current_profit >= 0.032):
                return 'roi 40 ( ' + buy_tag + ')'
        elif (current_time >= trade_time_87) and (current_time < trade_time_201):
            if(current_profit >= 0.018):
                return 'roi 87 ( ' + buy_tag + ')'
        elif (current_time >= trade_time_201):
            if(current_profit >= 0):
                return 'roi 201 ( ' + buy_tag + ')'

        last_candle = dataframe.iloc[-1]

        sell_1 = (   
            (last_candle['hma_50'] > last_candle['ema_100'])&
            (last_candle['close'] > last_candle['sma_9'])&
            (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset_2.value)) &
            (last_candle['rsi_fast'] > last_candle['rsi_slow'])
        )

        sell_2 = (   
            (last_candle['close'] < last_candle['ema_100'])&
            (last_candle['close'] > (last_candle[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
            (last_candle['rsi_fast'] > last_candle['rsi_slow'])       
        )

        sell_3 = (last_candle['rsi'] > 45 )

        sell_4 = (last_candle['hma_50'] < last_candle['ema_100'])

        if (sell_1 and sell_3):
            return 'sell 1-3 ( ' + buy_tag + ')'
        elif (sell_1 and sell_4):
            return 'sell 1-4 ( ' + buy_tag + ')'
        elif (sell_2 and sell_3):
            return 'sell 2-3 ( ' + buy_tag + ')'
        elif (sell_2 and sell_4):
            return 'sell 2-4 ( ' + buy_tag + ')'

        return None
    
    use_custom_stoploss = True
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle = df.iloc[-1].squeeze()
        
        if current_profit < 0.001 and current_time - timedelta(minutes=140) > trade.open_date_utc:
            return -0.005

        return 1


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)
        dataframe['hma_9'] = qtpylib.hull_moving_average(dataframe['close'], window=9)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)          
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['sma_9'] = ta.SMA(dataframe, timeperiod=9)
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)

        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        dataframe.loc[:, 'buy_tag'] = ''

        buy_1 = (
            (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] < dataframe['ema_100'])&
                (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
                (dataframe['rsi_fast'] <35)&
                (dataframe['rsi_fast'] >4)&
                (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi'] < self.rsi_buy_2.value) &
                (dataframe['volume'] > 0)&
                (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy_1, 'buy_tag'] += '1 '
        conditions.append(buy_1)

        buy_2 = (
            (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] < dataframe['ema_100'])&
            (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
            (dataframe['rsi_fast'] <35)&
            (dataframe['rsi_fast'] >4)&
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
            (dataframe['EWO'] > self.ewo_high_2.value) &
            (dataframe['rsi'] < self.rsi_buy_2.value) &
            (dataframe['volume'] > 0)&
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))&
            (dataframe['rsi']<25)
        )
        dataframe.loc[buy_2, 'buy_tag'] += '2 '
        conditions.append(buy_2)

        buy_3 = (
            (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] < dataframe['ema_100'])&
            (dataframe['sma_9'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'])&
            (dataframe['rsi_fast'] < 35)&
            (dataframe['rsi_fast'] >4)&
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['volume'] > 0)&
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy_3, 'buy_tag'] += '3 '
        conditions.append(buy_3)

        buy_4 = (
            (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] > dataframe['ema_100'])&
            (dataframe['rsi_fast'] <35)&
            (dataframe['rsi_fast'] >4)&
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] > self.ewo_high.value) &
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['volume'] > 0)&
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))
        )
        dataframe.loc[buy_4, 'buy_tag'] += '4 '
        conditions.append(buy_4)

        buy_5 = (
            (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] > dataframe['ema_100'])&
            (dataframe['rsi_fast'] <35)&
            (dataframe['rsi_fast'] >4)&
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)) &
            (dataframe['EWO'] > self.ewo_high_2.value) &
            (dataframe['rsi'] < self.rsi_buy.value) &
            (dataframe['volume'] > 0)&
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value))&
            (dataframe['rsi']<25)
        )
        dataframe.loc[buy_5, 'buy_tag'] += '5 '
        conditions.append(buy_5)

        buy_6 = (
            (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] > dataframe['ema_100'])&
            (dataframe['rsi_fast'] < 35)&
            (dataframe['rsi_fast'] >4)&
            (dataframe['close'] < (dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
            (dataframe['EWO'] < self.ewo_low.value) &
            (dataframe['volume'] > 0)&
            (dataframe['close'] < (dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) 
        )
        dataframe.loc[buy_6, 'buy_tag'] += '6 '
        conditions.append(buy_6)

        if conditions:
            dataframe.loc[:, 'buy'] = reduce(lambda x, y: x | y, conditions)

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'sell'] = 0

        return dataframe
