from datetime import datetime
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, stoploss_from_open, merge_informative_pair
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

def EWO(dataframe, ema_length=5, ema2_length=35):
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    return (ema1 - ema2) / dataframe['low'] * 100


class My_Updated_Strategy(IStrategy):
    """
    Updated Strategy with the latest hyperopt parameters.
    """

    # Buy hyperspace parameters (from your hyperopt results)
    buy_params = {
        "buy_cti_32": -0.8,
        "buy_rsi_32": 25,
        "buy_rsi_fast_32": 55,
        "buy_sma15_32": 0.979,
    }

    # Sell hyperspace parameters (from your hyperopt results)
    sell_params = {
        "sell_fastx": 56,
        "sell_loss_cci": 134,
        "sell_loss_cci_profit": 0.0,
    }

    # ROI table (from your hyperopt results)
    minimal_roi = {
        "0": 1
    }

    # Stoploss (from your hyperopt results)
    stoploss = -0.25

    # Trailing stop settings (from your hyperopt results)
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # Timeframe for the strategy
    timeframe = '5m'

    # Additional strategy settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count = 240
    process_only_new_candles = True

    # Informative pairs
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add indicators to the dataframe.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe['close'], length=20)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define buy conditions based on hyperopt results.
        """
        conditions = []
        conditions.append(dataframe['cti'] < self.buy_params['buy_cti_32'])
        conditions.append(dataframe['rsi'] < self.buy_params['buy_rsi_32'])
        conditions.append(dataframe['rsi'] > self.buy_params['buy_rsi_fast_32'])
        conditions.append(dataframe['sma15'] < self.buy_params['buy_sma15_32'])

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define sell conditions based on hyperopt results.
        """
        conditions = []
        conditions.append(dataframe['rsi'] > self.sell_params['sell_fastx'])
        conditions.append(dataframe['cti'] > self.sell_params['sell_loss_cci'])

        dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic based on trailing stop parameters.
        """
        sl_profit = self.trailing_stop_positive_offset
        if current_profit > sl_profit:
            return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)
        return self.stoploss
