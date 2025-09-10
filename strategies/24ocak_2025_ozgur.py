# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import DecimalParameter, CategoricalParameter
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
import logging

class Vedat22OcatRevized(IStrategy):
    """
    Revized version of the strategy with improved risk management, dynamic stop loss,
    multi-timeframe analysis, and additional filters for better signal quality.
    """

    # Strategy configuration
    INTERFACE_VERSION: int = 3
    timeframe = '15m'
    can_short: bool = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Risk management
    stoploss = -0.05  # Initial stop loss (5%)
    minimal_roi = {"0": 0.10}  # Take profit at 10%
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.15
    max_entry_position_adjustment = 3
    max_dca_multiplier = 4.6

    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Hyperparameters (optional, for optimization)
    buy_rsi = DecimalParameter(30, 70, default=50, space='buy', optimize=True)
    sell_rsi = DecimalParameter(70, 90, default=70, space='sell', optimize=True)
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, space='sell', optimize=True)

    def informative_pairs(self):
        # Add higher timeframe for trend confirmation
        return [(self.pair, "1h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['macd'] = ta.MACD(dataframe)['macd']
        dataframe['macdsignal'] = ta.MACD(dataframe)['macdsignal']
        dataframe['macdhist'] = ta.MACD(dataframe)['macdhist']
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Multi-timeframe indicators (1h)
        informative = self.dp.get_pair_dataframe(pair=self.pair, timeframe="1h")
        dataframe['ema200_1h'] = ta.EMA(informative, timeperiod=200)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry conditions
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['mfi'] < 30) &
                (dataframe['ema9'] > dataframe['ema26']) &
                (dataframe['ema26'] > dataframe['ema50']) &
                (dataframe['close'] > dataframe['ema200_1h'])  # Trend confirmation
            ),
            'enter_long'] = 1

        # Short entry conditions
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value) &
                (dataframe['mfi'] > 70) &
                (dataframe['ema9'] < dataframe['ema26']) &
                (dataframe['ema26'] < dataframe['ema50']) &
                (dataframe['close'] < dataframe['ema200_1h'])  # Trend confirmation
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit conditions
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) |
                (dataframe['ema9'] < dataframe['ema26'])  # Trend reversal
            ),
            'exit_long'] = 1

        # Short exit conditions
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) |
                (dataframe['ema9'] > dataframe['ema26'])  # Trend reversal
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        # Dynamic stop loss based on ATR
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        atr_stoploss = last_candle['atr'] * self.atr_multiplier.value
        return -atr_stoploss / current_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        # Adjust stake amount for DCA
        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Union[Optional[float], Optional[str]]:
        # DCA logic
        if current_profit < -0.10:  # Add to position if loss exceeds 10%
            filled_entries = trade.select_filled_orders(trade.entry_side)
            stake_amount = filled_entries[0].stake_amount * 1.5  # Increase stake by 50%
            return stake_amount, 'dca_adjustment'
        return None