# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter, DecimalParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Union, Tuple
import talib.abstract as ta
from technical import qtpylib


class ZaratustraV25(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True

    # DCA参数
    max_dca_orders = 2
    max_dca_multiplier = 2.0
    min_dca_profit = -0.05
    max_dca_profit = -0.1

    # 分批减仓参数
    position_adjustment_enable = True
    max_entry_position_adjustment = -1
    max_exit_position_adjustment = 3
    exit_portion_size = 0.3

    # ROI table:
    minimal_roi = {
        "0": 0.147,
        "32": 0.058,
        "48": 0.037,
        "63": 0
    }

    # Stoploss:
    stoploss = -0.313

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.064
    trailing_stop_positive_offset = 0.068
    trailing_only_offset_is_reached = False
    
    # Max Open Trades:
    max_open_trades = -1

    @property
    def plot_config(self):
        return {
            'main_plot': {
                'BBU': {},
                'BBM': {},
                'BBL': {},
            },
            'subplots': {
                'Momentum': {
                    'RSI': {'color': 'blue'},
                    'CCI': {'color': 'orange'},
                },
            },
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI and CCI
        dataframe['RSI'] = ta.RSI(dataframe)
        dataframe['CCI'] = ta.CCI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['BBU'] = bollinger['upper']
        dataframe['BBM'] = bollinger['mid']
        dataframe['BBL'] = bollinger['lower']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['RSI'] < 30) &
                (dataframe['CCI'] < -100) &
                (dataframe['close'] < dataframe['BBL'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Strong Trend Long')

        dataframe.loc[
            (
                (dataframe['RSI'] > 30) &
                (dataframe['CCI'] > 100) &
                (dataframe['close'] > dataframe['BBU'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Strong Trend Short')


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['RSI'] > 70) &
                (dataframe['CCI'] > 100)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'Long Exit')
        
        dataframe.loc[
            (
                (dataframe['RSI'] < 30) &
                (dataframe['CCI'] < -100)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'Short Exit')


        return dataframe
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """
        处理DCA和分批减仓
        FVGAdvancedStrategy_V2 - Rico X
        """
        if current_profit <= self.min_dca_profit and current_profit >= self.max_dca_profit:
            filled_entries = trade.select_filled_orders('enter_short' if trade.is_short else 'enter_long')
            count_entries = len(filled_entries)
            
            if count_entries < self.max_dca_orders:
                stake_amount = trade.stake_amount * self.max_dca_multiplier
                return stake_amount

        elif current_profit > 0.05:
            filled_exits = trade.select_filled_orders('exit_short' if trade.is_short else 'exit_long')
            count_exits = len(filled_exits)
            
            if count_exits < self.max_exit_position_adjustment:
                return -(trade.amount * self.exit_portion_size)

        return None

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, entry_tag: str, **kwargs) -> float:
        return proposed_stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return min(3, max_leverage)  # Use up to 3x leverage if available
