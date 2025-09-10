"""
Freqtrade Strategy Template.
Adapted to test the strategy in futures markets with leverage and enhanced risk management.
"""
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from freqtrade.strategy import IntParameter, DecimalParameter
from typing import Dict, List

class NOTankAi15(IStrategy):
    """
    Strategy intended for 15m timeframe.
    Adapted for futures markets with leverage and enhanced risk management.
    """

    INTERFACE_VERSION = 3

    # Strategy Parameters
    can_short = True
    timeframe = '15m'

    # ROI table
    minimal_roi = {
        "0": 0.1,
        "30": 0.05,
        "60": 0.025,
        "120": 0
    }

    # Stoploss
    stoploss = -0.10  # Adjusted for futures to be more conservative

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Hyperparameters for optimization
    buy_rsi = IntParameter(20, 80, default=30, space='buy', optimize=True)
    sell_rsi = IntParameter(60, 100, default=70, space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Add indicators used by the strategy.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)  # Average True Range for volatility-based position sizing
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Define conditions for entering trades.
        """
        # Long condition
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['ema50'] > dataframe['ema200'])
            ),
            'enter_long'
        ] = 1

        # Short condition
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value) &
                (dataframe['ema50'] < dataframe['ema200'])
            ),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Define conditions for exiting trades.
        """
        # Exit long condition
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi.value)
            ),
            'exit_long'
        ] = 1

        # Exit short condition
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value)
            ),
            'exit_short'
        ] = 1

        return dataframe

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        """
        Adjust leverage dynamically based on ATR (volatility).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe['atr'].iloc[-1]

        # Higher ATR (volatility) = lower leverage
        # Adjust the multiplier as needed
        dynamic_leverage = max(1.0, min(max_leverage, 2.0 / atr))

        return dynamic_leverage

    def custom_stake_amount(self, pair: str, current_time, current_rate: float, proposed_stake: float, min_stake: float, max_stake: float, **kwargs) -> float:
        """
        Dynamically adjust stake size based on ATR (volatility).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        atr = dataframe['atr'].iloc[-1]

        # Lower ATR (volatility) = higher stake size
        dynamic_stake = max(min_stake, min(max_stake, proposed_stake * (2.0 / atr)))

        return dynamic_stake
