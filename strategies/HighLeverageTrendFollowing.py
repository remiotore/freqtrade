from functools import reduce
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from datetime import datetime
from typing import Optional

class HighLeverageTrendFollowing(IStrategy):
    """
    High Leverage Trend Following Strategy

    This strategy maintains the core logic of the original TrendFollowingStrategy
    that achieved an 8% win rate, but adds optimized leverage settings to amplify returns.

    Key features:
    - Simple trend following using EMA crossovers
    - OBV confirmation for stronger signals
    - Dynamic leverage based on coin risk profile
    - Aggressive take profit targets for high leverage trading
    - Tight stop loss for capital protection
    """
    INTERFACE_VERSION: int = 3
    can_short = True  # Explicitly enable shorting
    trading_mode = "futures"
    margin_mode = "isolated"

    # Leverage settings
    leverage_optimization = True  # Enable dynamic leverage optimization
    max_leverage = 20  # Maximum leverage to use (more conservative than 100x)

    # ROI table - Balanced take profits for high leverage
    minimal_roi = {
        "0": 0.10,    # 10% profit immediately
        "30": 0.07,   # 7% profit after 30 minutes
        "60": 0.05,   # 5% profit after 60 minutes
        "120": 0.03   # 3% profit after 120 minutes
    }

    # Stoploss:
    stoploss = -0.15  # Slightly wider stop loss for fewer premature exits

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.02  # Start trailing once 2% profit is reached
    trailing_stop_positive_offset = 0.03  # Offset from current price
    trailing_only_offset_is_reached = True  # Only trail once offset is reached

    timeframe = "15m"

    # Strategy parameters
    ema_short = 9
    ema_long = 21
    obv_period = 14

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate OBV - same as original TrendFollowingStrategy
        dataframe['obv'] = ta.OBV(dataframe['close'], dataframe['volume'])

        # Add trend following indicator - same as original TrendFollowingStrategy
        dataframe['trend'] = dataframe['close'].ewm(span=20, adjust=False).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add trend following buy signals - same as original TrendFollowingStrategy
        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) &
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)),
            'enter_long'] = 1

        # Add trend following sell signals - same as original TrendFollowingStrategy
        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) &
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Add trend following exit signals for long positions - same as original TrendFollowingStrategy
        dataframe.loc[
            (dataframe['close'] < dataframe['trend']) &
            (dataframe['close'].shift(1) >= dataframe['trend'].shift(1)) &
            (dataframe['obv'] > dataframe['obv'].shift(1)),
            'exit_long'] = 1

        # Add trend following exit signals for short positions - same as original TrendFollowingStrategy
        dataframe.loc[
            (dataframe['close'] > dataframe['trend']) &
            (dataframe['close'].shift(1) <= dataframe['trend'].shift(1)) &
            (dataframe['obv'] < dataframe['obv'].shift(1)),
            'exit_short'] = 1

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str) -> float:
        """
        Customize leverage for each pair based on volatility and risk management
        """
        # High-cap coins (lower risk) - use higher leverage
        high_cap_coins = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
        # Mid-cap coins (medium risk)
        mid_cap_coins = ['SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT', 'DOT/USDT:USDT']

        # Set leverage based on coin category
        if pair in high_cap_coins:
            # Higher leverage for high-cap coins (less volatile)
            return min(20, max_leverage)  # Up to 20x for high-cap coins
        elif pair in mid_cap_coins:
            # Medium leverage for mid-cap coins
            return min(15, max_leverage)  # Up to 15x for mid-cap coins
        else:
            # Lower leverage for other coins (more volatile)
            return min(10, max_leverage)  # Up to 10x for other coins
