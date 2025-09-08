# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, Trade, informative, IntParameter, CategoricalParameter
import talib.abstract as ta
from technical import qtpylib


class ZaratustraV28(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    exit_profit_only = True
    position_adjustment_enable = True

    # ROI table:
    minimal_roi = {
        "0": 0.328,
        "167": 0.151,
        "406": 0.031,
        "1063": 0
    }

    # Stoploss:
    stoploss = -0.308

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.073
    trailing_stop_positive_offset = 0.081
    trailing_only_offset_is_reached = True
    
    # Max Open Trades:
    max_open_trades = 1

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI: Momentum and overbought/oversold indicator
        dataframe['RSI'] = ta.RSI(dataframe)

        # ADX and DI components: measure trend strength and direction
        dataframe['ADX'] = ta.ADX(dataframe)
        dataframe['PDI'] = ta.PLUS_DI(dataframe)
        dataframe['MDI'] = ta.MINUS_DI(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata['pair'] != 'BTC/USDT:USDT':
            return dataframe
        
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['RSI'], 70))
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['RSI'], 30))
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_below(dataframe['RSI'], 70))
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['RSI'], 30))
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'Bearish trend')

        return dataframe