# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Union, Tuple
import talib.abstract as ta
from technical import qtpylib



class ZaratustraV12(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = False
    exit_profit_only = True
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.25

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = -1

    # Hyperparams
    base_leverage = IntParameter(1, 50, default=10, space="buy")

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return self.base_leverage.value
        
    @informative('5m')
    @informative('15m')
    @informative('30m')
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        return dataframe

    @informative('5m', 'BTC/USDT:USDT', fmt='{column}_{base}_{timeframe}')
    def populate_indicators_btc_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['close_btc_5m'].shift(3) < dataframe['close_btc_5m'].shift(2)) &
                (dataframe['close_btc_5m'].shift(2) < dataframe['close_btc_5m'].shift(1)) &
                (dataframe['close_btc_5m'].shift(1) < dataframe['close_btc_5m']) &

                (dataframe['adx_30m'] > dataframe['mdi_30m']) &
                (dataframe['adx_15m'] > dataframe['mdi_15m']) &
                (dataframe['adx_5m']  > dataframe['mdi_5m']) &

                (dataframe['pdi_30m'] > dataframe['mdi_30m']) &
                (dataframe['pdi_15m'] > dataframe['mdi_15m']) &
                (dataframe['pdi_5m']  > dataframe['mdi_5m'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend enter')

        dataframe.loc[
            (
                (dataframe['close_btc_5m'].shift(3) > dataframe['close_btc_5m'].shift(2)) &
                (dataframe['close_btc_5m'].shift(2) > dataframe['close_btc_5m'].shift(1)) &
                (dataframe['close_btc_5m'].shift(1) > dataframe['close_btc_5m']) &

                (dataframe['adx_30m'] > dataframe['pdi_30m']) &
                (dataframe['adx_15m'] > dataframe['pdi_15m']) &
                (dataframe['adx_5m']  > dataframe['pdi_5m']) &

                (dataframe['mdi_30m'] > dataframe['pdi_30m']) &
                (dataframe['mdi_15m'] > dataframe['pdi_15m']) &
                (dataframe['mdi_5m']  > dataframe['pdi_5m'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe